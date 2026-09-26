"""Pair features for (S1, candidate) — the SAME function serves S1-vs-R and R-vs-R pairs.

Inputs are normalised frames (normalize.Normalizer.transform) keyed by entity_id.
Groups:
  name strings   rapidfuzz ratios on the normalised name / core / alias, letter-bag similarity (scrambles)
  name tokens    overlap counts, jaccards (tokens, skeletons), unexplained extra/missing words, legal-form relation
  numbers        set relation, house-number relation (equal / truncation / small shift / 1-edit / other), unit codes
  address text   token-set ratio, word jaccard, missing flags
  blocking       retrieval score/rank, arms, expansion, reverse preference (copied through from the candidate pool)
  context        rank / score ratio within the S1's candidate list, sibling-group size and agreement
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

REL_CODES = {"equal": 0, "subset": 1, "trunc": 2, "overlap": 3, "missing": 4, "disjoint": 5}


def _cp(a, b, scorer) -> np.ndarray:
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _join_side(P: pl.DataFrame, N: pl.DataFrame, key: str, sfx: str) -> pl.DataFrame:
    cols = [c for c in N.columns if c not in ("country",)]
    return P.join(N.select(cols).rename({c: f"{c}{sfx}" for c in cols if c != "entity_id"}).rename({"entity_id": key}), on=key, how="left")


def house_relation(a: str | None, b: str | None) -> int:
    """0 equal, 1 truncation (one digit dropped), 2 small shift (<=25), 3 one edit, 4 other, 5 missing."""
    if not a or not b:
        return 5
    if a == b:
        return 0
    if abs(len(a) - len(b)) == 1 and (a.endswith(b) or b.endswith(a) or a.startswith(b) or b.startswith(a)):
        return 1
    if a.isdigit() and b.isdigit() and abs(int(a) - int(b)) <= 25:
        return 2
    if Levenshtein.distance(a, b) == 1:
        return 3
    return 4


def pair_features(P: pl.DataFrame, QN: pl.DataFrame, RN: pl.DataFrame) -> pl.DataFrame:
    """P: candidate pool with at least [s1, r] (+ blocking columns). QN/RN: normalised frames of the two sides."""
    d = _join_side(_join_side(P, QN, "s1", "_s"), RN, "r", "_r").sort(["s1", "r"])  # deterministic order (rank ties)
    j = lambda c: pl.col(c).list.join(" ")
    d = d.with_columns(j("nw_s").alias("nm_s"), j("nw_r").alias("nm_r"), j("core_s").alias("co_s"), j("core_r").alias("co_r"),
                       j("alias_r").alias("al_r"), j("aw_s").alias("ad_s_txt"), j("aw_r").alias("ad_r_txt"),
                       pl.col("core_s").list.join("").str.split("").list.sort().list.join("").alias("bag_s"),
                       pl.col("core_r").list.join("").str.split("").list.sort().list.join("").alias("bag_r"))
    nm_s, nm_r, co_s, co_r = (d[c].to_list() for c in ("nm_s", "nm_r", "co_s", "co_r"))
    feats = {
        "nm_ratio": _cp(nm_s, nm_r, fuzz.ratio), "nm_tsort": _cp(nm_s, nm_r, fuzz.token_sort_ratio),
        "nm_tset": _cp(nm_s, nm_r, fuzz.token_set_ratio), "nm_partial": _cp(nm_s, nm_r, fuzz.partial_ratio),
        "co_jw": _cp(co_s, co_r, JaroWinkler.normalized_similarity), "co_tset": _cp(co_s, co_r, fuzz.token_set_ratio),
        "co_alias_tset": _cp(co_s, d["al_r"].to_list(), fuzz.token_set_ratio),
        "bag_ratio": _cp(d["bag_s"].to_list(), d["bag_r"].to_list(), fuzz.ratio),
        "ad_tset": _cp(d["ad_s_txt"].to_list(), d["ad_r_txt"].to_list(), fuzz.token_set_ratio),
    }
    d = d.with_columns(*[pl.Series(k, v) for k, v in feats.items()])
    inter = lambda a, b: pl.col(a).list.set_intersection(pl.col(b)).list.len()
    diff = lambda a, b: pl.col(a).list.set_difference(pl.col(b))
    d = d.with_columns(
        pl.col("core_s").list.len().alias("n_core_s"), pl.col("core_r").list.len().alias("n_core_r"),
        inter("core_s", "core_r").alias("n_core_common"),
        diff("core_r", "core_s").list.len().alias("n_extra_r"), diff("core_s", "core_r").list.len().alias("n_missing_r"),
        # extra words not explained by a skeleton match (transliteration/typo) = likely decoy qualifier
        diff("sk_r", "sk_s").list.len().alias("n_extra_sk"), diff("sk_s", "sk_r").list.len().alias("n_missing_sk"),
        (inter("sk_s", "sk_r") / pl.max_horizontal(pl.col("sk_s").list.len(), pl.col("sk_r").list.len(), pl.lit(1))).alias("sk_jacc"),
        (inter("nw_s", "nw_r") / pl.max_horizontal(pl.col("nw_s").list.len(), pl.col("nw_r").list.len(), pl.lit(1))).alias("nw_jacc"),
        (inter("aw_s", "aw_r") / pl.max_horizontal(pl.col("aw_s").list.len(), pl.col("aw_r").list.len(), pl.lit(1))).alias("aw_jacc"),
        (pl.col("aw_s").list.tail(3).list.set_intersection(pl.col("aw_r").list.tail(3)).list.len() /
         pl.max_horizontal(pl.col("aw_s").list.tail(3).list.unique().list.len(), pl.col("aw_r").list.tail(3).list.unique().list.len(), pl.lit(1))).alias("aw_tail_jacc"),
        pl.when((pl.col("core_s").list.first().is_not_null()) & (pl.col("core_r").list.first().is_not_null()) &
                (pl.col("core_s").list.first() == pl.col("core_r").list.first()) &
                (pl.col("core_s").list.first().str.len_chars() >= 3)).then(1).otherwise(0).cast(pl.Int8).alias("first_token_eq"),
        pl.when((pl.col("legal_s").list.len() == 0) | (pl.col("legal_r").list.len() == 0)).then(2)
          .when(inter("legal_s", "legal_r") > 0).then(0).otherwise(1).alias("legal_rel"),
        (pl.col("decoy_r").list.set_difference(pl.col("decoy_s")).list.len()).alias("n_decoy_extra"),
        # numbers
        pl.col("ad_s").list.len().alias("n_num_s"), pl.col("ad_r").list.len().alias("n_num_r"),
        (inter("ad_s", "ad_r") / pl.max_horizontal(pl.col("ad_s").list.unique().list.len(), pl.col("ad_r").list.unique().list.len(), pl.lit(1))).alias("num_jacc"),
        pl.when((pl.col("au_s").list.len() == 0) | (pl.col("au_r").list.len() == 0)).then(2)
          .when(inter("au_s", "au_r") > 0).then(0).otherwise(1).alias("unit_rel"),
        pl.col("noaddr_s").cast(pl.Int8), pl.col("noaddr_r").cast(pl.Int8),
        (pl.col("nm_s").str.len_chars() - pl.col("nm_r").str.len_chars()).abs().alias("len_diff"),
        pl.col("r").str.slice(0, 2).eq("S3").cast(pl.Int8).alias("src_s3"),
    )
    hs = [house_relation(a, b) for a, b in zip(d["ad_s"].list.first().to_list(), d["ad_r"].list.first().to_list())]
    d = d.with_columns(pl.Series("house_rel", hs, dtype=pl.Int8))
    d = d.with_columns(
        pl.when(pl.col("house_rel").is_in([2, 3, 4]) & (pl.col("nm_tset") >= 80) & (pl.col("noaddr_r") == 0) & (pl.col("noaddr_s") == 0))
          .then(1).otherwise(0).cast(pl.Int8).alias("digit_conflict")
    )
    if "s1_name_freq" in d.columns:
        d = d.with_columns(
            pl.when((pl.col("s1_name_freq") <= 1) & (pl.col("noaddr_r") == 1) & (pl.col("nm_tset") >= 88)).then(1).otherwise(0).cast(pl.Int8).alias("unique_brand_noaddr"),
            pl.when((pl.col("s1_name_freq") <= 2) & (pl.col("nm_tset") >= 90)).then(1).otherwise(0).cast(pl.Int8).alias("rare_brand_high_nm"),
        )
    else:
        d = d.with_columns(
            pl.lit(0, dtype=pl.Int8).alias("unique_brand_noaddr"),
            pl.lit(0, dtype=pl.Int8).alias("rare_brand_high_nm"),
        )
    if "rel" in d.columns:
        d = d.with_columns(pl.col("rel").replace_strict(REL_CODES, default=4, return_dtype=pl.Int8).alias("num_rel"))
    # context within the S1's candidate list
    if "sc" in d.columns:
        d = d.with_columns(
            (pl.col("sc") / pl.col("sc").max().over("s1")).fill_nan(0).alias("sc_ratio"),
            pl.col("sc").rank("ordinal", descending=True).over("s1").alias("sc_rank"),
            pl.len().over("s1").alias("n_cands"))
    if "gid" in d.columns:
        d = d.with_columns(pl.len().over(["s1", "gid"]).alias("grp_size"),
                           (pl.col("house_rel") == 0).sum().over(["s1", "gid"]).alias("grp_house_eq"),
                           pl.col("nm_tset").mean().over(["s1", "gid"]).alias("grp_nm_tset_mean"))
    drop = [c for c in d.columns if c.endswith("_s") or c.endswith("_r")] + ["nm_s", "nm_r", "co_s", "co_r", "al_r", "ad_s_txt", "ad_r_txt", "bag_s", "bag_r"]
    keep_r = {"noaddr_s", "noaddr_r", "n_core_s", "n_core_r", "n_num_s", "n_num_r", "n_extra_r", "n_missing_r"}
    return d.drop([c for c in set(drop) if c in d.columns and c not in keep_r])


FEATURES_EXCLUDE = {"s1", "r", "id", "id_r", "y", "gid", "rel", "country"}


def feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c, t in zip(df.columns, df.dtypes) if c not in FEATURES_EXCLUDE and t.is_numeric() or t == pl.Boolean and c not in FEATURES_EXCLUDE]
