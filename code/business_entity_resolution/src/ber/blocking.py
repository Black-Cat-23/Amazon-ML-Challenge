"""Blocking (candidate generation). NO learned pair-scoring model is used here (organiser answer 3):
only token statistics (IDF), deterministic keys and rules whose thresholds are tuned on train.

Pipeline for one country:
  build(R)            inverted index over normalised S2/S3 records (token hash -> record ids), IDF per token
  retrieve(S)         IDF-weighted overlap; primary arm = combined evidence; small auxiliary arms
  expand(cand)        sibling expansion: records sharing an exact number set / name skeleton with a top candidate
  annotate(...)       per-candidate facts used by the pruning rules: sibling group, number relation, reverse preference
  prune(ann, policy)  deterministic rules -> the emitted candidate set (= candidate_pairs.tsv)
"""
from __future__ import annotations

import gc

import numpy as np
import polars as pl

ARMS = {0: "primary", 3: "namepair", 4: "keys", 5: "nameonly", 6: "trigram", 7: "noaddr", 8: "namekey_x_addr"}
# recall lab r9 (full train pool): completeness US 94.0 -> 96.8, India 82.7 -> 90.4 vs the v5 config, 2x faster queries
CAP = {0: 100, 3: 30, 4: 30, 5: 40, 6: 25, 7: 20, 8: 10}
# address-word window for the number x word / name x word / key x word tokens: the first words plus the LAST ADDR_TAIL words.
# Indian copies keep "first number + city + state", and in the S1's long address the city is among the last words
# (C-split misses: number x word overlap 31.5% -> 64%). Recall lab r10 vs r9: India recall 96.58 -> 97.96, S1 complete
# 90.4 -> 93.8, Indian-script copies 95.1 -> 98.0; US 99.05 -> 99.32 / 96.8 -> 97.6; same candidate count. 0 = off.
ADDR_TAIL = 3


def _pairs4(col: str):
    h4 = pl.col(col).list.head(4)
    g = lambda i: h4.list.get(i, null_on_oob=True)
    return [pl.when(h4.list.len() > j).then(pl.min_horizontal(g(i), g(j)) + "|" + pl.max_horizontal(g(i), g(j)))
            for i in range(4) for j in range(i + 1, 4)]


def _win(col: str, n: int) -> pl.Expr:
    """first n words, plus the last ADDR_TAIL words when set (deduplicated, order kept)"""
    e = pl.col(col).list.head(n)
    return pl.concat_list(e, pl.col(col).list.tail(ADDR_TAIL)).list.unique(maintain_order=True) if ADDR_TAIL else e


def tokens(N: pl.DataFrame, query: bool = False) -> pl.DataFrame:
    """N: normalised frame with uint32 'id'. Returns [id, h(u64), arm(u8)]. Prefixes keep hashes unique per arm."""
    d = N
    pre = lambda p, c: pl.col(c).list.eval(pl.lit(p) + pl.element())
    arms = [
        (0, pl.concat_list(pre("n:", "core"), pre("k:", "sk"), pre("a:", "aw"), pre("#:", "ad"), pre("u:", "au"))),
        (5, pl.concat_list(pre("m:", "core"), pre("q:", "sk"), pre("m:", "alias"))),
        (3, pl.concat_list(*_pairs4("core")).list.drop_nulls().list.eval(pl.lit("p:") + pl.element())),
        (4, pl.concat_list(pl.lit("N:") + pl.col("core").list.sort().list.join(" "), pl.lit("C:") + pl.col("core").list.join(""),
                           pl.lit("C:") + pl.col("core").list.head(2).list.join(""), pl.lit("K:") + pl.col("sk").list.sort().list.join(" "))),
    ]
    out = []
    for a, e in arms:
        t = d.select("id", e.list.unique().alias("t")).explode("t").drop_nulls("t").filter(pl.col("t").str.len_chars() > 3)
        out.append(t.select("id", pl.col("t").hash().alias("h"), pl.lit(a, dtype=pl.UInt8).alias("arm")))
    # arm 7: name tokens of records WITHOUT an address, indexed on their own. A no-address copy's only evidence is its
    # name, and in the name-only arm it ties with every same-name record of the country (recall ~72% on such copies).
    # Index side: only no-address records; query side: every query.
    z = d if query else d.filter(pl.col("noaddr"))
    t = z.select("id", pl.concat_list(pre("z:", "core"), pre("zk:", "sk")).list.unique().alias("t")).explode("t").drop_nulls("t") \
         .filter(pl.col("t").str.len_chars() > 3)
    out.append(t.select("id", pl.col("t").hash().alias("h"), pl.lit(7, dtype=pl.UInt8).alias("arm")))
    # arm 8: the WHOLE core name (sorted key) x each address number / word. Common names ("Krishna Services") with a thin
    # address (city + one number, typical of Indian-script copies) are lost in the primary arm; the pair is rare.
    ck = d.select("id", pl.col("core").list.sort().list.join(" ").alias("ck")).filter(pl.col("ck").str.len_chars() >= 3)
    kn = ck.join(d.select("id", pl.col("ad").list.unique().list.head(4).alias("x")).explode("x").drop_nulls("x"), on="id")
    kw = ck.join(d.select("id", _win("aw", 8).alias("x")).explode("x").drop_nulls("x"), on="id")
    out.append(pl.concat([kn.select("id", (pl.lit("K#:") + pl.col("ck") + "|" + pl.col("x")).alias("t")),
                          kw.select("id", (pl.lit("Kw:") + pl.col("ck") + "|" + pl.col("x")).alias("t"))])
               .unique().select("id", pl.col("t").hash().alias("h"), pl.lit(8, dtype=pl.UInt8).alias("arm")))
    # arm 6: character trigrams of the first two core words (scrambled / typo'd names, esp. with no address)
    s = d.select("id", (pl.lit("^") + pl.col("core").list.head(2).list.join("") + pl.lit("$")).alias("s")).filter(pl.col("s").str.len_chars() >= 5)
    tri = s.with_columns(pl.int_ranges(0, pl.col("s").str.len_chars() - 2).alias("i")).explode("i") \
           .select("id", (pl.lit("g:") + pl.col("s").str.slice(pl.col("i"), 3)).alias("t")).unique()
    out.append(tri.select("id", pl.col("t").hash().alias("h"), pl.lit(6, dtype=pl.UInt8).alias("arm")))
    # primary-arm combination tokens
    nums = d.select("id", pl.col("ad").list.head(2).alias("x")).explode("x").drop_nulls("x")
    wrds = d.select("id", _win("aw", 6).alias("w")).explode("w").drop_nulls("w")
    out.append(nums.join(wrds, on="id").select("id", (pl.lit("x:") + pl.col("x") + "|" + pl.col("w")).hash().alias("h"), pl.lit(0, dtype=pl.UInt8).alias("arm")))
    aw5 = d.select("id", pl.col("aw").list.head(5).alias("w")).explode("w").drop_nulls("w")
    wp = aw5.join(aw5.select("id", pl.col("w").alias("w2")), on="id").filter(pl.col("w") < pl.col("w2"))
    out.append(wp.select("id", (pl.lit("w:") + pl.col("w") + "|" + pl.col("w2")).hash().alias("h"), pl.lit(0, dtype=pl.UInt8).alias("arm")))
    nw3 = d.select("id", pl.col("core").list.head(3).alias("n")).explode("n").drop_nulls("n")
    aw4 = aw5.group_by("id").head(4) if not ADDR_TAIL else d.select("id", _win("aw", 4).alias("w")).explode("w").drop_nulls("w")
    out.append(nw3.join(aw4, on="id").select("id", (pl.lit("y:") + pl.col("n") + "|" + pl.col("w")).hash().alias("h"), pl.lit(0, dtype=pl.UInt8).alias("arm")))
    if query:  # digit-drop variants of house numbers (the generator drops a first or last digit)
        dn = nums.filter(pl.col("x").str.len_chars() >= 3)
        dv = pl.concat([dn.with_columns(pl.col("x").str.slice(1)), dn.with_columns(pl.col("x").str.slice(0, pl.col("x").str.len_chars() - 1))])
        out.append(dv.join(wrds, on="id").select("id", (pl.lit("x:") + pl.col("x") + "|" + pl.col("w")).hash().alias("h"), pl.lit(0, dtype=pl.UInt8).alias("arm")))
    return pl.concat(out)


def signatures(N: pl.DataFrame) -> pl.DataFrame:
    """Sibling signatures: exact address-number set (>=3 digits) else exact name-skeleton key. -> [id, sig(u64), sig_kind]"""
    num = N.select("id", (pl.lit("#") + pl.col("ad").list.unique().list.sort().list.join("-")).hash().alias("sig"),
                   pl.col("ad").list.join("").str.len_chars().alias("nd"))
    nam = N.select("id", (pl.lit("K") + pl.col("sk").list.sort().list.join(" ")).hash().alias("sig2"), pl.col("sk").list.len().alias("ns"))
    j = num.join(nam, on="id")
    return j.select("id", pl.when(pl.col("nd") >= 3).then(pl.col("sig")).otherwise(pl.col("sig2")).alias("sig"),
                    pl.when(pl.col("nd") >= 3).then(pl.lit("num")).when(pl.col("ns") >= 2).then(pl.lit("name")).otherwise(pl.lit("none")).alias("kind"))


class Index:
    """Inverted index for one country's S2/S3 pool (or S1 pool, for reverse lookups)."""

    def __init__(self, N: pl.DataFrame, cap: int = 2000, key_cap: int = 200, chunk: int = 1_000_000, arms=(0, 3, 4, 5, 6, 7, 8)):
        parts = []
        for c0 in range(0, N.height, chunk):
            tk = tokens(N.slice(c0, chunk))
            parts.append(tk.filter(pl.col("arm").is_in(list(arms))).select(pl.col("id").cast(pl.UInt32).alias("id_r"), "h", "arm"))
            del tk; gc.collect()
        Rt = pl.concat(parts); del parts; gc.collect()
        self.n_tokens = Rt.height
        dfc = Rt.group_by("h").agg(pl.len().alias("df"), pl.col("arm").first())
        dfc = dfc.filter(pl.when(pl.col("arm") == 4).then(pl.col("df") <= key_cap).otherwise(pl.col("df") <= cap))
        # idf quantised to 1/1024 and summed as integers: exact in any summation order (deterministic ranks)
        self.dfc = dfc.select("h", ((np.log(N.height) - pl.col("df").cast(pl.Float64).log()) * 1024).round().cast(pl.Int32).alias("idf"),
                              pl.col("df").cast(pl.UInt32))
        Rt = Rt.join(self.dfc.select("h"), on="h", how="semi").select("id_r", "h").sort(["h", "id_r"])
        # postings as sorted arrays: a query chunk gathers only its own tokens' postings (binary search), never the whole index
        self._H = Rt["h"].to_numpy(); self._I = Rt["id_r"].to_numpy()
        del Rt; gc.collect()
        self.arms = arms

    @property
    def Rt(self) -> pl.DataFrame:  # compatibility (postings as a frame)
        return pl.DataFrame({"id_r": self._I, "h": self._H})

    def _postings(self, hs: np.ndarray) -> pl.DataFrame:
        uh = np.unique(hs)
        lo = np.searchsorted(self._H, uh, "left"); hi = np.searchsorted(self._H, uh, "right")
        cnt = hi - lo; keep = cnt > 0; uh, lo, cnt = uh[keep], lo[keep], cnt[keep]
        start = np.repeat(lo - np.concatenate([[0], np.cumsum(cnt)[:-1]]), cnt)
        pos = start + np.arange(cnt.sum())
        return pl.DataFrame({"h": np.repeat(uh, cnt), "id_r": self._I[pos]})

    def query(self, Q: pl.DataFrame, chunk: int = 10000, caps=None, budget: int = 10_000_000) -> pl.DataFrame:
        """Q: normalised query frame with uint32 'id'. Returns [id, id_r, sc, prk, xrk, a0..a6].
        Chunks hold at most `chunk` queries and ~`budget` joined posting rows (sum of token df)."""
        caps = caps or CAP
        St = tokens(Q, query=True).filter(pl.col("arm").is_in(list(self.arms))).join(self.dfc, on="h").sort("id")
        per = St.group_by("id").agg(pl.len().alias("n"), pl.col("df").cast(pl.Int64).sum().alias("cost")).sort("id")
        n, cost = per["n"].to_numpy(), per["cost"].to_numpy()
        row0 = np.concatenate([[0], np.cumsum(n)])
        res, i = [], 0
        while i < len(n):
            j = i + 1
            cc = cost[i]
            while j < len(n) and j - i < chunk and cc + cost[j] <= budget:
                cc += cost[j]; j += 1
            ch = St.slice(int(row0[i]), int(row0[j] - row0[i]))
            post = self._postings(ch["h"].to_numpy())
            a = ch.join(post, on="h").group_by(["id", "arm", "id_r"]).agg((pl.col("idf").cast(pl.Int64).sum() / 1024).cast(pl.Float32).alias("sc"))
            # deterministic: ties (records with identical token sets are common) broken by record id
            a = a.sort("id_r").with_columns(pl.col("sc").rank("ordinal", descending=True).over(["id", "arm"]).alias("rk"))
            a = a.filter(pl.col("rk") <= pl.col("arm").replace_strict(caps, default=10, return_dtype=pl.UInt32))
            f = a.group_by(["id", "id_r"]).agg(
                pl.col("sc").filter(pl.col("arm") == 0).max().fill_null(0.0).alias("sc"),
                pl.col("rk").filter(pl.col("arm") == 0).min().alias("prk"),
                pl.col("rk").filter(pl.col("arm") != 0).min().alias("xrk"),
                *[(pl.col("arm") == k).any().alias(f"a{k}") for k in ARMS])
            res.append(f)
            i = j
        if not res:
            return pl.DataFrame(schema={"id": pl.UInt32, "id_r": pl.UInt32, "sc": pl.Float32, "prk": pl.UInt32, "xrk": pl.UInt32,
                                        **{f"a{k}": pl.Boolean for k in ARMS}})
        return pl.concat(res)


def expand(cand: pl.DataFrame, sigs: pl.DataFrame, M: int = 10, maxgrp: int = 12) -> pl.DataFrame:
    """Records sharing a sibling signature with one of the top-M primary candidates (not already candidates)."""
    grp = sigs.filter(pl.col("kind") != "none").group_by("sig").agg(pl.col("id").alias("mem")).filter(pl.col("mem").list.len() <= maxgrp)
    top = cand.filter(pl.col("prk") <= M).select("id", "id_r")
    s = top.join(sigs.rename({"id": "id_r"}).select("id_r", "sig"), on="id_r").join(grp, on="sig") \
           .select("id", pl.col("mem")).explode("mem").rename({"mem": "id_r"}).unique()
    return s.join(cand.select("id", "id_r"), on=["id", "id_r"], how="anti")


def number_relation(pairs: pl.DataFrame, QS: pl.DataFrame, RS: pl.DataFrame) -> pl.DataFrame:
    """pairs [id, id_r]; QS/RS normalised frames ('id', 'ad'). Adds rel in {equal, subset, overlap, trunc, disjoint, missing}."""
    p = pairs.join(QS.select("id", pl.col("ad").list.unique().alias("A")), on="id") \
             .join(RS.select(pl.col("id").alias("id_r"), pl.col("ad").list.unique().alias("B")), on="id_r")
    inter = pl.col("A").list.set_intersection(pl.col("B")).list.len()
    p = p.with_columns(
        pl.when((pl.col("A").list.len() == 0) | (pl.col("B").list.len() == 0)).then(pl.lit("missing"))
        .when(pl.col("A").list.sort() == pl.col("B").list.sort()).then(pl.lit("equal"))
        .when((inter == pl.col("B").list.len()) | (inter == pl.col("A").list.len())).then(pl.lit("subset"))
        .when(inter > 0).then(pl.lit("overlap")).otherwise(pl.lit("disjoint")).alias("rel"))
    # disjoint but explained by a dropped first/last digit (generator noise on TRUE copies) -> 'trunc'
    dj = p.filter(pl.col("rel") == "disjoint").select("id", "id_r", "A", "B").explode("A").explode("B") \
          .filter((pl.col("A").str.len_chars() >= 3) | (pl.col("B").str.len_chars() >= 3))
    tr = dj.filter((pl.col("A").str.ends_with(pl.col("B")) | pl.col("B").str.ends_with(pl.col("A")) |
                    pl.col("A").str.starts_with(pl.col("B")) | pl.col("B").str.starts_with(pl.col("A")))
                   & ((pl.col("A").str.len_chars() - pl.col("B").str.len_chars()).abs() == 1)).select("id", "id_r").unique().with_columns(pl.lit(True).alias("t"))
    p = p.join(tr, on=["id", "id_r"], how="left").with_columns(
        pl.when(pl.col("t").fill_null(False)).then(pl.lit("trunc")).otherwise(pl.col("rel")).alias("rel"))
    return p.select("id", "id_r", "rel")


def prune(ann: pl.DataFrame, alpha: float = 0.0, G: int = 99, gate: bool = False, beta: float | None = None, kmax: int = 999) -> pl.DataFrame:
    """ann: [id, id_r, sc, gid, rel, a4, rev_margin]. Group-level deterministic rules; returns kept [id, id_r]."""
    g = ann.group_by(["id", "gid"]).agg(pl.col("sc").max().alias("gsc"), pl.col("a4").any().alias("gkey"),
                                        (pl.col("rel").is_in(["disjoint"])).all().alias("gconf"))
    g = g.sort("gid").with_columns((pl.col("gsc") / pl.col("gsc").max().over("id")).alias("ratio"),
                       pl.col("gsc").rank("ordinal", descending=True).over("id").alias("grk"))
    keep = (pl.col("ratio") >= alpha) & (pl.col("grk") <= G)
    if gate:
        keep = keep & (~pl.col("gconf") | pl.col("gkey"))
    k = ann.join(g.filter(keep).select("id", "gid"), on=["id", "gid"])
    if beta is not None:
        k = k.filter(pl.col("rev_margin").fill_null(0.0) >= -beta)
    return k.sort("id_r").with_columns(pl.col("sc").rank("ordinal", descending=True).over("id").alias("_r")).filter(pl.col("_r") <= kmax).select("id", "id_r")
