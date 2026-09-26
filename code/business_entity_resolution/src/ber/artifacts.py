"""Artefacts LEARNED from the provided records (no external data). Fit on TRAIN labels; the OOV fallback and the
vocabulary may also use unlabelled TEST records (allowed: organiser answer 11).

  fit_indic_lexicon   Indic-script word -> Latin word, from word-aligned true pairs (closed generator vocabulary)
  fit_oov_map         unseen Indic words -> nearest Latin vocabulary word (translit + skeleton, fuzzy fallback)
  fit_equivalences    token equivalences mined from true pairs (st~street, il~illinois, pvt~private, ...)
  fit_extra_words     per-token stats: inserted into TRUE copies (noise) vs present in look-alike NON-matches (decoy)
  alias_side_stats    in 'X f/k/a Y' names, which side is the real name
  fit_vocab           token frequencies of S1 names per country (for web-handle segmentation, OOV snapping)
"""
from __future__ import annotations

import polars as pl

from .tables import ALIAS_MARKERS
from .text import fold, skeleton, tokens, translit_col
from .translit import INDIC_RE, translit

TOK_ANY = r"[^\s.,()\[\]\-/]+"


# ----------------------------------------------------------------------------------------------------------------
def fit_indic_lexicon(pairs: pl.DataFrame, min_n: int = 3, min_share: float = 0.6) -> pl.DataFrame:
    """pairs: [s1_name, r_name] for TRUE pairs. Returns [indic, latin, n, share]. Indic side is NOT folded
    (NFKD + mark stripping would destroy Indic vowel signs)."""
    d = pairs.filter(pl.col("r_name").fill_null("").str.contains(INDIC_RE) & ~pl.col("s1_name").fill_null("").str.contains(INDIC_RE))
    d = d.select(fold(pl.col("s1_name")).str.extract_all(TOK_ANY).alias("a"), pl.col("r_name").str.extract_all(TOK_ANY).alias("b"))
    d = d.filter(pl.col("a").list.len() == pl.col("b").list.len()).explode(["a", "b"]).filter(pl.col("b").str.contains(INDIC_RE))
    c = d.group_by(["b", "a"]).len()
    tot = c.group_by("b").agg(pl.col("len").sum().alias("tot"))
    top = c.sort("len", descending=True).group_by("b").first().join(tot, on="b")
    return top.select(pl.col("b").alias("indic"), pl.col("a").alias("latin"), pl.col("len").alias("n"),
                      (pl.col("len") / pl.col("tot")).alias("share")).filter((pl.col("n") >= min_n) & (pl.col("share") >= min_share))


def fit_oov_map(words: list[str], vocab: pl.DataFrame) -> pl.DataFrame:
    """words: Indic words missing from the lexicon. vocab: [token, n] Latin name vocabulary.
    translit -> skeleton; pick the most frequent vocab token with the same skeleton, else best fuzzy match."""
    from rapidfuzz import process, fuzz
    v = vocab.filter(pl.col("token").str.len_chars() >= 2).with_columns(skeleton(pl.col("token")).alias("sk"))
    by_sk = v.sort("n", descending=True).group_by("sk").first()
    sk_map = dict(zip(by_sk["sk"].to_list(), by_sk["token"].to_list()))
    tl_list = [translit(w) for w in words]
    w = pl.DataFrame({"indic": words, "tl": tl_list})
    w = w.with_columns(skeleton(pl.col("tl")).alias("sk"))
    out = []
    for indic, tl, sk in w.iter_rows():
        hit = sk_map.get(sk)
        how = "skeleton"
        if hit is None:
            m = process.extractOne(tl, choices, scorer=fuzz.ratio, score_cutoff=70)
            hit, how = (m[0], "fuzzy") if m else (tl, "translit")
        out.append((indic, hit, tl, how))
    return pl.DataFrame(out, schema=["indic", "latin", "translit", "how"], orient="row")


# ----------------------------------------------------------------------------------------------------------------
def fit_equivalences(pairs: pl.DataFrame, a: str, b: str, min_n: int = 100, min_p: float = 0.3, max_diff: int = 3,
                      min_p_other: float = 0.1):
    """Mine token equivalences from TRUE pairs: tokens present on one side only, co-occurring with tokens present on
    the other side only. Returns (edges[da, db, n_ab, n_a, n_b, p_b_a, p_a_b], synonyms[token, equiv, n_ab])."""
    d = translit_col(pairs.select(pl.col(a).alias("a"), pl.col(b).alias("b")), "b")
    d = d.select(tokens(pl.col("a")).list.unique().alias("A"), tokens(pl.col("b")).list.unique().alias("B"))
    nonnum = lambda e: e.list.eval(pl.element().filter(~pl.element().str.contains(r"^\d+$")))
    d = d.select(nonnum(pl.col("A").list.set_difference(pl.col("B"))).alias("da"),
                 nonnum(pl.col("B").list.set_difference(pl.col("A"))).alias("db"))
    d = d.filter(pl.col("da").list.len().is_between(1, max_diff) & pl.col("db").list.len().is_between(1, max_diff))
    na = d.select("da").explode("da").group_by("da").len().rename({"len": "n_a"})
    nb = d.select("db").explode("db").group_by("db").len().rename({"len": "n_b"})
    e = d.explode("da").explode("db").group_by(["da", "db"]).len().rename({"len": "n_ab"})
    e = e.join(na, on="da").join(nb, on="db").with_columns((pl.col("n_ab") / pl.col("n_a")).alias("p_b_a"), (pl.col("n_ab") / pl.col("n_b")).alias("p_a_b"))
    e = e.filter((pl.col("n_ab") >= min_n) & (pl.max_horizontal("p_b_a", "p_a_b") >= min_p)).sort("n_ab", descending=True)
    # Synonym PAIRS (no transitive merging: 'ct' is both court and connecticut, 'tn' both tennessee and tamil nadu).
    # Keep an edge only if it is strong in at least one direction and not negligible in the other.
    e = e.filter((pl.max_horizontal("p_b_a", "p_a_b") >= 0.5) & (pl.min_horizontal("p_b_a", "p_a_b") >= min_p_other))
    syn = pl.concat([e.select(pl.col("da").alias("token"), pl.col("db").alias("equiv"), "n_ab"),
                     e.select(pl.col("db").alias("token"), pl.col("da").alias("equiv"), "n_ab")]).group_by(["token", "equiv"]).agg(pl.col("n_ab").sum())
    return e, syn


# ----------------------------------------------------------------------------------------------------------------
def _h40(e: pl.Expr) -> pl.Expr:
    return (e.hash() // 16_777_216).cast(pl.UInt64)  # top 40 bits: sums of <=9 terms cannot overflow


def _name_toks(e: pl.Expr) -> pl.Expr:
    return tokens(e).list.unique()


def fit_extra_words(s1: pl.DataFrame, R: pl.DataFrame, gt: pl.DataFrame, max_toks: int = 8) -> dict[str, pl.DataFrame]:
    """Order-free set hashing (sum of 40-bit token hashes) finds name pairs that differ by exactly ONE token:
      extra:   R name = S1 name + {t}   (t inserted in R)
      missing: R name = S1 name - {t}   (t dropped from R)
    For each token t, count how often such a pair is a TRUE match (noise word) vs a NON-match (decoy word).
    Also returns the exact-equal-name baseline. s1/R: [entity_id, business_name, country]; gt: [s1, r]."""
    def prep(df):
        d = df.filter(~pl.col("business_name").fill_null("").str.contains(INDIC_RE)).select(
            "entity_id", "country", _name_toks(pl.col("business_name")).alias("t"))
        d = d.filter(pl.col("t").list.len().is_between(1, max_toks))
        return d.with_columns((pl.col("t").list.eval(_h40(pl.element())).list.sum() + _h40(pl.col("country"))).alias("key"))
    S, Rr = prep(s1), prep(R)
    lab = gt.select(pl.col("s1"), pl.col("r"), pl.lit(True).alias("y"))
    matched = gt.select(pl.col("r").unique(), pl.lit(True).alias("m"))

    def label(j):  # j: [s1, r, t?]
        return j.join(lab, on=["s1", "r"], how="left").join(matched, on="r", how="left").with_columns(
            pl.col("y").fill_null(False), pl.col("m").fill_null(False))

    Sk = S.select(pl.col("entity_id").alias("s1"), "key")
    # exact equal names
    eq = label(Rr.select(pl.col("entity_id").alias("r"), "key").join(Sk, on="key"))
    base = eq.select(pl.len().alias("pairs"), pl.col("y").mean().alias("precision"))
    # R has one EXTRA token t
    rx = Rr.filter(pl.col("t").list.len() >= 2).select(pl.col("entity_id").alias("r"), "key", "t").explode("t") \
           .with_columns((pl.col("key") - _h40(pl.col("t"))).alias("k2")).drop("key")
    ex = label(rx.join(Sk.rename({"key": "k2"}), on="k2"))
    extra = ex.group_by("t").agg(pl.len().alias("n"), pl.col("y").sum().alias("n_true"),
                                 (~pl.col("m")).sum().alias("n_orphan")).with_columns(
        (pl.col("n_true") / pl.col("n")).alias("p_true")).sort("n", descending=True)
    # R is MISSING one token t of S1
    sx = S.filter(pl.col("t").list.len() >= 2).select(pl.col("entity_id").alias("s1"), "key", "t").explode("t") \
          .with_columns((pl.col("key") - _h40(pl.col("t"))).alias("k2")).drop("key")
    mx = label(sx.join(Rr.select(pl.col("entity_id").alias("r"), pl.col("key").alias("k2")), on="k2"))
    missing = mx.group_by("t").agg(pl.len().alias("n"), pl.col("y").sum().alias("n_true")).with_columns(
        (pl.col("n_true") / pl.col("n")).alias("p_true")).sort("n", descending=True)
    return {"equal_baseline": base, "extra": extra, "missing": missing}


# ----------------------------------------------------------------------------------------------------------------
def fit_lexicon_from_siblings(recs: pl.DataFrame, lex: pl.DataFrame, min_votes: int = 2, min_share: float = 0.5,
                              max_group: int = 15) -> pl.DataFrame:
    """UNSUPERVISED (usable on test, organiser answer 11): an Indic-script name and a Latin name that share the exact
    address number set are probably copies of one business. Align them word by word; words already in `lex` act as
    anchors (all anchors must agree), and unknown Indic words take the majority Latin word at their position.
    recs: [entity_id, business_name, business_address, country] (any sources). Returns [indic, latin, votes, share]."""
    sig = pl.lit("#") + pl.col("country") + "|" + fold(pl.col("business_address")).str.extract_all(r"\d+") \
        .list.eval(pl.element().str.strip_chars_start("0")).list.unique().list.sort().list.join("-")
    d = recs.with_columns(sig.alias("sig"), fold(pl.col("business_address")).str.extract_all(r"\d+").list.join("").str.len_chars().alias("nd")) \
            .filter(pl.col("nd") >= 3)
    d = d.join(d.group_by("sig").len().filter(pl.col("len") <= max_group).select("sig"), on="sig")
    isI = pl.col("business_name").fill_null("").str.contains(INDIC_RE)
    I = d.filter(isI).select("sig", pl.col("entity_id").alias("i"), pl.col("business_name").str.extract_all(TOK_ANY).alias("it"))
    L = d.filter(~isI).select("sig", pl.col("entity_id").alias("l"), fold(pl.col("business_name")).str.extract_all(TOK_ANY).alias("lt"))
    P = I.join(L, on="sig").filter(pl.col("it").list.len() == pl.col("lt").list.len()).with_row_index("pid")
    X = P.select("pid", "it", "lt").explode(["it", "lt"]).join(lex.select(pl.col("indic").alias("it"), pl.col("latin").alias("known")), on="it", how="left")
    ok = X.group_by("pid").agg(pl.col("known").is_not_null().sum().alias("n_known"),
                               (pl.col("known") == pl.col("lt")).sum().alias("n_agree"))
    ok = ok.filter((pl.col("n_known") >= 1) & (pl.col("n_agree") == pl.col("n_known"))).select("pid")
    V = X.join(ok, on="pid").filter(pl.col("known").is_null() & pl.col("it").str.contains(INDIC_RE)).group_by(["it", "lt"]).len()
    tot = V.group_by("it").agg(pl.col("len").sum().alias("tot"))
    top = V.sort("len", descending=True).group_by("it").first().join(tot, on="it")
    return top.select(pl.col("it").alias("indic"), pl.col("lt").alias("latin"), pl.col("len").alias("votes"),
                      (pl.col("len") / pl.col("tot")).alias("share")).filter((pl.col("votes") >= min_votes) & (pl.col("share") >= min_share))


# ----------------------------------------------------------------------------------------------------------------
def alias_side_stats(pairs: pl.DataFrame) -> pl.DataFrame:
    """pairs: [s1_name, r_name] TRUE pairs. For R names with an alias marker, which side overlaps the S1 name more."""
    pat = r"(?i)\s(" + "|".join(m.replace("/", r"\s*/\s*") for m in ALIAS_MARKERS) + r")[:\s]"
    d = pairs.filter(pl.col("r_name").fill_null("").str.contains(pat))
    d = d.with_columns(pl.col("r_name").str.replace(pat, "\x00").str.split("\x00").alias("parts"))
    d = d.filter(pl.col("parts").list.len() == 2).with_columns(
        _name_toks(pl.col("s1_name")).alias("S"),
        _name_toks(pl.col("parts").list.get(0, null_on_oob=True)).alias("L"),
        _name_toks(pl.col("parts").list.get(1, null_on_oob=True)).alias("Rt"))
    d = d.with_columns(pl.col("S").list.set_intersection(pl.col("L")).list.len().alias("ov_left"),
                       pl.col("S").list.set_intersection(pl.col("Rt")).list.len().alias("ov_right"))
    return d.select(pl.len().alias("alias_pairs"),
                    (pl.col("ov_right") > pl.col("ov_left")).mean().alias("right_better"),
                    (pl.col("ov_left") > pl.col("ov_right")).mean().alias("left_better"),
                    (pl.col("ov_left") == 0).mean().alias("left_no_overlap"))


def fit_vocab(names: pl.DataFrame, min_n: int = 2) -> pl.DataFrame:
    """names: [business_name, country] (S1 train+test). Returns [country, token, n]."""
    d = names.select("country", tokens(pl.col("business_name")).alias("token")).explode("token").drop_nulls("token")
    return d.group_by(["country", "token"]).len().rename({"len": "n"}).filter(pl.col("n") >= min_n)


# ---- test-time (unsupervised) ---------------------------------------------------------------------------------------
def pseudo_pairs(QN: pl.DataFrame, RN: pl.DataFrame, min_shared: int = 3, num_equal: bool = True) -> pl.DataFrame:
    """High-precision S1->R pairs WITHOUT labels: same first address number, same core-name set, >= min_shared common
    address words, and (num_equal) the same full number set. Train check (300k S1, num_equal=False): US precision 0.9975 /
    recall 0.52; India 0.914 / 0.50 -> with num_equal India 0.992 (India twins differ in a later number: 6-3-891 vs 6-3-896).
    QN/RN: normalised frames. Returns [s1, r]."""
    k = lambda N: N.filter((pl.col("ad").list.len() > 0) & (pl.col("core").list.len() > 0)).select(
        "entity_id", pl.col("ad").list.first().alias("hn"), pl.col("core").list.sort().list.join(" ").alias("ck"), "aw",
        pl.col("ad").list.unique().list.sort().list.join(" ").alias("nums"))
    j = k(QN).rename({"entity_id": "s1"}).join(k(RN).rename({"entity_id": "r", "aw": "aw_r", "nums": "nums_r"}), on=["hn", "ck"])
    j = j.filter(pl.col("aw").list.set_intersection(pl.col("aw_r")).list.len() >= min_shared)
    if num_equal:
        j = j.filter(pl.col("nums") == pl.col("nums_r"))
    return j.select("s1", "r")


def fit_country_addr_synonyms(nz, s1: pl.DataFrame, R: pl.DataFrame, min_n: int = 50, chunk: int = 500_000,
                              stop: set[str] | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per country: address token equivalences mined from pseudo_pairs (no labels), e.g. France region <-> department,
    'r' <-> 'rue'. nz: a Normalizer built on the base artefacts. s1/R: raw records incl. business_address.
    Returns (edges with country, synonyms[country, token, equiv, n_ab])."""
    edges, syns = [], []
    for c in s1["country"].unique().sort().to_list():
        S, Rc = s1.filter(pl.col("country") == c), R.filter(pl.col("country") == c)
        RN = pl.concat([nz.transform(Rc.slice(i, chunk)) for i in range(0, Rc.height, chunk)])
        pp = pseudo_pairs(nz.transform(S), RN)
        if pp.height < min_n:
            continue
        raw = pp.join(S.select(pl.col("entity_id").alias("s1"), pl.col("business_address").alias("a")), on="s1") \
                .join(Rc.select(pl.col("entity_id").alias("r"), pl.col("business_address").alias("b")), on="r")
        e, syn = fit_equivalences(raw, "a", "b", min_n=min_n)
        ok = lambda col: (pl.col(col).str.len_chars() >= 2) & ~pl.col(col).is_in(list(stop or ()))
        edges.append(e.filter(ok("da") & ok("db")).with_columns(pl.lit(c).alias("country"), pl.lit(pp.height).alias("n_pseudo")))
        syns.append(syn.filter(ok("token") & ok("equiv")).with_columns(pl.lit(c).alias("country")))
    if not syns:
        return pl.DataFrame(), pl.DataFrame(schema={"country": pl.String, "token": pl.String, "equiv": pl.String, "n_ab": pl.UInt32})
    return pl.concat(edges, how="vertical_relaxed"), pl.concat(syns, how="vertical_relaxed").select("country", "token", "equiv", "n_ab")
