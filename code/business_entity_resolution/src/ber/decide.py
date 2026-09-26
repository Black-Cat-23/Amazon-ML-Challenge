"""Decision layer: pair probabilities -> predicted match set per S1 (what matching_results.tsv contains).

  exclusive_posterior  each record belongs to at most one S1: turn independent p(s1, r) into the
                       "at most one owner" posterior over the S1s that retrieved the record
  rank_threshold       baseline: accept rank-1 if p > t1, others if p > t2
  expected_f_decode    per S1, choose the top-k set (k = 0..M) that maximises EXPECTED macro-F0.5 under
                       independent Bernoulli labels (exact Poisson-binomial DP, vectorised over S1s);
                       optional has-match head h(s1) replaces the independence estimate of P(no match)
  source_caps          per-source limits seen in train (S2 <= 5, S3 <= 6 per S1)

Frames are long [s1, r, p, ...]; outputs are [s1, r] of accepted pairs.
"""
from __future__ import annotations

import numpy as np
import polars as pl

CAPS = {"S2": 5, "S3": 6}


def exclusive_posterior(d: pl.DataFrame, p: str = "p", out: str = "q") -> pl.DataFrame:
    """q_s = p_s * prod_{t != s}(1 - p_t) / (prod_t(1 - p_t) + sum_u p_u * prod_{t != u}(1 - p_t)), per record r."""
    e = 1e-6
    odds = pl.col(p).clip(0.0, 1 - e) / (1 - pl.col(p)).clip(e, 1.0)
    # divide numerator and denominator by prod_t(1 - p_t): q_s = odds_s / (1 + sum_u odds_u)
    return d.with_columns((odds / (1 + odds.sum().over("r"))).alias(out))


def rank_threshold(d: pl.DataFrame, t1: float, t2: float, p: str = "p") -> pl.DataFrame:
    d = d.sort("r").with_columns(pl.col(p).rank("ordinal", descending=True).over("s1").alias("_rk"))
    return d.filter(((pl.col("_rk") == 1) & (pl.col(p) > t1)) | ((pl.col("_rk") > 1) & (pl.col(p) > t2))).select("s1", "r")


def conditioned_rank_threshold(d: pl.DataFrame, t1: float, t2: float, t2_strong: float = 0.58,
                               p_strong: float = 0.90, p: str = "p") -> pl.DataFrame:
    """Two-stage conditioned threshold:
       - Rank 1 candidate is accepted if p > t1 (protects singletons).
       - If Rank 1 is strong (p1 >= p_strong), S1 is a confirmed non-singleton;
         secondary candidates (_rk > 1) are accepted if p > t2_strong (recovers missed recall).
       - Otherwise, secondary candidates must pass the standard t2.
    """
    d = d.sort("r").with_columns(
        pl.col(p).rank("ordinal", descending=True).over("s1").alias("_rk"),
        pl.col(p).max().over("s1").alias("_p1")
    ).with_columns(
        pl.when(pl.col("_p1") >= p_strong).then(pl.lit(t2_strong)).otherwise(pl.lit(t2)).alias("_eff_t2")
    )
    return d.filter(((pl.col("_rk") == 1) & (pl.col(p) > t1)) | ((pl.col("_rk") > 1) & (pl.col(p) > pl.col("_eff_t2")))).select("s1", "r")



def source_caps(sel: pl.DataFrame, d: pl.DataFrame, p: str = "p", caps: dict = CAPS, total: int = 11) -> pl.DataFrame:
    """Keep at most caps[source] accepted records per S1 and source, and `total` per S1 (highest p first)."""
    x = sel.join(d.select("s1", "r", p), on=["s1", "r"], how="left").with_columns(pl.col("r").str.slice(0, 2).alias("_src"))
    x = x.sort("r").with_columns(pl.col(p).rank("ordinal", descending=True).over(["s1", "_src"]).alias("_k"))
    cap = pl.col("_src").replace_strict(caps, default=99, return_dtype=pl.Int64)
    x = x.filter(pl.col("_k") <= cap)
    return x.filter(pl.col(p).rank("ordinal", descending=True).over("s1") <= total).select("s1", "r")


def _pb_prefix(P: np.ndarray) -> np.ndarray:
    """P: (n, M) probs sorted desc. Returns D (M+1, n, M+1): D[k] = distribution of #positives among the first k."""
    n, M = P.shape
    D = np.zeros((M + 1, n, M + 1), dtype=np.float64)
    D[0, :, 0] = 1.0
    for k in range(1, M + 1):
        pk = P[:, k - 1:k]
        D[k] = D[k - 1] * (1 - pk)
        D[k, :, 1:] += D[k - 1, :, :-1] * pk
    return D


def expected_f_table(P: np.ndarray, h: np.ndarray | None = None, beta: float = 0.5) -> np.ndarray:
    """P: (n, M) probs sorted desc (pad with 0). Returns E (n, M+1): expected F_beta of predicting the top-k, k=0..M.

    F = (1+b2) TP / (b2 |T| + k) ; empty prediction scores 1 iff |T| = 0.
    With a has-match head h: E_k>0 is rescaled to condition on |T|>0 and multiplied by h; E_0 = 1 - h."""
    b2 = beta * beta
    n, M = P.shape
    pre = _pb_prefix(P)                        # TP distribution of the first k
    suf = _pb_prefix(P[:, ::-1].copy())        # FN distribution of the last M-k  -> suf[M-k]
    a = np.arange(M + 1, dtype=np.float64)
    E = np.zeros((n, M + 1))
    p_empty = pre[M][:, 0]
    E[:, 0] = p_empty
    for k in range(1, M + 1):
        tp, fn = pre[k], suf[M - k]            # (n, M+1) each
        # G[a, b] = (1+b2) a / (b2 (a + b) + k)
        G = (1 + b2) * a[:, None] / (b2 * (a[:, None] + a[None, :]) + k)
        E[:, k] = np.einsum("na,ab,nb->n", tp, G, fn, optimize=True)
    if h is not None:
        cond = E[:, 1:] / np.clip(1 - p_empty, 1e-9, None)[:, None]
        E[:, 1:] = cond * h[:, None]
        E[:, 0] = 1 - h
    return E


def expected_f_decode(d: pl.DataFrame, p: str = "p", has: pl.DataFrame | None = None, M: int = 15, beta: float = 0.5,
                      chunk: int = 50_000) -> pl.DataFrame:
    """d: [s1, r, p]; has: optional [s1, h] (P(S1 has >= 1 match)). Returns accepted [s1, r] (top-k* per S1)."""
    d = d.sort("r").with_columns(pl.col(p).rank("ordinal", descending=True).over("s1").alias("_rk")).filter(pl.col("_rk") <= M)
    s1s = d["s1"].unique().sort()
    out = []
    for i in range(0, s1s.len(), chunk):
        ids = s1s.slice(i, chunk)
        x = d.filter(pl.col("s1").is_in(ids.implode())).join(pl.DataFrame({"s1": ids}).with_row_index("_i"), on="s1")
        n = ids.len()
        P = np.zeros((n, M))
        P[x["_i"].to_numpy(), x["_rk"].to_numpy() - 1] = x[p].to_numpy()
        h = None
        if has is not None:
            h = pl.DataFrame({"s1": ids}).join(has, on="s1", how="left")["h"].fill_null(1.0).to_numpy()
        k = expected_f_table(P, h, beta).argmax(1)
        kk = pl.DataFrame({"_i": np.arange(n, dtype=np.uint32), "_k": k.astype(np.uint32)})
        out.append(x.join(kk, on="_i").filter(pl.col("_rk") <= pl.col("_k")).select("s1", "r"))
    return pl.concat(out) if out else d.head(0).select("s1", "r")


def _selftest():
    import itertools
    rng = np.random.default_rng(0)
    # exact expectation by enumeration vs the DP table
    for _ in range(20):
        m = rng.integers(1, 6)
        p = np.sort(rng.random(m))[::-1]
        E = expected_f_table(p[None, :])[0]
        for k in range(m + 1):
            ex = 0.0
            for lab in itertools.product([0, 1], repeat=m):
                pr = np.prod([pi if l else 1 - pi for pi, l in zip(p, lab)])
                T = sum(lab); tp = sum(lab[:k])
                f = (1.0 if T == 0 else 0.0) if k == 0 else (0.0 if tp == 0 else 1.25 * tp / (0.25 * T + k))
                ex += pr * f
            assert abs(ex - E[k]) < 1e-9, (p, k, ex, E[k])
    # decode picks argmax k
    d = pl.DataFrame({"s1": ["a"] * 3 + ["b"] * 2, "r": ["S2x", "S3y", "S2z", "S2u", "S2v"], "p": [0.95, 0.9, 0.2, 0.05, 0.02]})
    sel = expected_f_decode(d).sort(["s1", "r"])
    assert sel.rows() == [("a", "S2x"), ("a", "S3y")], sel
    # exclusivity: record retrieved by two S1s with p .9/.9 -> each q = .9/.1 / (1 + 18) = 0.47
    q = exclusive_posterior(pl.DataFrame({"s1": ["a", "b"], "r": ["S2x", "S2x"], "p": [0.9, 0.9]}))["q"].to_list()
    assert abs(q[0] - 9 / 19) < 1e-6, q
    return True


if __name__ == "__main__":
    print("selftest", _selftest())
