"""Blocking v5 "tight": v4 retrieval on NORMALISED records (learned artefacts) + sibling groups + deterministic pruning.
Sweeps pruning policies and reports the frontier: pair recall / entity completeness vs candidates per S1.
Also writes the annotated candidate pool of the train sample (training data for the matcher, phase 4)."""
import gc, glob, os, subprocess, sys, time
LOCAL = not os.path.exists("/kaggle")
if LOCAL:
    sys.path.insert(0, os.path.abspath("../../code/business_entity_resolution/src"))
else:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "git+https://github.com/Black-Cat-23/Amazon-ML-Challenge.git#subdirectory=code/business_entity_resolution"], check=True)
import numpy as np, polars as pl
from ber.normalize import Normalizer
from ber import blocking as B

T0 = time.time(); WD = "." if LOCAL else "/kaggle/working"
REP = open(f"{WD}/results_v5.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); REP.write(s + "\n"); REP.flush()
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(220)
IN = "../../research/eda/cache" if LOCAL else os.path.dirname(glob.glob("/kaggle/input/**/train_s1.parquet", recursive=True)[0])
ART = "../artifacts/kout/artifacts" if LOCAL else os.path.dirname(glob.glob("/kaggle/input/**/indic_lexicon.parquet", recursive=True)[0])
rd = lambda n: pl.read_parquet(f"{IN}/{n}.parquet")
NZ = Normalizer(ART); log(f"normalizer: {len(NZ.indic)} indic words, {len(NZ.noise)} noise, {len(NZ.decoy)} decoy  {time.time()-T0:.0f}s")

def norm_chunks(df, chunk=1_000_000):
    return pl.concat([NZ.transform(df.slice(i, chunk)) for i in range(0, df.height, chunk)])

def run_country(S_raw, R_raw, S_all_raw):
    """S_raw: query S1 sample; R_raw: S2/S3 pool; S_all_raw: all S1 of the country (reverse lookups). Returns annotated candidates."""
    R = norm_chunks(R_raw).with_columns(pl.Series("id", np.arange(R_raw.height, dtype=np.uint32)))
    idx = B.Index(R); log(f"  pool {R.height} -> postings {idx.Rt.height}  {time.time()-T0:.0f}s")
    Q = NZ.transform(S_raw).with_columns(pl.Series("id", np.arange(S_raw.height, dtype=np.uint32)))
    cand = idx.query(Q); sig = B.signatures(R)
    ex = B.expand(cand, sig)
    del idx; gc.collect()
    cand = pl.concat([cand, ex.with_columns(pl.lit(0.0, dtype=pl.Float32).alias("sc"), pl.lit(None, dtype=pl.UInt32).alias("prk"), pl.lit(None, dtype=pl.UInt32).alias("xrk"),
                                            *[pl.lit(False).alias(f"a{k}") for k in B.ARMS]).select(cand.columns)], how="vertical_relaxed")
    cand = cand.with_columns((pl.col("prk").is_null() & pl.col("xrk").is_null()).alias("exp"))
    # keep internal pool: primary top-60, aux, expansion
    cand = cand.filter((pl.col("prk") <= 60) | pl.col("xrk").is_not_null() | pl.col("exp"))
    # sibling group id (signature; records without one form their own group)
    cand = cand.join(sig.rename({"id": "id_r"}), on="id_r", how="left").with_columns(
        pl.when(pl.col("kind") == "none").then(pl.col("id_r").cast(pl.UInt64) + (1 << 62)).otherwise(pl.col("sig")).alias("gid")).drop("sig", "kind")
    # expansion members inherit their group's best score so groups stay whole
    cand = cand.with_columns(pl.col("sc").max().over(["id", "gid"]).alias("gsc_tmp")).with_columns(
        pl.when(pl.col("exp")).then(pl.col("gsc_tmp") * 0.999).otherwise(pl.col("sc")).alias("sc")).drop("gsc_tmp")
    cand = cand.join(B.number_relation(cand.select("id", "id_r"), Q, R), on=["id", "id_r"], how="left")
    # reverse preference: which S1 does each candidate record prefer? (index over ALL S1 of the country)
    S1n = norm_chunks(S_all_raw).with_columns(pl.Series("id", np.arange(S_all_raw.height, dtype=np.uint32)))
    sidx = B.Index(S1n, arms=(0,))
    ur = cand.select(pl.col("id_r").unique())
    RQ = R.join(ur.rename({"id_r": "id"}), on="id")
    rev = sidx.query(RQ, caps={0: 3})  # [id=r, id_r=s1 index in S_all]
    del sidx, S1n; gc.collect()
    samp_ids = S_all_raw.with_row_index("sall").join(S_raw.with_row_index("sq").select("entity_id", "sq"), on="entity_id").select("sall", "sq")
    rev = rev.select(pl.col("id").alias("id_r"), pl.col("id_r").alias("sall"), pl.col("sc").alias("rsc")).join(samp_ids, on="sall", how="left")
    best = rev.group_by("id_r").agg(pl.col("rsc").max().alias("rbest"))
    mine = rev.filter(pl.col("sq").is_not_null()).select("id_r", pl.col("sq").alias("id"), "rsc")
    other = rev.join(mine.select("id_r", pl.col("id").alias("mine_sq")), on="id_r", how="left")
    cand = cand.join(mine, on=["id", "id_r"], how="left").join(best, on="id_r", how="left")
    # margin of THIS S1 vs the best S1 for the record: 0 if this S1 is the best; negative if another S1 is preferred
    cand = cand.with_columns(((pl.col("rsc").fill_null(0.0) - pl.col("rbest")) / pl.col("rbest")).fill_null(0.0).alias("rev_margin"))
    ids = Q.select("id", pl.col("entity_id").alias("s1")); rids = R.select(pl.col("id").alias("id_r"), pl.col("entity_id").alias("r"))
    out = cand.join(ids, on="id").join(rids, on="id_r")
    log(f"  cands {out.height} ({out.height/Q.height:.1f}/S1)  {time.time()-T0:.0f}s")
    return out

def metrics(kept, gtp, n_s1):
    k = kept.select("s1", "r").with_columns(pl.lit(True).alias("k"))
    h = gtp.join(k, on=["s1", "r"], how="left").with_columns(pl.col("k").fill_null(False))
    ent = h.group_by("s1").agg(pl.col("k").all())
    per = kept.group_by("s1").len()["len"]
    return dict(recall=h["k"].mean(), complete=ent["k"].mean(), cands=kept.height / n_s1,
                p95=float(np.percentile(np.concatenate([per.to_numpy(), np.zeros(n_s1 - per.len())]), 95)))

# ------------------------------------------------ TRAIN sweep ------------------------------------------------------
s1 = rd("train_s1"); R = pl.concat([rd("train_s2"), rd("train_s3")])
gt = rd("gt_rows").with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids") \
     .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "r"})
if LOCAL: s1 = s1.head(100000); R = R.head(400000)
samp = s1.sample(2000 if LOCAL else 30000, seed=11)
anns = []
for c in samp["country"].unique().sort().to_list():
    log(f"-- {c}")
    anns.append(run_country(samp.filter(pl.col("country") == c), R.filter(pl.col("country") == c), s1.filter(pl.col("country") == c)))
    gc.collect()
ann = pl.concat(anns, how="vertical_relaxed")
gtp = gt.filter(pl.col("s1").is_in(samp["entity_id"].implode()))
ann = ann.join(gtp.with_columns(pl.lit(1).alias("y")), on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0))
ann.write_parquet(f"{WD}/cand_train_sample.parquet")
log(f"\ninternal pool: {ann.height/samp.height:.1f}/S1; positives in pool {ann['y'].sum()}/{gtp.height} = {ann['y'].sum()/gtp.height:.4f}")
log("number relation by label (share within label):")
log(ann.group_by(["y", "rel"]).len().with_columns((pl.col("len") / pl.col("len").sum().over("y")).round(4).alias("share")).sort(["y", "rel"]))

# prune(): needs [id, id_r, sc, gid, rel, a4, rev_margin]; use s1/r strings as ids for metrics
A = ann.select(pl.col("s1").alias("id"), pl.col("r").alias("id_r"), "sc", "gid", "rel", "a4", "rev_margin")
rows = []
for alpha in (0.0, 0.5, 0.7, 0.8, 0.9):
    for G in (1, 2, 3, 5, 99):
        for gate in (False, True):
            for beta in (None, 0.5, 0.25, 0.0):
                kept = B.prune(A, alpha=alpha, G=G, gate=gate, beta=beta).rename({"id": "s1", "id_r": "r"})
                m = metrics(kept, gtp, samp.height)
                rows.append(dict(alpha=alpha, G=G, gate=gate, beta=-1 if beta is None else beta, **m))
F = pl.DataFrame(rows).sort("cands")
F.write_csv(f"{WD}/frontier_v5.csv")
# Pareto frontier on (cands, complete)
best, pareto = -1, []
for r in F.iter_rows(named=True):
    if r["complete"] > best + 1e-4: pareto.append(r); best = r["complete"]
log("\n=== PARETO frontier (fewer candidates, more complete entities) ===")
log(pl.DataFrame(pareto))
log("\n=== full sweep (top 40 by completeness) ==="); log(F.sort("complete", descending=True).head(40))
log(f"\nDONE train  {time.time()-T0:.0f}s")

# ------------------------------------------------ TEST France: candidate volume only ------------------------------------
if not LOCAL:
    del ann, anns, A; gc.collect()
    t1 = rd("test_s1").filter(pl.col("country") == "France"); tR = pl.concat([rd("test_s2"), rd("test_s3")]).filter(pl.col("country") == "France")
    fs = t1.sample(10000, seed=3)
    fa = run_country(fs, tR, t1)
    FA = fa.select(pl.col("s1").alias("id"), pl.col("r").alias("id_r"), "sc", "gid", "rel", "a4", "rev_margin")
    for r in pareto[::max(1, len(pareto) // 8)]:
        kept = B.prune(FA, alpha=r["alpha"], G=r["G"], gate=r["gate"], beta=None if r["beta"] < 0 else r["beta"])
        per = kept.group_by("id").len()["len"]
        log(f"France policy a={r['alpha']} G={r['G']} gate={r['gate']} b={r['beta']}: cands/S1 {kept.height/fs.height:.2f} (train {r['cands']:.2f}); "
            f"S1 with 0 cands {1 - per.len()/fs.height:.3f}")
    log("France number relation distribution:"); log(fa.group_by("rel").len().with_columns((pl.col("len") / pl.col("len").sum()).round(4)))
log(f"\nDONE {time.time()-T0:.0f}s")
