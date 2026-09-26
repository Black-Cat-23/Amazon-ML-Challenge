"""Matcher on FULL-mode train pools (same candidate definition as test). Split blending by S1 hash: A 60 / B 30 / C 10.
Chooses the decision rule and the blocking policy on B; reports everything on held-out C.
Writes matcher.txt, head.txt, decision.json (consumed by the submission job)."""
import glob, json, os, subprocess, sys, time
REF = "__REF__"
LOCAL = not os.path.exists("/kaggle")
if LOCAL:
    sys.path.insert(0, os.path.abspath("../../code/business_entity_resolution/src"))
elif glob.glob("/kaggle/input/**/er-src/**/ber/__init__.py", recursive=True):   # account 2: no internet -> offline bundle
    _W = os.path.dirname(glob.glob("/kaggle/input/**/polars-1.44.2*.whl", recursive=True)[0])
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", _W, "polars==1.44.2", "rapidfuzz==3.14.6", "lightgbm==4.6.0"], check=True)
    sys.path.insert(0, os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/er-src/**/ber/__init__.py", recursive=True)[0])))
else:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm==4.6.0",
                    f"git+https://github.com/Black-Cat-23/Amazon-ML-Challenge.git@{REF}#subdirectory=code/business_entity_resolution"], check=True)
import numpy as np, polars as pl, lightgbm as lgb
from ber.normalize import Normalizer
from ber.metrics import macro_f05
from ber.decide import rank_threshold
from ber import model as M
from ber.artifacts import pseudo_pairs

T0 = time.time(); WD = "." if LOCAL else "/kaggle/working"
N_S1 = int(os.environ.get("N_S1", 3000 if LOCAL else 80_000))  # sampled S1 per country (~130 pool rows each)
REP = open(f"{WD}/results_matcher_full.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); REP.write(s + "\n"); REP.flush()
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(220)
find = lambda f: sorted(glob.glob(f"/kaggle/input/**/{f}", recursive=True))
IN = "../../research/eda/cache" if LOCAL else os.path.dirname(find("train_s1.parquet")[0])
ART = "../artifacts/kout/artifacts" if LOCAL else os.path.dirname(find("indic_lexicon.parquet")[0])
POOLS = sorted(glob.glob("../cands_full/pool_train_*.parquet")) if LOCAL else find("pool_train_*.parquet")
FRONTS = sorted(glob.glob("../cands_full/frontier_train_*.csv")) if LOCAL else find("frontier_train_*.csv")
log(f"ref {REF}; pools {POOLS}; N_S1/country {N_S1}")
NZ = Normalizer(ART)

# ---- sample S1s per country (deterministic hash), features --------------------------------------------------------
parts = []
S1ALL = pl.read_parquet(f"{IN}/train_s1.parquet")
for p in POOLS:
    # never load a whole country pool (100M+ rows): record-level stats by streaming, then only the sampled S1s' rows
    st = M.claim_stats(p)
    ids = pl.scan_parquet(p).select(pl.col("s1").unique()).collect(engine="streaming")["s1"].sort()
    thr = int(1_000_000 * min(1.0, N_S1 / len(ids)))
    P = pl.scan_parquet(p).filter((pl.col("s1").hash(7) % 1_000_000) < thr).collect(engine="streaming").join(st, on="r", how="left")
    P = P.drop("y", strict=False)   # labels always come from gt below (older pools carried a y column, newer do not)
    parts.append(P)
    log(f"{os.path.basename(p)}: {len(ids)} S1 -> sample {P['s1'].n_unique()} S1 / {P.height} rows  {time.time()-T0:.0f}s")
    del P, st
pool = pl.concat(parts, how="diagonal_relaxed"); del parts
ctry = pool.select("s1").unique().join(S1ALL.select(pl.col("entity_id").alias("s1"), "country"), on="s1")["country"].unique().to_list()
pool = pool.join(M.s1_name_freq(S1ALL.filter(pl.col("country").is_in(ctry)), NZ), on="s1", how="left")
log(f"context features: n_claim, n_s1_for_r, s1_name_freq  {time.time()-T0:.0f}s")
S1ids = pool["s1"].unique()
s1 = pl.scan_parquet(f"{IN}/train_s1.parquet").filter(pl.col("entity_id").is_in(S1ids.implode())).collect()
R = pl.scan_parquet([f"{IN}/train_s2.parquet", f"{IN}/train_s3.parquet"]).filter(pl.col("entity_id").is_in(pool["r"].unique().implode())).collect()
gt = pl.read_parquet(f"{IN}/gt_rows.parquet").filter(pl.col("source1_entity_id").is_in(S1ids.implode())) \
       .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids") \
       .filter(pl.col("matched_entity_ids") != "").select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("r"))
if "y" not in pool.columns:   # pools without labels (candidate jobs no longer join labels onto 100M+ rows)
    pool = pool.join(gt.with_columns(pl.lit(1, dtype=pl.Int8).alias("y")), on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0))
log(f"pool {pool.height} rows; positives in pool {pool['y'].sum()} / {gt.height} = {pool['y'].sum()/gt.height:.4f}")
F = M.pool_features(pool, s1, R, NZ).join(s1.select(pl.col("entity_id").alias("s1"), "country"), on="s1")
del pool; fc = M.feature_columns(F); log(f"features {len(fc)}  {time.time()-T0:.0f}s")

F = F.with_columns((pl.col("s1").hash(11) % 100).alias("h"))
A, Bv, Cv = F.filter(pl.col("h") < 60), F.filter((pl.col("h") >= 60) & (pl.col("h") < 90)), F.filter(pl.col("h") >= 90)
del F
params = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=os.cpu_count())
dA = lgb.Dataset(M.X(A, fc), A["y"].to_numpy(), feature_name=fc, free_raw_data=True)
dB = lgb.Dataset(M.X(Bv, fc), Bv["y"].to_numpy(), reference=dA)
m = lgb.train(params, dA, 4000, valid_sets=[dB], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(250)])
m.save_model(f"{WD}/matcher.txt"); del dA, dB, A
Bv = Bv.with_columns(pl.Series("p", m.predict(M.X(Bv, fc)))); Cv = Cv.with_columns(pl.Series("p", m.predict(M.X(Cv, fc))))
log("top features (gain):", [(f, int(g)) for f, g in sorted(zip(fc, m.feature_importance("gain")), key=lambda x: -x[1])[:25]])
sub = lambda g_, d: g_.filter(pl.col("s1").is_in(d["s1"].unique().implode()))
gB, gC = sub(gt, Bv), sub(gt, Cv); sB, sC = Bv["s1"].unique(), Cv["s1"].unique()
f05 = lambda sel, g_, s_: macro_f05(sel, g_, s_)

# ---- decision rules (tuned on B, reported on C) ----------------------------------------------------------------------
# coarse grid (0.1) then a local refinement (0.05) around the best point: each evaluation is a full pass over B
ev = lambda a, b: f05(rank_threshold(Bv, a, b), gB, sB)["f05"]
t1, t2, fb = max(((a, b, ev(a, b)) for a in np.arange(0.2, 0.95, 0.1) for b in np.arange(0.2, 0.95, 0.1)), key=lambda x: x[2])
t1, t2, _ = max(((a, b, ev(a, b)) for a in (t1 - 0.05, t1, t1 + 0.05) for b in (t2 - 0.05, t2, t2 + 0.05) if 0.05 < a < 0.99 and 0.05 < b < 0.99), key=lambda x: x[2])
gam = max(((g, f05(M.efd(Bv, g), gB, sB)["f05"]) for g in (0.6, 0.8, 1.0, 1.25, 1.5, 2.0)), key=lambda x: x[1])[0]
# has-head: trained on half of B (out-of-sample p), rules compared on the other half of B, reported on C
HB = M.s1_agg(Bv).join(gB.group_by("s1").len().rename({"len": "nt"}), on="s1", how="left").with_columns((pl.col("nt").fill_null(0) > 0).cast(pl.Int8).alias("y1"))
hp = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=200, verbose=-1, num_threads=os.cpu_count())
head = lgb.train(hp, lgb.Dataset(M.X(HB, M.HEAD_FEATURES), HB["y1"].to_numpy()), 400); head.save_model(f"{WD}/head.txt")
cfg0 = dict(t1=float(t1), t2=float(t2), gamma=float(gam))
log(f"\nrank thresholds t1={t1:.2f} t2={t2:.2f}; expected-F gamma={gam}")
log("=== decision rules (no extra pruning) ===")
resB, resC = {}, {}
for rule in M.RULES:
    cfg = dict(cfg0, rule=rule)
    resB[rule] = f05(M.decide(Bv, cfg, head), gB, sB)["f05"]   # has-head is in-sample on B: C is the honest number
    mm = f05(M.decide(Cv, cfg, head), gC, sC); resC[rule] = mm
    log(f"  {rule:26s} B {resB[rule]:.4f} | C {mm['f05']:.4f}  singleton {mm['singleton_f']:.4f}  non-singleton {mm['nonsingleton_f']:.4f}  P {mm['pair_precision']:.4f} R {mm['pair_recall']:.4f}")
rule = max((r for r in M.RULES if r != "expected-F + has-head"), key=lambda r: resB[r])
log(f"rule chosen on B (has-head excluded from choice: in-sample on B): {rule}; has-head on C for reference {resC['expected-F + has-head']['f05']:.4f}")
ceil = f05(Cv.filter(pl.col("y") == 1).select("s1", "r"), gC, sC); log(f"ceiling on C (perfect decisions on the pool): {ceil['f05']:.4f}")
per = resC[rule]["per"].join(s1.select(pl.col("entity_id").alias("s1"), "country"), on="s1")
log(per.group_by("country").agg(pl.col("f").mean(), pl.len()))
# reference for the label-free France check in the submission job: pseudo-pair hit rate on C (with true precision)
selC = M.decide(Cv, dict(cfg0, rule=rule), head)
for c in ("US", "India"):
    sc_ = s1.filter(pl.col("country") == c).filter(pl.col("entity_id").is_in(sC.implode()))
    pp = pseudo_pairs(NZ.transform(sc_), NZ.transform(R.filter(pl.col("country") == c))).join(Cv.select("s1", "r"), on=["s1", "r"], how="semi")
    prec = pp.join(gC, on=["s1", "r"], how="semi").height / max(pp.height, 1)
    hit = pp.join(selC.select("s1", "r"), on=["s1", "r"], how="semi").height / max(pp.height, 1)
    log(f"  {c} C pseudo-pairs {pp.height}: true precision {prec:.4f}, predicted as match {hit:.4f}  (compare with the France number in er-submit)")

# ---- blocking policy: end-to-end F per policy (Pareto points of the full-mode frontiers) --------------------------
best, pts = -1.0, []
if FRONTS:
    fr = pl.concat([pl.read_csv(f) for f in FRONTS]).group_by(["alpha", "G", "beta"]).agg(pl.col("cands").mean(), pl.col("complete").mean()).sort("cands")
    for r in fr.iter_rows(named=True):
        if r["complete"] > best + 1e-4: pts.append(r); best = r["complete"]
else:   # no frontier sweep upstream: a few group limits without the (lossy) reverse-preference filter
    pts = [dict(alpha=0.0, G=g, beta=-1.0, cands=None, complete=None) for g in (5, 10, 20)]
pts.append(dict(alpha=0.0, G=99, beta=-1.0, cands=None, complete=None))  # no pruning (internal pool)
pts = [dict(cutoff=n) for n in (5, 8, 9, 10, 11, 12, 14)] + pts   # deterministic feature cut-offs (ber.model._cut_rules)
rows = []
for r in pts:
    pol = dict(cutoff=r["cutoff"]) if "cutoff" in r else dict(alpha=r["alpha"], G=int(r["G"]), gate=False, beta=r["beta"])
    kb, kc = M.apply_policy(Bv, pol), M.apply_policy(Cv, pol)
    cfg = dict(cfg0, rule=rule)
    rows.append(dict(policy=json.dumps(pol), cands_B=kb.height / len(sB), recall_B=kb["y"].sum() / max(gB.height, 1),
                     F_B=f05(M.decide(kb, cfg, head), gB, sB)["f05"], cands_C=kc.height / len(sC), F_C=f05(M.decide(kc, cfg, head), gC, sC)["f05"]))
E = pl.DataFrame(rows).sort("cands_B"); log("\n=== END-TO-END per blocking policy ==="); log(E)
fmax = E["F_B"].max()
pick = E.filter(pl.col("F_B") >= fmax - 0.0005).sort("cands_B").row(0, named=True)   # smallest set within 0.0005 of best
log(f"policy chosen on B (smallest candidate set within 0.0005 of the best F): {pick}")
dec = dict(cfg0, rule=rule, policy=json.loads(pick["policy"]), features=fc,
           head_features=M.HEAD_FEATURES, ref=REF, holdout_C=dict(F=pick["F_C"], cands=pick["cands_C"]))
json.dump(dec, open(f"{WD}/decision.json", "w"), indent=1)
# level-1 outputs for stacking: B/C rows kept by the chosen policy (+ raw ids so heavy models can score the same pairs)
for nm, d in (("B", Bv), ("C", Cv)):
    M.apply_policy(d, dec["policy"]).drop("h").write_parquet(f"{WD}/level1_{nm}.parquet")
log(f"wrote level1_B/C.parquet (policy-pruned rows with features, p, y)")
log(json.dumps(dec)[:600]); log(f"\nDONE {time.time()-T0:.0f}s")
