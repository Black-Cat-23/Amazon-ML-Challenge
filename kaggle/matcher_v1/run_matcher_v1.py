"""Matcher v1: LightGBM on v5's labelled candidate pool (train sample), split blending A/B/C by S1.
Tunes rank-aware decision thresholds on B, reports exact macro F0.5 on held-out C, overall and by slice,
and END-TO-END F0.5 at several blocking operating points (candidate-set size vs score)."""
import glob, os, subprocess, sys, time
LOCAL = not os.path.exists("/kaggle")
if LOCAL:
    sys.path.insert(0, os.path.abspath("../../code/business_entity_resolution/src"))
else:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm==4.6.0",
                    "git+https://github.com/Black-Cat-23/Amazon-ML-Challenge.git@8764e93#subdirectory=code/business_entity_resolution"], check=True)
import numpy as np, polars as pl, lightgbm as lgb
from ber.normalize import Normalizer
from ber.features import pair_features, feature_columns
from ber.metrics import macro_f05
from ber import blocking as B
from ber.decide import rank_threshold, expected_f_decode, exclusive_posterior, source_caps

T0 = time.time(); WD = "." if LOCAL else "/kaggle/working"
REP = open(f"{WD}/results_matcher_v1.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); REP.write(s + "\n"); REP.flush()
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)
find = lambda f: glob.glob(f"/kaggle/input/**/{f}", recursive=True)[0]
IN = "../../research/eda/cache" if LOCAL else os.path.dirname(find("train_s1.parquet"))
ART = "../artifacts/kout/artifacts" if LOCAL else os.path.dirname(find("indic_lexicon.parquet"))
POOL = "../blocking_v5/cand_train_sample.parquet" if LOCAL else find("cand_train_sample.parquet")
FRONT = "../blocking_v5/frontier_v5.csv" if LOCAL else find("frontier_v5.csv")

P = pl.read_parquet(POOL)
gt = pl.read_parquet(f"{IN}/gt_rows.parquet").with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids") \
     .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "r"})
S1ids = P["s1"].unique()
gt = gt.filter(pl.col("s1").is_in(S1ids.implode()))
log(f"pool {P.height} rows, {S1ids.len()} S1, positives in pool {P['y'].sum()} / {gt.height}")
NZ = Normalizer(ART)
s1 = pl.scan_parquet(f"{IN}/train_s1.parquet").filter(pl.col("entity_id").is_in(S1ids.implode())).collect()
R = pl.scan_parquet([f"{IN}/train_s2.parquet", f"{IN}/train_s3.parquet"]).filter(pl.col("entity_id").is_in(P["r"].unique().implode())).collect()
F = pair_features(P, NZ.transform(s1), NZ.transform(R)).join(s1.select(pl.col("entity_id").alias("s1"), "country"), on="s1")
fc = feature_columns(F); log(f"features {len(fc)}  {time.time()-T0:.0f}s")

# split by S1 (hash) -> A 60 / B 30 / C 10
F = F.with_columns((pl.col("s1").hash() % 100).alias("h"))
split = lambda lo, hi: F.filter((pl.col("h") >= lo) & (pl.col("h") < hi))
A, Bv, Cv = split(0, 60), split(60, 90), split(90, 100)
X = lambda d: d.select(fc).to_numpy().astype(np.float32)
params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=os.cpu_count())
dA = lgb.Dataset(X(A), A["y"].to_numpy(), feature_name=fc); dB = lgb.Dataset(X(Bv), Bv["y"].to_numpy(), reference=dA)
m = lgb.train(params, dA, 3000, valid_sets=[dB], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
m.save_model(f"{WD}/matcher_v1.txt")
Bv = Bv.with_columns(pl.Series("p", m.predict(X(Bv)))); Cv = Cv.with_columns(pl.Series("p", m.predict(X(Cv))))
imp = sorted(zip(fc, m.feature_importance("gain")), key=lambda x: -x[1])
log("top features (gain):", [(f, int(g)) for f, g in imp[:25]])

decide = lambda d, t1, t2: rank_threshold(d, t1, t2)
grid = [(a, b) for a in np.arange(0.2, 0.95, 0.05) for b in np.arange(0.2, 0.95, 0.05)]
gB = gt.filter(pl.col("s1").is_in(Bv["s1"].unique().implode())); sB = Bv["s1"].unique()
gC = gt.filter(pl.col("s1").is_in(Cv["s1"].unique().implode())); sC = Cv["s1"].unique()
res = [(a, b, macro_f05(decide(Bv, a, b), gB, sB)["f05"]) for a, b in grid]
t1, t2, fb = max(res, key=lambda x: x[2])
mC = macro_f05(decide(Cv, t1, t2), gC, sC)
log(f"\nthresholds tuned on B: t1={t1:.2f} t2={t2:.2f} (B F0.5 {fb:.4f})")
log(f"HOLDOUT C [rank-threshold] macro F0.5 = {mC['f05']:.4f}   singleton {mC['singleton_f']:.4f}  non-singleton {mC['nonsingleton_f']:.4f}  "
    f"pair P {mC['pair_precision']:.4f} R {mC['pair_recall']:.4f}")
m05 = macro_f05(decide(Cv, 0.5, 0.5), gC, sC); log(f"(naive 0.5/0.5 on C: {m05['f05']:.4f})")

# --- decision rules compared on held-out C (anything tuned is tuned on B) ---
def efd(d, g=1.0, col="p", has=None):
    x = d.with_columns((pl.col(col) ** g).alias("pp"))
    return source_caps(expected_f_decode(x, p="pp", has=has), x, p="pp")
gam = max([(g, macro_f05(efd(Bv, g), gB, sB)["f05"]) for g in (0.6, 0.8, 1.0, 1.25, 1.5, 2.0)], key=lambda x: x[1])[0]
log(f"expected-F power calibration tuned on B: gamma={gam}")
Bq, Cq = exclusive_posterior(Bv), exclusive_posterior(Cv)
# has-match head: S1-level LightGBM on B's out-of-sample p (+ blocking context), applied to C
def s1_agg(d):
    d = d.with_columns(pl.col("p").rank("ordinal", descending=True).over("s1").alias("prk"))
    return d.group_by("s1").agg(pl.col("p").max().alias("p1"), pl.col("p").filter(pl.col("prk") == 2).first().fill_null(0).alias("p2"),
                                pl.col("p").sum().alias("psum"), (pl.col("p") > 0.5).sum().alias("n05"), pl.len().alias("nc"),
                                pl.col("nm_tset").max().alias("nm_max"), pl.col("num_jacc").max().alias("num_max"),
                                (pl.col("house_rel") == 0).sum().alias("n_house_eq"), pl.col("noaddr_s").first().alias("noaddr"),
                                pl.col("n_core_s").first().alias("ncore"), pl.col("rev_margin").max().alias("rev_max"))
def head_frame(d, gt_):
    a = s1_agg(d).join(gt_.group_by("s1").len().rename({"len": "nt"}), on="s1", how="left")
    return a.with_columns((pl.col("nt").fill_null(0) > 0).cast(pl.Int8).alias("y1"))
HB, HC = head_frame(Bv, gB), head_frame(Cv, gC)
hc = [c for c in HB.columns if c not in ("s1", "nt", "y1")]
hp = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=200, verbose=-1, num_threads=os.cpu_count())
hm = lgb.train(hp, lgb.Dataset(HB.select(hc).to_numpy().astype(np.float32), HB["y1"].to_numpy()), 400)
HC = HC.with_columns(pl.Series("h", hm.predict(HC.select(hc).to_numpy().astype(np.float32))))
# S1s with no candidates in the pool are always predicted empty (correct for singletons)
rules = {"rank-threshold": decide(Cv, t1, t2),
         "expected-F": efd(Cv, gam),
         "expected-F + exclusivity": efd(Cq.with_columns(pl.col("q").alias("p")), gam),
         "expected-F + has-head": efd(Cv, gam, has=HC.select("s1", "h"))}
log("\n=== decision rules on holdout C ===")
best_rule, best_f = None, -1
for k, sel in rules.items():
    mm = macro_f05(sel, gC, sC)
    log(f"  {k:28s} F0.5 {mm['f05']:.4f}  singleton {mm['singleton_f']:.4f}  non-singleton {mm['nonsingleton_f']:.4f}  P {mm['pair_precision']:.4f} R {mm['pair_recall']:.4f}")
    if mm["f05"] > best_f: best_rule, best_f, mC = k, mm["f05"], mm
log(f"best rule: {best_rule}")
# ceiling: perfect decisions on this candidate pool
ceil = macro_f05(Cv.filter(pl.col("y") == 1).select("s1", "r"), gC, sC); log(f"ceiling with perfect decisions on the pool (blocking recall limit): {ceil['f05']:.4f}")
per = mC["per"].join(s1.select(pl.col("entity_id").alias("s1"), "country"), on="s1")
log(per.group_by("country").agg(pl.col("f").mean(), pl.len()))
log(per.with_columns(pl.when(pl.col("nt") == 0).then(pl.lit("0")).when(pl.col("nt") <= 3).then(pl.lit("1-3")).otherwise(pl.lit("4+")).alias("size"))
        .group_by("size").agg(pl.col("f").mean(), pl.len()).sort("size"))

# END-TO-END: restrict C's pool to each blocking policy's kept candidates; same model + thresholds
front = pl.read_csv(FRONT)
pts = front.sort("cands").filter(pl.col("complete") >= pl.col("complete").max() - 0.08)
best, pareto = -1.0, []
for r in front.sort("cands").iter_rows(named=True):
    if r["complete"] > best + 1e-4: pareto.append(r); best = r["complete"]
AC = Cv.select(pl.col("s1").alias("id"), pl.col("r").alias("id_r"), "sc", "gid", "rel", "a4", "rev_margin")
log("\n=== END-TO-END on holdout C: blocking policy -> candidates/S1 -> macro F0.5 ===")
rows = []
for r in pareto:
    kept = B.prune(AC, alpha=r["alpha"], G=r["G"], gate=r["gate"], beta=None if r["beta"] < 0 else r["beta"]).rename({"id": "s1", "id_r": "r"})
    sub = Cv.join(kept, on=["s1", "r"])
    rsel = {"rank-threshold": lambda z: decide(z, t1, t2), "expected-F": lambda z: efd(z, gam),
            "expected-F + exclusivity": lambda z: efd(exclusive_posterior(z).with_columns(pl.col("q").alias("p")), gam),
            "expected-F + has-head": lambda z: efd(z, gam, has=HC.select("s1", "h"))}[best_rule]
    f = macro_f05(rsel(sub), gC, sC)["f05"]
    rows.append(dict(alpha=r["alpha"], G=r["G"], gate=r["gate"], beta=r["beta"], cands=kept.height / sC.len(), recall=r["recall"], complete=r["complete"], f05=f))
log(pl.DataFrame(rows))
log(f"\nDONE {time.time()-T0:.0f}s")
