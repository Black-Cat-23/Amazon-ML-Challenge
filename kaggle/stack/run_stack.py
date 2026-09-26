"""Level-2 combiner: GBDT on [level-1 features + level-1 p + cross-encoder logits] trained on B1 (half of B by hash),
decision tuned on B2, compared with level-1 alone (same protocol) on held-out C. Acceptance gate: C F0.5 +0.002."""
import glob, json, os, subprocess, sys, time
REF = "__REF__"
LOCAL = not os.path.exists("/kaggle")
if LOCAL:
    sys.path.insert(0, os.path.abspath("../../code/business_entity_resolution/src"))
else:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm==4.6.0",
                    f"git+https://github.com/Black-Cat-23/Amazon-ML-Challenge.git@{REF}#subdirectory=code/business_entity_resolution"], check=True)
import numpy as np, polars as pl, lightgbm as lgb
from ber.metrics import macro_f05
from ber import model as M
T0 = time.time(); WD = "." if LOCAL else "/kaggle/working"
REP = open(f"{WD}/results_stack.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); REP.write(s + "\n"); REP.flush()
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
find = lambda f: sorted(glob.glob(f"/kaggle/input/**/{f}", recursive=True))
IN = "../../research/eda/cache" if LOCAL else os.path.dirname(find("train_s1.parquet")[0])
MD = "../matcher_full" if LOCAL else os.path.dirname(find("decision.json")[0])
XD = "../xenc_score" if LOCAL else os.path.dirname(find("xenc_B.parquet")[0])
cfg = json.load(open(f"{MD}/decision.json")); fc = cfg["features"]
head = lgb.Booster(model_file=f"{MD}/head.txt")
def load(nm):
    d = pl.read_parquet(f"{MD}/level1_{nm}.parquet")
    x = pl.read_parquet(f"{XD}/xenc_{nm}.parquet")
    return d.join(x, on=["s1", "r"], how="left")
B, C = load("B"), load("C")
xc = [c for c in B.columns if c.startswith("lg_")]
log(f"B {B.height} rows, C {C.height}; cross-encoder columns {xc}; scored share B {B[xc[0]].is_not_null().mean():.3f}")
S1 = pl.concat([B.select("s1"), C.select("s1")]).unique()
gt = pl.read_parquet(f"{IN}/gt_rows.parquet").filter(pl.col("source1_entity_id").is_in(S1["s1"].implode())) \
       .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids") \
       .filter(pl.col("matched_entity_ids") != "").select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("r"))
# S1s left with no rows after pruning are absent from both arms, so the level-2 minus level-1 delta is unaffected
h2 = pl.col("s1").hash(13) % 2
B1, B2 = B.filter(h2 == 0), B.filter(h2 == 1)
g = lambda d: gt.filter(pl.col("s1").is_in(d["s1"].unique().implode()))
f05 = lambda sel, d: macro_f05(sel, g(d), d["s1"].unique())
# per-pair AUC of each signal on C band rows (diagnostic)
try:
    from sklearn.metrics import roc_auc_score
    band = C.filter(pl.col(xc[0]).is_not_null())
    for c in ["p"] + xc:
        log(f"AUC on C uncertain band ({band.height} rows, pos {band['y'].mean():.3f}): {c} {roc_auc_score(band['y'].to_numpy(), band[c].to_numpy()):.4f}")
except Exception as e:
    log("auc failed", e)
feat2 = fc + ["p"] + xc
params = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=os.cpu_count())
m2 = lgb.train(params, lgb.Dataset(M.X(B1, feat2), B1["y"].to_numpy()), 3000,
               valid_sets=[lgb.Dataset(M.X(B2, feat2), B2["y"].to_numpy())], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(250)])
m2.save_model(f"{WD}/stack.txt")
log("level-2 gain:", [(f, int(v)) for f, v in sorted(zip(feat2, m2.feature_importance("gain")), key=lambda x: -x[1])[:15]])
B2s = B2.with_columns(pl.Series("p", m2.predict(M.X(B2, feat2)))); Cs = C.with_columns(pl.Series("p", m2.predict(M.X(C, feat2))))
def tune(d):
    """re-tune the decision parameters on B2 for the configured rule"""
    rule = cfg["rule"]
    if rule == "rank-threshold":
        grid = [(a, b) for a in np.arange(0.2, 0.95, 0.05) for b in np.arange(0.2, 0.95, 0.05)]
        t1, t2, _ = max(((a, b, f05(M.decide(d, dict(cfg, t1=a, t2=b), head), d)["f05"]) for a, b in grid), key=lambda x: x[2])
        return dict(cfg, t1=float(t1), t2=float(t2))
    gam = max(((gm, f05(M.decide(d, dict(cfg, gamma=gm), head), d)["f05"]) for gm in (0.6, 0.8, 1.0, 1.25, 1.5, 2.0)), key=lambda x: x[1])[0]
    return dict(cfg, gamma=float(gam))
res = {}
for tag, b2, c in (("level-1", B2, C), ("level-2", B2s, Cs)):
    cf = tune(b2); mm = f05(M.decide(c, cf, head), c); res[tag] = (mm, cf)
    log(f"{tag}: C F0.5 {mm['f05']:.4f}  singleton {mm['singleton_f']:.4f}  non-singleton {mm['nonsingleton_f']:.4f}  P {mm['pair_precision']:.4f} R {mm['pair_recall']:.4f}  cfg {({k: cf[k] for k in ('rule','t1','t2','gamma') if k in cf})}")
d = res["level-2"][0]["f05"] - res["level-1"][0]["f05"]
per = res["level-2"][0]["per"].join(res["level-1"][0]["per"].select("s1", pl.col("f").alias("f1")), on="s1")
S1c = pl.scan_parquet(f"{IN}/train_s1.parquet").filter(pl.col("entity_id").is_in(per["s1"].implode())).select(pl.col("entity_id").alias("s1"), "country").collect()
log(per.join(S1c, on="s1").group_by("country").agg((pl.col("f") - pl.col("f1")).mean().alias("delta"), pl.len()))
log(f"DELTA level-2 minus level-1 on C: {d:+.4f}  -> {'ACCEPT' if d >= 0.002 else 'REJECT'} (gate +0.002)")
json.dump(dict(res["level-2"][1], stack_features=feat2, delta_C=d), open(f"{WD}/decision_stack.json", "w"), indent=1)
log(f"DONE {time.time()-T0:.0f}s")
