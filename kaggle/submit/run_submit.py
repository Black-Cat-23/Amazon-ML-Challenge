"""Submission: FULL-mode test pools -> blocking policy (candidate_pairs) -> matcher -> decision rule (matching_results).
Inputs: er-cands-test-* (pool_test_<country>.parquet), er-matcher-full (matcher.txt, head.txt, decision.json), artifacts, data."""
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
from ber.io import write_submission
from ber import model as M
from ber.artifacts import pseudo_pairs

T0 = time.time(); WD = "." if LOCAL else "/kaggle/working"
REP = open(f"{WD}/results_submit.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); REP.write(s + "\n"); REP.flush()
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
find = lambda f: sorted(glob.glob(f"/kaggle/input/**/{f}", recursive=True))
IN = "../../research/eda/cache" if LOCAL else os.path.dirname(find("test_s1.parquet")[0])
ART = "../artifacts/kout/artifacts" if LOCAL else os.path.dirname(find("indic_lexicon.parquet")[0])
POOLS = sorted(glob.glob("../cands_full/pool_test_*.parquet")) if LOCAL else find("pool_test_*.parquet")
MD = "../matcher_full" if LOCAL else os.path.dirname(find("decision.json")[0])
cfg = json.load(open(f"{MD}/decision.json")); m = lgb.Booster(model_file=f"{MD}/matcher.txt")
head = lgb.Booster(model_file=f"{MD}/head.txt") if os.path.exists(f"{MD}/head.txt") else None
fc = cfg["features"]
# optional level 2 (er-stack + cross-encoder scores on the test uncertain band from er-xenc-score)
SJ = find("decision_stack.json") if not LOCAL else []
stack = lgb.Booster(model_file=os.path.join(os.path.dirname(SJ[0]), "stack.txt")) if SJ else None
if SJ:
    cfg2 = json.load(open(SJ[0]))
    if cfg2.get("delta_C", 0) < 0.002:     # acceptance gate: level 2 only if it beat level 1 on holdout C by >= 0.002
        log(f"level 2 REJECTED by the gate (delta on C {cfg2.get('delta_C')}) -> level 1"); stack = None
    else:
        log(f"LEVEL 2 active: delta on C {cfg2.get('delta_C')}")
log(f"ref {REF}; model ref {cfg.get('ref')}; rule {cfg['rule']}; policy {cfg['policy']}; holdout {cfg.get('holdout_C')}")
NZ = Normalizer(ART)
s1_all = pl.read_parquet(f"{IN}/test_s1.parquet")
valid_r = pl.concat([pl.read_parquet(f"{IN}/test_s2.parquet", columns=["entity_id"]), pl.read_parquet(f"{IN}/test_s3.parquet", columns=["entity_id"])])["entity_id"]
cands, matches = [], []
NSH = int(os.environ.get("NSH", 1 if LOCAL else 8))   # S1 shards per country: features for 70M+ rows do not fit at once
for p in POOLS:
    ctry = os.path.basename(p)[len("pool_test_"):-len(".parquet")]
    st = M.claim_stats(p)                                   # record-level stats over the WHOLE pool (streaming)
    S = s1_all.filter(pl.col("country") == ctry)
    nf = M.s1_name_freq(S, NZ)
    # normalise every S1 / S2 / S3 record of the country ONCE (shards reuse them; a record appears under many S1s)
    SN = pl.concat([NZ.transform(S.slice(i, 1_000_000)) for i in range(0, S.height, 1_000_000)])
    Rall = pl.scan_parquet([f"{IN}/test_s2.parquet", f"{IN}/test_s3.parquet"]).filter(pl.col("country") == ctry).collect()
    RNall = pl.concat([NZ.transform(Rall.slice(i, 1_000_000)) for i in range(0, Rall.height, 1_000_000)])
    log(f"{ctry}: normalised {S.height} S1 + {Rall.height} S2/S3 once  {time.time()-T0:.0f}s")
    for k in range(NSH):
        pool = pl.scan_parquet(p).filter(pl.col("s1").hash(3) % NSH == k).collect(engine="streaming").join(st, on="r", how="left")
        pol = cfg["policy"]
        kept = pool if pol.get("cutoff") else M.prune_pool(pool, pol); del pool   # cut-off rules need the features first
        kept = kept.join(nf, on="s1", how="left")
        Sk = S.filter(pl.col("entity_id").is_in(kept["s1"].unique().implode()))
        QNk = SN.filter(pl.col("entity_id").is_in(kept["s1"].unique().implode()))
        RNk = RNall.filter(pl.col("entity_id").is_in(kept["r"].unique().implode()))
        F = M.pool_features(kept, Sk, None, NZ, QN=QNk, RN=RNk)
        if pol.get("cutoff"):
            F = F.filter(M.cutoff_mask(int(pol["cutoff"])))   # candidate_pairs = rows kept by the deterministic rules
            kept = F.select("s1", "r")
        F = F.with_columns(pl.Series("p", m.predict(M.X(F, fc))))
        F.write_parquet(f"{WD}/level1_test_{ctry}_{k}.parquet")   # for er-xenc-score (uncertain band) and level 2
        xf = find(f"xenc_test_{ctry}_{k}.parquet") if not LOCAL else []
        if stack is not None and xf:
            F = F.join(pl.read_parquet(xf[0]), on=["s1", "r"], how="left")
            F = F.with_columns(pl.Series("p", stack.predict(M.X(F, cfg2["stack_features"]))))
            sel = M.decide(F, cfg2, head)
        else:
            sel = M.decide(F, cfg, head)
        # label-free transfer check: pseudo-pairs (>=99% precise on US/India train) that are candidates -> share predicted
        pp = pseudo_pairs(QNk, RNk).join(kept.select("s1", "r"), on=["s1", "r"], how="semi")
        hit = pp.join(sel.select("s1", "r"), on=["s1", "r"], how="semi").height
        log(f"{ctry}[{k}]: S1 {Sk.height} candidates {kept.height} matches {sel.height}; pseudo-pairs {pp.height} predicted {hit/max(pp.height,1):.4f}  {time.time()-T0:.0f}s")
        cands.append(kept.select("s1", "r")); matches.append(sel.select("s1", "r"))
        del F, kept, QNk, RNk
    del SN, RNall, Rall
    cn = sum(c.height for c in cands[-NSH:]); mn = pl.concat(matches[-NSH:])
    log(f"{ctry}: S1 {S.height}  candidates {cn} ({cn/S.height:.2f}/S1)  matches {mn.height} ({mn.height/S.height:.2f}/S1)  S1 predicted empty {1 - mn['s1'].n_unique()/S.height:.4f}")
C = pl.concat(cands); Mt = pl.concat(matches)
write_submission(Mt, C, s1_all["entity_id"], valid_r, f"{WD}/matching_results.tsv", f"{WD}/candidate_pairs.tsv")
log(f"total: candidates/S1 {C.height/s1_all.height:.2f}, matches/S1 {Mt.height/s1_all.height:.2f}, S1 with no match {1 - Mt['s1'].n_unique()/s1_all.height:.4f}")
log(f"wrote matching_results.tsv + candidate_pairs.tsv  DONE {time.time()-T0:.0f}s")
