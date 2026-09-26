"""Recall lab (account 2, CPU, offline): one blocking variant per kernel. Full S2/S3 pool per country, 30k train S1 queried.
Reports pair recall, S1 completeness and internal-pool size by slice (country, no-address, Indian-script copies, scrambled
names), plus index/query time. Variant is baked in by make_jobs.py."""
import glob, json, os, subprocess, sys, time
VAR = json.loads('__VARIANT__') if not "__VARIANT__".startswith("__") else {"name": "local", "cap": 5000, "key_cap": 200, "caps": {}, "keep_prk": 60}
LOCAL = not os.path.exists("/kaggle")
find = lambda pat: sorted(glob.glob(f"/kaggle/input/**/{pat}", recursive=True))
if not LOCAL:
    W = os.path.dirname(find("polars-1.44.2*.whl")[0])
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", W, "polars==1.44.2", "rapidfuzz==3.14.6"], check=True)
    if VAR.get("ref"):   # account 2 is phone-verified now: install ber from GitHub at a pinned commit
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                        f"git+https://github.com/Black-Cat-23/Amazon-ML-Challenge.git@{VAR['ref']}#subdirectory=code/business_entity_resolution"], check=True)
    else:
        # code: prefer the small, frequently updated er-src dataset over the copy inside er-bundle
        cands = find("er-src/**/ber/__init__.py") or find("ber/__init__.py")
        sys.path.insert(0, os.path.dirname(os.path.dirname(cands[0])))
    IN = os.path.dirname(find("train_s1.parquet")[0]); ART = os.path.dirname(find("indic_lexicon.parquet")[0]); WD = "/kaggle/working"
else:
    sys.path.insert(0, os.path.abspath("../../code/business_entity_resolution/src"))
    IN, ART, WD = "../../research/eda/cache", "../../kaggle/artifacts/kout/artifacts", "."
if VAR.get("places"):
    os.environ["BER_PLACES"] = "1"      # read by ber.normalize at import
import numpy as np, polars as pl
from ber.normalize import Normalizer
from ber.pipeline import norm_chunks
from ber.translit import INDIC_RE
from ber import blocking as B
T0 = time.time(); N_Q = 2000 if LOCAL else 30000
REP = open(f"{WD}/results_{VAR['name']}.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); REP.write(s + "\n"); REP.flush()
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
log(f"variant {json.dumps(VAR)}; polars {pl.__version__}; ber from {os.path.dirname(B.__file__) if 'B' in dir() else '?'}")
NZ = Normalizer(ART)
gt = pl.read_parquet(f"{IN}/gt_rows.parquet").with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")) \
       .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "") \
       .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("r"))
caps = {**B.CAP, **{int(k): v for k, v in VAR.get("caps", {}).items()}}
B.ADDR_TAIL = int(VAR.get("tail", 0))
rows = []
for c in ("US", "India"):
    S = pl.scan_parquet(f"{IN}/train_s1.parquet").filter(pl.col("country") == c).collect().sort("entity_id")
    R = pl.scan_parquet([f"{IN}/train_s2.parquet", f"{IN}/train_s3.parquet"]).filter(pl.col("country") == c).collect().sort("entity_id")
    if LOCAL: R = R.head(150000)
    Q = S.filter(pl.col("entity_id").hash(21) % 1000 < max(1, int(1000 * N_Q / S.height)))
    t = time.time()
    RN = norm_chunks(NZ, R).with_columns(pl.Series("id", np.arange(R.height, dtype=np.uint32)))
    arms = tuple(VAR.get("arms", [a for a in B.ARMS]))
    idx = B.Index(RN, cap=VAR.get("cap", 5000), key_cap=VAR.get("key_cap", 200), arms=arms); tb = time.time() - t
    QN = NZ.transform(Q).with_columns(pl.Series("id", np.arange(Q.height, dtype=np.uint32)))
    t = time.time(); cand = idx.query(QN, caps=caps); tq = time.time() - t
    sig = B.signatures(RN); ex = B.expand(cand, sig)
    cand = pl.concat([cand.select("id", "id_r", "prk", "xrk"), ex.select("id", "id_r").with_columns(pl.lit(None, dtype=pl.UInt32).alias("prk"), pl.lit(None, dtype=pl.UInt32).alias("xrk"))], how="vertical_relaxed")
    cand = cand.filter((pl.col("prk") <= VAR.get("keep_prk", 60)) | pl.col("xrk").is_not_null() | (pl.col("prk").is_null() & pl.col("xrk").is_null()))
    cand = cand.join(QN.select("id", pl.col("entity_id").alias("s1")), on="id").join(RN.select(pl.col("id").alias("id_r"), pl.col("entity_id").alias("r")), on="id_r")
    g = gt.filter(pl.col("s1").is_in(Q["entity_id"].implode()))
    h = g.join(cand.select("s1", "r").with_columns(pl.lit(True).alias("k")), on=["s1", "r"], how="left").with_columns(pl.col("k").fill_null(False))
    # slices
    h = h.join(Q.select(pl.col("entity_id").alias("s1"), (pl.col("business_address").fill_null("").str.strip_chars().is_in(["", "None"])).alias("s1_noaddr")), on="s1") \
         .join(R.select(pl.col("entity_id").alias("r"), pl.col("business_name").fill_null("").str.contains(INDIC_RE).alias("r_indic"),
                        pl.col("business_address").fill_null("").str.strip_chars().is_in(["", "None"]).alias("r_noaddr")), on="r")
    ent = h.group_by("s1").agg(pl.col("k").all().alias("complete"), pl.col("s1_noaddr").first())
    per = cand.group_by("s1").len()
    row = dict(variant=VAR["name"], country=c, S1=Q.height, cands=cand.height / Q.height, p95=float(np.percentile(per["len"].to_numpy(), 95)),
               recall=h["k"].mean(), complete=ent["complete"].mean(),
               recall_r_indic=h.filter(pl.col("r_indic"))["k"].mean(), recall_r_noaddr=h.filter(pl.col("r_noaddr"))["k"].mean(),
               complete_s1_noaddr=ent.filter(pl.col("s1_noaddr"))["complete"].mean(), build_s=round(tb), query_s=round(tq),
               ms_per_query=1000 * tq / Q.height)
    rows.append(row); log(json.dumps(row))
    del idx, RN, QN, cand
pl.DataFrame(rows).write_csv(f"{WD}/recall_{VAR['name']}.csv")
log(pl.DataFrame(rows)); log(f"DONE {time.time()-T0:.0f}s")
