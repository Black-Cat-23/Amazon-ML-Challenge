"""[account 2 · CPU · internet] Blocking diagnosis for the true copies missing from the C-split candidate lists.
For each (S1, true copy) pair (missed ones + a sample of found ones): the UNCAPPED rank and score of the copy in every
search arm (r9 index), the number of competing records per arm, whether the default caps keep it, and the shared tokens
with their document frequency. Answers "which arm / cap / token rule loses it" before changing blocking."""
import glob, json, os, subprocess, sys, time
REF = "__REF__"
LOCAL = not os.path.exists("/kaggle")
find = lambda pat: sorted(glob.glob(f"/kaggle/input/**/{pat}", recursive=True))
if not LOCAL:
    W = os.path.dirname(find("polars-1.44.2*.whl")[0])
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", W, "polars==1.44.2", "rapidfuzz==3.14.6"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    f"git+https://github.com/Black-Cat-23/Amazon-ML-Challenge.git@{REF}#subdirectory=code/business_entity_resolution"], check=True)
    IN = os.path.dirname(find("train_s1.parquet")[0]); ART = os.path.dirname(find("indic_lexicon.parquet")[0])
    DP = find("diag_pairs.parquet")[0]; WD = "/kaggle/working"
else:
    sys.path.insert(0, os.path.abspath("../../code/business_entity_resolution/src"))
    IN, ART, DP, WD = "../../research/eda/cache", "../../kaggle/artifacts/kout/artifacts", "data/diag_pairs.parquet", "."
import numpy as np, polars as pl
from ber.normalize import Normalizer
from ber.pipeline import norm_chunks
from ber import blocking as B
T0 = time.time()
REP = open(f"{WD}/results_diag.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); REP.write(s + "\n"); REP.flush()
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220)
NZ = Normalizer(ART)
P = pl.read_parquet(DP)
out_rank, out_tok = [], []
for c in ("US", "India"):
    S = pl.scan_parquet(f"{IN}/train_s1.parquet").filter(pl.col("country") == c).collect()
    Pc = P.join(S.select(pl.col("entity_id").alias("s1")), on="s1", how="semi")
    if Pc.height == 0: continue
    R = pl.scan_parquet([f"{IN}/train_s2.parquet", f"{IN}/train_s3.parquet"]).filter(pl.col("country") == c).collect().sort("entity_id")
    RN = norm_chunks(NZ, R).with_columns(pl.Series("id", np.arange(R.height, dtype=np.uint32)))
    idx = B.Index(RN)                                      # r9 defaults: cap 2000, key_cap 200, arms 0,3..8
    log(f"{c}: index over {R.height} records  {time.time()-T0:.0f}s")
    Q = S.filter(pl.col("entity_id").is_in(Pc["s1"].unique().implode())).sort("entity_id")
    QN = NZ.transform(Q).with_columns(pl.Series("id", np.arange(Q.height, dtype=np.uint32)))
    tg = Pc.join(QN.select(pl.col("entity_id").alias("s1"), "id"), on="s1").join(RN.select(pl.col("entity_id").alias("r"), pl.col("id").alias("id_r")), on="r")
    # default-cap result (what the pool would contain before sibling expansion / pruning)
    cap = idx.query(QN).join(tg.select("id", "id_r"), on=["id", "id_r"], how="semi")
    # uncapped per-arm ranks of the target copies
    St = B.tokens(QN, query=True).filter(pl.col("arm").is_in(list(idx.arms))).join(idx.dfc, on="h")
    ids = QN["id"].to_numpy()
    for i in range(0, len(ids), 200):
        ch = St.filter(pl.col("id").is_in(ids[i:i + 200]))
        post = idx._postings(ch["h"].to_numpy())
        a = ch.join(post, on="h").group_by(["id", "arm", "id_r"]).agg((pl.col("idf").cast(pl.Int64).sum() / 1024).alias("sc"))
        a = a.sort("id_r").with_columns(pl.col("sc").rank("ordinal", descending=True).over(["id", "arm"]).alias("rk"),
                                        pl.len().over(["id", "arm"]).alias("n_arm"))
        out_rank.append(a.join(tg.select("id", "id_r"), on=["id", "id_r"], how="semi").join(tg, on=["id", "id_r"]).with_columns(pl.lit(c).alias("country")))
    # shared tokens (arm, df) of every target pair, from the raw token lists of both sides (before the df cap)
    rt = B.tokens(RN.filter(pl.col("id").is_in(tg["id_r"].implode()))).rename({"id": "id_r"})
    qt = B.tokens(QN, query=True)
    sh = tg.select("id", "id_r").join(qt, on="id").join(rt, on=["id_r", "h", "arm"])
    dfall = idx.dfc.select("h", "df")
    out_tok.append(sh.join(dfall, on="h", how="left").join(tg, on=["id", "id_r"]).group_by("s1", "r", "kind", "arm")
                   .agg(pl.len().alias("n_shared"), pl.col("df").is_null().sum().alias("n_over_cap"), pl.col("df").min().alias("min_df"))
                   .with_columns(pl.lit(c).alias("country")))
    cap = cap.join(tg, on=["id", "id_r"])
    out_rank.append(cap.select("s1", "r", "kind", pl.lit(255, dtype=pl.UInt8).alias("arm"), pl.col("prk").cast(pl.Float64).alias("sc"),
                               pl.col("xrk").cast(pl.UInt32).alias("rk"), pl.lit(0, dtype=pl.UInt32).alias("n_arm"), pl.lit(c).alias("country")))
    del idx, RN
RK = pl.concat([x.select("s1", "r", "kind", "country", pl.col("arm").cast(pl.UInt8), pl.col("sc").cast(pl.Float64), pl.col("rk").cast(pl.UInt32), pl.col("n_arm").cast(pl.UInt32)) for x in out_rank])
TK = pl.concat(out_tok)
RK.write_parquet(f"{WD}/diag_ranks.parquet"); TK.write_parquet(f"{WD}/diag_tokens.parquet")
caps = {**B.CAP}
# summary: per kind, share of pairs reached by each arm within its cap / at all
arm_rows = RK.filter(pl.col("arm") != 255).with_columns(pl.col("rk") <= pl.col("arm").replace_strict(caps, default=10, return_dtype=pl.UInt32).alias("in_cap"))
tot = P.group_by("kind").len()
log(arm_rows.group_by("kind", "arm").agg(pl.len().alias("reached"), pl.col("in_cap").sum().alias("in_cap"), pl.col("rk").median().alias("med_rk"))
    .join(tot, on="kind").with_columns((pl.col("reached") / pl.col("len")).round(3).alias("reach%"), (pl.col("in_cap") / pl.col("len")).round(3).alias("incap%")).sort("kind", "arm"))
anyc = arm_rows.group_by("s1", "r", "kind").agg(pl.col("in_cap").any().alias("any_in_cap"), pl.len().alias("n_arms"))
log(P.join(anyc, on=["s1", "r", "kind"], how="left").fill_null(False).group_by("kind").agg(pl.col("any_in_cap").mean().alias("retrieved by some arm within caps"), pl.len()))
log(TK.group_by("kind", "arm").agg(pl.len(), pl.col("n_over_cap").mean().alias("mean tokens over df cap"), pl.col("n_shared").mean().alias("mean shared")).sort("kind", "arm"))
log(f"DONE {time.time()-T0:.0f}s")
