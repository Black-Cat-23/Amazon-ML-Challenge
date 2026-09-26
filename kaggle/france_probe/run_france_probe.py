"""France probe: why are ~5% of France pseudo-pairs (label-free, >=99% precise on US/India) rejected by the matcher?
One France S1 shard: features + level-1 p; saves missed vs hit pseudo-pairs with raw text and features."""
import glob, json, os, subprocess, sys, time
REF = "__REF__"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm==4.6.0",
                f"git+https://github.com/Black-Cat-23/Amazon-ML-Challenge.git@{REF}#subdirectory=code/business_entity_resolution"], check=True)
import numpy as np, polars as pl, lightgbm as lgb
from ber.normalize import Normalizer
from ber.artifacts import pseudo_pairs
from ber import model as M
T0 = time.time(); WD = "/kaggle/working"
find = lambda f: sorted(glob.glob(f"/kaggle/input/**/{f}", recursive=True))
IN = os.path.dirname(find("test_s1.parquet")[0]); ART = os.path.dirname(find("indic_lexicon.parquet")[0])
MD = os.path.dirname(find("decision.json")[0]); POOL = find("pool_test_France.parquet")[0]
cfg = json.load(open(f"{MD}/decision.json")); m = lgb.Booster(model_file=f"{MD}/matcher.txt"); fc = cfg["features"]
NZ = Normalizer(ART)
S = pl.read_parquet(f"{IN}/test_s1.parquet").filter(pl.col("country") == "France")
st = M.claim_stats(POOL)
pool = pl.scan_parquet(POOL).filter(pl.col("s1").hash(3) % 8 == 0).collect().join(st, on="r", how="left")
pool = pool.join(M.s1_name_freq(S, NZ), on="s1", how="left")
Sk = S.filter(pl.col("entity_id").is_in(pool["s1"].unique().implode()))
R = pl.scan_parquet([f"{IN}/test_s2.parquet", f"{IN}/test_s3.parquet"]).filter(pl.col("entity_id").is_in(pool["r"].unique().implode())).collect()
QN, RN = NZ.transform(Sk), NZ.transform(R)
F = M.pool_features(pool, Sk, R, NZ, QN=QN, RN=RN)
F = F.with_columns(pl.Series("p", m.predict(M.X(F, fc))))
pp = pseudo_pairs(QN, RN).with_columns(pl.lit(1, dtype=pl.Int8).alias("pseudo"))
x = F.join(pp, on=["s1", "r"], how="inner")
x = x.with_columns((pl.col("p") > cfg["t2"]).alias("hit"))
print(f"pseudo-pairs in pool {x.height}; p > t2 ({cfg['t2']:.2f}): {x['hit'].mean():.4f}; kept by cut-off 11: {x.filter(M.cutoff_mask(11)).height / x.height:.4f}", flush=True)
num = [c for c in fc if x[c].dtype.is_numeric()]
summ = pl.concat([x.filter(pl.col("hit") == h).select([pl.col(c).cast(pl.Float64).mean().alias(c) for c in num]).with_columns(pl.lit(h).alias("hit")) for h in (True, False)])
summ.write_csv(f"{WD}/feature_means_hit_vs_missed.csv")
txt = lambda df, k: df.select(pl.col("entity_id").alias(k), pl.col("business_name").alias(f"name_{k}"), pl.col("business_address").alias(f"addr_{k}"))
miss = x.filter(~pl.col("hit")).join(txt(Sk, "s1"), on="s1").join(txt(R, "r"), on="r")
miss.write_parquet(f"{WD}/missed_pseudo_pairs.parquet")
x.filter(pl.col("hit")).sample(min(5000, x.filter(pl.col("hit")).height), seed=1).join(txt(Sk, "s1"), on="s1").join(txt(R, "r"), on="r").write_parquet(f"{WD}/hit_pseudo_pairs_sample.parquet")
print(f"missed {miss.height}; DONE {time.time()-T0:.0f}s", flush=True)
