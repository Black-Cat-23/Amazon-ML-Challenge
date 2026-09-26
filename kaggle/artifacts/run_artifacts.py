"""Kaggle job: fit all learned artefacts (phase 1) on FULL train (+ unlabelled test for vocabulary/OOV) and report."""
import glob, os, subprocess, sys, time
LOCAL = not os.path.exists("/kaggle")
if LOCAL:
    sys.path.insert(0, os.path.abspath("../../code/business_entity_resolution/src"))
else:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "git+https://github.com/Black-Cat-23/Amazon-ML-Challenge.git#subdirectory=code/business_entity_resolution"], check=True)
import polars as pl
from ber import artifacts as A
from ber.translit import INDIC_RE

T0 = time.time()
WD = "." if LOCAL else "/kaggle/working"; OUT = f"{WD}/artifacts"; os.makedirs(OUT, exist_ok=True)
REP = open(f"{WD}/artifacts_report.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); REP.write(s + "\n"); REP.flush()
pl.Config.set_tbl_rows(70); pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_width_chars(200)
IN = "../../research/eda/cache" if LOCAL else os.path.dirname(glob.glob("/kaggle/input/**/train_s1.parquet", recursive=True)[0])
rd = lambda n: pl.read_parquet(f"{IN}/{n}.parquet")
EMPTY_OOV = pl.DataFrame(schema={"indic": pl.String, "latin": pl.String, "translit": pl.String, "how": pl.String})

s1 = rd("train_s1"); R = pl.concat([rd("train_s2"), rd("train_s3")])
gt = rd("gt_rows").with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids") \
     .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "r"})
t1 = rd("test_s1"); tR = pl.concat([rd("test_s2"), rd("test_s3")])
if LOCAL:
    s1 = s1.head(60000); gt = gt.filter(pl.col("s1").is_in(s1["entity_id"].implode()))
    R = R.filter(pl.col("entity_id").is_in(gt["r"].implode()) | (pl.int_range(pl.len()) < 100000))
    t1 = t1.head(60000); tR = tR.head(300000)
pairs = gt.join(s1.select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("s1_name"), pl.col("business_address").alias("s1_addr")), on="s1") \
          .join(R.select(pl.col("entity_id").alias("r"), pl.col("business_name").alias("r_name"), pl.col("business_address").alias("r_addr")), on="r")
log(f"true pairs {pairs.height}  S1 {s1.height}  R {R.height}  {time.time()-T0:.0f}s")

# 1. Indic lexicon (supervised, train) ------------------------------------------------------------------------------
lex = A.fit_indic_lexicon(pairs.select("s1_name", "r_name"))
lex.write_parquet(f"{OUT}/indic_lexicon.parquet")
log(f"\n[1] Indic lexicon: {lex.height} words; mean share {lex['share'].mean():.4f}")
log(lex.sort("n", descending=True).head(12))

# 2. vocabulary + unseen-word handling ------------------------------------------------------------------------------
vocab = A.fit_vocab(pl.concat([s1.select("business_name", "country"), t1.select("business_name", "country")]))
vocab.write_parquet(f"{OUT}/vocab.parquet"); vall = vocab.group_by("token").agg(pl.col("n").sum())
log(f"\n[2] vocab tokens {vocab.height}")
# 2a validate on TRAIN: hide 30% of lexicon words; recover by (i) siblings (unsupervised) (ii) skeleton/fuzzy fallback
hide = lex.sample(fraction=0.3, seed=0); keep = lex.join(hide.select("indic"), on="indic", how="anti")
recs_tr = pl.concat([s1, R]).select("entity_id", "business_name", "business_address", "country")
sib_tr = A.fit_lexicon_from_siblings(recs_tr, keep)
chk = hide.select("indic", pl.col("latin").alias("gold")).join(sib_tr, on="indic", how="left")
cov = chk.filter(pl.col("latin").is_not_null())
log(f"siblings on TRAIN hidden words ({hide.height}): coverage {cov.height/max(hide.height,1):.3f}, accuracy {(cov['latin'] == cov['gold']).mean() if cov.height else float('nan'):.3f}")
fb = A.fit_oov_map(hide["indic"].to_list(), vall).join(hide.select("indic", pl.col("latin").alias("gold")), on="indic")
log(f"skeleton/fuzzy fallback on the same hidden words: accuracy {(fb['latin'] == fb['gold']).mean():.3f}")
# 2b apply on TEST unseen words
known = set(lex["indic"].to_list())
tw = tR.filter(pl.col("business_name").fill_null("").str.contains(INDIC_RE)).select(pl.col("business_name").str.extract_all(A.TOK_ANY).alias("w")) \
       .explode("w").filter(pl.col("w").str.contains(INDIC_RE)).group_by("w").len()
oov_words = [w for w in tw["w"].to_list() if w not in known]
log(f"test Indic word types {tw.height}; unseen {len(oov_words)} = {tw.filter(pl.col('w').is_in(oov_words))['len'].sum()/max(tw['len'].sum(),1):.4f} of tokens")
recs_te = pl.concat([t1, tR]).select("entity_id", "business_name", "business_address", "country")
sib_te = A.fit_lexicon_from_siblings(recs_te, lex).filter(pl.col("indic").is_in(oov_words))
rest = [w for w in oov_words if w not in set(sib_te["indic"].to_list())]
fbt = A.fit_oov_map(rest, vall) if rest else EMPTY_OOV
oov = pl.concat([sib_te.select("indic", "latin", pl.lit("siblings").alias("how")), fbt.select("indic", "latin", "how")])
oov.write_parquet(f"{OUT}/oov_map.parquet")
log(f"test unseen words: {sib_te.height} via siblings, {fbt.height} via fallback"); log(oov.head(40))

# 3. synonym pairs (addresses, names) --------------------------------------------------------------------------------
sp = pairs.sample(min(pairs.height, 3_000_000), seed=1)
mn = 50 if LOCAL else 200
ae, asyn = A.fit_equivalences(sp, "s1_addr", "r_addr", min_n=mn)
ae.write_parquet(f"{OUT}/addr_edges.parquet"); asyn.write_parquet(f"{OUT}/addr_synonyms.parquet")
log(f"\n[3a] address synonym edges {ae.height}"); log(ae.head(70))
ne, nsyn = A.fit_equivalences(sp.filter(~pl.col("r_name").fill_null("").str.contains(INDIC_RE)), "s1_name", "r_name", min_n=mn)
ne.write_parquet(f"{OUT}/name_edges.parquet"); nsyn.write_parquet(f"{OUT}/name_synonyms.parquet")
log(f"\n[3b] name synonym edges {ne.height}"); log(ne.head(60))

# 4. noise vs decoy words ---------------------------------------------------------------------------------------------
xw = A.fit_extra_words(s1.select("entity_id", "business_name", "country"), R.select("entity_id", "business_name", "country"), gt)
for k, v in xw.items(): v.write_parquet(f"{OUT}/{k}_words.parquet")
log(f"\n[4] equal-name baseline: {xw['equal_baseline'].rows(named=True)}")
mw = 20 if LOCAL else 200
ex = xw["extra"].filter(pl.col("n") >= mw)
log("EXTRA token in R, mostly TRUE copies (noise words):"); log(ex.sort("p_true", descending=True).head(40))
log("EXTRA token in R, mostly NON-matches (decoy words):"); log(ex.sort("p_true").head(40))
log("MISSING token (in S1, dropped from R):"); log(xw["missing"].filter(pl.col("n") >= mw).sort("n", descending=True).head(30))

# 5. alias side --------------------------------------------------------------------------------------------------------
log("\n[5] alias side stats:"); log(A.alias_side_stats(pairs.select("s1_name", "r_name")))
# 6. TEST-TIME per-country address synonyms from pseudo-pairs (unsupervised; covers countries absent from train) ------
from ber.normalize import Normalizer
from ber.tables import ADDR_STOP
NZ = Normalizer(OUT)
ce, csyn = A.fit_country_addr_synonyms(NZ, t1, tR, min_n=20 if LOCAL else 50, stop=ADDR_STOP)
csyn.write_parquet(f"{OUT}/addr_synonyms_country.parquet"); ce.write_parquet(f"{OUT}/addr_edges_country.parquet")
log(f"\n[6] test-time per-country address synonyms: {csyn.height} (pseudo-pairs per country: "
    f"{ce.group_by('country').agg(pl.col('n_pseudo').first()).rows()})")
for c in ce["country"].unique().sort().to_list():
    log(f"-- {c}"); log(ce.filter(pl.col("country") == c).head(40))
log(f"\nDONE {time.time()-T0:.0f}s")
