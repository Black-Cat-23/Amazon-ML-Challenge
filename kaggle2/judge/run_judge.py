"""[account 2 · GPU T4 x2 · internet] Level-3 judge.
GPU0 Qwen/Qwen3.5-2B, GPU1 Qwen/Qwen3-Reranker-4B (both Apache-2.0, LoRA), fine-tuned on the HARD pairs of split B1
(level-1 uncertain band 0.005 < p1 < 0.995; B1 = the half of B that trained the level-2 stack, same hash).
Both then score the level-2 uncertain band (0.01 < p2 < 0.99) of B2 and C. Level 3 = logistic blend of logit(p2) and the
judge logits, fitted on B2 per band width; decision thresholds re-tuned on B2; reported on held-out C (gate +0.002).
Inputs: satvik006/er-bundle (raw data), satvik006/er-judge-in (level1_B/C, xenc_B/C, stack.txt, decision_stack.json).
Short prompt (about half the reranker template's tokens) and length-sorted scoring batches for speed."""
import glob, json, os, subprocess, sys, time
REF = "__REF__"
LOCAL = not os.path.exists("/kaggle")
WD = "." if LOCAL else "/kaggle/working"
if not LOCAL:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)   # old torchao breaks peft
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "peft>=0.13"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "flash-linear-attention", "causal-conv1d"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm==4.6.0", "polars==1.44.2",
                    f"git+https://github.com/satvik-A/amazon-mlss.git@{REF}#subdirectory=code/business_entity_resolution"], check=True)
else:
    sys.path.insert(0, os.path.abspath("../../code/business_entity_resolution/src"))
import numpy as np, polars as pl
T0 = time.time()
REP = open(f"{WD}/results_judge.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); REP.write(s + "\n"); REP.flush()
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
find = lambda f: sorted(glob.glob(f"/kaggle/input/**/{f}", recursive=True))
IN = "../../research/eda/cache" if LOCAL else os.path.dirname(find("train_s1.parquet")[0])
JD = "data" if LOCAL else os.path.dirname(find("level1_B.parquet")[0])
TRAIN_MIN = float(os.environ.get("TRAIN_MIN", 2 if LOCAL else 400))
EPOCHS = float(os.environ.get("EPOCHS", 3))
N_EASY = int(os.environ.get("N_EASY", 6000))
KINDS = os.environ.get("KINDS", "__KINDS__" if not "__KINDS__".startswith("__") else "qwen35:2B,qwen4b").split(",")
LO, HI = 0.01, 0.99                     # level-2 band that the judge scores (same rule at test time)


# ------------------------------------------------ 1. pairs (CPU) ----------------------------------------------------
def build_pairs():
    import lightgbm as lgb
    from ber import model as M
    cfg2 = json.load(open(f"{JD}/decision_stack.json")); m2 = lgb.Booster(model_file=f"{JD}/stack.txt")
    def load(n):
        d = pl.read_parquet(f"{JD}/level1_{n}.parquet").join(pl.read_parquet(f"{JD}/xenc_{n}.parquet"), on=["s1", "r"], how="left")
        return d.with_columns(pl.col("p").alias("p1"), pl.Series("p2", m2.predict(M.X(d, cfg2["stack_features"]))))
    B, C = load("B"), load("C")
    h2 = pl.col("s1").hash(13) % 2                     # the stack's B1 / B2 split
    B1, B2 = B.filter(h2 == 0), B.filter(h2 == 1)
    band1 = (pl.col("p1") > 0.005) & (pl.col("p1") < 0.995)
    tr = pl.concat([B1.filter(band1), B1.filter(~band1).sample(min(N_EASY, B1.filter(~band1).height), seed=4)]).select("s1", "r", "y")
    band2 = (pl.col("p2") > LO) & (pl.col("p2") < HI)
    # all rows of B2 / C (s1, r, p2, y) for the level-3 decision; judge texts only for the band rows
    for nm, d in (("B2", B2), ("C", C)):
        d.select("s1", "r", "y", "p1", "p2").write_parquet(f"{WD}/rows_{nm}.parquet")
    S = pl.read_parquet(f"{IN}/train_s1.parquet"); R = pl.concat([pl.read_parquet(f"{IN}/train_s2.parquet"), pl.read_parquet(f"{IN}/train_s3.parquet")])
    txt = lambda df, k: df.select(pl.col("entity_id").alias(k), (pl.col("business_name").fill_null("").str.slice(0, 120) + " | "
                                  + pl.col("business_address").fill_null("").str.slice(0, 160)).alias(f"t_{k}"))
    ctry = S.select(pl.col("entity_id").alias("s1"), "country")
    def texts(d):
        return d.join(txt(S, "s1"), on="s1").join(txt(R, "r"), on="r").join(ctry, on="s1")
    tr = texts(tr).sample(fraction=1.0, shuffle=True, seed=5)
    ev = pl.concat([texts(B2.filter(band2).select("s1", "r", "y")).with_columns(pl.lit("B2").alias("part")),
                    texts(C.filter(band2).select("s1", "r", "y")).with_columns(pl.lit("C").alias("part"))])
    tr.write_parquet(f"{WD}/pairs_train.parquet"); ev.write_parquet(f"{WD}/pairs_eval.parquet")
    log(f"pairs: train {tr.height} (pos {tr['y'].mean():.3f}; B1 band + {N_EASY} easy); "
        f"eval B2 {ev.filter(pl.col('part') == 'B2').height} + C {ev.filter(pl.col('part') == 'C').height} (band {LO}..{HI})  {time.time()-T0:.0f}s")


# ------------------------------------------------ 2. worker (one per GPU) ------------------------------------------
WORKER = r'''
import json, os, sys, time
import numpy as np, polars as pl, torch
from sklearn.metrics import roc_auc_score
kind, WD, TRAIN_MIN, EPOCHS = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
tag = kind.split("/")[-1].replace(":", "_").replace(".", "_").replace("-", "_")
torch.manual_seed(0); np.random.seed(0)
dev = "cuda"
tr = pl.read_parquet(f"{WD}/pairs_train.parquet"); ev = pl.read_parquet(f"{WD}/pairs_eval.parquet")
L = open(f"{WD}/log_{tag}.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(f"[{kind}] " + s, flush=True); L.write(s + "\n"); L.flush()
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
if kind.startswith("qwen35"):
    from transformers import AutoModelForImageTextToText as AM
    name = "Qwen/Qwen3.5-" + kind.split(":")[1]
    tm = r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj\w*|out_proj)$"
    lr, bs, sbs = (1e-4, 8, 32) if "4B" in kind else (1e-4, 16, 64)
else:
    # "qwen4b" = Qwen3-Reranker-4B; "causal:<hf id>" = any causal-LM reranker (e.g. mixedbread-ai/mxbai-rerank-large-v2)
    from transformers import AutoModelForCausalLM as AM
    name = kind.split(":", 1)[1] if kind.startswith("causal:") else "Qwen/Qwen3-Reranker-4B"
    tm = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lr, bs, sbs = (1e-4, 8, 32) if "4B" in name else (1e-4, 16, 64)
tok = AutoTokenizer.from_pretrained(name, padding_side="left")
model = AM.from_pretrained(name, torch_dtype=torch.float16).to(dev)
model.gradient_checkpointing_enable(); model.enable_input_require_grads()
yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
def prompt(x, y):
    return (f"<|im_start|>user\nEntity Resolution: Do record A and B refer to the same real-world business? "
            f"Account for legal forms (Pvt/Ltd), transliteration, and truncated address. "
            f"If building/store numbers conflict, answer no. Answer strictly yes or no.\nA: {x}\nB: {y}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")
def enc(a, b):
    ids = tok([prompt(x, y) for x, y in zip(a, b)], add_special_tokens=False)["input_ids"]
    m = max(map(len, ids)); pad = tok.pad_token_id
    return dict(input_ids=torch.tensor([[pad] * (m - len(i)) + i for i in ids], device=dev),
                attention_mask=torch.tensor([[0] * (m - len(i)) + [1] * len(i) for i in ids], device=dev))
def fwd(b):
    try:
        lg = model(**b, logits_to_keep=1).logits[:, -1, :]
    except TypeError:
        lg = model(**b).logits[:, -1, :]
    return (lg[:, yes] - lg[:, no]).float()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=tm))
params = [p for p in model.parameters() if p.requires_grad]
log(f"model {name} on {torch.cuda.get_device_name(0)}; trainable {sum(p.numel() for p in params)/1e6:.1f}M; train {tr.height} pairs")

@torch.no_grad()
def score(d):
    """length-sorted batches (much less padding), returned in the input order"""
    model.eval()
    a, b = d["t_s1"].to_list(), d["t_r"].to_list()
    order = np.argsort([len(x) + len(y) for x, y in zip(a, b)])
    out = np.zeros(len(a), dtype=np.float32)
    for i in range(0, len(a), sbs):
        ix = order[i:i + sbs]
        with torch.autocast("cuda", dtype=torch.float16):
            out[ix] = fwd(enc([a[j] for j in ix], [b[j] for j in ix])).cpu().numpy()
    return out

opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
scaler = torch.amp.GradScaler("cuda")
lossf = torch.nn.BCEWithLogitsLoss()
total = int(EPOCHS * tr.height / bs)
steps, seen, t1, run, stop = 0, 0, time.time(), [], False
model.train()
for ep in range(int(np.ceil(EPOCHS))):
    d = tr.sample(fraction=1.0, shuffle=True, seed=100 + ep)
    a, b = d["t_s1"].to_list(), d["t_r"].to_list(); y = torch.tensor(d["y"].to_numpy(), dtype=torch.float32, device=dev)
    for i in range(0, len(a), bs):
        if time.time() - t1 > TRAIN_MIN * 60 or steps >= total:
            stop = True; break
        # linear warm-up 200 steps, then cosine decay over the planned steps
        w = min(1.0, (steps + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(1.0, steps / max(total, 1)))))
        for g in opt.param_groups: g["lr"] = lr * w
        with torch.autocast("cuda", dtype=torch.float16):
            loss = lossf(fwd(enc(a[i:i + bs], b[i:i + bs])), y[i:i + bs])
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update()
        steps += 1; seen += len(a[i:i + bs]); run.append(loss.item())
        if steps % 200 == 0:
            log(f"ep {ep} step {steps}/{total} seen {seen} loss {np.mean(run[-200:]):.4f} {seen/(time.time()-t1):.1f} pairs/s")
        if steps % 2000 == 0:
            model.save_pretrained(f"{WD}/model_{tag}")          # checkpoint: survive a session time-out
    if stop: break
log(f"trained {steps} steps, {seen} pairs ({seen/tr.height:.2f} epochs) in {(time.time()-t1)/60:.1f} min")
model.save_pretrained(f"{WD}/model_{tag}"); tok.save_pretrained(f"{WD}/model_{tag}")
t2 = time.time(); lg = score(ev); log(f"scored {ev.height} band pairs in {time.time()-t2:.0f}s ({ev.height/(time.time()-t2):.1f} pairs/s)")
x = ev.select("s1", "r", "y", "part", "country").with_columns(pl.Series(f"lg_{tag}", lg))
x.write_parquet(f"{WD}/judge_{tag}.parquet")
for p in ("B2", "C"):
    xp = x.filter(pl.col("part") == p)
    log(f"AUC on {p} band: {roc_auc_score(xp['y'].to_numpy(), xp[f'lg_{tag}'].to_numpy()):.4f}  ({xp.height} pairs)")
'''


# ------------------------------------------------ 3. level 3 (CPU) --------------------------------------------------
def level3():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from ber import model as M
    from ber.metrics import macro_f05
    cfg2 = json.load(open(f"{JD}/decision_stack.json"))
    J = [pl.read_parquet(f) for f in sorted(glob.glob(f"{WD}/judge_*.parquet"))]
    if not J:
        log("no judge scores -> nothing to do"); return
    tags = [[c for c in j.columns if c.startswith("lg_")][0] for j in J]
    jj = J[0].select("s1", "r", tags[0])
    for j, t in zip(J[1:], tags[1:]):
        jj = jj.join(j.select("s1", "r", t), on=["s1", "r"], how="full", coalesce=True)
    rows = {nm: pl.read_parquet(f"{WD}/rows_{nm}.parquet").join(jj, on=["s1", "r"], how="left") for nm in ("B2", "C")}
    S1 = pl.concat([d.select("s1") for d in rows.values()]).unique()
    gt = pl.read_parquet(f"{IN}/gt_rows.parquet").filter(pl.col("source1_entity_id").is_in(S1["s1"].implode())) \
           .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids") \
           .filter(pl.col("matched_entity_ids") != "").select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("r"))
    g = lambda d: gt.filter(pl.col("s1").is_in(d["s1"].unique().implode()))
    f05 = lambda d, cf: macro_f05(M.decide(d, cf, None), g(d), d["s1"].unique())
    def tune(d):
        grid = [(a, b) for a in np.arange(0.5, 0.93, 0.05) for b in np.arange(0.5, 0.93, 0.05)]
        t1, t2, f = max(((a, b, f05(d, dict(cfg2, t1=a, t2=b))["f05"]) for a, b in grid), key=lambda x: x[2])
        return dict(cfg2, t1=float(t1), t2=float(t2)), f
    lgt = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    band_C = rows["C"].filter(pl.col(tags[0]).is_not_null())
    for c in ["p2"] + tags:
        log(f"AUC on C band ({band_C.height} rows, pos {band_C['y'].mean():.3f}): {c} {roc_auc_score(band_C['y'].to_numpy(), band_C[c].to_numpy()):.4f}")
    base = {nm: d.with_columns(pl.col("p2").alias("p")) for nm, d in rows.items()}
    cf0, fB0 = tune(base["B2"]); m0 = f05(base["C"], cf0)
    log(f"level 2: B2 F0.5 {fB0:.4f}; C F0.5 {m0['f05']:.4f} (singleton {m0['singleton_f']:.4f}, P {m0['pair_precision']:.4f} R {m0['pair_recall']:.4f})")
    combos = [[t] for t in tags] + ([tags] if len(tags) > 1 else [])
    res = []
    for cb in combos:
        for lo, hi in ((0.01, 0.99), (0.02, 0.98), (0.05, 0.95), (0.1, 0.9)):
            band = (pl.col("p2") > lo) & (pl.col("p2") < hi) & pl.all_horizontal([pl.col(t).is_not_null() for t in cb])
            tb = rows["B2"].filter(band)
            Xf = lambda d: np.column_stack([lgt(d["p2"].to_numpy())] + [d[t].to_numpy() for t in cb])
            lr = LogisticRegression(C=1.0, max_iter=1000).fit(Xf(tb), tb["y"].to_numpy())
            def apply(d):
                ib = d.select(band.alias("b"))["b"].to_numpy()
                p = d["p2"].to_numpy().copy()
                if ib.any():
                    p[ib] = lr.predict_proba(Xf(d.filter(band)))[:, 1]
                return d.with_columns(pl.Series("p", p))
            dB, dC = apply(rows["B2"]), apply(rows["C"])
            cf, fB = tune(dB); mC = f05(dC, cf)
            share = rows["C"].filter(band).height / rows["C"].height
            res.append(dict(judges=cb, lo=lo, hi=hi, F_B2=fB, F_C=mC["f05"], d_C=mC["f05"] - m0["f05"], share_C=share,
                            singleton_C=mC["singleton_f"], t1=cf["t1"], t2=cf["t2"],
                            coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0])))
            log(f"{'+'.join(cb):40s} band ({lo},{hi}) share {share:.4f}: B2 {fB:.4f} (+{fB-fB0:.4f})  C {mC['f05']:.4f} ({mC['f05']-m0['f05']:+.4f})  "
                f"singleton {mC['singleton_f']:.4f}  coef {np.round(lr.coef_[0], 3).tolist()}")
    best = max(res, key=lambda r: (round(r["F_B2"], 4), -r["share_C"]))     # chosen on B2; smaller band on ties
    ok = best["d_C"] >= 0.002
    log(f"CHOSEN on B2: {best['judges']} band ({best['lo']},{best['hi']}) -> C {best['F_C']:.4f} ({best['d_C']:+.4f}) -> {'ACCEPT' if ok else 'REJECT'} (gate +0.002)")
    json.dump(dict(cfg2, **best, level2_C=m0["f05"], accept=ok, all=res), open(f"{WD}/decision_judge.json", "w"), indent=1)


if __name__ == "__main__":
    build_pairs()
    open(f"{WD}/worker.py", "w").write(WORKER)
    import torch
    ng = torch.cuda.device_count(); log(f"GPUs: {ng}; kinds {KINDS}; train {TRAIN_MIN} min, {EPOCHS} epochs max")
    if ng >= 2:
        ps = [subprocess.Popen([sys.executable, f"{WD}/worker.py", k, WD, str(TRAIN_MIN), str(EPOCHS)],
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i)}) for i, k in enumerate(KINDS[:ng])]
        codes = [p.wait() for p in ps]
    else:
        codes = [subprocess.run([sys.executable, f"{WD}/worker.py", k, WD, str(TRAIN_MIN / max(1, len(KINDS))), str(EPOCHS)]).returncode for k in KINDS]
    log(f"worker exit codes {codes}")
    for k in KINDS:
        f = f"{WD}/log_{k.split('/')[-1].replace(':', '_').replace('.', '_').replace('-', '_')}.txt"
        if os.path.exists(f): log(f"--- {k}"); log(open(f).read()[-2500:])
    level3()
    log(f"DONE {time.time()-T0:.0f}s")
