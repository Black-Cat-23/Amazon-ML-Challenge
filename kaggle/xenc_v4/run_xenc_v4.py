"""[v4: CANINE-c and MuRIL-base on 3x more training pairs] Cross-encoder bake-off on Kaggle T4 x2 (one model per GPU, in parallel):
  GPU0  Qwen/Qwen3-Reranker-0.6B  (Apache-2.0)  LoRA, yes/no logit
  GPU1  BAAI/bge-reranker-v2-m3   (Apache-2.0, XLM-R base MIT)  full fine-tune
Pairs are built here with our blocking (CPU): TRAIN on A-split S1s (hash(11)%100 < 60, same split as the matcher, so
scores on B/C stay out-of-sample for stacking), EVAL on C-split S1s (>= 90). Raw text ("name | address"), no normalisation.
Reports zero-shot vs fine-tuned AUC / logloss / top-1 accuracy by country, saves weights + eval predictions."""
import glob, json, os, subprocess, sys, time
REF = "__REF__"
LOCAL = not os.path.exists("/kaggle")
WD = "." if LOCAL else "/kaggle/working"
if not LOCAL:
    # the image's torchao 0.10 makes recent peft refuse to build LoRA layers; we do not use torchao
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    if any(k.startswith("qwen35") for k in os.environ.get("KINDS_HINT", "hf:google/canine-c,hf:google/muril-base-cased").split(",")):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "flash-linear-attention", "causal-conv1d"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.51", "peft>=0.13",
                    f"git+https://github.com/Black-Cat-23/Amazon-ML-Challenge.git@{REF}#subdirectory=code/business_entity_resolution"], check=True)
else:
    sys.path.insert(0, os.path.abspath("../../code/business_entity_resolution/src"))
import numpy as np, polars as pl
T0 = time.time()
REP = open(f"{WD}/results_xenc_v4.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); REP.write(s + "\n"); REP.flush()
N_A = int(os.environ.get("N_A", 300 if LOCAL else 40000))    # train S1 per country
N_C = int(os.environ.get("N_C", 100 if LOCAL else 2000))     # eval S1 per country
TRAIN_MIN = float(os.environ.get("TRAIN_MIN", 1 if LOCAL else 150))
KINDS = os.environ.get("KINDS", "hf:google/canine-c,hf:google/muril-base-cased").split(",")   # one model per GPU

# ------------------------------------------------ 1. pairs (CPU) ----------------------------------------------------
def build_pairs():
    from ber.normalize import Normalizer
    from ber.pipeline import norm_chunks
    from ber import blocking as B
    find = lambda f: sorted(glob.glob(f"/kaggle/input/**/{f}", recursive=True))
    IN = "../../research/eda/cache" if LOCAL else os.path.dirname(find("train_s1.parquet")[0])
    ART = "../artifacts/kout/artifacts" if LOCAL else os.path.dirname(find("indic_lexicon.parquet")[0])
    NZ = Normalizer(ART)
    gt = pl.read_parquet(f"{IN}/gt_rows.parquet").with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")) \
           .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "") \
           .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("r"))
    txt = lambda df, k: df.select(pl.col("entity_id").alias(k), (pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")).alias(f"t_{k}"))
    out = []
    for c in ("US", "India"):
        S = pl.scan_parquet(f"{IN}/train_s1.parquet").filter(pl.col("country") == c).collect().sort("entity_id")
        R = pl.scan_parquet([f"{IN}/train_s2.parquet", f"{IN}/train_s3.parquet"]).filter(pl.col("country") == c).collect().sort("entity_id")
        if LOCAL: R = R.head(200000)
        h = pl.col("entity_id").hash(11) % 100
        SA = S.filter(h < 60).sample(min(N_A, S.height), seed=1); SC = S.filter(h >= 90).sample(min(N_C, S.height), seed=2)
        Q = pl.concat([SA, SC])
        RN = norm_chunks(NZ, R).with_columns(pl.Series("id", np.arange(R.height, dtype=np.uint32)))
        idx = B.Index(RN)
        QN = NZ.transform(Q).with_columns(pl.Series("id", np.arange(Q.height, dtype=np.uint32)))
        cand = idx.query(QN); del idx
        cand = cand.filter((pl.col("prk") <= 60) | pl.col("xrk").is_not_null())
        cand = cand.join(QN.select("id", pl.col("entity_id").alias("s1")), on="id").join(RN.select(pl.col("id").alias("id_r"), pl.col("entity_id").alias("r")), on="id_r")
        g = gt.filter(pl.col("s1").is_in(Q["entity_id"].implode()))
        cand = cand.join(g.with_columns(pl.lit(1, dtype=pl.Int8).alias("y")), on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0))
        cand = cand.with_columns(pl.col("sc").rank("ordinal", descending=True).over("s1").alias("rk"))
        log(f"{c}: pool {cand.height/Q.height:.1f}/S1, recall {cand['y'].sum()/max(g.height,1):.4f}  {time.time()-T0:.0f}s")
        isA = pl.col("s1").is_in(SA["entity_id"].implode())
        # train: all retrieved positives + missed positives + 10 hardest negatives + 2 random from ranks 11..60
        neg = cand.filter(isA & (pl.col("y") == 0))
        tr = pl.concat([cand.filter(isA & (pl.col("y") == 1)).select("s1", "r", "y"),
                        g.filter(pl.col("s1").is_in(SA["entity_id"].implode())).join(cand.select("s1", "r"), on=["s1", "r"], how="anti").with_columns(pl.lit(1, dtype=pl.Int8).alias("y")),
                        neg.filter(pl.col("rk") <= 10).select("s1", "r", "y"),
                        neg.filter(pl.col("rk") > 10).sample(fraction=1.0, shuffle=True, seed=3).group_by("s1").head(2).select("s1", "r", "y")])
        # eval: what the matcher would see (top-15 by blocking score + auxiliary hits), labels from gt
        ev = cand.filter(~isA & ((pl.col("rk") <= 15) | pl.col("xrk").is_not_null())).select("s1", "r", "y", "sc", "rk")
        for d, name in ((tr, "train"), (ev, "eval")):
            d = d.join(txt(Q, "s1"), on="s1").join(txt(R, "r"), on="r").with_columns(pl.lit(c).alias("country"))
            out.append((name, d))
        del RN, QN, cand
    tr = pl.concat([d for n, d in out if n == "train"], how="diagonal_relaxed").sample(fraction=1.0, shuffle=True, seed=5)
    ev = pl.concat([d for n, d in out if n == "eval"], how="diagonal_relaxed")
    tr.write_parquet(f"{WD}/pairs_train.parquet"); ev.write_parquet(f"{WD}/pairs_eval.parquet")
    log(f"pairs: train {tr.height} (pos {tr['y'].mean():.3f}), eval {ev.height} (pos {ev['y'].mean():.3f})  {time.time()-T0:.0f}s")

# ------------------------------------------------ 2. worker (GPU) ---------------------------------------------------
WORKER = r'''
import json, math, os, sys, time
import numpy as np, polars as pl, torch
from sklearn.metrics import roc_auc_score, log_loss
kind, WD, TRAIN_MIN = sys.argv[1], sys.argv[2], float(sys.argv[3])
tag = kind.replace("hf:", "").replace("/", "_").replace(":", "_")
torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
tr = pl.read_parquet(f"{WD}/pairs_train.parquet"); ev = pl.read_parquet(f"{WD}/pairs_eval.parquet")
L = open(f"{WD}/log_{tag}.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(f"[{kind}] " + s, flush=True); L.write(s + "\n"); L.flush()
from transformers import AutoTokenizer
EVAL_MAX = 10**9
if kind.startswith("qwen35"):
    from transformers import AutoModelForImageTextToText
    from peft import LoraConfig, get_peft_model
    name = "Qwen/Qwen3.5-" + kind.split(":")[1]
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    big = not kind.endswith("0.8B")
    model = AutoModelForImageTextToText.from_pretrained(name, **({"torch_dtype": torch.float16} if big else {})).to(dev)
    if big:
        model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    pre = tok.encode('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n', add_special_tokens=False)
    suf = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)
    INST = "Are the Query business record and the Document business record the same business entity (same business at the same location)?"
    def enc(a, b):
        body = tok([f"<Instruct>: {INST}\n<Query>: {x}\n<Document>: {y}" for x, y in zip(a, b)], add_special_tokens=False)["input_ids"]
        ids = [pre + t[:200] + suf for t in body]; m = max(map(len, ids)); pad = tok.pad_token_id
        return dict(input_ids=torch.tensor([[pad] * (m - len(i)) + i for i in ids], device=dev),
                    attention_mask=torch.tensor([[0] * (m - len(i)) + [1] * len(i) for i in ids], device=dev))
    def fwd(b):
        try:
            lg = model(**b, logits_to_keep=1).logits[:, -1, :]
        except TypeError:
            lg = model(**b).logits[:, -1, :]
        return (lg[:, yes] - lg[:, no]).float()
    def make_trainable():
        global model
        # LoRA on the language model's linear layers only (attention, gated-DeltaNet projections, MLP), not the vision tower
        tm = r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj\w*|out_proj)$"
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=tm))
        return (1e-4, 12) if big else (2e-4, 24)
elif kind.startswith("qwen"):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    name = "Qwen/Qwen3-Reranker-4B" if kind == "qwen4b" else "Qwen/Qwen3-Reranker-0.6B"
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    if kind == "qwen4b":   # 8 GB of fp16 weights on a 16 GB T4: frozen fp16 base + fp32 LoRA, gradient checkpointing
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(dev)
        model.gradient_checkpointing_enable(); model.enable_input_require_grads(); EVAL_MAX = 20000
    else:
        model = AutoModelForCausalLM.from_pretrained(name).to(dev)
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    pre = tok.encode('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n', add_special_tokens=False)
    suf = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)
    INST = "Are the Query business record and the Document business record the same business entity (same business at the same location)?"
    def enc(a, b):
        body = tok([f"<Instruct>: {INST}\n<Query>: {x}\n<Document>: {y}" for x, y in zip(a, b)], add_special_tokens=False)["input_ids"]
        ids = [pre + t[:200] + suf for t in body]; m = max(map(len, ids)); pad = tok.pad_token_id
        return dict(input_ids=torch.tensor([[pad] * (m - len(i)) + i for i in ids], device=dev),
                    attention_mask=torch.tensor([[0] * (m - len(i)) + [1] * len(i) for i in ids], device=dev))
    def fwd(b):
        # only the last position's logits (full-sequence x 152k-vocab logits is ~7.6 GB per scoring batch)
        lg = model(**b, logits_to_keep=1).logits[:, -1, :]
        return (lg[:, yes] - lg[:, no]).float()
    def make_trainable():
        global model
        if kind == "qwenfull":        # full fine-tune of the 0.6B (compare with LoRA)
            return 1e-5, 16
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                               target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        return (1e-4, 8) if kind == "qwen4b" else (2e-4, 24)
else:
    # "bge" = BAAI/bge-reranker-v2-m3; "hf:<id>" = any HF encoder / encoder-decoder with a sequence-classification head
    from transformers import AutoModelForSequenceClassification
    name = kind[3:] if kind.startswith("hf:") else "BAAI/bge-reranker-v2-m3"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1, trust_remote_code=True, ignore_mismatched_sizes=True).to(dev)
    def enc(a, b):
        return {k: v.to(dev) for k, v in tok(list(a), list(b), padding=True, truncation=True, max_length=192, return_tensors="pt").items()}
    def fwd(b):
        return model(**b).logits.squeeze(-1).float()
    def make_trainable():
        if "t5" in name.lower():      # byte sequences are long: checkpointing + smaller batch (585M full FT OOMs at 24)
            model.gradient_checkpointing_enable(); return 1e-4, 12
        if "canine" in name.lower():
            return 3e-5, 32
        return 2e-5, 24

@torch.no_grad()
def score(d, bs=None):
    bs = bs or (16 if kind == "qwen4b" else 32 if kind.startswith("qwen35") else 128)   # 4B activations at 128 x ~220 tokens overflow a T4
    model.eval(); out = []
    a, b = d["t_s1"].to_list(), d["t_r"].to_list()
    for i in range(0, len(a), bs):
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            out.append(fwd(enc(a[i:i + bs], b[i:i + bs])).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)

def report(tag, d, lg):
    p = 1 / (1 + np.exp(-np.clip(lg, -30, 30))); y = d["y"].to_numpy()
    x = d.with_columns(pl.Series("lg", lg), pl.Series("p", p))
    res = {"auc": float(roc_auc_score(y, lg)), "logloss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))}
    top = x.sort("lg", descending=True).group_by("s1", maintain_order=True).first()
    has = x.group_by("s1").agg(pl.col("y").max().alias("any"))
    t = top.join(has, on="s1")
    res["top1_acc_nonsingleton"] = float(t.filter(pl.col("any") == 1)["y"].mean())
    res["singleton_max_p_below_0.5"] = float((t.filter(pl.col("any") == 0)["p"] < 0.5).mean()) if t.filter(pl.col("any") == 0).height else None
    for c in x["country"].unique().to_list():
        xc = x.filter(pl.col("country") == c)
        res[f"auc_{c}"] = float(roc_auc_score(xc["y"].to_numpy(), xc["lg"].to_numpy()))
    log(tag, json.dumps(res)); return res, x

log(f"model {name} on {dev} ({torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'})")
t0 = time.time()
zs = ev.sample(min(ev.height, 6000), seed=0)
r0, _ = report("ZERO-SHOT", zs, score(zs)); log(f"zero-shot scored {zs.height} in {time.time()-t0:.0f}s")
lr, bs = make_trainable()
params = [p for p in model.parameters() if p.requires_grad]
log(f"trainable params {sum(p.numel() for p in params)/1e6:.1f}M")
opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
lossf = torch.nn.BCEWithLogitsLoss()
a, b, y = tr["t_s1"].to_list(), tr["t_r"].to_list(), torch.tensor(tr["y"].to_numpy(), dtype=torch.float32, device=dev)
steps, seen, t1, run = 0, 0, time.time(), []
model.train()
for i in range(0, len(a), bs):
    if time.time() - t1 > TRAIN_MIN * 60: break
    for g in opt.param_groups: g["lr"] = lr * min(1.0, (steps + 1) / 200)
    with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
        loss = lossf(fwd(enc(a[i:i + bs], b[i:i + bs])), y[i:i + bs])
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update()
    steps += 1; seen += len(a[i:i + bs]); run.append(loss.item())
    if steps % 200 == 0:
        log(f"step {steps} seen {seen} loss {np.mean(run[-200:]):.4f} {seen/(time.time()-t1):.1f} pairs/s")
log(f"trained {steps} steps, {seen} pairs ({seen/len(a):.2f} epoch) in {(time.time()-t1)/60:.1f} min")
ev = ev if ev.height <= EVAL_MAX else ev.filter(pl.col("s1").is_in(ev["s1"].unique().sort().head(int(EVAL_MAX / 18)).implode()))
t2 = time.time(); lg = score(ev); log(f"eval scored {ev.height} pairs in {time.time()-t2:.0f}s ({ev.height/(time.time()-t2):.0f} pairs/s)")
r1, x = report("FINE-TUNED", ev, lg)
x.select("s1", "r", "y", "country", "sc", "rk", "lg").write_parquet(f"{WD}/eval_{tag}.parquet")
model.save_pretrained(f"{WD}/model_{tag}"); tok.save_pretrained(f"{WD}/model_{tag}")
json.dump({"zero_shot": r0, "fine_tuned": r1, "pairs_seen": seen, "steps": steps, "model": name}, open(f"{WD}/metrics_{tag}.json", "w"), indent=1)
'''

if __name__ == "__main__":
    build_pairs()
    open(f"{WD}/worker.py", "w").write(WORKER)
    import torch
    ng = torch.cuda.device_count(); log(f"GPUs: {ng}")
    kinds = KINDS
    if ng >= 2:
        ps = [subprocess.Popen([sys.executable, f"{WD}/worker.py", k, WD, str(TRAIN_MIN)], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i)}) for i, k in enumerate(kinds)]
        codes = [p.wait() for p in ps]
    else:
        codes = [subprocess.run([sys.executable, f"{WD}/worker.py", k, WD, str(TRAIN_MIN / max(1, len(kinds)))]).returncode for k in kinds]
    log(f"worker exit codes {codes}")
    for k in kinds:
        for f in (f"{WD}/log_{k.replace('hf:', '').replace('/', '_').replace(':', '_')}.txt",):
            if os.path.exists(f): log(f"--- {k}"); log(open(f).read()[-3000:])
    # baseline for reference: blocking score alone on the same eval pairs
    ev = pl.read_parquet(f"{WD}/pairs_eval.parquet")
    try:
        from sklearn.metrics import roc_auc_score
        log(f"baseline blocking-score AUC on eval: {roc_auc_score(ev['y'].to_numpy(), ev['sc'].to_numpy()):.4f}")
    except Exception as e:
        log("baseline failed", e)
    log(f"DONE {time.time()-T0:.0f}s")
