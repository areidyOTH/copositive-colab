"""Paper V -- LoRA-adapted QA-abstention (HEAVIER + RESUMABLE). Does letting the backbone EMBEDDINGS
adapt (LoRA inside the LLM + head, trained jointly) widen copositive's abstention advantage? Trains, per
seed, for each read {softmax, copositive}, BOTH the frozen-head variant (R14) and a LoRA-adapted variant.

CHECKPOINTING / RESUME (for Kaggle disconnects): every ~20 min (and at each stage) it saves the LoRA
adapter + head + optimizer + step + RNG to --out, and records completed stages in out/progress.json.
Re-running with the same --out RESUMES: completed stages are skipped, and an interrupted LoRA stage
continues from its last checkpoint. Best run via Kaggle "Save Version -> Save & Run All (Commit)" so it
runs headless and survives a closed tab.

Reuses qa_abstention.py. Run (GPU):
    pip install -U transformers datasets peft ; pip uninstall -y torchao
    python qa_abstention_lora.py --model Qwen/Qwen3-0.6B --device cuda
"""
import argparse, os, json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import qa_abstention as QA

# --------------------------------------------------------------- embedding through a (LoRA) backbone
def embed_texts(model, tok, texts, device, pooling, max_len=64):
    t = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    t = {k: v.to(device) for k, v in t.items()}
    h = model(**t).last_hidden_state
    m = t["attention_mask"]
    emb = h[torch.arange(h.size(0), device=device), m.sum(1) - 1] if pooling == "last" \
        else (h * m.unsqueeze(-1).float()).sum(1) / m.unsqueeze(-1).float().sum(1).clamp(min=1)
    return F.normalize(emb.float(), dim=-1)

def encode_batch(model, tok, ex, device, pooling):
    qE = embed_texts(model, tok, [e["q"] for e in ex], device, pooling)
    sizes = [len(e["cands"]) for e in ex]; flat = [c for e in ex for c in e["cands"]]
    cflat = embed_texts(model, tok, flat, device, pooling)
    cE = torch.nn.utils.rnn.pad_sequence(torch.split(cflat, sizes), batch_first=True)
    cM = torch.zeros(len(ex), cE.shape[1], device=device)
    for i, s in enumerate(sizes):
        cM[i, :s] = 1
    y = torch.tensor([e["answerable"] for e in ex], dtype=torch.float, device=device)
    return qE, cE, cM, y

@torch.no_grad()
def encode_set(model, tok, ex, device, pooling, bs=64):
    model.eval()
    qE = torch.cat([embed_texts(model, tok, [e["q"] for e in ex[i:i+bs]], device, pooling) for i in range(0, len(ex), bs)])
    flat, sizes = [c for e in ex for c in e["cands"]], [len(e["cands"]) for e in ex]
    cflat = torch.cat([embed_texts(model, tok, flat[i:i+bs], device, pooling) for i in range(0, len(flat), bs)])
    splits = list(torch.split(cflat, sizes)); maxC = max(sizes)
    cE = torch.zeros(len(ex), maxC, qE.shape[1], device=device); cM = torch.zeros(len(ex), maxC, device=device)
    for i, s in enumerate(splits):
        cE[i, :s.shape[0]] = s; cM[i, :s.shape[0]] = 1
    y = torch.tensor([e["answerable"] for e in ex], dtype=torch.float, device=device)
    return qE, cE, cM, y

@torch.no_grad()
def signals_on(head, qE, cE, cM, ans, easyC, easyM, hard):
    return head(qE[ans], cE[ans], cM[ans]), head(qE, easyC, easyM), head(qE[hard], cE[hard], cM[hard])

def table(s_ans, s_easy, s_hard):
    out = {}
    for r in (0.80, 0.90, 0.95):
        thr = torch.quantile(s_ans, 1.0 - r)
        out[f"{r:.2f}"] = [round((s_easy >= thr).float().mean().item(), 3), round((s_hard >= thr).float().mean().item(), 3)]
    return out

# --------------------------------------------------------------- LoRA training (resumable + checkpointed)
def train_lora(read, examples, tok, mid, device, total_steps, batch, lr, pooling, rank, out, seed, ckpt_sec):
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import AutoModel
    base = AutoModel.from_pretrained(mid).to(device).float(); base.config.use_cache = False
    base.gradient_checkpointing_enable()
    adir = f"{out}/lora_{read}_s{seed}"; sfile = f"{adir}_state.pt"
    head = QA.Head(base.config.hidden_size, read).to(device)
    if os.path.isdir(adir) and os.path.exists(sfile):                       # ---- resume ----
        model = PeftModel.from_pretrained(base, adir, is_trainable=True)
        st = torch.load(sfile, map_location=device)
        head.load_state_dict(st["head"]); start = st["step"]
        rng = np.random.default_rng(); rng.bit_generator.state = st["np_state"]
        torch.set_rng_state(st["torch_state"].to("cpu", torch.uint8))
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad] + list(head.parameters()), lr=lr)
        opt.load_state_dict(st["opt"])
        print(f"    [RESUME lora:{read} s{seed}] from step {start}/{total_steps}", flush=True)
    else:                                                                   # ---- fresh ----
        cfg = LoraConfig(r=rank, lora_alpha=2 * rank, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
        model = get_peft_model(base, cfg); start = 0
        rng = np.random.default_rng(1000 + seed)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad] + list(head.parameters()), lr=lr)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.train()

    def save(step):
        model.save_pretrained(adir)
        torch.save({"head": head.state_dict(), "opt": opt.state_dict(), "step": step,
                    "np_state": rng.bit_generator.state, "torch_state": torch.get_rng_state()}, sfile)

    last = time.time()
    for step in range(start, total_steps):
        ex = [examples[i] for i in rng.choice(len(examples), batch, replace=False)]
        qE, cE, cM, y = encode_batch(model, tok, ex, device, pooling)
        sig = head(qE, cE, cM)
        loss = F.binary_cross_entropy_with_logits(12 * sig if read == "copositive" else 8 * (sig - sig.mean()), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, total_steps // 8) == 0:
            print(f"    [lora:{read} s{seed}] step {step}/{total_steps} loss {loss.item():.3f}", flush=True)
        if time.time() - last > ckpt_sec:
            save(step + 1); last = time.time(); print(f"    [ckpt] {read} s{seed} step {step+1} saved", flush=True)
    save(total_steps)
    return model, head

# --------------------------------------------------------------- progress
def load_prog(out):
    p = f"{out}/progress.json"
    return json.load(open(p)) if os.path.exists(p) else {"done": [], "results": {}}

def save_prog(out, prog):
    json.dump(prog, open(f"{out}/progress.json", "w"), indent=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pooling", default="last", choices=["mean", "last"])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--n_train", type=int, default=4000); ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--lora_steps", type=int, default=1500); ap.add_argument("--lora_batch", type=int, default=8)
    ap.add_argument("--rank", type=int, default=32); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head_steps", type=int, default=400); ap.add_argument("--ckpt_sec", type=int, default=1200)
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    prog = load_prog(a.out); reads = ("softmax", "copositive")
    print(f"model={a.model} device={a.device} pooling={a.pooling} | seeds={a.seeds} lora_steps={a.lora_steps} "
          f"rank={a.rank} batch={a.lora_batch} | resuming: {len(prog['done'])} stages already done")

    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        tr = QA.build("train", a.n_train, rng); ev = QA.build("validation", a.n_eval, rng)
        perm = rng.permutation(len(ev))
        # ---- FROZEN ----
        if any(f"frozen_{r}_s{seed}" not in prog["done"] for r in reads):
            emb = QA.Embedder(a.model, a.device, a.pooling); d = emb.d
            qTr, cTr, mTr, yTr = QA.encode_set(emb, tr); qEv, cEv, mEv, yEv = QA.encode_set(emb, ev)
            ans = yEv == 1; easyC = cEv[perm]; easyM = mEv[perm]
            for read in reads:
                stage = f"frozen_{read}_s{seed}"
                if stage in prog["done"]:
                    continue
                torch.manual_seed(seed)
                head = QA.train_head(QA.Head(d, read), qTr, cTr, mTr, yTr, steps=a.head_steps)
                torch.save(head.state_dict(), f"{a.out}/frozen_{read}_s{seed}_head.pt")
                sa, se, sh = signals_on(head, qEv, cEv, mEv, ans, easyC, easyM, ~ans)
                prog["results"][stage] = table(sa, se, sh); prog["done"].append(stage); save_prog(a.out, prog)
                print(f"  [done] {stage}", flush=True)
            del emb
            if a.device == "cuda":
                torch.cuda.empty_cache()
        # ---- LoRA ----
        if any(f"lora_{r}_s{seed}" not in prog["done"] for r in reads):
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "right"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            for read in reads:
                stage = f"lora_{read}_s{seed}"
                if stage in prog["done"]:
                    continue
                torch.manual_seed(seed)
                model, head = train_lora(read, tr, tok, a.model, a.device, a.lora_steps, a.lora_batch,
                                         a.lr, a.pooling, a.rank, a.out, seed, a.ckpt_sec)
                qEv2, cEv2, mEv2, yEv2 = encode_set(model, tok, ev, a.device, a.pooling)
                ans2 = yEv2 == 1; easyC2 = cEv2[perm]; easyM2 = mEv2[perm]
                sa, se, sh = signals_on(head, qEv2, cEv2, mEv2, ans2, easyC2, easyM2, ~ans2)
                prog["results"][stage] = table(sa, se, sh); prog["done"].append(stage); save_prog(a.out, prog)
                print(f"  [done] {stage}", flush=True)
                del model
                if a.device == "cuda":
                    torch.cuda.empty_cache()

    # ---- aggregated table (mean over completed seeds) ----
    print("\n" + "=" * 80)
    print(f"  FALSE-ANSWER at matched recall (easy/hard, lower=better) -- mean over completed seeds")
    print(f"  {'variant':<20}{'recall .80':>14}{'recall .90':>14}{'recall .95':>14}")
    agg = {}
    for v in ("frozen_softmax", "frozen_copositive", "lora_softmax", "lora_copositive"):
        seeds_done = [s for s in range(a.seeds) if f"{v}_s{s}" in prog["results"]]
        if not seeds_done:
            continue
        m = {r: np.mean([prog["results"][f"{v}_s{s}"][r] for s in seeds_done], axis=0) for r in ("0.80", "0.90", "0.95")}
        agg[v] = m
        print(f"  {v:<20}" + "".join(f"{m[r][0]:.2f}/{m[r][1]:.2f}{'':>5}" for r in ("0.80", "0.90", "0.95"))
              + f"  (n={len(seeds_done)})")
    if "frozen_copositive" in agg and "lora_copositive" in agg:
        fc, lc = agg["frozen_copositive"]["0.90"], agg["lora_copositive"]["0.90"]
        print(f"\n  copositive @ .90 frozen->LoRA:  easy {fc[0]:.2f}->{lc[0]:.2f}  hard {fc[1]:.2f}->{lc[1]:.2f}"
              f"  ({'helps BOTH' if lc[0]<fc[0]-.02 and lc[1]<fc[1]-.02 else ('TRADED easy<->hard' if lc[0]<fc[0]-.02 and lc[1]>fc[1]+.02 else 'no clear effect')})")
    print(f"  artifacts in ./{a.out}/  (frozen heads + LoRA adapters + heads + progress.json)")

if __name__ == "__main__":
    main()
