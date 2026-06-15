"""Paper V -- LoRA-adapted QA-abstention: does letting the backbone EMBEDDINGS adapt (not just a read
head on frozen embeddings) widen copositive's abstention advantage? Trains, for each read {softmax,
copositive}, BOTH:
  * FROZEN  : backbone frozen, only the read head trained (== R14).
  * LoRA    : small LoRA adapters trained INSIDE the backbone + the head, jointly (embeddings can move).
Reports the matched-recall false-answer table (easy/hard) for all four, and SAVES every trained model to
./artifacts so you can download them and compare outputs locally.

Reuses qa_abstention.py (must be in the same dir). Run (GPU):
    pip install -U transformers datasets peft
    python qa_abstention_lora.py --model Qwen/Qwen3-0.6B --device cuda
"""
import argparse, os, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import qa_abstention as QA                       # sent_split, build, closed_weights, Head, Embedder, encode_set, train_head

# --------------------------------------------------------------- embedding through a (LoRA) backbone
def embed_texts(model, tok, texts, device, pooling, max_len=64):
    t = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    t = {k: v.to(device) for k, v in t.items()}
    h = model(**t).last_hidden_state
    m = t["attention_mask"]
    if pooling == "last":
        emb = h[torch.arange(h.size(0), device=device), m.sum(1) - 1]
    else:
        mf = m.unsqueeze(-1).float(); emb = (h * mf).sum(1) / mf.sum(1).clamp(min=1)
    return F.normalize(emb.float(), dim=-1)

def encode_batch(model, tok, ex, device, pooling):
    """grad-enabled embedding of one batch of examples -> qE (B,d), cE (B,maxC,d), cM (B,maxC), y (B,)."""
    qE = embed_texts(model, tok, [e["q"] for e in ex], device, pooling)
    sizes = [len(e["cands"]) for e in ex]
    flat = [c for e in ex for c in e["cands"]]
    cflat = embed_texts(model, tok, flat, device, pooling)
    cE = torch.nn.utils.rnn.pad_sequence(torch.split(cflat, sizes), batch_first=True)   # (B,maxC,d), grad
    cM = torch.zeros(len(ex), cE.shape[1], device=device)
    for i, s in enumerate(sizes):
        cM[i, :s] = 1
    y = torch.tensor([e["answerable"] for e in ex], dtype=torch.float, device=device)
    return qE, cE, cM, y

@torch.no_grad()
def encode_set(model, tok, ex, device, pooling, bs=64):
    """no-grad embedding of a whole eval set -> qE,cE,cM,y (cE padded to global maxC)."""
    model.eval()
    qE = torch.cat([embed_texts(model, tok, [e["q"] for e in ex[i:i+bs]], device, pooling)
                    for i in range(0, len(ex), bs)])
    flat, sizes = [c for e in ex for c in e["cands"]], [len(e["cands"]) for e in ex]
    cflat = torch.cat([embed_texts(model, tok, flat[i:i+bs], device, pooling)
                       for i in range(0, len(flat), bs)])
    splits = list(torch.split(cflat, sizes))
    maxC = max(sizes)
    cE = torch.zeros(len(ex), maxC, qE.shape[1], device=device); cM = torch.zeros(len(ex), maxC, device=device)
    for i, s in enumerate(splits):
        cE[i, :s.shape[0]] = s; cM[i, :s.shape[0]] = 1
    y = torch.tensor([e["answerable"] for e in ex], dtype=torch.float, device=device)
    return qE, cE, cM, y

# --------------------------------------------------------------- matched-recall table (shared)
@torch.no_grad()
def signals_on(head, qE, cE, cM, ans, easyC, easyM, hardmask):
    s_ans = head(qE[ans], cE[ans], cM[ans])
    s_hard = head(qE[hardmask], cE[hardmask], cM[hardmask])
    s_easy = head(qE, easyC, easyM)
    return s_ans, s_easy, s_hard

def table(s_ans, s_easy, s_hard, targets=(0.80, 0.90, 0.95)):
    out = {}
    for r in targets:
        thr = torch.quantile(s_ans, 1.0 - r)
        out[f"{r:.2f}"] = [round((s_easy >= thr).float().mean().item(), 3),
                           round((s_hard >= thr).float().mean().item(), 3)]
    return out

# --------------------------------------------------------------- LoRA training
def train_lora(read, examples, tok, mid, device, steps, batch, lr, pooling, ckpt=True):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel
    base = AutoModel.from_pretrained(mid).to(device).float()
    base.config.use_cache = False
    if ckpt:
        base.gradient_checkpointing_enable()
    cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
    model = get_peft_model(base, cfg)
    if ckpt and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    d = base.config.hidden_size
    head = QA.Head(d, read).to(device)
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    rng = np.random.default_rng(0); model.train()
    for step in range(steps):
        ex = [examples[i] for i in rng.choice(len(examples), batch, replace=False)]
        qE, cE, cM, y = encode_batch(model, tok, ex, device, pooling)
        sig = head(qE, cE, cM)
        logit = 12 * sig if read == "copositive" else 8 * (sig - sig.mean())
        loss = F.binary_cross_entropy_with_logits(logit, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 6) == 0:
            print(f"    [lora:{read}] step {step}/{steps} loss {loss.item():.3f}", flush=True)
    return model, head, d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B"); ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pooling", default="last", choices=["mean", "last"])
    ap.add_argument("--n_train", type=int, default=1500); ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--lora_steps", type=int, default=250); ap.add_argument("--lora_batch", type=int, default=4)
    ap.add_argument("--head_steps", type=int, default=400); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args(); rng = np.random.default_rng(a.seed); os.makedirs(a.out, exist_ok=True)
    print(f"model={a.model} device={a.device} pooling={a.pooling}")
    tr = QA.build("train", a.n_train, rng); ev = QA.build("validation", a.n_eval, rng)
    print(f"built {len(tr)} train / {len(ev)} eval examples")
    perm = rng.permutation(len(ev)); results = {}

    # ---- FROZEN (== R14): frozen backbone, train only the head ----
    emb = QA.Embedder(a.model, a.device, a.pooling); d = emb.d
    qTr, cTr, mTr, yTr = QA.encode_set(emb, tr); qEv, cEv, mEv, yEv = QA.encode_set(emb, ev)
    ans = yEv == 1; easyC = cEv[perm]; easyM = mEv[perm]
    for read in ("softmax", "copositive"):
        torch.manual_seed(a.seed)
        head = QA.train_head(QA.Head(d, read), qTr, cTr, mTr, yTr, steps=a.head_steps)
        sa, se, sh = signals_on(head, qEv, cEv, mEv, ans, easyC, easyM, ~ans)
        results[f"frozen_{read}"] = table(sa, se, sh)
        torch.save(head.state_dict(), f"{a.out}/frozen_{read}_head.pt")
    del emb
    if a.device == "cuda":
        torch.cuda.empty_cache()

    # ---- LoRA: adapt the backbone + head jointly ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for read in ("softmax", "copositive"):
        torch.manual_seed(a.seed)
        model, head, d = train_lora(read, tr, tok, a.model, a.device, a.lora_steps, a.lora_batch, 1e-4, a.pooling)
        qEv2, cEv2, mEv2, yEv2 = encode_set(model, tok, ev, a.device, a.pooling)
        ans2 = yEv2 == 1; easyC2 = cEv2[perm]; easyM2 = mEv2[perm]
        sa, se, sh = signals_on(head, qEv2, cEv2, mEv2, ans2, easyC2, easyM2, ~ans2)
        results[f"lora_{read}"] = table(sa, se, sh)
        model.save_pretrained(f"{a.out}/lora_{read}"); torch.save(head.state_dict(), f"{a.out}/lora_{read}_head.pt")
        del model
        if a.device == "cuda":
            torch.cuda.empty_cache()

    json.dump({"model": a.model, "pooling": a.pooling, "hidden": d, "results": results},
              open(f"{a.out}/metadata.json", "w"), indent=1)
    print("\n" + "=" * 78)
    print("  FALSE-ANSWER at matched recall  (easy / hard ; lower=better)")
    print(f"  {'variant':<22}{'recall .80':>14}{'recall .90':>14}{'recall .95':>14}")
    for k in ("frozen_softmax", "frozen_copositive", "lora_softmax", "lora_copositive"):
        t = results[k]
        print(f"  {k:<22}" + "".join(f"{t[r][0]:.2f}/{t[r][1]:.2f}{'':>5}" for r in ("0.80", "0.90", "0.95")))
    fc, lc = results["frozen_copositive"]["0.90"], results["lora_copositive"]["0.90"]
    print(f"\n  copositive EASY false-answer @ recall .90:  frozen {fc[0]:.2f}  ->  LoRA {lc[0]:.2f}"
          f"   ({'WIDER gap (LoRA helps)' if lc[0] < fc[0] - 0.03 else 'no clear improvement'})")
    print(f"  artifacts saved to ./{a.out}/  (frozen heads + LoRA adapters + heads + metadata)")

if __name__ == "__main__":
    main()
