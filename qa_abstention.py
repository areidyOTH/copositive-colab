"""Paper V -- fine-tuning test: certified (copositive null) vs calibrated (softmax confidence) abstention
on a REAL pretrained backbone, under distribution shift. Tests the central claim (R3/R8 at LLM scale):
copositive's null gives OOD-robust 'no match', while a calibrated softmax threshold leaks OOD.

Setup. Frozen backbone (default Qwen3-0.6B) embeds a question and the candidate sentences of its context
(mean-pooled last hidden state). A small TRAINED read decides answerable (some candidate clears the bar)
vs abstain:
    copositive : z = closed-form competitive read over candidate scores with learned theta (threshold),
                 beta; abstain iff 1^T z == 0  (a key clears theta -> answerable). The null is structural.
    softmax    : a = softmax(alpha * scores); confidence = max a; abstain iff confidence < tau, tau
                 CALIBRATED on a held-out set to a target operating point.
Only a small projection + read params train (backbone frozen) -> cheap. Data: SQuAD v2 (answerable +
adversarial unanswerable). Distribution-shift test (mirrors FINDINGS R3): calibrate on EASY unanswerables
(question paired with an unrelated context) -> measure false-answer rate on HARD unanswerables (SQuAD v2's
native, topically-relevant unanswerables). Hypothesis: softmax-tau leaks easy->hard; copositive holds.

Colab-ready. See FINETUNE_GPU.md. Run (GPU):
    python qa_abstention.py --model Qwen/Qwen3-0.6B --n_train 4000 --n_eval 2000 --device cuda
"""
import argparse, re, os, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

# ----------------------------------------------------------------- closed-form copositive read
def closed_weights(scores, theta, beta):
    """scores,(...,C); theta,beta scalars/learnable. Returns weights (sum<=1; 0 row = NULL/abstain)."""
    b = scores - theta
    srt, _ = torch.sort(b, dim=-1, descending=True); css = srt.cumsum(-1)
    C = b.shape[-1]; Km1 = torch.arange(0, C, device=b.device, dtype=b.dtype)
    denom = 1 + Km1 * beta
    kstar = ((srt * denom - beta * css) > 0).sum(-1, keepdim=True).clamp(min=1)
    tau = beta * (css / denom).gather(-1, (kstar - 1).long())
    z = torch.relu(b - tau)
    return z / (z.sum(-1, keepdim=True) + 1e-9), z.sum(-1)               # weights, total (null signal)

# ----------------------------------------------------------------- data (SQuAD v2 -> question, candidates)
def sent_split(text):
    s = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if len(x.strip()) > 0]
    return s[:8] if s else [text[:200]]                                  # cap candidates

def build(split, n, rng):
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad_v2", split=split)
    idx = rng.permutation(len(ds))[: n * 2]
    ex = []
    for i in idx:
        r = ds[int(i)]
        cands = sent_split(r["context"])
        ans = r["answers"]["text"]
        if ans:                                                          # answerable
            a = ans[0]; tgt = next((j for j, c in enumerate(cands) if a[:25].lower() in c.lower()), -1)
            if tgt < 0:
                continue
            ex.append(dict(q=r["question"], cands=cands, answerable=1, tgt=tgt, ctx=r["context"]))
        else:                                                            # hard unanswerable (relevant ctx)
            ex.append(dict(q=r["question"], cands=cands, answerable=0, tgt=-1, ctx=r["context"]))
        if len(ex) >= n:
            break
    return ex

# ----------------------------------------------------------------- frozen backbone embedding
class Embedder:
    def __init__(self, model_id, device, pooling="mean"):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "right"                                  # so last real token = mask.sum-1
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.m = AutoModel.from_pretrained(model_id).eval().to(device)
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.device = device; self.d = self.m.config.hidden_size; self.pooling = pooling

    @torch.no_grad()
    def embed(self, texts, bs=32):
        out = []
        for i in range(0, len(texts), bs):
            t = self.tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=64)
            t = {k: v.to(self.device) for k, v in t.items()}
            h = self.m(**t).last_hidden_state                            # (B,T,d)
            m = t["attention_mask"]
            if self.pooling == "last":                                  # last real token (best for decoders)
                emb = h[torch.arange(h.size(0)), m.sum(1) - 1]
            else:                                                        # mean-pool (best for encoders)
                mf = m.unsqueeze(-1).float(); emb = (h * mf).sum(1) / mf.sum(1).clamp(min=1)
            out.append(F.normalize(emb, dim=-1).cpu())
        return torch.cat(out)

def encode_set(emb, ex):
    """-> qE (N,d), list of cand embeddings (N, Ci, d as padded tensor + mask), labels."""
    qE = emb.embed([e["q"] for e in ex])
    maxC = max(len(e["cands"]) for e in ex)
    cE = torch.zeros(len(ex), maxC, emb.d); cM = torch.zeros(len(ex), maxC)
    flat, owner = [], []
    for i, e in enumerate(ex):
        for c in e["cands"]:
            flat.append(c); owner.append(i)
    fE = emb.embed(flat); p = 0
    for i, e in enumerate(ex):
        k = len(e["cands"]); cE[i, :k] = fE[p:p + k]; cM[i, :k] = 1; p += k
    y = torch.tensor([e["answerable"] for e in ex]).float()
    return qE, cE, cM, y

# ----------------------------------------------------------------- read head (trained; backbone frozen)
class Head(nn.Module):
    def __init__(self, d, read):
        super().__init__(); self.read = read
        self.proj = nn.Linear(d, d, bias=False); nn.init.eye_(self.proj.weight)
        self.theta = nn.Parameter(torch.tensor(0.3)); self.logbeta = nn.Parameter(torch.tensor(-0.5))
        self.logalpha = nn.Parameter(torch.tensor(2.0))                  # softmax inverse-temp
    def scores(self, qE, cE, cM):
        q = F.normalize(self.proj(qE), dim=-1); c = F.normalize(self.proj(cE), dim=-1)
        s = torch.einsum("nd,ncd->nc", q, c)                            # cosine-like score per candidate
        return s.masked_fill(cM == 0, -1e4)
    def forward(self, qE, cE, cM):
        s = self.scores(qE, cE, cM)
        if self.read == "copositive":
            return s.max(-1).values - self.theta     # ABSOLUTE margin over threshold; structural null = (<0)
        a = F.softmax(torch.exp(self.logalpha) * s, -1)
        return a.max(-1).values                      # confidence (scale-invariant: the R3 weakness)

def train_head(head, qE, cE, cM, y, steps=400, lr=5e-3):
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(steps):
        sig = head(qE, cE, cM)
        # copositive: supervise theta as the ABSOLUTE answerable/unanswerable boundary (sig>0 => answer).
        # softmax: relative separation of the (scale-invariant) confidence; tau is calibrated at eval.
        logit = 12 * sig if head.read == "copositive" else 8 * (sig - sig.mean())
        loss = F.binary_cross_entropy_with_logits(logit, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return head

# ----------------------------------------------------------------- distribution-shift abstention eval
# Faithful R3 test: copositive uses its STRUCTURAL null (answer iff 1^T z > 0, i.e. some candidate clears
# the learned absolute threshold theta -- distribution-free), while softmax abstains via a threshold tau
# CALIBRATED on EASY unanswerables. Both matched at the same easy-FA operating point; then we ask which
# one leaks on HARD (topically-relevant) unanswerables. Softmax's max-prob is scale-invariant (normalized)
# so a weak best-match still looks confident -> it should leak; copositive's absolute theta should hold.
@torch.no_grad()
def sigset(head, qE, cE, cM):
    return head(qE, cE, cM)                                            # higher = more 'answerable'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n_train", type=int, default=4000); ap.add_argument("--n_eval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--pooling", choices=["mean", "last"], default="mean",
                    help="mean for encoder backbones (MiniLM); last for decoder LLMs (Qwen, etc.)")
    a = ap.parse_args(); rng = np.random.default_rng(a.seed)
    print(f"model={a.model} device={a.device} pooling={a.pooling}  (frozen backbone + trained read)")
    emb = Embedder(a.model, a.device, a.pooling)
    tr = build("train", a.n_train, rng); ev = build("validation", a.n_eval, rng)
    print(f"built {len(tr)} train / {len(ev)} eval examples")
    qTr, cTr, mTr, yTr = encode_set(emb, tr); qEv, cEv, mEv, yEv = encode_set(emb, ev)
    ans = yEv == 1; un = yEv == 0
    # EASY unanswerables: every eval question paired with a RANDOM (unrelated) context's candidates
    perm = rng.permutation(len(ev)); easyC = cEv[perm]; easyM = mEv[perm]
    raw = {}
    for read in ("softmax", "copositive"):
        torch.manual_seed(a.seed)
        head = train_head(Head(emb.d, read).to("cpu"), qTr, cTr, mTr, yTr, steps=a.steps)
        raw[read] = dict(ansd=sigset(head, qEv[ans], cEv[ans], mEv[ans]),
                         hard=sigset(head, qEv[un], cEv[un], mEv[un]),
                         easy=sigset(head, qEv, easyC, easyM))           # all easy = unanswerable
    # Confound-free comparison: at MATCHED answerable recall (sweep each signal's threshold to the same
    # recall), read off the false-answer rate on EASY and HARD unanswerables. The abstention signal is
    # max_score-theta (copositive, absolute) vs softmax confidence (scale-invariant). Lower hard-FA at
    # matched recall = the better OOD abstention signal.
    def at_recall(d, r):
        thr = torch.quantile(d["ansd"], 1.0 - r)
        return (round((d["ansd"] >= thr).float().mean().item(), 3),
                round((d["easy"] >= thr).float().mean().item(), 3),
                round((d["hard"] >= thr).float().mean().item(), 3))
    targets = [0.80, 0.90, 0.95]
    res = {}
    print("\n" + "=" * 84)
    print("  ABSTENTION SIGNAL QUALITY  (false-answer on EASY / HARD unanswerables, at matched recall)")
    print(f"  {'recall target':<16}{'softmax  easy / hard':>24}{'copositive  easy / hard':>26}")
    for r in targets:
        sm = at_recall(raw["softmax"], r); cp = at_recall(raw["copositive"], r)
        res[f"recall{int(r*100)}"] = {"softmax": sm, "copositive": cp}
        print(f"  {r:<14.2f}{f'{sm[1]:.2f} / {sm[2]:.2f}':>24}{f'{cp[1]:.2f} / {cp[2]:.2f}':>26}")
    # headline at recall 0.90
    sm90 = res["recall90"]["softmax"]; cp90 = res["recall90"]["copositive"]
    print(f"\n  HEADLINE @ recall 0.90 -- HARD-unanswerable false-answer (lower=better): "
          f"softmax {sm90[2]:.3f}  vs  copositive {cp90[2]:.3f}")
    if cp90[2] < sm90[2] - 0.03:
        print("  -> copositive abstains better OOD (supports R3 at this scale)")
    elif sm90[2] < cp90[2] - 0.03:
        print("  -> softmax abstains better here (no copositive advantage on these embeddings)")
    else:
        print("  -> roughly equal (certificate's OOD edge does not separate on these entangled embeddings)")
    json.dump(res, open(f"/tmp/qa_abstention_s{a.seed}.json", "w"))
    return res

if __name__ == "__main__":
    main()
