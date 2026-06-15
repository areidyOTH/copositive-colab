"""Load the 4 trained models from Colab's artifacts/ (download artifacts.zip, unzip) on top of the Qwen3
backbone LOCALLY and compare how each one answers vs. abstains on the same question/context pairs.
Shows the per-model abstention signal: copositive -> margin (max_score - theta), answers iff > 0 (the
structural null); softmax -> confidence (max prob, no structural abstain).

Run:  python compare_local.py --artifacts artifacts --model Qwen/Qwen3-0.6B   (CPU is fine)
"""
import argparse, torch, torch.nn.functional as F
import qa_abstention as QA
from transformers import AutoModel, AutoTokenizer

# (question, [candidate sentences], answerable?) -- 2 answerable, 2 unanswerable (unrelated context)
DEMOS = [
    ("What is the capital of France?",
     ["Paris is the capital and largest city of France.", "France borders Spain and Germany.",
      "The Eiffel Tower was completed in 1889."], True),
    ("Who painted the Mona Lisa?",
     ["The Mona Lisa was painted by Leonardo da Vinci.", "It hangs in the Louvre museum.",
      "Many tourists visit it each year."], True),
    ("How many moons does Mars have?",
     ["The recipe calls for two cups of flour.", "Preheat the oven to 350 degrees.",
      "Let the dough rest for an hour."], False),
    ("When was the printing press invented?",
     ["The quarterback threw for three touchdowns.", "The game went into overtime.",
      "Fans celebrated late into the night."], False),
]

def embed(model, tok, texts, pooling="last"):
    t = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        h = model(**t).last_hidden_state
    m = t["attention_mask"]
    if pooling == "last":
        emb = h[torch.arange(h.size(0)), m.sum(1) - 1]
    else:
        mf = m.unsqueeze(-1).float(); emb = (h * mf).sum(1) / mf.sum(1).clamp(min=1)
    return F.normalize(emb.float(), dim=-1)

def load_head(art, variant, d):
    read = "copositive" if "copositive" in variant else "softmax"
    head = QA.Head(d, read); head.load_state_dict(torch.load(f"{art}/{variant}_head.pt", map_location="cpu"))
    return head.eval()

def backbone_for(variant, mid, art):
    base = AutoModel.from_pretrained(mid).eval().float()
    if variant.startswith("lora"):
        from peft import PeftModel
        return PeftModel.from_pretrained(base, f"{art}/{variant}").eval()
    return base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts"); ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--pooling", default="last")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    variants = ["frozen_softmax", "frozen_copositive", "lora_softmax", "lora_copositive"]
    # precompute each model's signal on each demo
    rows = {v: [] for v in variants}
    d = None
    for v in variants:
        bb = backbone_for(v, a.model, a.artifacts)
        d = bb.config.hidden_size if hasattr(bb, "config") else bb.base_model.config.hidden_size
        head = load_head(a.artifacts, v, d)
        for q, cands, _ in DEMOS:
            qE = embed(bb, tok, [q], a.pooling); cE = embed(bb, tok, cands, a.pooling).unsqueeze(0)
            cM = torch.ones(1, cE.shape[1])
            sig = head(qE, cE, cM).item()
            rows[v].append(sig)
        del bb
    print("\nPer-model abstention signal on each question (copositive: margin>0=ANSWER; softmax: confidence)\n")
    for i, (q, cands, ans) in enumerate(DEMOS):
        print(f"[{'ANSWERABLE' if ans else 'UNANSWERABLE'}] {q}")
        for v in variants:
            sig = rows[v][i]
            if "copositive" in v:
                tag = "ANSWER " if sig > 0 else "ABSTAIN"
                print(f"    {v:<20} margin {sig:+.3f}  -> {tag}")
            else:
                print(f"    {v:<20} confidence {sig:.3f}  (no structural abstain)")
        print()
    print("Read it: a good model ANSWERS the answerable ones and ABSTAINS (margin<0) on the unanswerable")
    print("ones. Compare frozen vs LoRA copositive: LoRA should push unanswerable margins more negative.")

if __name__ == "__main__":
    main()
