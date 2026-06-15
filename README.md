# Copositive QA-abstention — Colab runner

One self-contained experiment: does a **copositive** (absolute-threshold) abstention read beat
**softmax** (scale-invariant confidence) at refusing to answer ungrounded questions **under
distribution shift**? A *frozen* LLM backbone embeds SQuAD-v2 questions + context sentences; a tiny
trained read decides answer-vs-abstain. We compare false-answer rate at *matched recall* on EASY
(question × unrelated context) and HARD (topically-relevant) unanswerables. Already validated on CPU
with a MiniLM encoder; this runs it on a real **decoder LLM** (Qwen3-0.6B).

## Run on Colab — least effort (free T4, ~10–15 min)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/areidyOTH/copositive-colab/blob/main/run_colab.ipynb)

1. Click the badge ☝ (GPU is preset in the notebook).
2. **Runtime → Run all**.
3. Paste the three printed tables back.

## Or via git clone (a Colab cell)
```python
!git clone https://github.com/areidyOTH/copositive-colab.git
%cd copositive-colab
!pip -q install -U transformers datasets
for s in range(3):
    print(f"\n===== seed {s} =====")
    !python qa_abstention.py --model Qwen/Qwen3-0.6B --device cuda --pooling last \
        --n_train 4000 --n_eval 2000 --steps 600 --seed {s}
```
Fallback model if `Qwen3-0.6B` errors: `--model Qwen/Qwen2.5-0.5B`.

## What to send back
For each seed, the `ABSTENTION SIGNAL QUALITY` table (false-answer easy/hard at matched recall) and the
`HEADLINE` line. Lower copositive numbers ⇒ its abstention transfers across the shift where softmax's
calibrated confidence leaks.
