# AGENTS.md

Guide for AI coding agents working in this repo. Humans: see `README.md`.

## Project

COSC 4368 final project. Classify FAA Service Difficulty Reports (SDR) by aircraft system. The input is the `Discrepancy` text, and the label is the first two digits of `JASCCode` (25 classes). We compare a TF-IDF + Logistic Regression baseline against fine-tuned DistilBERT. The presentation is Sun Nov 29, 2026, and the report is due Dec 11, 2026.

## Setup and commands

Run everything from the repo root.

```bash
# raw data (git-ignored): data/raw/SDR-2024.csv, data/raw/SDR-2025.csv
# from https://external.apic4e.faa.gov/sdrs/retrieve/SDR-{year}.csv
python3 prepare.py                 # -> data/{train,val,test}.csv, data/summary.txt
python3 metrics.py                 # self-check, prints "ok"
python3 baseline.py                # ~10 min CPU -> results/baseline_*
python3 -m venv --system-site-packages .venv && .venv/bin/pip install transformers==4.57.1
.venv/bin/python bert.py           # ~40 min on Apple MPS -> results/bert_frac1_*
.venv/bin/python bert.py --train-frac 0.1   # learning-curve point -> results/bert_frac0.1_*
```

These are plain scripts, with no package, no test suite and no CI. `python3 metrics.py` is the only automated check.

## Rules that protect the results

- **Keep the masking.** Manual references such as `AMM 52-11-21`, `ATA 32` or `CH 32` give away the label. `prepare.py` replaces them with `REF`. Models train on `text`, which is masked. `raw_text`, which is unmasked, is used only for the leakage ablation.
- **Keep the temporal split.** Train/val come from 2024 (90/10, stratified, seed 42), and test is all of 2025. Never tune on the test set. Choose hyperparameters and epochs by validation macro-F1.
- **Score every model with `metrics.report()`.** This keeps accuracy, macro P/R/F1, per-class scores and the confusion matrix comparable. Macro-F1 is the headline metric because the classes are very imbalanced (`53` is about 40% of the data).
- **Only the Discrepancy text is a model input.** Never feed JASC codes or other structured fields to a model.
- Use seed 42 everywhere.

## Code style

- Keep scripts short and flat, and follow the existing files. Use pandas, scikit-learn, PyTorch and `transformers`, and add no new dependencies without a reason.
- Name outputs after the run: `results/<name>.json` and `results/<name>_confusion.png`. `bert.py` uses `bert_frac{train_frac}` as its tag.
- When a new result lands, update the results and current-state tables in `README.md`.

## Git

- Commit and push straight to `main`. Do not create branches or PRs, and do not force-push. If a push is rejected, rebase on `origin/main`.
- Do not add AI co-author or session trailers to commit messages.
- Never commit `data/raw/`, `data/*.csv`, `results/*_predictions.csv`, `results/*.log` or `.venv/`. These are already git-ignored.
