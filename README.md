# COSC 4368 – Intro to AI Final Project

**Classifying Aircraft Maintenance Reports by Affected System**
Joshua Nguyen Au · Shriyans Sai · Matthew Rodriguez · Angel Garcia

Given the free-text description in an FAA Service Difficulty Report (SDR), predict which aircraft system it concerns: the first two digits of its JASC/ATA code (e.g. `32` landing gear, `53` fuselage). We compare a classical TF-IDF + Logistic Regression baseline with a fine-tuned DistilBERT transformer.

## Deadlines

| Date | Deliverable | Status |
| --- | --- | --- |
| Sep 28 | One-page proposal | Submitted (`COSC_4368_Final_Project_Proposal.pdf`) |
| **Sun Nov 29, night** | Presentation (10 min + 2 min Q&A) | In progress |
| Dec 11, 11:59 pm | 4–6 page report on the course template + supplementary ZIP | Not started |

## Current state (Nov 26)

| Step | Status |
| --- | --- |
| Download data | Done: FAA SDR 2024 and 2025 CSVs in `data/raw/` (133,735 reports) |
| Clean + split (`prepare.py`) | Done |
| Leakage check + masking | Done: see below |
| Baseline (`baseline.py`) | Done: test accuracy 0.905, macro-F1 0.799 |
| DistilBERT (`bert.py`) | Smoke run passed on MPS (2% of training data, 1 epoch; pipeline check only, not a result). Full 2-epoch run started Nov 26 |
| Learning curve (training-set size) | Planned |
| Error analysis | Planned |
| Slides | Planned |
| Report | Planned |

### Data and splits

- **Input:** the `Discrepancy` text only. JASC code and all other structured fields are never model inputs.
- **Label:** first two digits of `JASCCode`.
- **Cleaning:** drop texts shorter than 20 characters, drop every copy of a text that appears with different labels (1,331 rows), and drop near-duplicates (8,269 rows; same text once digits and punctuation are removed).
- **Classes:** the 25 systems with at least 200 reports in 2024.
- **Split by time:** train/validation = 2024 (90/10, stratified, seed 42); test = all of 2025. This matches real use (train on the past, label new reports) and no 2025 report duplicates a 2024 one.

| Split | Reports |
| --- | ---: |
| Train (2024) | 54,522 |
| Validation (2024) | 6,059 |
| Test (2025) | 61,160 |

Classes are very imbalanced: `53` fuselage is 40% of the data, while the smallest kept classes have about 200 training reports. This is why we report **macro-F1** (every system counts equally) alongside accuracy. Per-class counts are in `data/summary.txt`.

### Finding: label leakage in the text

Mechanics cite manual sections such as `AMM 52-11-21` or `SRM 51-40-02`, whose first two digits are the ATA chapter. **74% of reports contain such a reference, and in 44% the cited chapter equals the label.** A model could score well by reading the answer instead of understanding the description. `prepare.py` replaces these references (and `ATA 32`, `CH 32`) with `REF`. The baseline is also trained on unmasked text as an ablation to measure how much the leak inflates scores.

### Results so far

**Test set (2025, 61,160 reports, 25 systems):**

| Model | Text | Accuracy | Macro-F1 |
| --- | --- | ---: | ---: |
| TF-IDF + Logistic Regression (C=16, balanced) | masked | **0.905** | **0.799** |
| TF-IDF + Logistic Regression (C=16, balanced) | unmasked (leakage ablation) | 0.909 | 0.812 |
| DistilBERT | masked | running | running |

Baseline validation sweep (2024 validation split, macro-F1; manual, no tuning library):

| C | none | balanced |
| ---: | ---: | ---: |
| 1 | 0.797 | 0.831 |
| 4 | 0.836 | 0.849 |
| 16 | 0.844 | **0.851** |

What the baseline shows so far:

- **Balanced class weights help** at every C, because they raise recall on rare systems.
- **The leak is real but small for TF-IDF:** unmasked text adds 1.4 macro-F1 points. The description words already carry most of the signal.
- **Year-to-year drift:** macro-F1 drops from 0.851 on 2024 validation to 0.799 on 2025 test.
- **Weakest systems (test F1):** `31` indicating/recording systems 0.23, `51` standard practices/structures 0.42, `30` ice and rain protection 0.61. `51` is a generic structures chapter that overlaps with fuselage `53` and wings `57`. Best: `33` lights 0.98, `35` oxygen 0.96, `53` fuselage 0.95.

Files: `results/baseline_test.json` (per-class scores), `results/baseline_test_confusion.png`, `results/baseline_top_words.json`.

## Files

| File | Purpose |
| --- | --- |
| `prepare.py` | Load raw CSVs, label, mask references, dedupe, split → `data/{train,val,test}.csv`, `data/summary.txt` |
| `metrics.py` | Shared scoring: accuracy, macro P/R/F1, per-class scores → `results/<name>.json`, confusion-matrix PNG. `python3 metrics.py` runs a self-check |
| `baseline.py` | TF-IDF (1–2-grams) + Logistic Regression: validation sweep, test score, unmasked ablation, top words per class |
| `bert.py` | DistilBERT fine-tuning on MPS/CUDA; keeps the epoch with best validation macro-F1; `--train-frac` for the learning curve |
| `results/` | Metrics, confusion matrices, predictions, logs |

## How to run

```bash
python3 prepare.py                      # ~1 min
python3 baseline.py                     # ~10 min on CPU
python3 -m venv --system-site-packages .venv && .venv/bin/pip install transformers==4.57.1
.venv/bin/python bert.py                # full run; downloads distilbert-base-uncased on first use
.venv/bin/python bert.py --train-frac 0.1   # learning-curve point
```

## Plan to the presentation (Sun Nov 29)

1. **Thu night:** baseline test results; full DistilBERT run.
2. **Fri:** learning curve: both models at 10%, 30%, 100% of the training data. Error analysis: most-confused system pairs, examples where the models disagree.
3. **Sat:** slides: problem, data and class distribution, leakage finding, methods, results table, confusion matrix, learning curve, takeaways.
4. **Sun:** rehearse (10 minutes), final checks.

Team members not running experiments can start the slide template and the background reading (TF-IDF/linear text classification, transformers/DistilBERT, FAA SDR and JASC documentation) now.

## Future vision

- **Report (by Dec 11):** expand the slides into the 4–6 page template with abstract, background, data, methods, experiments, conclusion, references, and the author-contribution footnote with percentages.
- **Is the transformer worth it?** Compare the accuracy gain with training and inference cost (minutes on CPU vs. GPU fine-tuning).
- **Hierarchical labels:** predict the full four-digit JASC code (sub-system), using the two-digit system as the first stage.
- **More years:** train on several years and test how performance drifts over time.
- **Practical use:** rank the top-3 likely systems so a reviewer can confirm quickly, and flag low-confidence reports for manual review.

## Data sources

- FAA, Download Service Difficulty Reports (SDR) by Year: https://www.faa.gov/av-info/download_SDR
- FAA, Joint Aircraft System/Component Code Table and Definitions: https://www.faa.gov/sites/faa.gov/files/JASC_Code.pdf
