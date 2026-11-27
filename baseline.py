"""TF-IDF + Logistic Regression baseline.

C and class weighting are chosen by hand on the validation set, then the chosen
model is scored once on the 2025 test set. Also runs the leakage ablation:
the same model trained and tested on unmasked text.
"""
import json
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline

from metrics import report

train, val, test = (pd.read_csv(f"data/{s}.csv", dtype=str) for s in ("train", "val", "test"))
labels = sorted(train.label.unique())


def model(C, weight):
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000),
        LogisticRegression(C=C, class_weight=weight, max_iter=2000))


# Manual validation sweep (no tuning library).
sweep = []
for C in (1, 4, 16):
    for weight in (None, "balanced"):
        start = time.time()
        m = model(C, weight).fit(train.text, train.label)
        f1 = f1_score(val.label, m.predict(val.text), average="macro")
        sweep.append({"C": C, "class_weight": weight or "none", "val_macro_f1": f1, "seconds": time.time() - start})
        print(sweep[-1], flush=True)
best = max(sweep, key=lambda r: r["val_macro_f1"])
C, weight = best["C"], None if best["class_weight"] == "none" else "balanced"
pd.DataFrame(sweep).to_csv("results/baseline_sweep.csv", index=False)

final = model(C, weight).fit(train.text, train.label)
report("baseline_test", test.label, final.predict(test.text), labels)
pd.DataFrame({"label": test.label, "pred": final.predict(test.text), "text": test.text}).to_csv(
    "results/baseline_test_predictions.csv", index=False)

# Leakage ablation: unmasked manual references.
leaky = model(C, weight).fit(train.raw_text, train.label)
report("baseline_test_unmasked", test.label, leaky.predict(test.raw_text), labels)

# Most informative words per class, for the slides.
vec, clf = final.named_steps.values()
words = vec.get_feature_names_out()
top = {c: [words[i] for i in clf.coef_[k].argsort()[-8:][::-1]] for k, c in enumerate(clf.classes_)}
with open("results/baseline_top_words.json", "w") as f:
    json.dump({"chosen": best, "top_words": top}, f, indent=2)
