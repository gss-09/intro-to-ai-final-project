"""Shared scoring so every model is evaluated the same way."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def report(name, y_true, y_pred, labels, out_dir="results"):
    """Write <name>.json (overall + per-class) and <name>_confusion.png; return the overall dict."""
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": float(p.mean()), "macro_recall": float(r.mean()),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "per_class": {str(l): {"precision": float(a), "recall": float(b), "f1": float(c), "support": int(d)}
                      for l, a, b, c, d in zip(labels, p, r, f, s)},
    }
    (out / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")

    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    cm /= cm.sum(1, keepdims=True).clip(min=1)  # row-normalized: share of each true class
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted system (JASC chapter)")
    ax.set_ylabel("True system (JASC chapter)")
    ax.set_title(f"{name}: acc {result['accuracy']:.3f}, macro-F1 {result['macro_f1']:.3f}")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / f"{name}_confusion.png", dpi=150)
    plt.close(fig)
    print(f"{name}: accuracy {result['accuracy']:.4f}  macro-F1 {result['macro_f1']:.4f}")
    return result


if __name__ == "__main__":
    # self-check: perfect predictions score 1.0, one wrong label lowers both scores
    y = np.array(["21", "32", "32", "53"])
    assert report("_selfcheck", y, y, ["21", "32", "53"], "/tmp")["macro_f1"] == 1.0
    assert report("_selfcheck", y, np.array(["21", "32", "53", "53"]), ["21", "32", "53"], "/tmp")["accuracy"] == 0.75
    print("ok")
