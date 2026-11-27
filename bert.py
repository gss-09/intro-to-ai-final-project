"""Fine-tune DistilBERT on the masked SDR descriptions; score on the 2025 test set.

Run with the project venv:  .venv/bin/python bert.py [--train-frac 0.1] [--epochs 2]
"""
import argparse
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from metrics import report

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--train-frac", type=float, default=1.0, help="Stratified share of training data (learning curve)")
p.add_argument("--max-len", type=int, default=128)
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--lr", type=float, default=5e-5)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
NAME = "distilbert-base-uncased"
tag = f"bert_frac{args.train_frac:g}"

train, val, test = (pd.read_csv(f"data/{s}.csv", dtype=str) for s in ("train", "val", "test"))
if args.train_frac < 1:
    train = train.groupby("label", group_keys=False).sample(frac=args.train_frac, random_state=args.seed)
labels = sorted(pd.read_csv("data/train.csv", dtype=str).label.unique())
to_id = {l: i for i, l in enumerate(labels)}

tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=len(labels)).to(device)


def batches(df, shuffle):
    order = np.random.permutation(len(df)) if shuffle else np.argsort(df.text.str.len().values)  # length-sorted eval = less padding
    for i in range(0, len(df), args.batch_size):
        part = df.iloc[order[i:i + args.batch_size]]
        enc = tok(list(part.text), truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
        yield {k: v.to(device) for k, v in enc.items()}, torch.tensor([to_id[l] for l in part.label], device=device), order[i:i + args.batch_size]


@torch.no_grad()
def predict(df):
    model.eval()
    out = np.empty(len(df), dtype=object)
    for enc, _, idx in batches(df, shuffle=False):
        out[idx] = [labels[i] for i in model(**enc).logits.argmax(1).cpu().tolist()]
    return out


steps = args.epochs * -(-len(train) // args.batch_size)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
best_f1, best_state, log = -1, None, []
print(f"{tag}: device {device}, train {len(train)}, steps {steps}", flush=True)
for epoch in range(1, args.epochs + 1):
    model.train()
    start, total = time.time(), 0.0
    for step, (enc, y, _) in enumerate(batches(train, shuffle=True), 1):
        loss = torch.nn.functional.cross_entropy(model(**enc).logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        total += loss.item()
        if step % 200 == 0:
            print(f"  epoch {epoch} step {step}: loss {total / step:.4f} ({time.time() - start:.0f}s)", flush=True)
    f1 = f1_score(val.label, predict(val), average="macro")
    log.append({"epoch": epoch, "train_loss": total / step, "val_macro_f1": f1, "seconds": time.time() - start})
    print(log[-1], flush=True)
    if f1 > best_f1:
        best_f1, best_state = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

pd.DataFrame(log).to_csv(f"results/{tag}_log.csv", index=False)
model.load_state_dict(best_state)
pred = predict(test)
report(f"{tag}_test", test.label, pred, labels)
pd.DataFrame({"label": test.label, "pred": pred, "text": test.text}).to_csv(f"results/{tag}_test_predictions.csv", index=False)
