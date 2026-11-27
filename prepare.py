"""Build train/val/test splits from FAA SDR CSVs.

Label = first two digits of the JASC code (aircraft system / ATA chapter).
Train + val come from 2024 (90/10, stratified, seed 42); test is all of 2025.
Manual references such as "AMM 52-11-21" give away the chapter, so they are masked.
"""
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW, OUT = Path("data/raw"), Path("data")
MIN_TRAIN = 200  # keep systems with at least this many 2024 reports
# "52-11-21", "21-31", "ATA 32", "CHAPTER 32", "CH 32": all reveal the label
REF = re.compile(r"\b\d{2}-\d{2}(?:-\d{2,3})?\b|\b(?:ATA|CHAPTER|CH)\.?\s*\d{2}\b")


def mask(text):
    return REF.sub(" REF ", text)


def key(text):
    """Near-duplicate key: lowercase letters only, so reports differing only in numbers collide."""
    return re.sub(r"[^a-z]+", " ", text.lower()).strip()


def load(year):
    df = pd.read_csv(RAW / f"SDR-{year}.csv", dtype=str, usecols=["JASCCode", "Discrepancy"])
    df = pd.DataFrame({"label": df.JASCCode.str.strip().str[:2],
                       "raw_text": df.Discrepancy.fillna("").str.strip(), "year": year})
    df = df[df.label.str.fullmatch(r"\d{2}") & (df.raw_text.str.len() >= 20)]
    df["text"] = df.raw_text.map(mask)
    df["key"] = df.text.map(key)
    return df


def main():
    df = pd.concat([load(2024), load(2025)], ignore_index=True)
    n0 = len(df)
    # The same text with different labels cannot be learned; drop every copy of it.
    conflicting = df.groupby("key").label.nunique() > 1
    df = df[~df.key.isin(conflicting[conflicting].index)]
    n1 = len(df)
    # Keep the first copy of each near-duplicate; 2024 sorts first, so 2025 copies of
    # 2024 reports leave the test set.
    df = df.drop_duplicates("key")
    n2 = len(df)
    counts = df[df.year == 2024].label.value_counts()
    keep = sorted(counts[counts >= MIN_TRAIN].index)
    df = df[df.label.isin(keep)]

    old, test = df[df.year == 2024], df[df.year == 2025]
    train, val = train_test_split(old, test_size=0.1, stratify=old.label, random_state=42)
    assert not (set(train.key) & set(test.key)) and not (set(train.key) & set(val.key))
    for name, part in (("train", train), ("val", val), ("test", test)):
        part[["label", "text", "raw_text"]].to_csv(OUT / f"{name}.csv", index=False)

    summary = (f"rows read {n0}; label-conflict texts dropped {n0 - n1}; "
               f"near-duplicates dropped {n1 - n2}; classes kept {len(keep)} (>= {MIN_TRAIN} in 2024), "
               f"rows in dropped classes {n2 - len(df)}\n"
               f"train {len(train)}  val {len(val)}  test {len(test)}\n"
               f"reports with a masked reference: {(df.text != df.raw_text).mean():.1%}\n")
    counts = pd.DataFrame({"train": train.label.value_counts(), "val": val.label.value_counts(),
                           "test": test.label.value_counts()}).fillna(0).astype(int)
    (OUT / "summary.txt").write_text(summary + counts.sort_values("train", ascending=False).to_string() + "\n")
    print((OUT / "summary.txt").read_text())


if __name__ == "__main__":
    main()
