from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data" / "raw" / "parkinsons" / "parkinsons.data"


def load_dataset():
    df = pd.read_csv(DATA_FILE)

    y = df["status"]

    X = df.drop(
        columns=["name", "status"],
        errors="ignore"
    )

    groups = pd.Series(
        df["name"],
        name="subject"
    )

    return X, y, groups