from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

ROOT = Path(__file__).resolve().parents[2]


def load_dataset():
    data = load_breast_cancer()

    X = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )

    y = pd.Series(
        1 - data.target,
        name="target"
    )

    groups = pd.Series(
        range(len(X)),
        name="group"
    )

    return X, y, groups