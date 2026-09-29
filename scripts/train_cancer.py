import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from training import train_module

from modules.cancer.explainability import (
    generate_xgboost_shap
)


if __name__ == "__main__":

    print(
        "\nTraining Cancer Module\n"
    )

    train_module(
        module="cancer",
        include_vqc=True
    )

    generate_xgboost_shap(
        "cancer"
    )

    print(
        "\nCancer Module Complete\n"
    )