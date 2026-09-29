import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from training import train_module

from modules.parkinsons.explainability import (
    generate_xgboost_shap
)


if __name__ == "__main__":

    print(
        "\nTraining Parkinsons Module\n"
    )

    train_module(
        module="parkinsons",
        include_vqc=True
    )

    generate_xgboost_shap(
        "parkinsons"
    )

    print(
        "\nParkinsons Module Complete\n"
    )