import json

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

from pathlib import Path

ARTIFACT_DIR = Path("artifacts")


def generate_xgboost_shap(
    module_name,
    max_patients=100
):

    artifact = (
        ARTIFACT_DIR
        / module_name
    )

    test = pd.read_csv(
        artifact / "test_patients.csv"
    )

    X = test.drop(
        columns=["y_true"]
    )

    manifest = json.loads(
        (artifact / "manifest.json")
        .read_text()
    )

    feature_names = (
        manifest["feature_names"]
    )

    preprocessor = joblib.load(
        artifact / "preprocessor_full.joblib"
    )

    model = joblib.load(
        artifact
        / "models"
        / "classical_models.joblib"
    )["XGBoost"]

    transformed = pd.DataFrame(
        preprocessor.transform(
            X.iloc[:max_patients]
        ),
        columns=feature_names
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        transformed
    )

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap":
                abs(shap_values).mean(axis=0)
        }
    )

    importance = importance.sort_values(
        "mean_abs_shap",
        ascending=False
    )

    importance.to_csv(
        artifact
        / "shap_importance_xgboost.csv",
        index=False
    )

    plt.figure(figsize=(10, 6))

    shap.summary_plot(
        shap_values,
        transformed,
        plot_type="bar",
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        artifact
        / "shap_xgboost.png",
        dpi=200
    )

    plt.close()