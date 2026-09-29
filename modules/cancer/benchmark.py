from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc
)

ARTIFACT_DIR = Path("artifacts")


def load_metrics(module_name):

    return pd.read_csv(
        ARTIFACT_DIR /
        module_name /
        "metrics.csv"
    )


def load_predictions(module_name):

    return pd.read_csv(
        ARTIFACT_DIR /
        module_name /
        "predictions.csv"
    )


def leaderboard(module_name):

    metrics = load_metrics(module_name)

    return metrics.sort_values(
        "roc_auc",
        ascending=False
    )


def create_metric_plot(module_name):

    metrics = load_metrics(module_name)

    metrics = metrics.set_index("model")

    metric_columns = [
        "accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc"
    ]

    available = [
        c
        for c in metric_columns
        if c in metrics.columns
    ]

    metrics[available].plot(
        kind="bar",
        figsize=(12, 6)
    )

    plt.title(
        f"{module_name.upper()} Model Comparison"
    )

    plt.ylabel("Score")

    plt.tight_layout()

    plt.savefig(
        ARTIFACT_DIR /
        module_name /
        "metric_comparison.png"
    )

    plt.close()


def create_roc_plot(module_name):

    predictions = load_predictions(
        module_name
    )

    plt.figure(figsize=(8, 6))

    for model in predictions["model"].unique():

        subset = predictions[
            predictions["model"] == model
        ]

        if "probability" not in subset.columns:
            continue

        fpr, tpr, _ = roc_curve(
            subset["y_true"],
            subset["probability"]
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{model} ({roc_auc:.3f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        "--"
    )

    plt.legend()

    plt.title(
        f"{module_name.upper()} ROC Curves"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.tight_layout()

    plt.savefig(
        ARTIFACT_DIR /
        module_name /
        "roc_curve.png"
    )

    plt.close()


def create_confusion_matrices(
    module_name
):

    predictions = load_predictions(
        module_name
    )

    for model in predictions[
        "model"
    ].unique():

        subset = predictions[
            predictions["model"] == model
        ]

        y_true = subset["y_true"]

        y_pred = (
            subset["probability"] >= 0.5
        ).astype(int)

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        plt.figure(figsize=(5, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues"
        )

        plt.title(model)

        plt.tight_layout()

        safe_name = (
            model
            .replace("/", "_")
            .replace(" ", "_")
        )

        plt.savefig(
            ARTIFACT_DIR /
            module_name /
            f"confusion_{safe_name}.png"
        )

        plt.close()


def run_benchmark(
    module_name
):

    create_metric_plot(
        module_name
    )

    create_roc_plot(
        module_name
    )

    create_confusion_matrices(
        module_name
    )

    print(
        f"Benchmark complete: {module_name}"
    )