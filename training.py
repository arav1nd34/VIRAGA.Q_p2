"""Leakage-safe training, evaluation, and artifact generation."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from data import load_module
from quantum import build_qsvc, build_vqc
from settings import ARTIFACT_DIR, QUBIT_COUNTS, RANDOM_STATE


def _split(X, y, groups, module):
    if module == "parkinsons":
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        return train_idx, test_idx
    return train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def _inner_split(X, y, groups, module):
    """Validation split used for choosing the QML qubit count; never use test rows."""
    if module == "parkinsons":
        splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE + 1)
        fit_idx, validation_idx = next(splitter.split(X, y, groups))
        return fit_idx, validation_idx
    return train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=RANDOM_STATE + 1)


def _balanced_subset(y: pd.Series, max_rows: int, seed: int = RANDOM_STATE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = y.to_numpy()
    pos, neg = np.where(values == 1)[0], np.where(values == 0)[0]
    per_class = min(len(pos), len(neg), max_rows // 2)
    chosen = np.concatenate([rng.choice(pos, per_class, replace=False), rng.choice(neg, per_class, replace=False)])
    rng.shuffle(chosen)
    return chosen


def _metrics(y_true, probability, train_seconds, model_name):
    prediction = (probability >= 0.5).astype(int)
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "sensitivity": recall_score(y_true, prediction, zero_division=0),
        "specificity": recall_score(y_true, prediction, pos_label=0, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
        "train_seconds": train_seconds,
    }


def _probabilities(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    score = model.decision_function(X)
    return 1 / (1 + np.exp(-score))


def train_module(module: str, include_vqc: bool = False, quantum_sample_size: int = 160) -> Path:
    """Train models once and save all objects required by Streamlit.

    QSVC has an O(N^2) kernel cost. The QML branch therefore trains on a balanced
    training subset; compact classical SVM is evaluated on the identical subset.
    """
    X, y, groups = load_module(module)
    feature_names = X.columns.tolist()
    train_idx, test_idx = _split(X, y, groups, module)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    fit_idx, validation_idx = _inner_split(
        X_train, y_train, groups.iloc[train_idx].reset_index(drop=True), module
    )

    output = ARTIFACT_DIR / module
    (output / "models").mkdir(parents=True, exist_ok=True)

    full_prep = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    X_train_full = full_prep.fit_transform(X_train)
    X_test_full = full_prep.transform(X_test)
    joblib.dump(full_prep, output / "preprocessor_full.joblib")

    classical = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "RBF SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=350, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, eval_metric="logloss", random_state=RANDOM_STATE),
    }
    rows, prediction_frames = [], []
    saved_models = {}
    for name, model in classical.items():
        start = time.perf_counter(); model.fit(X_train_full, y_train); elapsed = time.perf_counter() - start
        probability = _probabilities(model, X_test_full)
        rows.append(_metrics(y_test.to_numpy(), probability, elapsed, name))
        prediction_frames.append(pd.DataFrame({"model": name, "y_true": y_test.to_numpy(), "probability": probability}))
        saved_models[name] = model
    joblib.dump(saved_models, output / "models" / "classical_models.joblib")

    # The selector is fitted only on the training data. It defines the compact QML input.
    best_qml = None
    qml_metadata = []
    for n_qubits in QUBIT_COUNTS:
        if n_qubits > X_train.shape[1]:
            continue
        # First, choose circuit width using validation patients only.
        validation_prep = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("select", SelectKBest(mutual_info_classif, k=n_qubits)),
            ("angles", MinMaxScaler(feature_range=(0, np.pi))),
        ])
        X_fit_q = validation_prep.fit_transform(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
        X_validation_q = validation_prep.transform(X_train.iloc[validation_idx])
        validation_subset = _balanced_subset(y_train.iloc[fit_idx], quantum_sample_size)
        validation_qsvc = build_qsvc(n_qubits=n_qubits)
        validation_qsvc.fit(X_fit_q[validation_subset], y_train.iloc[fit_idx].iloc[validation_subset])
        validation_probability = _probabilities(validation_qsvc, X_validation_q)
        validation_auc = roc_auc_score(y_train.iloc[validation_idx], validation_probability)

        # Refit the selected preprocessing form on the entire training partition, then test once.
        compact = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("select", SelectKBest(mutual_info_classif, k=n_qubits)),
            ("angles", MinMaxScaler(feature_range=(0, np.pi))),
        ])
        X_train_q = compact.fit_transform(X_train, y_train)
        X_test_q = compact.transform(X_test)
        selected = np.asarray(feature_names)[compact.named_steps["select"].get_support()].tolist()

        # Balanced, fixed quantum subset keeps kernel training practical and comparable.
        subset = _balanced_subset(y_train, quantum_sample_size)

        compact_svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)
        start = time.perf_counter(); compact_svm.fit(X_train_q[subset], y_train.iloc[subset]); elapsed = time.perf_counter() - start
        compact_prob = _probabilities(compact_svm, X_test_q)
        compact_name = f"RBF SVM ({n_qubits} compact features)"
        rows.append(_metrics(y_test.to_numpy(), compact_prob, elapsed, compact_name))
        prediction_frames.append(pd.DataFrame({"model": compact_name, "y_true": y_test.to_numpy(), "probability": compact_prob}))

        qsvc = build_qsvc(n_qubits=n_qubits)
        start = time.perf_counter(); qsvc.fit(X_train_q[subset], y_train.iloc[subset]); elapsed = time.perf_counter() - start
        q_prob = _probabilities(qsvc, X_test_q)
        q_name = f"QSVC ({n_qubits} qubits)"
        metric = _metrics(y_test.to_numpy(), q_prob, elapsed, q_name)
        rows.append(metric)
        prediction_frames.append(pd.DataFrame({"model": q_name, "y_true": y_test.to_numpy(), "probability": q_prob}))
        joblib.dump(compact, output / "models" / f"qsvc_{n_qubits}_preprocessor.joblib")
        joblib.dump(qsvc, output / "models" / f"qsvc_{n_qubits}.joblib")
        qml_metadata.append({"qubits": n_qubits, "selected_features": selected, "validation_roc_auc": validation_auc, "test_metrics": metric})
        if best_qml is None or validation_auc > best_qml["validation_roc_auc"]:
            best_qml = {"qubits": n_qubits, "validation_roc_auc": validation_auc, "test_metrics": metric, "selected_features": selected}

    if include_vqc and best_qml is not None:
        n_qubits = best_qml["qubits"]
        prep = joblib.load(output / "models" / f"qsvc_{n_qubits}_preprocessor.joblib")
        X_train_q, X_test_q = prep.transform(X_train), prep.transform(X_test)
        vqc = build_vqc(n_qubits=n_qubits)
        start = time.perf_counter(); vqc.fit(X_train_q, y_train); elapsed = time.perf_counter() - start
        vqc_prob = _probabilities(vqc, X_test_q)
        rows.append(_metrics(y_test.to_numpy(), vqc_prob, elapsed, f"VQC ({n_qubits} qubits)"))
        prediction_frames.append(pd.DataFrame({"model": f"VQC ({n_qubits} qubits)", "y_true": y_test.to_numpy(), "probability": vqc_prob}))
        joblib.dump(vqc, output / "models" / f"vqc_{n_qubits}.joblib")

    metrics = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    X_test.assign(y_true=y_test.to_numpy()).to_csv(output / "test_patients.csv", index=False)
    with open(output / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"module": module, "feature_names": feature_names, "best_qsvc": best_qml, "qsvc_runs": qml_metadata}, handle, indent=2)
    return output
