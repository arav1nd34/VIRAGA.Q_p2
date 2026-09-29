import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier


def build_models():

    return {
        "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            ),

        "RBF SVM":
            SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),

        "XGBoost":
            XGBClassifier(
                n_estimators=350,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="logloss",
                random_state=42
            )
    }


def train_models(models, X_train, y_train):

    trained = {}

    timings = {}

    for name, model in models.items():

        start = time.perf_counter()

        model.fit(X_train, y_train)

        timings[name] = (
            time.perf_counter() - start
        )

        trained[name] = model

    return trained, timings