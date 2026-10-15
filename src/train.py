
from pathlib import Path
import json

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


def main():
    X_train = pd.read_csv(DATA / "X_train_clean.csv")
    y_train = pd.read_csv(DATA / "y_train.csv").iloc[:, 0]
    X_test = pd.read_csv(DATA / "X_test_clean.csv")
    y_test = pd.read_csv(DATA / "y_test.csv").iloc[:, 0]

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Training and testing feature columns do not match.")

    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature and label counts do not match.")

    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    all_metrics = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(
                y_test, predictions, zero_division=0
            )),
            "recall": float(recall_score(
                y_test, predictions, zero_division=0
            )),
            "f1_score": float(f1_score(
                y_test, predictions, zero_division=0
            )),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "classification_report": classification_report(
                y_test, predictions, output_dict=True, zero_division=0
            ),
        }

        all_metrics[name] = metrics

        joblib.dump(
            {
                "model": model,
                "feature_names": list(X_train.columns),
                "target_name": "ransomware/goodware",
            },
            MODELS / f"{name}.joblib",
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(y_test, predictions),
            display_labels=["Class 0", "Class 1"],
        )
        display.plot(values_format="d")
        plt.title(f"Confusion Matrix: {name}")
        plt.tight_layout()
        plt.savefig(RESULTS / f"{name}_confusion_matrix.png", dpi=150)
        plt.close()

        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-score:  {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

    with open(RESULTS / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=2)

    summary = pd.DataFrame({
        name: {
            key: values[key]
            for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        }
        for name, values in all_metrics.items()
    }).T

    summary.to_csv(RESULTS / "model_comparison.csv")

    print("\nMODEL COMPARISON")
    print(summary.round(4))
    print(f"\nModels saved in: {MODELS}")
    print(f"Results saved in: {RESULTS}")


if __name__ == "__main__":
    main()
