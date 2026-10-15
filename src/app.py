
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

st.set_page_config(
    page_title="Behavior-Aware Ransomware Detection",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Behavior-Aware Ransomware Detection")
st.caption(
    "Explainable ML research prototype | Offline CSV analysis only"
)

st.warning(
    "Research prototype only. Predictions are not a guarantee "
    "of security and should not replace endpoint protection."
)

# Show evaluation results
metrics_path = RESULTS / "metrics.json"
if metrics_path.exists():
    st.subheader("Model Evaluation")
    metrics = pd.read_json(metrics_path, orient="index")
    st.dataframe(
        metrics[["accuracy", "precision", "recall", "f1_score", "roc_auc"]],
        use_container_width=True,
    )

# Feature importance chart
importance_path = RESULTS / "feature_importance.csv"
if importance_path.exists():
    st.subheader("Why does the model make predictions?")
    importance = pd.read_csv(importance_path).head(15)
    st.bar_chart(importance.set_index("feature")["importance"])

# Upload feature data for prediction
st.subheader("Predict from behavioral features")
uploaded = st.file_uploader(
    "Upload a CSV containing the 426 expected feature columns",
    type=["csv"],
)

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
        st.dataframe(df.head(10), use_container_width=True)

        model_path = MODELS / "random_forest.joblib"

        if not model_path.exists():
            st.error("Model not found. Run: python src/train.py")
        else:
            bundle = joblib.load(model_path)
            model = bundle["model"]
            expected = bundle["feature_names"]

            missing = [col for col in expected if col not in df.columns]
            extra = [col for col in df.columns if col not in expected]

            if missing:
                st.error(
                    f"Missing {len(missing)} required columns. "
                    f"Examples: {missing[:10]}"
                )
            elif extra:
                st.warning(
                    f"{len(extra)} extra columns will be ignored."
                )

            if not missing and st.button("Predict uploaded rows"):
                X = df[expected].apply(pd.to_numeric, errors="coerce")

                if X.isna().any().any():
                    st.error(
                        "Feature columns contain missing or non-numeric "
                        "values. Please clean the CSV and try again."
                    )
                else:
                    predictions = model.predict(X)
                    probabilities = model.predict_proba(X)

                    output = df.copy()
                    output["predicted_class"] = predictions
                    output["prediction"] = [
                        "Ransomware" if value == 1 else "Goodware"
                        for value in predictions
                    ]
                    output["ransomware_probability"] = probabilities[:, 1]

                    st.subheader("Prediction results")
                    st.dataframe(output, use_container_width=True)

                    st.download_button(
                        "Download predictions",
                        data=output.to_csv(index=False).encode("utf-8"),
                        file_name="ransomware_predictions.csv",
                        mime="text/csv",
                    )

    except Exception as exc:
        st.error(f"Could not process this CSV: {exc}")

st.caption(
    "Predictions depend on the quality and provenance of the input data. "
    "This app does not monitor a computer or detect live ransomware."
)
