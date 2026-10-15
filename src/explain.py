
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "random_forest.joblib"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Load the trained Random Forest model
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_names = bundle["feature_names"]

# Extract feature importance
classifier = model
importance = classifier.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance,
}).sort_values("importance", ascending=False)

# Save all feature importance scores
importance_df.to_csv(
    RESULTS / "feature_importance.csv",
    index=False,
)

# Plot the top 15 features
top_features = importance_df.head(15).sort_values("importance")

plt.figure(figsize=(10, 7))
plt.barh(top_features["feature"], top_features["importance"])
plt.xlabel("Feature Importance")
plt.ylabel("Behavioral Feature")
plt.title("Top 15 Features — Random Forest")
plt.tight_layout()
plt.savefig(RESULTS / "feature_importance.png", dpi=150)
plt.close()

print("\nTop 15 important features:")
print(importance_df.head(15).to_string(index=False))
print(f"\nSaved: {RESULTS / 'feature_importance.csv'}")
print(f"Saved: {RESULTS / 'feature_importance.png'}")
