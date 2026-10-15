# Behavior-Aware Early Ransomware Detection Using Explainable Machine Learning

An explainable machine learning research prototype that classifies ransomware and goodware using Windows API call and behavioral signature features.

## Overview

Ransomware detection is a cybersecurity challenge because malicious and benign programs can exhibit overlapping behaviors. This project evaluates machine learning classifiers on pre-extracted behavioral features and provides feature-importance explanations to help understand model predictions.

## Objectives

- Train and compare Logistic Regression and Random Forest classifiers.
- Evaluate models using accuracy, precision, recall, F1-score, and ROC-AUC.
- Identify influential behavioral features using Random Forest feature importance.
- Provide an interactive Streamlit interface for offline CSV predictions.

## Technology Stack

- Python
- Pandas and scikit-learn
- Matplotlib
- Joblib
- Streamlit

## Project Structure

```text
Ransomware-Detection-XAI/
├── data/
│   └── raw/                 # Local datasets; excluded from Git
├── models/                  # Local trained models; excluded from Git
├── results/                 # Evaluation outputs
├── src/
│   ├── train.py             # Model training and evaluation
│   ├── explain.py           # Feature-importance analysis
│   └── app.py               # Streamlit interface
├── .gitignore
├── requirements.txt
└── README.md
```

## Models and Evaluation

Two baseline classifiers are evaluated on the supplied fixed training and test split.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 95.63% | 96.88% |
| Precision | 96.20% | 96.30% |
| Recall | 95.00% | 97.50% |
| F1-score | 95.60% | 96.89% |
| ROC-AUC | 0.985 | 0.993 |

These results are from a test set of 160 samples. Performance should be validated on additional, independently sourced data before drawing conclusions about real-world effectiveness.

## Explainability

The project calculates Random Forest feature importance and saves a ranked CSV file and visualization. These scores indicate which features contribute most to the model's overall decisions; they do not establish causation.

## Setup

1. Install Python 3.12 or a compatible Python version.
2. Clone the repository.
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Place the permitted dataset files in `data/raw/`, using the filenames expected by the training script.
5. Train the models:

   ```bash
   python src/train.py
   ```

6. Generate feature-importance outputs:

   ```bash
   python src/explain.py
   ```

7. Launch the dashboard:

   ```bash
   python -m streamlit run src/app.py
   ```

## Dataset

The project uses pre-extracted behavioral features and separate training and test labels. Consult the original dataset source and license before downloading, redistributing, or publishing any dataset files. Raw data is excluded from this repository by default.

## Limitations

- This is an offline research prototype, not a real-time endpoint monitoring system.
- The input features are already extracted; live collection and temporal early-detection behavior are not implemented.
- Model performance may differ on unseen programs, new malware families, or different data sources.
- Feature importance is not a causal explanation.
- Predictions should not replace professional endpoint protection or security analysis.

## Future Work

- Add stronger validation and evaluate generalization across malware families.
- Explore SHAP-based local explanations.
- Evaluate false-positive rates and decision thresholds.
- Investigate time-ordered behavioral data for genuine early-detection evaluation.

## Disclaimer

For defensive cybersecurity research and educational use only. Do not execute live malware as part of this project.
