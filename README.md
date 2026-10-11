# Explainable Federated Learning Framework for Predicting Customer Churn in Telecom Networks

An experimental machine learning framework that compares centralized and simulated federated learning approaches for telecom customer churn prediction, with model evaluation and explainability using SHAP.

## Overview

Customer churn is an important challenge in the telecommunications industry. Predicting which customers may leave can help organizations understand churn patterns and make better-informed retention decisions.

This project implements a centralized Logistic Regression baseline and a simulated federated learning approach. It evaluates both approaches using classification metrics and uses SHAP to explain the centralized model's predictions.

The project is developed as an academic capstone prototype using the publicly available IBM Telco Customer Churn dataset.

## Objectives

- Build a machine learning model to predict customer churn.
- Establish a centralized learning baseline.
- Implement a simulated federated learning workflow using multiple clients and parameter aggregation.
- Compare model performance using Accuracy, Precision, Recall, F1-score, and ROC-AUC.
- Analyze feature importance using permutation importance.
- Explain global feature influence and an individual prediction using SHAP.
- Document the limitations of the current implementation and identify future improvements.

## Current Implementation

| Component | Implementation |
|---|---|
| Dataset | IBM Telco Customer Churn |
| Data preprocessing | Missing-value imputation, categorical encoding, numerical scaling |
| Centralized model | Logistic Regression |
| Federated learning | Custom simulated federated training with weighted parameter averaging |
| Simulated clients | 3 |
| Federated training rounds | 12 |
| Evaluation | Accuracy, Precision, Recall, F1-score, ROC-AUC |
| Feature importance | Permutation importance |
| Explainability | SHAP for the centralized model |
| Outputs | CSV result files and PNG visualizations |

## Dataset

**Source:** [IBM Telco Customer Churn Dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

- **Records:** 7,043
- **Attributes:** 21 columns in the original CSV
- **Prediction target:** `Churn`
- **Target classes:** `Yes` and `No`

The dataset contains customer information related to services, account characteristics, billing, and churn status.

The target is converted into a binary variable for model training. The `customerID` column is excluded from the model features, and `TotalCharges` is converted to a numeric representation during preprocessing.

The dataset is publicly available. This project does not establish that the records represent current, live telecom customers.

## Methodology

### 1. Data preprocessing

The implementation prepares the data for machine learning by:

- Cleaning the churn target.
- Converting `TotalCharges` to a numeric type.
- Separating the target variable from the input features.
- Creating a stratified 80:20 training and testing split.
- Imputing missing numerical and categorical values.
- Scaling numerical features and one-hot encoding categorical features.

The current implementation fits preprocessing transformations on the training split before simulating the federated clients. This is a limitation to address in a more realistic federated setup.

### 2. Centralized learning

A Logistic Regression classifier is trained on the centralized training data. Class weighting is used to account for the imbalance between churn and non-churn customers.

This model serves as the baseline for evaluating the simulated federated approach.

### 3. Simulated federated learning

The project implements a custom federated-style training workflow:

1. The training data is divided among three simulated clients.
2. Each client performs local model updates.
3. Client model parameters are aggregated using sample-count-weighted averaging.
4. The process repeats over 12 communication rounds.
5. The resulting model is evaluated on the held-out test set.

The clients are simulated on one computer. The implementation does not currently demonstrate communication between independent telecom organizations or provide formal privacy guarantees.

### 4. Model evaluation

The centralized and federated models are evaluated using:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted churn cases that are actual churn cases.
- **Recall:** Proportion of actual churn cases detected by the model.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Ability to distinguish between the two classes across classification thresholds.

### 5. Explainable AI

**Permutation importance** measures how model performance changes when individual features are shuffled.

**SHAP (SHapley Additive exPlanations)** is used to analyze the centralized model's predictions through:

- Global feature importance based on mean absolute SHAP values.
- Individual prediction explanations showing each feature's contribution to the model output.

SHAP explanations are currently generated for the centralized model only. Feature contributions describe model behavior and should not be interpreted as proof of causal relationships.

## Experimental Results

The following results were obtained from the current implementation on the held-out test split.

| Metric | Centralized Logistic Regression | Simulated Federated Model |
|---|---:|---:|
| Accuracy | 0.7381 | 0.7786 |
| Precision | 0.5043 | 0.7422 |
| Recall | 0.7834 | 0.2540 |
| F1-score | 0.6136 | 0.3785 |
| ROC-AUC | 0.8413 | 0.8363 |

### Interpretation

The simulated federated model achieved higher accuracy and precision in this experiment. However, its recall and F1-score were substantially lower than those of the centralized baseline.

This means the federated model missed a larger proportion of actual churn cases at the default classification threshold. Consequently, the results do **not** demonstrate that federated learning is universally better than centralized learning.

The difference may be affected by the models' training procedures, optimization settings, class weighting, and the single train-test split used. Further controlled experiments are needed before drawing broader conclusions.

### SHAP and feature importance

The current global SHAP analysis identifies the following among the most influential features for the centralized model:

1. `tenure`
2. `MonthlyCharges`
3. `TotalCharges`
4. `InternetService_Fiber optic`
5. `Contract_Month-to-month`

These rankings reflect the trained model's behavior on the analyzed data. They do not establish that these features independently cause customer churn.

## Project Structure

```text
explainable-federated-learning-telecom-churn/
│
├── train.py
├── train_with_shap.py
├── requirements.txt
├── README.md
│
├── data/
│   └── Telco-Customer-Churn.csv
│
└── outputs/
    ├── model_comparison.csv
    ├── confusion_matrices.png
    ├── feature_importance.csv
    ├── feature_importance.png
    ├── shap_global_importance.csv
    ├── shap_global_importance.png
    ├── shap_individual_explanation.csv
    ├── shap_individual_explanation.png
    └── example_prediction.csv
```

This is the intended project structure. Keep only files that actually exist in your repository, and update the list if your local filenames differ.

## Installation and Execution

### Prerequisites

- Python 3.10 or a compatible Python version.
- pip.
- The project source files and dataset.

### Step 1: Clone the repository

Replace the URL below with your actual GitHub repository URL.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd explainable-federated-learning-telecom-churn
```

### Step 2: Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

If `requirements.txt` is present and contains the required packages:

```bash
python -m pip install -r requirements.txt
```

### Step 4: Prepare the dataset

Place the downloaded CSV at:

```text
data/Telco-Customer-Churn.csv
```

Ensure the dataset path matches the path configured in the Python script.

### Step 5: Run the baseline experiment

```bash
python train.py
```

### Step 6: Run the explainability experiment

```bash
python train_with_shap.py
```

The explainability script requires SHAP and the other dependencies used by the project. The scripts generate evaluation results and visualizations in the configured output directory.

Run the scripts from the project root directory. Exact dependency versions should be documented in `requirements.txt` after checking the environment used for the successful run.

## Limitations

The current prototype has several limitations:

- **Simulated clients:** Federated training is simulated locally rather than deployed across independent organizations or devices.
- **Privacy:** Local training and parameter aggregation alone do not guarantee privacy. Secure aggregation and differential privacy have not been implemented.
- **Preprocessing:** Preprocessing is fitted before the training data is partitioned among simulated clients.
- **Model comparability:** The centralized and federated implementations use different training procedures, so their results are not a perfectly controlled comparison.
- **Evaluation:** Results are based on a single stratified train-test split and a fixed classification threshold.
- **Explainability coverage:** SHAP currently explains the centralized model, not the federated model.
- **Real-world applicability:** The experiment does not establish production readiness or demonstrate actual churn reduction from a retention campaign.

These limitations should be considered when interpreting the reported metrics.

## Future Work

Potential extensions include:

- Improving the fairness of the centralized-versus-federated comparison by aligning model objectives and evaluation settings.
- Evaluating performance across multiple random seeds and train-test splits.
- Tuning the classification threshold and analyzing precision-recall trade-offs.
- Implementing client-specific preprocessing and more realistic non-IID data partitions.
- Exploring secure aggregation and differential privacy.
- Extending explainability to the federated model where technically appropriate.
- Evaluating the framework on additional datasets and investigating its usefulness for retention decisions.

These are proposed extensions, not claims of completed implementation.

## Technologies Used

Python, pandas, NumPy, scikit-learn, SHAP, Matplotlib, and CSV-based experiment outputs.

## Academic Project

**Project title:** Explainable Federated Learning Framework for Predicting Customer Churn in Telecom Networks

**Project type:** B.Tech Computer Science Engineering capstone project

The repository is intended to document the implementation, experiments, results, and limitations of the academic prototype.

## Reproducibility and Integrity

The reported results correspond to the current local implementation and its recorded experimental settings. They should be reproduced by running the scripts in the documented environment before being treated as independently verified.

Only implemented and tested functionality should be described as completed. Planned enhancements are identified separately from current functionality.
