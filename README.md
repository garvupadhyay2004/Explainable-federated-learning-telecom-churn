# Explainable Federated Learning Framework for Predicting Customer Churn in Telecom Networks

An academic machine learning project that compares centralized learning with simulated federated learning for telecom customer churn prediction, with model evaluation, explainable AI (SHAP), and classification-threshold analysis.

## Overview

Customer churn prediction helps telecommunications companies identify customers who may discontinue their services. This project explores machine learning approaches for predicting churn while investigating model interpretability and the trade-offs involved in detecting potential churners.

The framework implements a centralized Logistic Regression baseline and a custom simulated Federated Averaging (FedAvg) workflow. It evaluates both approaches using standard classification metrics and uses SHAP to explain the centralized model's predictions.

A separate threshold-validation experiment investigates how changing classification thresholds affects precision, recall, F1-score, and accuracy.

## Objectives

- Develop a machine learning model for customer churn prediction.
- Establish a centralized Logistic Regression baseline.
- Implement a simulated federated learning workflow with multiple clients.
- Compare model performance using standard classification metrics.
- Analyze feature importance using permutation importance.
- Explain global feature influence and individual predictions using SHAP.
- Evaluate validation-selected classification thresholds and their impact on churn detection.
- Document implementation limitations and potential future improvements.

## Current Implementation

| Component | Implementation |
|---|---|
| Dataset | IBM Telco Customer Churn |
| Data preprocessing | Missing-value imputation, categorical encoding, numerical scaling |
| Centralized model | Logistic Regression |
| Federated learning | Custom simulated FedAvg training |
| Simulated clients | 3 |
| Federated training rounds | 12 |
| Evaluation metrics | Accuracy, Precision, Recall, F1-score, ROC-AUC |
| Feature importance | Permutation importance |
| Explainability | Global and individual SHAP explanations for the centralized model |
| Threshold experiment | Validation-based threshold selection for centralized and federated models |
| Dashboard | Streamlit application |
| Outputs | CSV results and PNG visualizations |

## Dataset

**Source:** [IBM Telco Customer Churn Dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

- **Records:** 7,043
- **Original columns:** 21
- **Target variable:** `Churn`
- **Target classes:** `Yes` and `No`

The dataset contains customer information related to services, account characteristics, billing, and churn status.

The `customerID` column is excluded from model features, and `TotalCharges` is converted to a numeric representation during preprocessing.

This is a publicly available dataset used for experimentation. This project does not establish that the records represent current, live telecom customers.

The dataset file is maintained in the repository root:

```text
Telco-Customer-Churn.csv
```

## Methodology

### 1. Data preprocessing

The workflow includes:

- Cleaning the churn target.
- Converting `TotalCharges` to numeric format.
- Separating target and input features.
- Imputing missing numerical and categorical values.
- Scaling numerical features.
- One-hot encoding categorical features.

The original baseline experiment uses a stratified 80:20 training-test split.

The separate threshold experiment uses a stratified training-validation-test split. Its preprocessing transformations are fitted on the training partition before evaluation.

### 2. Centralized learning

A Logistic Regression classifier is trained using centralized training data. Class weighting is used to account for the imbalance between churn and non-churn customers.

This model serves as the baseline for comparison with the simulated federated approach.

### 3. Simulated federated learning

The custom federated workflow follows these steps:

1. Divide the training data among three simulated clients.
2. Perform local model updates at each client.
3. Aggregate client parameters using sample-count-weighted averaging.
4. Repeat the aggregation process over 12 communication rounds.
5. Evaluate the resulting model on held-out test data.

The clients are simulated on a single computer. The implementation does not demonstrate communication between independent telecom organizations or devices.

### 4. Model evaluation

The models are evaluated using:

- **Accuracy:** Proportion of all predictions that are correct.
- **Precision:** Proportion of predicted churn cases that are actual churn cases.
- **Recall:** Proportion of actual churn cases detected.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Ability to rank churn cases above non-churn cases across classification thresholds.

### 5. Explainable AI

**Permutation importance** evaluates how model performance changes when individual features are shuffled.

**SHAP (SHapley Additive exPlanations)** is used to explain the centralized model through:

- Global feature importance based on mean absolute SHAP values.
- Individual prediction explanations showing feature contributions to model output.

SHAP explanations are currently generated for the centralized model only. Feature contributions describe model behavior and do not establish causal relationships.

### 6. Classification-threshold experiment

A separate experiment evaluates the effect of selecting classification thresholds using validation data.

The dataset was divided as follows:

| Partition | Records |
|---|---:|
| Training | 4,507 |
| Validation | 1,127 |
| Final test | 1,409 |
| **Total** | **7,043** |

The threshold for each model was selected using the validation set to maximize F1-score among the tested thresholds. The selected thresholds were then evaluated on the final test set.

Selected thresholds:

- **Centralized Logistic Regression:** 0.62
- **Simulated federated model:** 0.33

The threshold-selection process uses validation data rather than selecting thresholds from final test results.

## Experimental Results

### Experiment 1: Original baseline

The following metrics were obtained from the original held-out test split.

| Metric | Centralized Logistic Regression | Simulated Federated Model |
|---|---:|---:|
| Accuracy | 0.7381 | 0.7786 |
| Precision | 0.5043 | 0.7422 |
| Recall | 0.7834 | 0.2540 |
| F1-score | 0.6136 | 0.3785 |
| ROC-AUC | 0.8413 | 0.8363 |

#### Interpretation

The simulated federated model achieved higher accuracy and precision in this experiment. However, its recall and F1-score were lower than those of the centralized baseline.

At the default classification threshold, the federated model missed a larger proportion of actual churn cases. The results therefore do not demonstrate that federated learning is universally better than centralized learning.

### Experiment 2: Threshold-validation results

The following results were obtained from the final test partition of the separate threshold experiment.

| Model | Threshold | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Centralized | 0.50 | 0.7417 | 0.5086 | 0.7888 | 0.6184 | 0.8421 |
| Centralized | 0.62 | 0.7708 | 0.5544 | 0.6952 | 0.6168 | 0.8421 |
| Simulated Federated | 0.50 | 0.7771 | 0.7344 | 0.2513 | 0.3745 | 0.8362 |
| Simulated Federated | 0.33 | 0.7523 | 0.5234 | 0.7487 | 0.6161 | 0.8362 |

#### Interpretation

For the simulated federated model, reducing the threshold from 0.50 to 0.33 increased:

- **Recall:** 25.13% to 74.87%.
- **F1-score:** 37.45% to 61.61%.

Precision decreased from 73.44% to 52.34%, while accuracy decreased from 77.71% to 75.23%.

For the centralized model, the validation-selected threshold of 0.62 increased accuracy compared with the default threshold, while F1-score decreased slightly.

ROC-AUC remained unchanged within each model because changing the classification threshold does not change the underlying prediction scores or their ranking.

These results illustrate the trade-off between detecting more churn cases and generating additional false-positive predictions. They are specific to this experiment and do not establish universal superiority of either approach.

**Important:** The original baseline and threshold-validation results come from different train-test splits and should not be compared as though they were one experiment.

## SHAP Findings

The global SHAP analysis identified the following among the most influential features for the centralized model:

1. `tenure`
2. `MonthlyCharges`
3. `TotalCharges`
4. `InternetService_Fiber optic`
5. `Contract_Month-to-month`

These rankings reflect the behavior of the trained centralized model on the analyzed data. They do not prove that these features independently cause churn.

## Project Structure

The repository contains the following main files and result artifacts:

```text
Explainable-federated-learning-telecom-churn/
├── app.py
├── train.py
├── train_with_shap.py
├── threshold_validation_experiment.py
├── requirements.txt
├── README.md
├── .gitignore
├── Telco-Customer-Churn.csv
├── Capstone_Phase_II_Progress_Report.docx
├── Capstone_Report(till phase 3 ).pdf
└── Output/
    ├── central_validation_thresholds.csv
    ├── fedavg_validation_thresholds.csv
    ├── test_threshold_comparison.csv
    ├── test_confusion_matrices.csv
    ├── model_comparison.csv
    ├── confusion_matrices.png
    ├── example_prediction.csv
    ├── feature_importance.csv
    ├── feature_importance.png
    ├── shap_global_importance.csv
    ├── shap_global_importance.png
    ├── shap_individual_explanation.csv
    ├── shap_individual_explanation.png
    └── ...
```

The structure reflects the files shown in the repository screenshots. The `Output/` folder contains generated results and visualizations. Exact filenames and paths should be checked against the current repository and source code.

## Installation and Execution

### Prerequisites

- Python 3.10 or a compatible Python version.
- pip.
- The project source files and dataset.

### 1. Clone the repository

Replace the placeholder with the actual GitHub repository URL.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Explainable-federated-learning-telecom-churn
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the dataset

The dataset must remain in the repository root:

```text
Telco-Customer-Churn.csv
```

Ensure that `train.py`, `train_with_shap.py`, and `threshold_validation_experiment.py` all load the dataset from this location. No separate `data/` directory is required.

### 5. Run the baseline experiment

```bash
python train.py
```

### 6. Run the explainability experiment

```bash
python train_with_shap.py
```

### 7. Run the threshold-validation experiment

```bash
python threshold_validation_experiment.py
```

The threshold experiment generates CSV files containing validation-selected thresholds, final test comparisons, and confusion-matrix data in its configured output directory.

### 8. Launch the Streamlit dashboard

```bash
python -m streamlit run app.py
```

The dashboard displays the features and results implemented in the current version of `app.py`. Threshold-experiment results should only be considered part of the dashboard if the application explicitly loads and displays them.

Run the commands from the project root. Check `requirements.txt` for the actual dependencies and versions used in the working environment.

## Limitations

- **Simulated clients:** Federated training is simulated locally rather than deployed across independent organizations or devices.
- **Privacy:** Secure aggregation and differential privacy have not been implemented. Local training and parameter aggregation alone do not guarantee privacy.
- **Preprocessing:** The original baseline fits preprocessing transformations before dividing the training data among simulated clients. The threshold experiment fits its transformations on the training partition before evaluation.
- **Model comparability:** The centralized and federated implementations use different training procedures, so the comparison is not perfectly controlled.
- **Evaluation robustness:** Repeated-seed experiments have not been established.
- **Threshold generalization:** Validation-selected thresholds may not generalize to other datasets or deployment populations.
- **Explainability coverage:** SHAP explanations are currently limited to the centralized model.
- **Real-world applicability:** The prototype does not establish production readiness or demonstrate actual churn reduction from a retention campaign.

These limitations should be considered when interpreting the results.

## Future Work

Potential extensions include:

- Improving the fairness of the centralized-versus-federated comparison by aligning training objectives and evaluation settings.
- Evaluating performance across multiple random seeds and data splits.
- Testing additional threshold-selection strategies and precision-recall trade-offs.
- Implementing client-specific preprocessing and more realistic non-IID data partitions.
- Exploring secure aggregation and differential privacy.
- Extending explainability to the federated model where technically appropriate.
- Evaluating the framework on additional datasets.
- Investigating how model predictions could support customer-retention decisions.

These are proposed extensions, not claims of completed implementation.

## Technologies Used

Python, pandas, NumPy, scikit-learn, SHAP, Matplotlib, Streamlit, and CSV-based experiment outputs.

## Academic Project

**Project title:** Explainable Federated Learning Framework for Predicting Customer Churn in Telecom Networks

**Degree:** B.Tech Computer Science Engineering

**Project type:** Academic capstone project

This repository documents the implementation, experimental results, explainability analysis, and limitations of the project prototype.
