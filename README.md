# Explainable Federated Learning Framework for Telecom Customer Churn

A starter implementation for the capstone project. It includes:
1. A centralized Logistic Regression baseline.
2. A simulated three-client Federated Averaging (FedAvg) logistic model.
3. Evaluation metrics and a confusion matrix.
4. Model-level permutation feature importance.
5. Optional SHAP explanations.
6. A Streamlit demonstration interface.

## Important research honesty
- The default dataset is the public IBM Telco Customer Churn dataset. You can also provide a local CSV.
- The client split is a simulation on one computer, not three real telecom organizations.
- This educational FedAvg implementation exchanges model parameters. It is not a complete privacy/security solution; model updates can leak information, and no differential privacy or secure aggregation is implemented.
- The demo preprocessing is fitted on the overall training split before partitioning it among simulated clients. For a stronger experimental study, document this design choice and later improve the protocol if required.
- Do not report metrics until you have run the code. Results depend on the dataset and settings.

## Setup in VS Code (Windows)
1. Install Python 3.10 or 3.11.
2. Open this folder in VS Code.
3. In the terminal, create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   If PowerShell blocks activation, use Command Prompt:
   ```bat
   .venv\Scripts\activate.bat
   ```

4. Run the command-line experiment:
   ```powershell
   python train.py
   ```

5. Run the optional dashboard:
   ```powershell
   streamlit run app.py
   ```

The first run downloads the public CSV if no local dataset is found. If the download fails, download the IBM Telco Customer Churn CSV manually and save it as `data/Telco-Customer-Churn.csv`.

## Expected CSV columns
The IBM dataset contains `Churn` as the target and columns such as `customerID`, `tenure`, `MonthlyCharges`, and `TotalCharges`. A local file can be passed using:
```powershell
python train.py --data path/to/your.csv
```

## What to demonstrate
1. Dataset shape and target distribution.
2. Centralized metrics: accuracy, precision, recall, F1 and ROC-AUC.
3. Three simulated clients and FedAvg rounds.
4. Centralized vs federated metric comparison.
5. Feature importance plot and one example prediction.
6. Explain the limitations honestly.

## Suggested next steps after the first run
- Save your actual results and screenshots.
- Add SHAP after the baseline works; permutation importance is included as a dependable fallback.
- Try non-IID client partitions (e.g. sort by tenure or use different churn proportions) and discuss how client heterogeneity affects results.
- Add repeated runs/seeds and a more rigorous evaluation before final submission.
