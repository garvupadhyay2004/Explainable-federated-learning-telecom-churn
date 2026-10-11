"""Optional Streamlit interface for demonstrating the capstone."""
import os
import pandas as pd
import streamlit as st
import subprocess
import sys

st.set_page_config(page_title="Explainable Federated Churn", layout="wide")
st.title("Explainable Federated Learning for Telecom Churn")
st.caption("Capstone prototype — simulated clients; not a production privacy guarantee.")

st.markdown("""
This interface launches the experiment script. Run the experiment once before the mentor demo
so the outputs are already available. The first run may need internet access to download the dataset.
""")

if st.button("Run experiment", type="primary"):
    with st.spinner("Training centralized and federated models; this may take a minute..."):
        result = subprocess.run([sys.executable, "train.py"], capture_output=True, text=True)
    st.code(result.stdout or "(No standard output)")
    if result.returncode != 0:
        st.error("The experiment failed. Check the error below.")
        st.code(result.stderr)
    else:
        st.success("Experiment completed.")

if os.path.exists("outputs/model_comparison.csv"):
    st.subheader("Centralized vs Federated results")
    st.dataframe(pd.read_csv("outputs/model_comparison.csv"), use_container_width=True)
if os.path.exists("outputs/confusion_matrices.png"):
    st.subheader("Confusion matrices")
    st.image("outputs/confusion_matrices.png", use_container_width=True)
if os.path.exists("outputs/feature_importance.png"):
    st.subheader("Explainability: permutation feature importance")
    st.image("outputs/feature_importance.png", use_container_width=True)
if os.path.exists("outputs/example_prediction.csv"):
    st.subheader("Example prediction")
    st.dataframe(pd.read_csv("outputs/example_prediction.csv"), use_container_width=True)
