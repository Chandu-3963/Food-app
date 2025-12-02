
import streamlit as st
import pandas as pd
import joblib
import json

MODEL_FILENAME = "churn_rf_model.pkl"
FEATURES_FILENAME = "feature_columns.json"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_FILENAME)
    with open(FEATURES_FILENAME, "r") as f:
        feature_info = json.load(f)
    return model, feature_info

model, feature_info = load_artifacts()

st.title("Food Delivery Customer Churn Prediction App")

categorical_cols = feature_info["categorical_cols"]
numeric_cols = feature_info["numeric_cols"]
categories = feature_info["categories"]
input_cols = feature_info["input_columns"]

input_values = {}

with st.form("input_form"):
    for col in input_cols:
        if col in categorical_cols:
            opts = categories.get(col, [])
            if len(opts) > 0:
                val = st.selectbox(col, opts)
            else:
                val = st.text_input(col)
            input_values[col] = val
        else:
            val = st.number_input(col, value=0.0)
            input_values[col] = val
    
    submit = st.form_submit_button("Predict")

if submit:
    df_input = pd.DataFrame([input_values])
    prob = model.predict_proba(df_input)[0][1]
    pred = model.predict(df_input)[0]

    st.subheader("Prediction Results")
    st.write(f"Churn Probability: **{prob:.2f}**")
    st.write(f"Predicted Class: **{'Churned' if pred == 1 else 'Active'}**")
