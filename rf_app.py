# rf_app.py

import streamlit as st
import pandas as pd
import joblib
import yaml

# Very First Command
st.set_page_config(page_title="VisionBudget Random Forest", layout="centered")

# Load config
@st.cache_resource
def load_config():
    with open("rf_config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

# Load model and encoder
@st.cache_resource
def load_model():
    bundle = joblib.load(config['model_bundle_path'])
    return bundle['model'], bundle['encoder']

model, encoder = load_model()

# App title
st.title("🌳 VisionBudget: Random Forest Execution Status Predictor")

# Select input mode
mode = st.radio("Choose Input Mode", ["Upload Full CSV", "Manual Entry (Single Prediction)"])

# Define features
features = [
    'Approved Budget (UGX, Millions)',
    'Released Budget (UGX)',
    'Actual Expenditure (UGX)',
    'Approved Budget (% of GDP)',
    'Actual Expenditure (% of GDP)',
    'Performance (%)',
    'Deviation (UGX)',
    'Nominal GDP (UGX Trillions)'
]

# Input and predictions
if mode == "Upload Full CSV":
    uploaded_file = st.file_uploader("📂 Upload your budget data CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if not all(col in df.columns for col in features):
            st.error("❌ Uploaded file is missing one or more required columns.")
        else:
            X = df[features].fillna(df[features].median())
            y_pred = model.predict(X)
            df['Predicted Execution Status'] = encoder.inverse_transform(y_pred)
            st.success("✅ Predictions completed.")
            st.dataframe(df[['Vote', 'Sector', 'Financial Year', 'Predicted Execution Status']].head(10))

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Full Prediction CSV", csv, file_name="rf_prediction_results.csv", mime='text/csv')

elif mode == "Manual Entry (Single Prediction)":
    st.markdown("Enter feature values below to predict execution status for a single budget line:")

    input_values = {}
    for feature in features:
        input_values[feature] = st.number_input(feature, value=0.0)

    if st.button("🔎 Predict"):
        X_input = pd.DataFrame([input_values])
        prediction = model.predict(X_input)
        result = encoder.inverse_transform(prediction)[0]
        st.success(f"📈 Predicted Execution Status: **{result}**")
