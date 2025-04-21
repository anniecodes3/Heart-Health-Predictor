import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from streamlit_shap import st_shap  


model = joblib.load("heart_model.joblib")


st.set_page_config(page_title="Heart Health Predictor", page_icon="🫀", layout="centered")

st.title("🫀 Heart Health Predictor")
st.markdown("This app predicts the risk of heart disease based on patient data.")

st.subheader("Enter your info here")


age = st.selectbox("Age", range(20, 100))
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["typical", "atypical", "nonanginal", "asymptomatic"])
restbp = st.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Serum Cholesterol (mg/dl)", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
maxhr = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.select_slider("Oldpeak (ST depression)", options=np.round(np.arange(0.0, 6.1, 0.1), 1), value=1.0)
slope = st.selectbox("Slope of peak exercise ST segment", [0, 1, 2])
ca = st.selectbox("Number of major vessels colored by fluoroscopy", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia", ["normal", "fixed", "reversible"])


sex_map = {"Male": 1, "Female": 0}
fbs_map = {"Yes": 1, "No": 0}
exang_map = {"Yes": 1, "No": 0}


input_dict = {
    "Age": age,
    "Sex": sex_map[sex],
    "RestBP": restbp,
    "Chol": chol,
    "Fbs": fbs_map[fbs],
    "RestECG": restecg,
    "MaxHR": maxhr,
    "ExAng": exang_map[exang],
    "Oldpeak": oldpeak,
    "Slope": slope,
    "Ca": ca,
    
    f"ChestPain_{cp}": 1,
    
    f"Thal_{thal}": 1
}


all_features = model.feature_names_in_
input_df = pd.DataFrame([input_dict])
for col in all_features:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[all_features]  

st.subheader("Heart Disease Prediction with SHAP")
st.write("### Input Data")
st.dataframe(input_df)


prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

st.write("### Prediction")
if prediction == 1:
    st.error("⚠️ Heart Disease Detected")
else:
    st.success("✅ No Heart Disease")

st.write(f"**Probability of Heart Disease:** `{probability:.2f}`")


import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

explainer = shap.Explainer(model, input_df)
shap_values = explainer(input_df)

st.write("### SHAP Explanation")
st_shap(shap.plots.waterfall(shap_values[0], max_display=10), height=400)

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile

st.write("### 📥 Download PDF Report")


def generate_pdf():
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        c = canvas.Canvas(tmpfile.name, pagesize=letter)
        width, height = letter
        y = height - 50

        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y, "🩺 HEART DISEASE PREDICTION REPORT 🩺")
        y -= 40

        c.setFont("Helvetica", 12)
        inputs = [
            ("Age", age), ("Sex", sex), ("Chest Pain Type", cp), ("Resting BP", restbp),
            ("Serum Cholesterol", chol), ("Fasting Blood Sugar > 120", fbs),
            ("Resting ECG", restecg), ("Max Heart Rate", maxhr), ("Exercise Angina", exang),
            ("Oldpeak", oldpeak), ("Slope", slope), ("Major Vessels (Ca)", ca), ("Thalassemia", thal)
        ]
        for label, value in inputs:
            c.drawString(50, y, f"{label}: {value}")
            y -= 20

        y -= 10
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, f"Prediction: {'Heart Disease Detected ❌' if prediction == 1 else 'No Heart Disease ✅'}")
        y -= 20
        c.drawString(50, y, f"Probability of Heart Disease: {probability:.2f}")
        y -= 40
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(50, y, "Generated using Heart Health Predictor App ❤️")
        c.save()

        return tmpfile.name


pdf_file_path = generate_pdf()
with open(pdf_file_path, "rb") as file:
    st.download_button(
        label="📄 Download Prediction Report (PDF)",
        data=file,
        file_name="Heart_Prediction_Report.pdf",
        mime="application/pdf"
    )
