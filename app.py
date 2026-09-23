import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time


# Set up the web page title and description
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Prediction System")
st.write("""
This tool uses a Support Vector Machine (SVM) model to predict the likelihood of heart disease based on clinical parameters. 
Please enter the patient's details below.
""")

# --- 1. Load the model and preprocessor with error handling ---
@st.cache_resource # Cache the model so it doesn't reload on every interaction
def load_models():
    try:
        model = joblib.load('svm_heart_model.pkl')
        preprocessor = joblib.load('heart_preprocessor.pkl')
        return model, preprocessor
    except FileNotFoundError:
        st.error("⚠️ Error: Model files not found! Please ensure 'svm_heart_model.pkl' and 'heart_preprocessor.pkl' are in the same folder as this app.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ An unexpected error occurred while loading the model: {e}")
        st.stop()

model, preprocessor = load_models()

# --- 2. Define the exact column order expected by the preprocessor ---
# This MUST match the order of X_train from Step 3
expected_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# --- 3. Create the UI Input Fields ---
st.subheader("Patient Clinical Parameters")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50, step=1)
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120, step=1)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, step=1)
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150, step=1)
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

with col2:
    sex = st.selectbox("Sex", options=[(1, "Male"), (0, "Female")], format_func=lambda x: x[1])[0]
    cp = st.selectbox("Chest Pain Type", options=[(1, "Typical Angina"), (2, "Atypical Angina"), (3, "Non-anginal Pain"), (4, "Asymptomatic")], format_func=lambda x: x[1])[0]
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])[0]
    restecg = st.selectbox("Resting ECG", options=[(0, "Normal"), (1, "ST-T Wave Abnormality"), (2, "Left Ventricular Hypertrophy")], format_func=lambda x: x[1])[0]

with col3:
    exang = st.selectbox("Exercise Induced Angina", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])[0]
    slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[(1, "Upsloping"), (2, "Flat"), (3, "Downsloping")], format_func=lambda x: x[1])[0]
    ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", options=["normal", "fixed", "reversible"])

# --- 4. Prediction Logic ---
if st.button("🔍 Predict Heart Disease Risk", use_container_width=True):
    with st.spinner("Analyzing patient data..."):
        time.sleep(1) # Artificial delay to show the spinner
        
        # Create a dictionary with the input data
        input_data = {
            'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps],
            'chol': [chol], 'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach],
            'exang': [exang], 'oldpeak': [oldpeak], 'slope': [slope], 
            'ca': [ca], 'thal': [thal]
        }
        
        # Convert to DataFrame and ENFORCE the expected column order
        input_df = pd.DataFrame(input_data)[expected_columns]
        
        try:
            # Preprocess the input using our saved pipeline
            input_processed = preprocessor.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_processed)[0]
            probability = model.predict_proba(input_processed)[0][1]
            
            # Display the result
            st.divider()
            if prediction == 1:
                st.error(f"⚠️ **High Risk Detected:** The model predicts the presence of heart disease with a probability of **{probability*100:.2f}%**.")
                st.write("👉 **Recommendation:** Please consult a cardiologist for further diagnostic tests.")
            else:
                st.success(f"✅ **Low Risk:** The model predicts NO heart disease with a confidence of **{(1-probability)*100:.2f}%**.")
                st.write("👉 **Recommendation:** Maintain a healthy lifestyle and regular checkups.")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

st.divider()
st.caption("⚠️ **Disclaimer:** This is a machine learning prototype for educational and portfolio purposes. It is NOT a substitute for professional medical diagnosis.")