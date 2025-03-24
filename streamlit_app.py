import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the pre-trained model
@st.cache_resource
def load_model():
    model = joblib.load("heart_disease_model.pkl")
    return model

model = load_model()

# Title and description
st.title("Heart Disease Prediction App")
st.write("""
This app predicts the likelihood of heart disease based on patient data.
Enter the patient details below to get a prediction.
""")

# Create input fields for all features
st.header("Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])

with col2:
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=200, value=120)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

with col3:
    restecg = st.selectbox("Resting ECG", options=[0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate", min_value=50, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

col4, col5 = st.columns(2)

with col4:
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2])

with col5:
    ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3])

# Prediction function
def predict_heart_disease(features):
    feature_array = np.array(features).reshape(1, -1)
    prediction = model.predict(feature_array)
    probability = model.predict_proba(feature_array)[0]
    return prediction[0], probability

# Predict button
if st.button("Predict"):
    # Collect all features in the correct order
    features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    
    # Make prediction
    prediction, probability = predict_heart_disease(features)
    
    # Display results
    st.subheader("Prediction Results")
    if prediction == 1:
        st.error(f"High Risk of Heart Disease (Probability: {probability[1]:.2%})")
    else:
        st.success(f"Low Risk of Heart Disease (Probability of no disease: {probability[0]:.2%})")
    
    # Display probability bars
    st.write("Prediction Confidence:")
    col_prob1, col_prob2 = st.columns(2)
    with col_prob1:
        st.write("No Heart Disease")
        st.progress(probability[0])
    with col_prob2:
        st.write("Heart Disease")
        st.progress(probability[1])

# Add some footer information
st.write("---")
st.write("Note: This is a predictive model and should not be used as a substitute for professional medical advice.")
