import streamlit as st
import joblib
import numpy as np

# Load saved model and scaler
scaler = joblib.load('scaler.pkl')
model = joblib.load('student_rf_model.pkl')

st.title("🎓 Student Performance Predictor")
st.write("Enter academic metrics below to predict final marks.")

col1, col2 = st.columns(2)
with col1:
    study_hours = st.number_input("Study Hours (per week)", min_value=0.0, max_value=20.0, value=5.0)
    attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=80.0)
with col2:
    prev_score = st.number_input("Previous Score (0-100)", min_value=0.0, max_value=100.0, value=75.0)
    assignments = st.number_input("Assignments Score (0-100)", min_value=0.0, max_value=100.0, value=85.0)

if st.button("Predict Final Marks"):
    input_data = np.array([[study_hours, attendance, prev_score, assignments]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    st.success(f"Predicted Final Marks: {prediction:.2f}/100")
