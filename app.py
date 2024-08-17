# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from feature_engineering import feature_engineering  # Import the function

# Load the model
model = joblib.load('model.pkl')

# Define the feature names expected by the model
expected_features = ['age', 'gender', 'hypertension', 'heart_disease', 'ever_married', 
                     'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 
                     'smoking_status', 'bmi_category', 'glucose_category', 
                     'age_bmi_interaction', 'married_work_interaction', 
                     'high_risk_indicators', 'age_group', 'risk_factor', 
                     'glucose_bmi_ratio', 'married_working', 'urban_smoker', 
                     'age_work_interaction', 'smoking_residence_interaction', 
                     'age_hypertension_interaction', 'age_heart_disease_interaction', 
                     'glucose_hypertension_interaction', 'glucose_heart_disease_interaction', 
                     'bmi_hypertension_interaction', 'bmi_heart_disease_interaction', 
                     'age_group_glucose_bmi_ratio', 'glucose_age_group_interaction', 
                     'age_glucose_interaction', 'bmi_glucose_interaction', 
                     'hypertension_glucose_category_interaction', 
                     'heart_disease_glucose_category_interaction', 
                     'age_group_glucose_category_interaction', 
                     'bmi_category_glucose_category_interaction', 
                     'age_group_bmi_category_interaction', 
                     'age_high_risk_interaction', 
                     'glucose_bmi_ratio_risk_factor_interaction', 
                     'work_type_risk_factor_interaction']

# Function to center content
def center_content():
    st.markdown("""
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    <style>
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }
    .container {
        width: 80%;
        margin: auto;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="container">', unsafe_allow_html=True)

def end_center_content():
    st.markdown('</div>', unsafe_allow_html=True)

# Center content
center_content()

# App title
st.markdown('<h1 class="text-center mb-4">Stroke Prediction App</h1>', unsafe_allow_html=True)

# Bootstrap grid layout for input fields
st.markdown("""
<div class="row">
    <div class="col-md-6">
        <label for="age">Age</label>
        <input type="number" class="form-control" id="age" value="30">
    </div>
    <div class="col-md-6">
        <label for="gender">Gender</label>
        <select class="form-control" id="gender">
            <option>Male</option>
            <option>Female</option>
        </select>
    </div>
</div>
<div class="row mt-3">
    <div class="col-md-6">
        <label for="hypertension">Hypertension</label>
        <select class="form-control" id="hypertension">
            <option>0</option>
            <option>1</option>
        </select>
    </div>
    <div class="col-md-6">
        <label for="heart_disease">Heart Disease</label>
        <select class="form-control" id="heart_disease">
            <option>0</option>
            <option>1</option>
        </select>
    </div>
</div>
<div class="row mt-3">
    <div class="col-md-6">
        <label for="ever_married">Ever Married</label>
        <select class="form-control" id="ever_married">
            <option>No</option>
            <option>Yes</option>
        </select>
    </div>
    <div class="col-md-6">
        <label for="work_type">Work Type</label>
        <select class="form-control" id="work_type">
            <option>Govt_job</option>
            <option>Never_worked</option>
            <option>Private</option>
            <option>Self_employed</option>
        </select>
    </div>
</div>
<div class="row mt-3">
    <div class="col-md-6">
        <label for="Residence_type">Residence Type</label>
        <select class="form-control" id="Residence_type">
            <option>Rural</option>
            <option>Urban</option>
        </select>
    </div>
    <div class="col-md-6">
        <label for="avg_glucose_level">Average Glucose Level</label>
        <input type="number" class="form-control" id="avg_glucose_level" value="100.0">
    </div>
</div>
<div class="row mt-3">
    <div class="col-md-6">
        <label for="bmi">BMI</label>
        <input type="number" class="form-control" id="bmi" value="25.0">
    </div>
    <div class="col-md-6">
        <label for="smoking_status">Smoking Status</label>
        <select class="form-control" id="smoking_status">
            <option>Unknown</option>
            <option>formerly smoked</option>
            <option>never smoked</option>
            <option>smokes</option>
        </select>
    </div>
</div>
""", unsafe_allow_html=True)

# Convert input data
age = st.number_input('Age', min_value=0, max_value=120, value=30)
gender = st.selectbox('Gender', ['Male', 'Female'])
hypertension = st.selectbox('Hypertension', [0, 1])
heart_disease = st.selectbox('Heart Disease', [0, 1])
ever_married = st.selectbox('Ever Married', ['No', 'Yes'])
work_type = st.selectbox('Work Type', ['Govt_job', 'Never_worked', 'Private', 'Self_employed'])
Residence_type = st.selectbox('Residence Type', ['Rural', 'Urban'])
avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0, value=100.0)
bmi = st.number_input('BMI', min_value=0.0, value=25.0)
smoking_status = st.selectbox('Smoking Status', ['Unknown', 'formerly smoked', 'never smoked', 'smokes'])

# Prepare input data for prediction
input_data = feature_engineering({
    'age': age,
    'gender': 0 if gender == 'Male' else 1,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'ever_married': 0 if ever_married == 'No' else 1,
    'work_type': ['Govt_job', 'Never_worked', 'Private', 'Self_employed'].index(work_type),
    'Residence_type': 0 if Residence_type == 'Rural' else 1,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'smoking_status': ['Unknown', 'formerly smoked', 'never smoked', 'smokes'].index(smoking_status)
})

# Prediction
if st.button('Predict'):
    try:
        prediction = model.predict(input_data)
        st.markdown(f"""
        <div class="alert alert-primary" role="alert">
            <h4 class="alert-heading">Prediction Result</h4>
            <p class="mb-0">The prediction is: <strong>{'Stroke' if prediction[0] == 1 else 'No Stroke'}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error making prediction: {e}")

# End centered content
end_center_content()
