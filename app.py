import streamlit as st
import pandas as pd
import numpy as np
import joblib

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
    <style>
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }
    .container {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="center">', unsafe_allow_html=True)
    st.markdown('<div class="container">', unsafe_allow_html=True)

def end_center_content():
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Center content
center_content()

# App title
st.title('Stroke Prediction App')

# User input
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
        st.write(f"Prediction: {'Stroke' if prediction[0] == 1 else 'No Stroke'}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")

# End centered content
end_center_content()
