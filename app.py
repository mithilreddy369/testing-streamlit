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

# Feature engineering
def feature_engineering(data):
    def bmi_category(bmi):
        if bmi < 18.5:
            return 0
        elif bmi < 24.9:
            return 1
        elif bmi < 29.9:
            return 2
        else:
            return 3

    def glucose_category(glucose):
        if glucose < 70:
            return 0
        elif glucose < 100:
            return 1
        elif glucose < 126:
            return 2
        else:
            return 3

    data['bmi_category'] = bmi_category(data['bmi'])
    data['glucose_category'] = glucose_category(data['avg_glucose_level'])
    data['age_bmi_interaction'] = data['age'] * data['bmi']
    data['married_work_interaction'] = data['ever_married'] * data['work_type']
    data['high_risk_indicators'] = (
        (data['hypertension'] == 1) |
        (data['heart_disease'] == 1) |
        (data['glucose_category'] == 3) |
        (data['bmi_category'] == 3)
    ).astype(int)
    data['age_group'] = 0 if data['age'] < 30 else (1 if data['age'] < 60 else 2)
    data['risk_factor'] = data['hypertension'] + data['heart_disease'] + data['age_group']
    data['glucose_bmi_ratio'] = data['avg_glucose_level'] / data['bmi']
    data['married_working'] = data['ever_married'] * (data['work_type'] != 1).astype(int)
    data['urban_smoker'] = data['Residence_type'] * (data['smoking_status'] == 3).astype(int)
    data['age_work_interaction'] = data['age_group'] * data['work_type']
    data['smoking_residence_interaction'] = data['smoking_status'] * data['Residence_type']
    data['age_hypertension_interaction'] = data['age'] * data['hypertension']
    data['age_heart_disease_interaction'] = data['age'] * data['heart_disease']
    data['glucose_hypertension_interaction'] = data['avg_glucose_level'] * data['hypertension']
    data['glucose_heart_disease_interaction'] = data['avg_glucose_level'] * data['heart_disease']
    data['bmi_hypertension_interaction'] = data['bmi'] * data['hypertension']
    data['bmi_heart_disease_interaction'] = data['bmi'] * data['heart_disease']
    data['age_group_glucose_bmi_ratio'] = data['age_group'] * data['glucose_bmi_ratio']
    data['glucose_age_group_interaction'] = data['avg_glucose_level'] * data['age_group']
    data['age_glucose_interaction'] = data['age'] * data['avg_glucose_level']
    data['bmi_glucose_interaction'] = data['bmi'] * data['avg_glucose_level']
    data['hypertension_glucose_category_interaction'] = data['hypertension'] * data['glucose_category']
    data['heart_disease_glucose_category_interaction'] = data['heart_disease'] * data['glucose_category']
    data['age_group_glucose_category_interaction'] = data['age_group'] * data['glucose_category']
    data['bmi_category_glucose_category_interaction'] = data['bmi_category'] * data['glucose_category']
    data['age_group_bmi_category_interaction'] = data['age_group'] * data['bmi_category']
    data['age_high_risk_interaction'] = data['age'] * data['high_risk_indicators']
    data['glucose_bmi_ratio_risk_factor_interaction'] = data['glucose_bmi_ratio'] * data['risk_factor']
    data['work_type_risk_factor_interaction'] = data['work_type'] * data['risk_factor']

    # Create DataFrame
    df = pd.DataFrame([data])
    # Reorder columns to match model
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0  # Add missing features with default value
    df = df[expected_features]
    return df

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
