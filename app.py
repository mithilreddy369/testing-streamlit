import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the model
model = joblib.load('model.pkl')

# Load the scaler if you used one during training
scaler = joblib.load('scaler.pkl')

# Define functions for feature engineering
def bmi_category(bmi):
    if bmi < 18.5:
        return 0  # Underweight
    elif bmi < 24.9:
        return 1  # Normal weight
    elif bmi < 29.9:
        return 2  # Overweight
    else:
        return 3  # Obesity

def glucose_category(glucose):
    if glucose < 70:
        return 0  # Low
    elif glucose < 100:
        return 1  # Normal
    elif glucose < 126:
        return 2  # Pre-diabetic
    else:
        return 3  # Diabetic

def age_group(age):
    if age < 30:
        return 0  # Young
    elif age < 60:
        return 1  # Middle-aged
    else:
        return 2  # Senior

# Streamlit app interface
st.title('Stroke Prediction App')

# Collect user inputs
st.sidebar.header('User Input Parameters')

def get_user_input():
    gender = st.sidebar.selectbox('Gender', [0, 1])  # Assuming 0 for Male, 1 for Female
    age = st.sidebar.slider('Age', 0, 100, 50)
    hypertension = st.sidebar.selectbox('Hypertension', [0, 1])
    heart_disease = st.sidebar.selectbox('Heart Disease', [0, 1])
    ever_married = st.sidebar.selectbox('Ever Married', [0, 1])
    work_type = st.sidebar.selectbox('Work Type', [0, 1, 2, 3])
    residence_type = st.sidebar.selectbox('Residence Type', [0, 1])
    avg_glucose_level = st.sidebar.slider('Average Glucose Level', 0.0, 300.0, 100.0)
    bmi = st.sidebar.slider('BMI', 0.0, 50.0, 25.0)
    smoking_status = st.sidebar.selectbox('Smoking Status', [0, 1, 2, 3])

    # Create a DataFrame
    data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married],
        'work_type': [work_type],
        'Residence_type': [residence_type],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status]
    })

    # Feature engineering
    data['bmi_category'] = data['bmi'].apply(bmi_category)
    data['glucose_category'] = data['avg_glucose_level'].apply(glucose_category)
    data['age_group'] = data['age'].apply(age_group)
    data['age_bmi_interaction'] = data['age'] * data['bmi']
    data['married_work_interaction'] = data['ever_married'] * data['work_type']
    data['high_risk_indicators'] = (
        (data['hypertension'] == 1) |
        (data['heart_disease'] == 1) |
        (data['glucose_category'] == 3) |
        (data['bmi_category'] == 3)
    ).astype(int)
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
    data['age_high_risk_interaction'] = data['age'] * data['high_risk_indicators']
    data['glucose_bmi_ratio_risk_factor_interaction'] = data['glucose_bmi_ratio'] * data['risk_factor']
    data['work_type_risk_factor_interaction'] = data['work_type'] * data['risk_factor']
    
    # Return the prepared DataFrame
    return data

user_input = get_user_input()

# Display user input
st.subheader('User Input:')
st.write(user_input)

# Scale features if necessary
scaled_features = scaler.transform(user_input)

# Make prediction
prediction = model.predict(scaled_features)

# Display result
st.subheader('Prediction:')
st.write('Stroke Risk: {}'.format('Yes' if prediction[0] == 1 else 'No'))
