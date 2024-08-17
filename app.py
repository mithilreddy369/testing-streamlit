import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the model with error handling
try:
    model = joblib.load('model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please upload the model file.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load the scaler if you used one during training
try:
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    scaler = None
except Exception as e:
    st.error(f"Error loading scaler: {e}")
    st.stop()

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
st.sidebar.header('User Input')

age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=30)
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
hypertension = st.sidebar.selectbox('Hypertension', ['No', 'Yes'])
heart_disease = st.sidebar.selectbox('Heart Disease', ['No', 'Yes'])
ever_married = st.sidebar.selectbox('Ever Married', ['No', 'Yes'])
work_type = st.sidebar.selectbox('Work Type', ['Govt_job', 'Never_worked', 'Private', 'Self_employed'])
residence_type = st.sidebar.selectbox('Residence Type', ['Rural', 'Urban'])
avg_glucose_level = st.sidebar.number_input('Average Glucose Level', min_value=0.0, value=70.0)
bmi = st.sidebar.number_input('BMI', min_value=0.0, value=22.0)
smoking_status = st.sidebar.selectbox('Smoking Status', ['Unknown', 'formerly smoked', 'never smoked', 'smokes'])

# Convert user input to DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'gender': [1 if gender == 'Female' else 0],
    'hypertension': [1 if hypertension == 'Yes' else 0],
    'heart_disease': [1 if heart_disease == 'Yes' else 0],
    'ever_married': [1 if ever_married == 'Yes' else 0],
    'work_type': [['Govt_job', 'Never_worked', 'Private', 'Self_employed'].index(work_type)],
    'Residence_type': [['Rural', 'Urban'].index(residence_type)],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'smoking_status': [['Unknown', 'formerly smoked', 'never smoked', 'smokes'].index(smoking_status)]
})

# Feature engineering
input_data['bmi_category'] = input_data['bmi'].apply(bmi_category)
input_data['glucose_category'] = input_data['avg_glucose_level'].apply(glucose_category)
input_data['age_bmi_interaction'] = input_data['age'] * input_data['bmi']
input_data['married_work_interaction'] = input_data['ever_married'] * input_data['work_type']
input_data['high_risk_indicators'] = (
    (input_data['hypertension'] == 1) |
    (input_data['heart_disease'] == 1) |
    (input_data['glucose_category'] == 3) |
    (input_data['bmi_category'] == 3)
).astype(int)
input_data['age_group'] = input_data['age'].apply(age_group)
input_data['risk_factor'] = input_data['hypertension'] + input_data['heart_disease'] + input_data['age_group']
input_data['glucose_bmi_ratio'] = input_data['avg_glucose_level'] / input_data['bmi']
input_data['married_working'] = input_data['ever_married'] * (input_data['work_type'] != 1).astype(int)
input_data['urban_smoker'] = input_data['Residence_type'] * (input_data['smoking_status'] == 3).astype(int)
input_data['age_work_interaction'] = input_data['age_group'] * input_data['work_type']
input_data['smoking_residence_interaction'] = input_data['smoking_status'] * input_data['Residence_type']
input_data['age_hypertension_interaction'] = input_data['age'] * input_data['hypertension']
input_data['age_heart_disease_interaction'] = input_data['age'] * input_data['heart_disease']
input_data['glucose_hypertension_interaction'] = input_data['avg_glucose_level'] * input_data['hypertension']
input_data['glucose_heart_disease_interaction'] = input_data['avg_glucose_level'] * input_data['heart_disease']
input_data['bmi_hypertension_interaction'] = input_data['bmi'] * input_data['hypertension']
input_data['bmi_heart_disease_interaction'] = input_data['bmi'] * input_data['heart_disease']
input_data['age_group_glucose_bmi_ratio'] = input_data['age_group'] * input_data['glucose_bmi_ratio']
input_data['glucose_age_group_interaction'] = input_data['avg_glucose_level'] * input_data['age_group']
input_data['age_glucose_interaction'] = input_data['age'] * input_data['avg_glucose_level']
input_data['bmi_glucose_interaction'] = input_data['bmi'] * input_data['avg_glucose_level']
input_data['hypertension_glucose_category_interaction'] = input_data['hypertension'] * input_data['glucose_category']
input_data['heart_disease_glucose_category_interaction'] = input_data['heart_disease'] * input_data['glucose_category']
input_data['age_group_glucose_category_interaction'] = input_data['age_group'] * input_data['glucose_category']
input_data['bmi_category_glucose_category_interaction'] = input_data['bmi_category'] * input_data['glucose_category']
input_data['age_group_bmi_category_interaction'] = input_data['age_group'] * input_data['bmi_category']
input_data['age_high_risk_interaction'] = input_data['age'] * input_data['high_risk_indicators']
input_data['glucose_bmi_ratio_risk_factor_interaction'] = input_data['glucose_bmi_ratio'] * input_data['risk_factor']
input_data['work_type_risk_factor_interaction'] = input_data['work_type'] * input_data['risk_factor']

# Scale features if scaler is available
if scaler:
    scaled_features = scaler.transform(input_data)
else:
    scaled_features = input_data

# Make prediction
try:
    prediction = model.predict(scaled_features)
except Exception as e:
    st.error(f"Error making prediction: {e}")
    st.stop()

# Display result
st.subheader('Prediction:')
st.write('Stroke Risk: {}'.format('Yes' if prediction[0] == 1 else 'No'))
