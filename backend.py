# backend.py
import joblib
import pandas as pd
from feature_engineering import feature_engineering

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

def prepare_input_data(age, gender, hypertension, heart_disease, ever_married, work_type,
                       Residence_type, avg_glucose_level, bmi, smoking_status):
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
    return input_data

def make_prediction(input_data):
    try:
        prediction = model.predict(input_data)
        return 'Stroke' if prediction[0] == 1 else 'No Stroke'
    except Exception as e:
        return f"Error making prediction: {e}"
