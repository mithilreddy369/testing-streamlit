import streamlit as st
import pickle
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Load the models
with open('catboost_model1.pkl', 'rb') as file:
    catboost_model = pickle.load(file)

with open('lgb_model1.pkl', 'rb') as file:
    lgb_model = pickle.load(file)

with open('xgb_model1.pkl', 'rb') as file:
    xgb_model = pickle.load(file)

with open('gbm_model1.pkl', 'rb') as file:
    gbm_model = pickle.load(file)

# Function to predict using all models
def predict_stroke(features_array):
    predictions = {}
    models = {
        'CatBoost': catboost_model,
        'LightGBM': lgb_model,
        'XGBoost': xgb_model,
        'Gradient Boosting': gbm_model
    }
    for name, model in models.items():
        pred = model.predict(features_array)[0]
        predictions[name] = pred
    return predictions

# Streamlit app
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
        }
        .title {
            text-align: center;
            color: #343a40;
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .form-row {
            display: flex;
            justify-content: space-between;
        }
        .form-group {
            flex: 1;
            margin-right: 10px;
        }
        .form-group:last-child {
            margin-right: 0;
        }
        .predict-button {
            display: block;
            width: 100%;
            margin-top: 20px;
            background-color: #007bff;
            color: white;
            padding: 10px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .predict-button:hover {
            background-color: #0056b3;
        }
        .result-box {
            display: inline-block;
            width: 22%;
            padding: 10px;
            margin: 5px;
            border-radius: 5px;
            color: white;
            text-align: center;
        }
        .result-box.green {
            background-color: #28a745;
        }
        .result-box.red {
            background-color: #dc3545;
        }
        .results-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Brain Stroke Prediction App</div>', unsafe_allow_html=True)

# Input form
with st.form(key='prediction_form'):
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox('Gender', ['Male', 'Female'])
    with col2:
        age = st.slider('Age', min_value=0, max_value=100, value=20)
    with col3:
        hypertension = st.selectbox('Hypertension', [0, 1])
        
    col4, col5, col6 = st.columns(3)
    with col4:
        heart_disease = st.selectbox('Heart Disease', [0, 1])
    with col5:
        ever_married = st.selectbox('Ever Married', ['Yes', 'No'])
    with col6:
        work_type = st.selectbox('Work Type', ['Govt_job', 'Never_worked', 'Private', 'Self_employed', 'children'])
    
    col7, col8, col9 = st.columns(3)
    with col7:
        residence_type = st.selectbox('Residence Type', ['Rural', 'Urban'])
    with col8:
        avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0, max_value=300.0, value=80.13)
    with col9:
        bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=23.4)
    
    col10, col11 = st.columns(2)
    with col10:
        smoking_status = st.selectbox('Smoking Status', ['Unknown', 'formerly smoked', 'never smoked', 'smokes'])
    with col11:
        st.write("")  # Empty column for spacing

    submit_button = st.form_submit_button(label='Predict', key='predict_button', help='Click to predict stroke risk')

# Map categorical values to numerical values
def map_data(data):
    return {
        'gender': 0 if data['gender'] == 'Male' else 1,
        'age': data['age'],
        'hypertension': data['hypertension'],
        'heart_disease': data['heart_disease'],
        'ever_married': 1 if data['ever_married'] == 'Yes' else 0,
        'work_type': {'Govt_job': 0, 'Never_worked': 1, 'Private': 2, 'Self_employed': 3, 'children': 4}[data['work_type']],
        'Residence_type': 0 if data['residence_type'] == 'Rural' else 1,
        'avg_glucose_level': data['avg_glucose_level'],
        'bmi': data['bmi'],
        'smoking_status': {'Unknown': 0, 'formerly smoked': 1, 'never smoked': 2, 'smokes': 3}[data['smoking_status']]
    }

if submit_button:
    input_data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'residence_type': residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }
    
    data_mapped = map_data(input_data)
    
    features = [
        data_mapped['gender'],
        data_mapped['age'],
        data_mapped['hypertension'],
        data_mapped['heart_disease'],
        data_mapped['ever_married'],
        data_mapped['work_type'],
        data_mapped['Residence_type'],
        data_mapped['avg_glucose_level'],
        data_mapped['bmi'],
        data_mapped['smoking_status']
    ]
    
    features_array = np.array(features).reshape(1, -1)
    
    predictions = predict_stroke(features_array)
    
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    
    for model_name, pred in predictions.items():
        color = 'green' if pred == 0 else 'red'
        result = 'No Stroke' if pred == 0 else 'Stroke'
        st.markdown(f'<div class="result-box {color}">{model_name}: {result}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
