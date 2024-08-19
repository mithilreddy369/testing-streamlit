import streamlit as st
import pickle
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt
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

# Create a function for SHAP explanations
def explain_model(model, features_df):
    explainer = shap.Explainer(model)
    shap_values = explainer(features_df)
    return shap_values

# Streamlit app
st.markdown("""
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        .input-group { margin-bottom: 15px; }
        .prediction-box { padding: 10px; border-radius: 5px; margin-bottom: 10px; }
        .green { background-color: #28a745; color: white; }
        .red { background-color: #dc3545; color: white; }
        .prediction-row { display: flex; justify-content: space-around; }
        .xai-buttons { margin-top: 20px; }
        .xai-button { padding: 10px 20px; border-radius: 5px; cursor: pointer; }
    </style>
""", unsafe_allow_html=True)

st.title('Brain Stroke Prediction App')

# Initialize session state
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'CatBoost'

if 'input_data' not in st.session_state:
    st.session_state.input_data = {
        'gender': 'Male',
        'age': 20,
        'hypertension': 0,
        'heart_disease': 0,
        'ever_married': 'Yes',
        'work_type': 'Govt_job',
        'residence_type': 'Rural',
        'avg_glucose_level': 80.13,
        'bmi': 23.4,
        'smoking_status': 'Unknown'
    }

# Input form
with st.form(key='prediction_form'):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.input_data['gender'] = st.selectbox('Gender', ['Male', 'Female'], index=['Male', 'Female'].index(st.session_state.input_data['gender']))
    with col2:
        st.session_state.input_data['age'] = st.slider('Age', min_value=0, max_value=100, value=st.session_state.input_data['age'])
    with col3:
        st.session_state.input_data['hypertension'] = st.selectbox('Hypertension', [0, 1], index=[0, 1].index(st.session_state.input_data['hypertension']))

    col4, col5, col6 = st.columns(3)
    with col4:
        st.session_state.input_data['heart_disease'] = st.selectbox('Heart Disease', [0, 1], index=[0, 1].index(st.session_state.input_data['heart_disease']))
    with col5:
        st.session_state.input_data['ever_married'] = st.selectbox('Ever Married', ['Yes', 'No'], index=['Yes', 'No'].index(st.session_state.input_data['ever_married']))
    with col6:
        st.session_state.input_data['work_type'] = st.selectbox('Work Type', ['Govt_job', 'Never_worked', 'Private', 'Self_employed', 'children'], index=['Govt_job', 'Never_worked', 'Private', 'Self_employed', 'children'].index(st.session_state.input_data['work_type']))

    col7, col8, col9 = st.columns(3)
    with col7:
        st.session_state.input_data['residence_type'] = st.selectbox('Residence Type', ['Rural', 'Urban'], index=['Rural', 'Urban'].index(st.session_state.input_data['residence_type']))
    with col8:
        st.session_state.input_data['avg_glucose_level'] = st.number_input('Average Glucose Level', min_value=0.0, max_value=300.0, value=st.session_state.input_data['avg_glucose_level'])
    with col9:
        st.session_state.input_data['bmi'] = st.number_input('BMI', min_value=0.0, max_value=100.0, value=st.session_state.input_data['bmi'])

    col10, col11 = st.columns(2)
    with col10:
        st.session_state.input_data['smoking_status'] = st.selectbox('Smoking Status', ['Unknown', 'formerly smoked', 'never smoked', 'smokes'], index=['Unknown', 'formerly smoked', 'never smoked', 'smokes'].index(st.session_state.input_data['smoking_status']))
    
    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    data_mapped = map_data(st.session_state.input_data)
    
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
    
    # Create a DataFrame with feature names
    feature_names = [
        'gender',
        'age',
        'hypertension',
        'heart_disease',
        'ever_married',
        'work_type',
        'Residence_type',
        'avg_glucose_level',
        'bmi',
        'smoking_status'
    ]
    
    features_df = pd.DataFrame(features_array, columns=feature_names)
    
    # Make predictions
    predictions = predict_stroke(features_array)
    
    st.write("## Predictions")

    prediction_rows = []
    for model_name, pred in predictions.items():
        color_class = 'green' if pred == 0 else 'red'
        result = 'No Stroke' if pred == 0 else 'Stroke'
        prediction_rows.append(f'<div class="prediction-box {color_class}">{model_name}: {result}</div>')

    st.markdown('<div class="prediction-row">' + ''.join(prediction_rows) + '</div>', unsafe_allow_html=True)

    # XAI explanation buttons
    st.write("## XAI Explanations")
    
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button('CatBoost XAI'):
                st.session_state.selected_model = 'CatBoost'

        with col2:
            if st.button('LightGBM XAI'):
                st.session_state.selected_model = 'LightGBM'

        with col3:
            if st.button('XGBoost XAI'):
                st.session_state.selected_model = 'XGBoost'

        with col4:
            if st.button('Gradient Boosting XAI'):
                st.session_state.selected_model = 'Gradient Boosting'
    
    selected_model = st.session_state.selected_model

    # Display SHAP explanation for the selected model
    if selected_model == 'CatBoost':
        st.write("### SHAP Explanation for CatBoost Model")
        shap_values = explain_model(catboost_model, features_df)
    elif selected_model == 'LightGBM':
        st.write("### SHAP Explanation for LightGBM Model")
        shap_values = explain_model(lgb_model, features_df)
    elif selected_model == 'XGBoost':
        st.write("### SHAP Explanation for XGBoost Model")
        shap_values = explain_model(xgb_model, features_df)
    elif selected_model == 'Gradient Boosting':
        st.write("### SHAP Explanation for Gradient Boosting Model")
        shap_values = explain_model(gbm_model, features_df)

    # Plot the SHAP explanation
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0])
    st.pyplot(fig)
