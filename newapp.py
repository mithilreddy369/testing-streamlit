import streamlit as st
import pickle
import numpy as np
import shap
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

# Load models
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

# Function to display SHAP and LIME plots for the chosen model
def display_xai(model_name, model, features_array):
    st.write(f"### XAI for {model_name}")
    
    # SHAP
    st.write("#### SHAP Waterfall Plot")
    explainer = shap.Explainer(model)
    shap_values = explainer(pd.DataFrame(features_array, columns=feature_names))
    shap.plots.waterfall(shap_values[0])
    
    # LIME
    st.write("#### LIME Explanation")
    explainer = LimeTabularExplainer(training_data=features_array, feature_names=feature_names, class_names=['No Stroke', 'Stroke'], mode='classification')
    explanation = explainer.explain_instance(features_array[0], model.predict_proba)
    fig = explanation.as_pyplot_figure()
    st.pyplot(fig)

# Streamlit app
st.title('Brain Stroke Prediction App')

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

    submit_button = st.form_submit_button(label='Predict')

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

# Prediction process
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
    features_array = np.array(list(data_mapped.values())).reshape(1, -1)

    predictions = predict_stroke(features_array)

    # Display prediction boxes
    st.write("## Predictions (Click to view XAI)")
    for model_name in predictions.keys():
        if st.button(f'{model_name}: {"No Stroke" if predictions[model_name] == 0 else "Stroke"}'):
            display_xai(model_name, globals()[f"{model_name.lower().replace(' ', '_')}_model"], features_array)
