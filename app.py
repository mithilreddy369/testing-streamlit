import streamlit as st
import numpy as np
import joblib
from feature_engineering import feature_engineering
from css import add_custom_css

# Apply custom CSS
add_custom_css()

# Load models
catboost_model = joblib.load('catboost_model.pkl')
lgb_model = joblib.load('lgb_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')
gbm_model = joblib.load('gbm_model.pkl')

# Center content
def center_content():
    st.markdown("""
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    """, unsafe_allow_html=True)

center_content()

# App title
st.markdown('<div class="container"><div class="header"><h1>Stroke Prediction App</h1></div>', unsafe_allow_html=True)

# Form input details
form_data = {
    'Age': ('age', 'number_input', 0, 120, 30),
    'Hypertension': ('hypertension', 'selectbox', [0, 1]),
    'Ever Married': ('ever_married', 'selectbox', ['No', 'Yes']),
    'Average Glucose Level': ('avg_glucose_level', 'number_input', 0.0, 200.0, 100.0),
    'Smoking Status': ('smoking_status', 'selectbox', ['Unknown', 'formerly smoked', 'never smoked', 'smokes']),
    'Gender': ('gender', 'selectbox', ['Male', 'Female']),
    'Heart Disease': ('heart_disease', 'selectbox', [0, 1]),
    'Work Type': ('work_type', 'selectbox', ['Govt_job', 'Never_worked', 'Private', 'Self_employed']),
    'Residence Type': ('Residence_type', 'selectbox', ['Rural', 'Urban']),
    'BMI': ('bmi', 'number_input', 0.0, 50.0, 25.0),
}

# Input form
with st.form(key='input_form'):
    st.markdown('<div class="form-row">', unsafe_allow_html=True)
    cols = st.columns(3)
    
    user_input = {}
    for i, (label, (var_name, input_type, *args)) in enumerate(form_data.items()):
        with cols[i % 3]:
            if input_type == 'number_input':
                user_input[var_name] = st.number_input(label, min_value=args[0], max_value=args[1], value=args[2])
            elif input_type == 'selectbox':
                user_input[var_name] = st.selectbox(label, args[0])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Submit button
    submit_button = st.form_submit_button(label='Predict')

# Prepare input data
if submit_button:
    features = {
        'age': user_input['age'],
        'gender': 0 if user_input['gender'] == 'Male' else 1,
        'hypertension': user_input['hypertension'],
        'heart_disease': user_input['heart_disease'],
        'ever_married': 0 if user_input['ever_married'] == 'No' else 1,
        'work_type': ['Govt_job', 'Never_worked', 'Private', 'Self_employed'].index(user_input['work_type']),
        'Residence_type': 0 if user_input['Residence_type'] == 'Rural' else 1,
        'avg_glucose_level': user_input['avg_glucose_level'],
        'bmi': user_input['bmi'],
        'smoking_status': ['Unknown', 'formerly smoked', 'never smoked', 'smokes'].index(user_input['smoking_status']),
    }

    # Apply feature engineering
    input_data = feature_engineering(features)
    input_data = np.array(input_data).reshape(1, -1)

    try:
        # Predictions
        catboost_pred = catboost_model.predict(input_data)[0]
        lgb_pred = lgb_model.predict(input_data)[0]
        xgb_pred = xgb_model.predict(input_data)[0]
        gbm_pred = gbm_model.predict(input_data)[0]

        predictions = {
            'CatBoost Model': 'Stroke' if catboost_pred == 1 else 'No Stroke',
            'LightGBM Model': 'Stroke' if lgb_pred == 1 else 'No Stroke',
            'XGBoost Model': 'Stroke' if xgb_pred == 1 else 'No Stroke',
            'Gradient Boosting Model': 'Stroke' if gbm_pred == 1 else 'No Stroke'
        }

        # Display predictions
        for model_name, result in predictions.items():
            color = 'success' if result == 'No Stroke' else 'danger'
            st.markdown(f"""
            <div class="alert alert-{color} mt-4" role="alert">
                <h4 class="alert-heading">{model_name}</h4>
                <p class="mb-0">The prediction is: <strong>{result}</strong></p>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error making prediction: {e}")
