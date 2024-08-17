import streamlit as st
import numpy as np
import joblib
from feature_engineering import feature_engineering  # Import the function
from css import add_custom_css  # Import the custom CSS function

# Apply custom CSS
add_custom_css()

# Load models using joblib
catboost_model = joblib.load('catboost_model.pkl')
lgb_model = joblib.load('lgb_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')
gbm_model = joblib.load('gbm_model.pkl')

# Function to center content
def center_content():
    st.markdown("""
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    """, unsafe_allow_html=True)

# Center content
center_content()

# App title
st.markdown('<div class="container"><div class="header"><h1>Stroke Prediction App</h1></div>', unsafe_allow_html=True)

# Bootstrap grid layout for input fields
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

# Create form layout
with st.form(key='input_form'):
    st.markdown('<div class="form-row">', unsafe_allow_html=True)
    cols = st.columns(3)  # Create 3 columns per row
    for i, (label, (var_name, input_type, *args)) in enumerate(form_data.items()):
        with cols[i % 3]:  # Ensure 3 columns per row
            if input_type == 'number_input':
                locals()[var_name] = st.number_input(label, min_value=args[0], max_value=args[1], value=args[2])
            elif input_type == 'selectbox':
                locals()[var_name] = st.selectbox(label, args[0])

    st.markdown('</div>', unsafe_allow_html=True)

    # Submit button
    submit_button = st.form_submit_button(label='Predict')

# Prepare input data for prediction
features = {
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
}

# Apply feature engineering
input_data = feature_engineering(features)

# Ensure input_data is a 2D array
input_data = np.array(input_data).reshape(1, -1)  # Flatten and reshape to (1, -1)

# Prediction
if submit_button:
    try:
        # Get predictions from all models
        catboost_prob = catboost_model.predict_proba(input_data)[0][1]  # Probability of "Stroke" class
        lgb_prob = lgb_model.predict_proba(input_data)[0][1]
        xgb_prob = xgb_model.predict_proba(input_data)[0][1]
        gbm_prob = gbm_model.predict_proba(input_data)[0][1]

        predictions = {
            'CatBoost Model': 'Stroke' if catboost_prob > 0.2 else 'No Stroke',
            'LightGBM Model': 'Stroke' if lgb_prob > 0.2 else 'No Stroke',
            'XGBoost Model': 'Stroke' if xgb_prob > 0.2 else 'No Stroke',
            'Gradient Boosting Model': 'Stroke' if gbm_prob > 0.2 else 'No Stroke'
        }

        # Display predictions side by side using columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            color = 'success' if predictions['CatBoost Model'] == 'No Stroke' else 'danger'
            st.markdown(f"""
            <div class="alert alert-{color}" role="alert">
                <h4 class="alert-heading">CatBoost Model</h4>
                <p>{predictions['CatBoost Model']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            color = 'success' if predictions['LightGBM Model'] == 'No Stroke' else 'danger'
            st.markdown(f"""
            <div class="alert alert-{color}" role="alert">
                <h4 class="alert-heading">LightGBM Model</h4>
                <p>{predictions['LightGBM Model']}</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            color = 'success' if predictions['XGBoost Model'] == 'No Stroke' else 'danger'
            st.markdown(f"""
            <div class="alert alert-{color}" role="alert">
                <h4 class="alert-heading">XGBoost Model</h4>
                <p>{predictions['XGBoost Model']}</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            color = 'success' if predictions['Gradient Boosting Model'] == 'No Stroke' else 'danger'
            st.markdown(f"""
            <div class="alert alert-{color}" role="alert">
                <h4 class="alert-heading">Gradient Boosting</h4>
                <p>{predictions['Gradient Boosting Model']}</p>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error making prediction: {e}")

# End centered content
st.markdown('</div>', unsafe_allow_html=True)
