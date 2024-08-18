import streamlit as st
import numpy as np
import pickle
from feature_engineering import feature_engineering  # Ensure this is used if needed
from css import add_custom_css  # Ensure this is correctly implemented

# Apply custom CSS
add_custom_css()

def load_model(file_path):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except pickle.UnpicklingError as e:
        st.error(f"Error loading {file_path}: {e}")
    except FileNotFoundError as e:
        st.error(f"File not found: {file_path}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Load models
catboost_model = load_model('catboost_model1.pkl')
lgb_model = load_model('lgb_model1.pkl')
xgb_model = load_model('xgb_model1.pkl')
gbm_model = load_model('gbm_model1.pkl')

# Function to center content
def center_content():
    st.markdown("""
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    """, unsafe_allow_html=True)

# Center content
center_content()

# App title
st.markdown('<div class="container"><div class="header"><h1>Brain Stroke Prediction App</h1></div>', unsafe_allow_html=True)

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
    # Create a grid layout with 3 columns per row
    rows = []
    for i, (label, (var_name, input_type, *args)) in enumerate(form_data.items()):
        if i % 3 == 0:
            # Start a new row for every 3 inputs
            row = st.columns(3)
            rows.append(row)
        
        # Determine the correct column for the input field
        col = rows[-1][i % 3]
        
        if input_type == 'number_input':
            locals()[var_name] = col.number_input(label, min_value=args[0], max_value=args[1], value=args[2])
        elif input_type == 'selectbox':
            locals()[var_name] = col.selectbox(label, args[0])
    
    # Handle remaining fields if not divisible by 3
    if len(form_data) % 3 != 0:
        remaining_cols = 3 - (len(form_data) % 3)
        for _ in range(remaining_cols):
            st.empty()  # Add empty columns to align the last row properly

    # Submit button
    submit_button = st.form_submit_button(label='Predict')

# Categorical mappings
mappings = {
    'gender': {'Female': 0, 'Male': 1, 'Other': 2},
    'ever_married': {'No': 0, 'Yes': 1},
    'work_type': {'Govt_job': 0, 'Never_worked': 1, 'Private': 2, 'Self_employed': 3, 'children': 4},
    'Residence_type': {'Rural': 0, 'Urban': 1},
    'smoking_status': {'Unknown': 0, 'formerly smoked': 1, 'never smoked': 2, 'smokes': 3}
}

# Prepare input data
features = {
    'age': age,
    'gender': mappings['gender'][gender],
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'ever_married': mappings['ever_married'][ever_married],
    'work_type': mappings['work_type'][work_type],
    'Residence_type': mappings['Residence_type'][Residence_type],
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'smoking_status': mappings['smoking_status'][smoking_status]
}

# Convert feature values to NumPy array
input_data = np.array(list(features.values())).reshape(1, -1)

# Prediction
if submit_button:
    try:
        # Get predictions from all models
        catboost_prob = catboost_model.predict_proba(input_data)[0][1]
        lgb_prob = lgb_model.predict_proba(input_data)[0][1]
        xgb_prob = xgb_model.predict_proba(input_data)[0][1]
        gbm_prob = gbm_model.predict_proba(input_data)[0][1]

        predictions = {
            'CatBoost Model': 'Stroke' if catboost_prob > 0.5 else 'No Stroke',
            'LightGBM Model': 'Stroke' if lgb_prob > 0.5 else 'No Stroke',
            'XGBoost Model': 'Stroke' if xgb_prob > 0.5 else 'No Stroke',
            'Gradient Boosting Model': 'Stroke' if gbm_prob > 0.5 else 'No Stroke'
        }

        # Display predictions side by side
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
