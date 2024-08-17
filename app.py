import streamlit as st
import numpy as np
import joblib
from feature_engineering import feature_engineering  # Import the function
from css import add_custom_css  # Import the custom CSS function

# Apply custom CSS
add_custom_css()

# Load the model
model = joblib.load('model.pkl')

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
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', min_value=0, max_value=120, value=30)
    hypertension = st.selectbox('Hypertension', [0, 1])
    ever_married = st.selectbox('Ever Married', ['No', 'Yes'])
    avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0, value=100.0)
    smoking_status = st.selectbox('Smoking Status', ['Unknown', 'formerly smoked', 'never smoked', 'smokes'])

with col2:
    gender = st.selectbox('Gender', ['Male', 'Female'])
    heart_disease = st.selectbox('Heart Disease', [0, 1])
    work_type = st.selectbox('Work Type', ['Govt_job', 'Never_worked', 'Private', 'Self_employed'])
    Residence_type = st.selectbox('Residence Type', ['Rural', 'Urban'])
    bmi = st.number_input('BMI', min_value=0.0, value=25.0)

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

# Debug: Check the shape and type of input_data
st.write(f"Type of Input Data: {type(input_data)}")
st.write(f"Shape of Input Data: {np.array(input_data).shape}")

# Ensure input_data is a 2D array
input_data = np.array(input_data).reshape(1, -1)  # Flatten and reshape to (1, 40)

# Prediction
if st.button('Predict', key='predict_button'):
    try:
        prediction = model.predict(input_data)
        result = 'Stroke' if prediction[0] == 1 else 'No Stroke'
        st.markdown(f"""
        <div class="alert alert-primary mt-4" role="alert">
            <h4 class="alert-heading">Prediction Result</h4>
            <p class="mb-0">The prediction is: <strong>{result}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error making prediction: {e}")

# End centered content
st.markdown('</div>', unsafe_allow_html=True)
