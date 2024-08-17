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
    for i, (label, (var_name, input_type, *args)) in enumerate(form_data.items()):
        if i % 3 == 0 and i > 0:
            st.markdown('</div><div class="form-row">', unsafe_allow_html=True)
        with st.beta_expander(label, expanded=True):
            if input_type == 'number_input':
                locals()[var_name] = st.number_input(label, min_value=args[0], max_value=args[1], value=args[2])
            elif input_type == 'selectbox':
                locals()[var_name] = st.selectbox(label, args[0])
            # Add other input types if needed
    st.markdown('</div>', unsafe_allow_html=True)

    # Submit button
    submit_button = st.button(label='Predict')

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
input_data = np.array(input_data).reshape(1, -1)  # Flatten and reshape to (1, 40)

# Prediction
if submit_button:
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
