# app.py
import streamlit as st
from backend import prepare_input_data, make_prediction

# Function to center content
def center_content():
    st.markdown("""
    <style>
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }
    .container {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="center">', unsafe_allow_html=True)
    st.markdown('<div class="container">', unsafe_allow_html=True)

def end_center_content():
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Center content
center_content()

# App title
st.title('Stroke Prediction App')

# User input
age = st.number_input('Age', min_value=0, max_value=120, value=30)
gender = st.selectbox('Gender', ['Male', 'Female'])
hypertension = st.selectbox('Hypertension', [0, 1])
heart_disease = st.selectbox('Heart Disease', [0, 1])
ever_married = st.selectbox('Ever Married', ['No', 'Yes'])
work_type = st.selectbox('Work Type', ['Govt_job', 'Never_worked', 'Private', 'Self_employed'])
Residence_type = st.selectbox('Residence Type', ['Rural', 'Urban'])
avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0, value=100.0)
bmi = st.number_input('BMI', min_value=0.0, value=25.0)
smoking_status = st.selectbox('Smoking Status', ['Unknown', 'formerly smoked', 'never smoked', 'smokes'])

# Prepare input data for prediction
input_data = prepare_input_data(age, gender, hypertension, heart_disease, ever_married, work_type,
                                Residence_type, avg_glucose_level, bmi, smoking_status)

# Prediction
if st.button('Predict'):
    prediction = make_prediction(input_data)
    st.write(f"Prediction: {prediction}")

# End centered content
end_center_content()
