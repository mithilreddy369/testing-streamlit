import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('model.pkl')

# Title of the app
st.title('Stroke Prediction App')

# Create input fields for user data
age = st.number_input('Age', min_value=0)
hypertension = st.selectbox('Hypertension', [0, 1])
heart_disease = st.selectbox('Heart Disease', [0, 1])
avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0)
bmi = st.number_input('BMI', min_value=0.0)

# Create a DataFrame for prediction
input_data = pd.DataFrame([[age, hypertension, heart_disease, avg_glucose_level, bmi]],
                          columns=['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi'])

# Make a prediction
if st.button('Predict'):
    prediction = model.predict(input_data)
    st.write(f'Prediction: {"Stroke" if prediction[0] == 1 else "No Stroke"}')
