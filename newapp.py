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

# Define the feature names
feature_names = [
    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
]

# Placeholder for storing explanations
shap_lime_output = st.empty()

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

# Function to display SHAP and LIME explanations when a box is clicked
def display_explanations(model_name, features_array):
    # Create a DataFrame with feature names
    features_df = pd.DataFrame(features_array, columns=feature_names)

    if model_name == "CatBoost":
        # Initialize the SHAP explainer for CatBoost
        explainer = shap.Explainer(catboost_model)
        shap_values = explainer(features_df)

        # Plot SHAP values for the first prediction
        st.subheader(f"{model_name} SHAP Waterfall Chart")
        shap.plots.waterfall(shap_values[0])

        # Initialize the LIME explainer
        explainer_lime = LimeTabularExplainer(
            training_data=features_array,
            feature_names=feature_names,
            class_names=['No Stroke', 'Stroke'],
            mode='classification'
        )

        # Explain the first instance
        instance_index = 0  # For demo, explain the first instance
        instance = features_array[instance_index]
        explanation = explainer_lime.explain_instance(
            data_row=instance,
            predict_fn=catboost_model.predict_proba
        )

        # Display the LIME explanation
        st.subheader(f"{model_name} LIME Explanation")
        explanation.show_in_notebook(show_table=True, show_all=False)

# Function to create clickable boxes
def create_clickable_boxes(predictions, features_array):
    prediction_rows = []
    for name, pred in predictions.items():
        color = 'green' if pred == 0 else 'red'
        prediction_row = f"""
            <div class="prediction-box" onclick="document.getElementById('explanation_{name}').style.display='block'">
                <strong>{name}</strong><br>
                Prediction: <span style="color:{color}">{'No Stroke' if pred == 0 else 'Stroke'}</span>
            </div>
        """
        prediction_rows.append(prediction_row)

    # Render clickable boxes
    st.markdown('<div class="prediction-row">' + ''.join(prediction_rows) + '</div>', unsafe_allow_html=True)

    # Display explanation boxes (hidden by default)
    for name in predictions.keys():
        shap_lime_output.markdown(f"""
            <div id="explanation_{name}" style="display:none;">
                <h3>{name} Explanation:</h3>
            </div>
        """, unsafe_allow_html=True)

# Streamlit app
st.markdown("""
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        .input-group { margin-bottom: 15px; }
        .prediction-box { padding: 10px; border-radius: 5px; margin-bottom: 10px; }
        .green { background-color: #28a745; color: white; }
        .red { background-color: #dc3545; color: white; }
        .prediction-row { display: flex; justify-content: space-around; }
    </style>
""", unsafe_allow_html=True)

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
    
    st.write("## Predictions")

    prediction_rows = []
    for model_name, pred in predictions.items():
        color_class = 'green' if pred == 0 else 'red'
        result = 'No Stroke' if pred == 0 else 'Stroke'
        prediction_rows.append(f'<div class="prediction-box {color_class}">{model_name}: {result}</div>')

    st.markdown("""
        <script>
        function showExplanations(modelName) {
            if (modelName === 'CatBoost') {
                // Call Python function to show explanations (SHAP, LIME)
                Streamlit.setComponentValue('CatBoost');
            }
        }
        </script>""", unsafe_allow_html=True)
