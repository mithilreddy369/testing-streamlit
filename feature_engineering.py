def feature_engineering(data):
  # 1. Body Mass Index (BMI) Category
def bmi_category(bmi):
    if bmi < 18.5:
        return 0  # Underweight
    elif bmi < 24.9:
        return 1  # Normal weight
    elif bmi < 29.9:
        return 2  # Overweight
    else:
        return 3  # Obesity

data['bmi_category'] = data['bmi'].apply(bmi_category)

# 2. Glucose Level Category
def glucose_category(glucose):
    if glucose < 70:
        return 0  # Low
    elif glucose < 100:
        return 1  # Normal
    elif glucose < 126:
        return 2  # Pre-diabetic
    else:
        return 3  # Diabetic

data['glucose_category'] = data['avg_glucose_level'].apply(glucose_category)

# 3. Age and BMI Interaction
data['age_bmi_interaction'] = data['age'] * data['bmi']

# 4. Married and Work Type Interaction
data['married_work_interaction'] = data['ever_married'] * data['work_type']

# 5. High Risk Indicators
data['high_risk_indicators'] = (
    (data['hypertension'] == 1) |
    (data['heart_disease'] == 1) |
    (data['glucose_category'] == 3) |
    (data['bmi_category'] == 3)
).astype(int)

# 6. Age Group
def age_group(age):
    if age < 30:
        return 0  # Young
    elif age < 60:
        return 1  # Middle-aged
    else:
        return 2  # Senior

data['age_group'] = data['age'].apply(age_group)

# 7. Risk Factor
data['risk_factor'] = data['hypertension'] + data['heart_disease'] + data['age_group']

# 8. Glucose-BMI Ratio
data['glucose_bmi_ratio'] = data['avg_glucose_level'] / data['bmi']

# 9. Married Working
data['married_working'] = data['ever_married'] * (data['work_type'] != 1).astype(int)

# 10. Urban Smoker
data['urban_smoker'] = data['Residence_type'] * (data['smoking_status'] == 3).astype(int)

# 11. Age Group and Work Type Interaction
data['age_work_interaction'] = data['age_group'] * data['work_type']

# 12. Smoking Status and Residence Type Interaction
data['smoking_residence_interaction'] = data['smoking_status'] * data['Residence_type']

# 13. Age and Hypertension Interaction
data['age_hypertension_interaction'] = data['age'] * data['hypertension']

# 14. Age and Heart Disease Interaction
data['age_heart_disease_interaction'] = data['age'] * data['heart_disease']

# 15. Glucose Level and Hypertension Interaction
data['glucose_hypertension_interaction'] = data['avg_glucose_level'] * data['hypertension']

# 16. Glucose Level and Heart Disease Interaction
data['glucose_heart_disease_interaction'] = data['avg_glucose_level'] * data['heart_disease']

# 17. BMI and Hypertension Interaction
data['bmi_hypertension_interaction'] = data['bmi'] * data['hypertension']

# 18. BMI and Heart Disease Interaction
data['bmi_heart_disease_interaction'] = data['bmi'] * data['heart_disease']

# 19. Glucose and BMI Ratio by Age Group
data['age_group_glucose_bmi_ratio'] = data['age_group'] * data['glucose_bmi_ratio']

# 20. Glucose Level and Age Group Interaction
data['glucose_age_group_interaction'] = data['avg_glucose_level'] * data['age_group']

# New Features

# 21. Age and Glucose Level Interaction
data['age_glucose_interaction'] = data['age'] * data['avg_glucose_level']

# 22. BMI and Glucose Level Interaction
data['bmi_glucose_interaction'] = data['bmi'] * data['avg_glucose_level']

# 23. Hypertension and Glucose Level Category Interaction
data['hypertension_glucose_category_interaction'] = data['hypertension'] * data['glucose_category']

# 24. Heart Disease and Glucose Level Category Interaction
data['heart_disease_glucose_category_interaction'] = data['heart_disease'] * data['glucose_category']

# 25. Age Group and Glucose Category Interaction
data['age_group_glucose_category_interaction'] = data['age_group'] * data['glucose_category']

# 26. BMI Category and Glucose Level Category Interaction
data['bmi_category_glucose_category_interaction'] = data['bmi_category'] * data['glucose_category']

# 27. Age Group and BMI Category Interaction
data['age_group_bmi_category_interaction'] = data['age_group'] * data['bmi_category']

# 28. Age and High Risk Indicators Interaction
data['age_high_risk_interaction'] = data['age'] * data['high_risk_indicators']

# 29. Glucose-BMI Ratio and Risk Factor Interaction
data['glucose_bmi_ratio_risk_factor_interaction'] = data['glucose_bmi_ratio'] * data['risk_factor']

# 30. Work Type and Risk Factor Interaction
data['work_type_risk_factor_interaction'] = data['work_type'] * data['risk_factor']

# Create DataFrame
    df = pd.DataFrame([data])
    # Reorder columns to match model
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0  # Add missing features with default value
    df = df[expected_features]
    
    return df
