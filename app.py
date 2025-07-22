import streamlit as st
import pandas as pd
from Main import model, label_encoder, X  

#  Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
        color: #2c3e50;
        font-family: 'Arial', sans-serif;
    }
    .title {
        color: #ffffff;
        background: linear-gradient(90deg, #6dd5ed, #2193b0);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        font-size: 40px;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #2196F3;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0b7dda;
        color: white;
    }
    .stForm {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 2px 2px 15px rgba(0,0,0,0.1);
    }
    .success {
        background-color: #4CAF50;
        color: white;
        padding: 12px;
        border-radius: 8px;
        font-size: 18px;
        text-align: center;
    }
    .info {
        background-color: #2196F3;
        color: white;
        padding: 12px;
        border-radius: 8px;
        font-size: 18px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

#  App Title
st.markdown('<div class="title"> AI Employee Salary Predictor</div>', unsafe_allow_html=True)
st.write("This app predicts whether an employee earns **<=50K** or **>50K** using your trained model.")

#  Input Form
st.subheader("🔎 Enter Employee Details")
with st.form("prediction_form"):
    age = st.number_input(" Age", min_value=18, max_value=100, value=30)
    workclass = st.selectbox(" Workclass", [
        'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
        'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'
    ])
    education = st.selectbox(" Education", [
        'Bachelors', 'HS-grad', 'Masters', 'Some-college', 'Assoc-acdm',
        'Assoc-voc', 'Doctorate', 'Prof-school', '12th', '11th', '10th',
        '9th', '7th-8th', '5th-6th', '1st-4th', 'Preschool'
    ])
    marital_status = st.selectbox(" Marital Status", [
        'Married-civ-spouse', 'Divorced', 'Never-married',
        'Separated', 'Widowed', 'Married-spouse-absent'
    ])
    occupation = st.selectbox(" Occupation", [
        'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
        'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
        'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
        'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'
    ])
    relationship = st.selectbox(" Relationship", [
        'Wife', 'Own-child', 'Husband', 'Not-in-family',
        'Other-relative', 'Unmarried'
    ])
    race = st.selectbox(" Race", [
        'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'
    ])
    gender = st.radio(" Gender", ['Male', 'Female'])
    capital_gain = st.number_input(" Capital Gain", min_value=0, value=0)
    capital_loss = st.number_input(" Capital Loss", min_value=0, value=0)
    hours_per_week = st.slider(" Hours per Week", min_value=1, max_value=99, value=40)
    native_country = st.selectbox(" Native Country", [
        'United-States', 'Mexico', 'Philippines', 'Germany',
        'Canada', 'India', 'England', 'China', 'Other'
    ])

    submit = st.form_submit_button(" Predict Salary Class")

#  Prediction Logic
if submit:
    #  Create DataFrame from input
    new_employee = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'education': [education],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'gender': [gender],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week': [hours_per_week],
        'native-country': [native_country],
        'fnlwgt': [150000],           # Dummy default value
        'educational-num': [10]       # Dummy default value
    })

    #  Align new_employee with model’s expected columns
    for col in X.columns:
        if col not in new_employee.columns:
            new_employee[col] = 0
    new_employee = new_employee[X.columns]

    st.write(" Input Data Prepared for Prediction:")
    st.dataframe(new_employee)

    #  Predict
    prediction = model.predict(new_employee)
    predicted_class = label_encoder.inverse_transform(prediction)[0]

    #  Check if model supports predict_proba
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(new_employee)[0][1] * 100
    else:
        probability = None

    # Show result
    st.markdown(f'<div class="success">Predicted Income Class: <b>{predicted_class}</b></div>', unsafe_allow_html=True)
    if probability is not None:
        st.markdown(f'<div class="info"> Probability of earning >50K: <b>{probability:.2f}%</b></div>', unsafe_allow_html=True)
