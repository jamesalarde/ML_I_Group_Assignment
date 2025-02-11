# -*- coding: utf-8 -*-

import streamlit as st
import pickle
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

# Load the model and scaler
model = joblib.load('linearregressionclassifier_final.pkl')
scaler = pickle.load(open('scaler_final.pkl', 'rb'))

# Create the prediction function
def diabetes_prediction(input_data):

    # Change the input data into an array and reshape
    input_data_new = np.asarray(input_data).reshape(1, -1)

    # Apply feature scaling
    input_data_scaled = scaler.transform(input_data_new)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    if (prediction[0] == 0):
      st.success(':white_check_mark: The patient is likely not diabetic.')
    else:
      st.error('⚠️ The patient is likely diabetic. Please consider further testing of the patient to confirm.')

def project_description():
    st.title('ML I Group Assignment')
    st.subheader('Group 8 : James Alarde, Omer Althwaini, Africa Bajils, Maria do Carmo Brito e Abreu, Emiliano Puertas')
    st.markdown('---')

    st.write("""
    
    #### Project Overview
    This project uses Machine Learning to predict diabetes based on user-inputted health parameters. The goal is to improve diabetes screening protocols,
    which have proved to be demanding on time and resource.

    #### Methodology
    The group obtained the dataset of female Pima Indians aged at least 21 years old from the National Institute of Diabetes and Digestion and Kidney Diseases. The dataset can be accessed here : https://www.kaggle.com/datasets/nancyalaswad90/review.
    The model was trained using the above dataset containing health features, such as glucose, BMI, age, etc., and was developed with the following methodology.
    1. Data Preparation
    2. Model Development (Logistic Regression)
    3. Model Evaluation
    4. User Interface Deployment

    #### Features
    The key features considered in this model were :
    - **Age**
    """)   

    with st.expander('Explanation'):
       st.write('What it measures : The person’s age in years.')
       st.write('Relation to Diabetes : Older adults are more likely to develop Type 2 Diabetes. Metabolism slows down, and insulin sensitivity decreases with age.')

    st.write("""
    - **BMI**
    """)

    with st.expander('Explanation'):
       st.write('What it measures : A person’s weight relative to their height.')
       st.write('Relation to Diabetes : Higher BMI (>30) is a major risk factor for Type 2 Diabetes. Excess fat, especially around the abdomen, leads to insulin resistance.')

    st.write("""
    - **Pregnancies**
    """)

    with st.expander('Explanation'):
       st.write('What it measures : The number of times a woman has been pregnant.')
       st.write('Relation to Diabetes : A higher number of pregnancies is linked to a higher risk of gestational diabetes, which increases the likelihood of developing Type 2 Diabetes later in life. Pregnancy-related weight gain and hormonal changes can contribute to insulin resistance.')    

    st.write("""
    - **Glucose Level**
    """)

    with st.expander('Explanation'):
       st.write('What it measures : Blood sugar levels.')
       st.write('Relation to Diabetes : High fasting glucose levels (above 126 mg/dL) are a strong indicator of diabetes. Diabetes occurs when the body doesn’t produce enough insulin or cells don’t respond to insulin properly, leading to high blood sugar.')

    st.write("""
    - **Insulin Level**
    """)

    with st.expander('Explanation'):
       st.write('What it measures : Insulin levels in the blood.')
       st.write('Relation to Diabetes : Low insulin levels indicate Type 1 Diabete, where insulin production is impaired. Type 2 Diabetes may show high insulin levels at first due to insulin resistance, but over time, the pancreas may fail to produce enough insulin.')

    st.write("""
    - **Blood Pressure**
    """)

    with st.expander('Explanation'):
       st.write('What it measures : The force of blood against artery walls.')
       st.write('Relation to Diabetes : High blood pressure (hypertension) is common in people with diabetes. It contributes to insulin resistance and increases the risk of heart disease, a major complication of diabetes.')

    st.write("""
    - **Skin Thickness**
    """)

    with st.expander('Explanation'):
       st.write('What it measures : Subcutaneous fat thickness and measured in the triceps area.')
       st.write('Relation to Diabetes : A higher skin thickness suggests more body fat, which is linked to insulin resistance. More fat around organs (visceral fat) worsens glucose metabolism.')

    st.write("""
    - **Diabetes Pedigree Function (DPF) Value**
    """)

    with st.expander('Explanation'):
       st.write('What it measures : A score estimating genetic predisposition to diabetes.')
       st.write('Relation to Diabetes : A higher DPF means a stronger family history of diabetes. Genetics play a role in how the body processes glucose and responds to insulin.')

    st.image('./features.png')

    st.write("""
    #### Findings
    Below are the weights the model assigns to each of the features - how relevant it is to the diagnosis. Based on the model, glucose level is the best indicator for likelihood of diabetes.
    """)

    # Data visualization
    data = {
    'Parameter': ['Glucose Level', 'BMI', 'Pregnancies', 'DPF Value', 'Blood Pressure', 'Insulin Level', 'Age', 'Skin Thickness'],
    'Value': [5.83, 3.07, 1.63, 1.53, 1.07, 0.47, 0.45, 0.00],
    }
    df = pd.DataFrame(data)
    fig = px.bar(df, x = 'Parameter', y = 'Value', color = 'Value', title = 'Feature Importance', text = 'Value')
    st.plotly_chart(fig, use_container_width = True)

    st.write("""

    #### Conclusion
    The model's accuracy is at 84% after testing, and it effectively flags high-risk patients for further testing. With this, the group hopes that the medical industry
    can adopt this tool as a first touch approach to screening diabetes. 
    """)

def prediction_interface():
    col1, col2 = st.columns([1,5])
    with col1:
       st.title(':dna: :clipboard:')
    with col2:
       st.title('Diabetes Prediction App')
    
    st.write('The app aims to predict a diabetes diagnosis using several factors provided by the user.')
    st.write("""

    #### How to Use
    1. Enter all required details.
    2. Review all values in the User Input Review section.
    3. Click on the **Predict** button to get a diagnosis.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
      Age = st.number_input('Age', min_value = 0, max_value = 122, value = 0, step = 1)
      st.write('')
    with col1:
      Pregnancies = st.number_input('Pregnancies', min_value = 0 ,max_value = 20, value = 0, step = 1)
      st.write('')
    with col1:
      Glucose = st.number_input('Glucose level', min_value = 40, max_value = 600, value = 40, step = 1)
      st.write('')
    with col1:
      BMI = st.slider('BMI value', min_value = 12.0, max_value = 100.0, value = 12.0, step = 0.01)
    with col2:
      BloodPressure = st.number_input('Blood Pressure level in mm Hg', min_value = 30, max_value = 300, value = 30, step = 1)
      st.write('')
    with col2:
      SkinThickness = st.number_input('Skin Thickness in mm', min_value = 0, max_value = 100, value = 0, step = 1)
      st.write('')
    with col2:
      Insulin = st.number_input('Insulin level in mu U/ml', min_value = 0, max_value = 300, value = 0, step = 1)
      st.write('')
    with col2:
      DiabetesPedigreeFunction = st.slider('Diabetes Pedigree Function value', min_value = 0.000, max_value = 2.500, value = 0.000, step = 0.001, format="%3f")
    
    # Store user input
    user_data = [Age, Pregnancies, Glucose, BMI, Insulin, BloodPressure, SkinThickness, DiabetesPedigreeFunction]

    # Display user input values as a table
    st.write('### User Input Review')
    st.write('Please review the table below to ensure all values are valid and correct.')

    user_df = pd.DataFrame([user_data], columns=['Age', 'Pregnancies', 'Glucose', 'BMI', 'Insulin', 'Blood Pressure', 'Skin Thickness', 'DPF value'])
    st.dataframe(user_df, hide_index=True)

    # Make prediction
    if st.button('Predict'):
        diagnosis = diabetes_prediction(user_data)     

def main():
  tab1, tab2 = st.tabs(['Project Overview', 'Diabetes Prediction App'])
  with tab1:
    project_description()
  with tab2:
    prediction_interface()

if __name__ == '__main__':
     main()