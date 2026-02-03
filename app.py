import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load your trained ANN model
model = load_model('model.h5')

# Load encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    lb_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    oh_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit UI
st.title('Customer Churn Prediction')

# Inputs
geography = st.selectbox('Geography', oh_geo.categories_[0])
gender = st.selectbox('Gender', lb_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('No Of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare base input data (as DataFrame)
input_data = pd.DataFrame({
    'CreditScore': [credit_score], 
    'Gender': [lb_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography
geo_encoded = oh_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=oh_geo.get_feature_names_out(['Geography'])
)

# Concatenate numerical + one-hot columns
final_input = pd.concat([input_data, geo_encoded_df], axis=1)

# Scale the input
final_input_scaled = scaler.transform(final_input)

# ANN Prediction
prediction = model.predict(final_input_scaled)
prob = prediction[0][0]

# st.write(f"Churn Probability:{prob:.2f}")

# Output
st.subheader("Prediction Result:")
if prob > 0.5:
    st.error(f"The customer is likely to **CHURN** ({prob:.2f} probability)")
else:
    st.success(f"The customer is **NOT likely to churn** ({prob:.2f} probability)")
