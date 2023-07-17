#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import pickle

# Load the trained model from the pickle file
with open('model7.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit app
st.title('Property Evaluation Prediction')

# Input features for prediction
zipcode = st.number_input('ZipCode', min_value=10000, max_value=99999)
bldg_class = st.number_input('Building Class Category (0 for Residential, 1 for Commercial)', min_value=0, max_value=1)
land_area = st.number_input('Land Area (in SqFt)')
gross_area = st.number_input('Gross Area (in SqFt)')
year_of_construction = st.number_input('Year of Construction', min_value=1800, max_value=2023)

# Create a DataFrame from user inputs
input_df = pd.DataFrame({
    'ZipCode': [zipcode],
    'BldgClassCategory': [bldg_class],
    'LandAreaInSqFt': [land_area],
    'GrossAreaInSqFt': [gross_area],
    'YearOfConstruction': [year_of_construction]
})

# Make prediction using the loaded model
prediction = model.predict(input_df)[0]

# Display the predicted property evaluation value
st.subheader('Predicted Property Evaluation Value')
st.write(f'${round(prediction, 2)}')


# In[ ]:




