#!/usr/bin/env python
# coding: utf-8

# In[8]:


import streamlit as st
import joblib
import numpy as np
#import xgboost as xgb
# Load your trained model
model = joblib.load("C:\\Users\\victo\\Downloads\\AI_work")


# In[9]:


# Streamlit app
def main():
    st.title('Player Rating Prediction')
    st.markdown('Enter the following attributes to predict the player rating:')
    
    # Define input fields
    movement_reactions = st.number_input('Movement Reactions', min_value=0, step=1)
    mentality_composure = st.number_input('Mentality Composure', min_value=0, step=1)
    passing = st.number_input('Passing', min_value=0, step=1)
    potential = st.number_input('potential', min_value=0, step=1)
    wage_eur = st.number_input('wage_eur', min_value=0, step=1)
    value_eur = st.number_input('value_eur', min_value=0, step=1)
    dribbling = st.number_input('dribbling', min_value=0, step=1)
    attacking_short_passing = st.number_input('attacking_short_passing', min_value=0, step=1)
    mentality_vision = st.number_input('mentality_vision', min_value=0, step=1)
    international_reputation = st.number_input('international_reputation', min_value=0, step=1)
   
    if st.button('Predict'):
        # Prepare input data as numpy array
        input_data = np.array([[international_reputation,mentality_vision,attacking_short_passing,dribbling,value_eur,wage_eur,potential,movement_reactions, mentality_composure, passing]])
        
        # Perform prediction
        predicted_rating = model.predict(input_data)
        
        # Display prediction result
        st.success(f'Predicted Rating: {predicted_rating[0]}')

if __name__ == '__main__':
    main()


# In[ ]:




