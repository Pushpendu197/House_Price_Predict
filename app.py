import pandas as pd 
import pickle as pk
import streamlit as st

model = pk.load(open('price_predict.pkl', 'rb'))

st.header('Bangalore House Price Predictor')
data = pd.read_csv('Cleaned_data.csv')

loc = st.selectbox('Choose the Location', data['location'].unique())
sqft = st.number_input('Enter total Sqft')
beds = st.number_input('Enter Number of Bedrooms')
bal = st.number_input('Enter Number of Balconies')
bath = st.number_input('Enter Number of Bathrooms')

input_df = pd.DataFrame([[loc, sqft, bath, bal, beds]],
                        columns=['location', 'total_sqft', 'bath', 'balcony', 'bedrooms'])

if st.button('Predict'):
    output = model.predict(input_df)
    out_str = 'Predicted Price of the House is ₹ ' + str(int(output[0] * 100000))
    st.success(out_str)  # <- This line displays the result!
