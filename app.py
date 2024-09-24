import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
model_public_transit = joblib.load('public_transit.pkl')
model_ridesharing = joblib.load('ridesharing.pkl')
model_cycling = joblib.load('cycling.pkl')


df = pd.read_csv('transportation_demand_dataset.csv')


st.title('Transportation Demand Forecasting')



st.sidebar.header('Input Features')

population = st.sidebar.number_input('Population', min_value=0, value=10000)
median_income = st.sidebar.number_input('Median Income', min_value=0, value=50000)
employment_rate = st.sidebar.number_input('Employment Rate (%)', min_value=0.0, max_value=100.0, value=75.0)
bus_routes = st.sidebar.number_input('Number of Bus Routes', min_value=0, value=5)
train_stations = st.sidebar.number_input('Number of Train Stations', min_value=0, value=2)
bike_lanes = st.sidebar.number_input('Bike Lanes (m)', min_value=0, value=20)
road_quality_index = st.sidebar.number_input('Road Quality Index (1-10)', min_value=1, max_value=10, value=7)
weather_score = st.sidebar.number_input('Weather Score (1-10)', min_value=1, max_value=10, value=6)
event_count = st.sidebar.number_input('Event Count', min_value=0, value=2)



input_data = np.array([[population, median_income, employment_rate, bus_routes, 
                        train_stations, bike_lanes, road_quality_index, 
                        weather_score, event_count]])



if st.button('Predict Public Transit Riders'):
    prediction_public_transit = model_public_transit.predict(input_data)
    st.write(f'Predicted Public Transit Riders: {int(prediction_public_transit[0])}')




if st.button('Predict Ridesharing Trips'):
    prediction_ridesharing = model_ridesharing.predict(input_data)
    st.write(f'Predicted Ridesharing Trips: {int(prediction_ridesharing[0])}')




if st.button('Predict Cycling/Walking Trips'):
    prediction_cycling = model_cycling.predict(input_data)
    st.write(f'Predicted Cycling/Walking Trips: {int(prediction_cycling[0])}')
