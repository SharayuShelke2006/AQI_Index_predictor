
# Air Quality Index Prediction

## About the Project

This project predicts Air Quality Index (AQI) using
pollutant concentrations and Machine Learning algorithms.

The project compares three regression algorithms:

1. Multiple Linear Regression
2. Support Vector Regression (SVR)
3. Random Forest Regression

The best-performing model is saved and integrated
with a Flask-based website.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- HTML
- CSS

## Input Features

- PM2.5
- PM10
- NO2
- SO2
- CO
- O3

## Project Structure

AQI_Prediction/
- app.py
- model.pkl
- requirements.txt
- templates/index.html
- static/style.css

## How to Run

1. Clone or download the project.

2. Install dependencies:
   
      flask
      pandas
      numpy
      scikit-learn
      joblib

3. Make sure model.pkl is in the main project folder.

4. Start the Flask application:

   python app.py

5. Open your browser:

   http://127.0.0.1:5000

## Features

- AQI prediction using Machine Learning
- Comparison of regression algorithms
- Simple web interface
- AQI category classification

## Disclaimer

This is an educational project.
Predictions are estimates and are not official
air quality measurements or health advice.
