import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import streamlit as st
warnings.filterwarnings('ignore')

df = pd.read_csv('bodyfat.csv')

# Create the features and target variables
X = df[["age", "weight", "height", "neck", "chest", "abdomen", "hip", "thigh", "knee", "ankle", "biceps", "forearm", "wrist"]]
y = df["body_fat"]

# Create the linear regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Create the Random Forest model
rf_model = RandomForestRegressor(n_estimators = 100, random_state = 45)

# Train the Random Forest model
rf_model.fit(X, y)

# Create the Decision Tree model
dt_model = DecisionTreeRegressor()

# Train the Decision Tree model
dt_model.fit(X, y)

# Create the Streamlit app
st.title("Body Fat Prediction")

# Get the user input
age = st.slider("Age", min=18, max=80, value=25)
weight = st.number_input("Weight (lbs)", value=180)
height = st.number_input("Height (in)", value=72)
neck = st.number_input("Neck circumference (in)", value=15)
chest = st.number_input("Chest circumference (in)", value=40)
abdomen = st.number_input("Abdominal circumference (in)", value=35)
hip = st.number_input("Hip circumference (in)", value=45)
thigh = st.number_input("Thigh circumference (in)", value=30)
knee = st.number_input("Knee circumference (in)", value=18)
ankle = st.number_input("Ankle circumference (in)", value=10)
biceps = st.number_input("Biceps circumference (in)", value=12)
forearm = st.number_input("Forearm circumference (in)", value=10)
wrist = st.number_input("Wrist circumference (in)", value=7)

# Predict the body fat percentage using the linear regression model
predicted_body_fat_lr = model.predict([[age, weight, height, neck, chest, abdomen, hip, thigh, knee, ankle, biceps, forearm, wrist]])

# Predict the body fat percentage using the Random Forest model
predicted_body_fat_rf = rf_model.predict([[age, weight, height, neck, chest, abdomen, hip, thigh, knee, ankle, biceps, forearm, wrist]])

# Predict the body fat percentage using the Decision Tree model
predicted_body_fat_dt = dt_model.predict([[age, weight, height, neck, chest, abdomen, hip, thigh, knee, ankle, biceps, forearm, wrist]])

# Display the predictions
st.write(f"The predicted body fat percentage using linear regression is {predicted_body_fat_lr[0]:.2f}%")
st.write(f"The predicted body fat percentage using Random Forest is {predicted_body_fat_rf[0]:.2f}%")
# st.write(f"The predicted body fat percentage using Decision Tree is {predicted_body_fat_dt[0]:.2f}%")
