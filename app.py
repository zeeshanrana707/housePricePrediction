import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# App title
st.title("🏡 House Price Prediction App")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("house.csv")
    return df

data = load_data()

# Display raw data
st.subheader("📋 Raw Dataset")
st.dataframe(data.head())

# EDA Section
st.subheader("📊 Exploratory Data Analysis")

with st.expander("Show Histograms"):
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    cols = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors']
    for idx, col in enumerate(cols):
        sns.histplot(data[col], bins=30, ax=ax[idx // 3, idx % 3], kde=True)
        ax[idx // 3, idx % 3].set_title(f'Distribution of {col}')
    st.pyplot(fig)

with st.expander("Show Correlation Heatmap"):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

# Feature selection
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors']
X = data[features]
y = data['price']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.subheader("📈 Model Evaluation")
st.write(f"**R² Score:** {r2:.4f}")
st.write(f"**Mean Absolute Error (MAE):** ${mae:,.2f}")

# Save model and scaler
model_path = "model.pkl"
scaler_path = "scaler.pkl"

with open(model_path, "wb") as f:
    pickle.dump(model, f)

with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)

st.success("✅ Model and scaler saved as 'model.pkl' and 'scaler.pkl'")

# Download buttons
with open(model_path, "rb") as f:
    st.download_button("⬇️ Download Trained Model", f, file_name="model.pkl")

with open(scaler_path, "rb") as f:
    st.download_button("⬇️ Download Scaler", f, file_name="scaler.pkl")

# Sidebar inputs
st.sidebar.header("🔧 Input House Features")
def user_input():
    bedrooms = st.sidebar.slider('Bedrooms', 0, 10, 3)
    bathrooms = st.sidebar.slider('Bathrooms', 0.0, 5.0, 2.0, 0.25)
    sqft_living = st.sidebar.slider('Sqft Living', 300, 10000, 1800)
    sqft_lot = st.sidebar.slider('Sqft Lot', 500, 50000, 7500)
    floors = st.sidebar.slider('Floors', 1, 3, 1)
    return pd.DataFrame([[bedrooms, bathrooms, sqft_living, sqft_lot, floors]], columns=features)

input_df = user_input()

# Prediction
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)

st.subheader("🎯 Prediction Result")
st.write(f"**Estimated House Price:** ${prediction[0]:,.2f}")
