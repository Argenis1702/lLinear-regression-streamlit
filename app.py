import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Cargar datos
df = pd.read_csv("Advertising.csv")

# Entrenar modelo
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
model = LinearRegression()
model.fit(X, y)

# Interfaz en Streamlit
st.title("Advertising Sales Prediction")

st.sidebar.header("Enter Advertising Budget")
tv_budget = st.sidebar.number_input("TV Budget ($)", min_value=0, max_value=500, value=100)
radio_budget = st.sidebar.number_input("Radio Budget ($)", min_value=0, max_value=500, value=50)
newspaper_budget = st.sidebar.number_input("Newspaper Budget ($)", min_value=0, max_value=500, value=25)

# Predicción
input_data = np.array([[tv_budget, radio_budget, newspaper_budget]])
predicted_sales = model.predict(input_data)[0]

st.subheader("Predicted Sales:")
st.write(f"${predicted_sales:.2f}")
