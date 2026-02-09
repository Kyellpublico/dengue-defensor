import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Dengue-Defensor | AI Dashboard",
    page_icon="🦟",
    layout="wide"
)

# --- HEADER & SIDEBAR ---
st.title("🦟 Dengue-Defensor PH")
st.markdown("**AI-Powered Early Warning System for Davao City**")
st.markdown("---")

st.sidebar.header("⚙️ Outbreak Simulator")
st.sidebar.write("Adjust drivers to simulate future risk:")

# --- 1. LOAD DATA & MODEL (Cached for Speed) ---
@st.cache_resource
def load_assets():
    DATA_PATH = 'data/processed/02_features.csv' 
    MODEL_PATH = 'models/dengue_lstm_best.keras'

    if not os.path.exists(DATA_PATH):
        return None, None, None, None
    
    df = pd.read_csv(DATA_PATH, index_col='date', parse_dates=True)
    
    # Setup Scalers (Fit on historical data)
    scaler_X = MinMaxScaler()
    feature_cols = [c for c in df.columns if c != 'cases']
    scaler_X.fit(df[feature_cols])
    
    scaler_y = MinMaxScaler()
    scaler_y.fit(df[['cases']])
    
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
    else:
        model = None
        
    return df, model, scaler_X, scaler_y

df, model, scaler_X, scaler_y = load_assets()

if df is None:
    st.error("❌ Critical Error: Data file not found.")
    st.stop()

# --- 2. SIDEBAR CONTROLS ---
# Get default values from the last historical row
last_row = df.iloc[-1]
default_cases = float(last_row.get('cases_lag1w', 50.0))  # Default to 50 if missing
default_rain = float(last_row.get('rain_roll_sum_8w', 100.0))
default_temp = float(last_row.get('tmean_lag4w', 28.0))

# --- NEW: MOMENTUM SLIDER ---
cases_sim = st.sidebar.slider("⚠️ Current Weekly Cases (Momentum)", 0, 500, int(default_cases))
st.sidebar.caption("☝️ **Tip:** Increasing this has the biggest immediate impact.")

rain_sim = st.sidebar.slider("🌧️ 8-Week Rainfall (mm)", 0.0, 600.0, default_rain)
temp_sim = st.sidebar.slider("🌡️ Avg Temperature (°C)", 24.0, 34.0, default_temp)

# --- 3. PREDICTION ENGINE ---
if model is not None:
    # Prepare input
    input_data = df.iloc[[-1]].copy()
    
    # Update features with slider values
    if 'cases_lag1w' in input_data.columns:
        input_data['cases_lag1w'] = cases_sim  # <--- THIS DRIVES THE MODEL
    if 'rain_roll_sum_8w' in input_data.columns:
        input_data['rain_roll_sum_8w'] = rain_sim
    if 'tmean_lag4w' in input_data.columns:
        input_data['tmean_lag4w'] = temp_sim
    
    # Scale & Reshape
    feature_cols = [c for c in df.columns if c != 'cases']
    X_input = scaler_X.transform(input_data[feature_cols])
    X_input = X_input.reshape((1, 1, len(feature_cols)))
    
    # Predict
    pred_scaled = model.predict(X_input, verbose=0)
    pred_cases = scaler_y.inverse_transform(pred_scaled)[0][0]
    pred_cases = max(0, int(pred_cases))

    # --- 4. DASHBOARD METRICS ---
    col1, col2, col3 = st.columns(3)
    
    # Dynamic Risk Thresholds
    if pred_cases > 150:
        risk_color = "red"
        risk_label = "CRITICAL OUTBREAK"
    elif pred_cases > 80:
        risk_color = "orange"
        risk_label = "HIGH ALERT"
    else:
        risk_color = "green"
        risk_label = "NORMAL"

    col1.metric("📅 Forecast Horizon", "1 Week Ahead")
    col2.metric("🦟 Predicted Cases", f"{pred_cases}", delta_color="inverse")
    col3.markdown(f"### Status: :{risk_color}[{risk_label}]")

    # --- 5. VISUALIZATION ---
    st.subheader("📉 Forecast Trend Analysis")
    fig = go.Figure()
    
    # Historical Data
    history_df = df.tail(52) # Last 1 year
    fig.add_trace(go.Scatter(
        x=history_df.index, y=history_df['cases'],
        mode='lines', name='Historical Cases',
        line=dict(color='gray', width=1)
    ))
    
    # Prediction Point
    next_week = history_df.index[-1] + pd.Timedelta(weeks=1)
    fig.add_trace(go.Scatter(
        x=[next_week], y=[pred_cases],
        mode='markers', name='AI Prediction',
        marker=dict(color='red', size=20, symbol='diamond')
    ))

    fig.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("⚠️ Model not found.")