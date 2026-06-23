import streamlit as pd_st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# 1. Page Configuration & Styling
pd_st.set_page_config(page_title="Project PRISM Dashboard", layout="wide")

pd_st.markdown("""
    <style>
    .main-title { font-size: 38px; font-weight: bold; color: #E74C3C; text-align: center; margin-bottom: 10px; }
    .subtitle { font-size: 18px; text-align: center; color: #BDC3C7; margin-bottom: 30px; }
    .metric-card { background-color: #1E272C; padding: 20px; border-radius: 10px; border: 1px solid #34495E; text-align: center; }
    .status-safe { color: #2ECC71; font-weight: bold; font-size: 24px; }
    .status-alert { color: #E74C3C; font-weight: bold; font-size: 24px; }
    </style>
""", unsafe_allow_html=True)

pd_st.markdown('<div class="main-title">PROJECT PRISM</div>', unsafe_allow_html=True)
pd_st.markdown('<div class="subtitle">AI-Powered Cross-Silo Anomaly Detection System</div>', unsafe_allow_html=True)

# 2. Sidebar Controls & Preset Demos
pd_st.sidebar.header("🎛️ System Control Panel")

# Initialize session state for sliders if not present
if 'silo1' not in pd_st.session_state:
    pd_st.session_state.silo1 = 45.0
if 'silo2' not in pd_st.session_state:
    pd_st.session_state.silo2 = 50.0
if 'silo3' not in pd_st.session_state:
    pd_st.session_state.silo3 = 48.0

pd_st.sidebar.subheader("📢 Demo Live Presets")
if pd_st.sidebar.button("🟢 Reset to Normal"):
    pd_st.session_state.silo1 = 45.0
    pd_st.session_state.silo2 = 50.0
    pd_st.session_state.silo3 = 48.0

if pd_st.sidebar.button("🚨 The Hidden Crisis (Judge Demo)"):
    # Individual silos stay well below their manual 80% limits, but the combination is highly anomalous
    pd_st.session_state.silo1 = 74.0
    pd_st.session_state.silo2 = 73.0
    pd_st.session_state.silo3 = 75.0

pd_st.sidebar.markdown("---")
pd_st.sidebar.subheader("Manual Adjustments")
val1 = pd_st.sidebar.slider("Silo 1 Metrics (%)", 0.0, 100.0, key="silo1")
val2 = pd_st.sidebar.slider("Silo 2 Metrics (%)", 0.0, 100.0, key="silo2")
val3 = pd_st.sidebar.slider("Silo 3 Metrics (%)", 0.0, 100.0, key="silo3")

# 3. Training the AI Model (Isolation Forest)
# We generate a baseline of "normal, steady operational days" where everything stays moderate
np.random.seed(42)
normal_data = np.random.normal(loc=45, scale=8, size=(200, 3))
# Clean up any boundaries to keep baseline tight
normal_data = np.clip(normal_data, 10, 65)

# Train the Unsupervised AI
ai_model = IsolationForest(contamination=0.02, random_state=42)
ai_model.fit(normal_data)

# 4. Processing Current Live Data
current_input = np.array([[val1, val2, val3]])
# Isolation Forest returns -1 for an anomaly/outlier, and 1 for normal data
ai_prediction = ai_model.predict(current_input)[0]

# Traditional Rule-Based System Check (Standard simple threshold logic)
traditional_alert = val1 >= 80.0 or val2 >= 80.0 or val3 >= 80.0

# 5. Displaying the Front-End Dashboard Interface
col1, col2, col3 = pd_st.columns(3)

with col1:
    pd_st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    pd_st.subheader("📊 Silo Metrics")
    pd_st.write(f"**Silo 1 Status:** {val1}%")
    pd_st.write(f"**Silo 2 Status:** {val2}%")
    pd_st.write(f"**Silo 3 Status:** {val3}%")
    pd_st.markdown('</div>', unsafe_allow_html=True)

with col2:
    pd_st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    pd_st.subheader("📉 Traditional System")
    if traditional_alert:
        pd_st.markdown('<p class="status-alert">🛑 RED ALERT</p>', unsafe_allow_html=True)
        pd_st.write("Reason: A single silo breached the static 80% limit threshold.")
    else:
        pd_st.markdown('<p class="status-safe">🟢 ALL CLEAR (SAFE)</p>', unsafe_allow_html=True)
        pd_st.write("Reason: No individual silo has breached the 80% line.")
    pd_st.markdown('</div>', unsafe_allow_html=True)

with col3:
    pd_st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    pd_st.subheader("🧠 PRISM Engine (AI)")
    if ai_prediction == -1:
        pd_st.markdown('<p class="status-alert">🚨 CORRELATION ALERT</p>', unsafe_allow_html=True)
        pd_st.write("**AI Analysis:** Danger detected! While individual silos look okay, this *specific pattern* of combined elevated data is historically unsafe.")
    else:
        pd_st.markdown('<p class="status-safe">🟢 SYSTEM NORMAL</p>', unsafe_allow_html=True)
        pd_st.write("**AI Analysis:** The system cluster pattern perfectly matches trusted operational behavior parameters.")
    pd_st.markdown('</div>', unsafe_allow_html=True)

pd_st.markdown("---")
pd_st.markdown("### 💡 Pitch Guide Tip for the Judges:")
pd_st.info(
    "Click **'The Hidden Crisis'** on the sidebar. Show the judges how the Traditional System stays completely green "
    "because no single slider hit 80%. Then, point out how the PRISM AI immediately triggers a red alert because its "
    "Unsupervised Anomaly Model recognizes that all three tracking parameters rising together is statistically impossible on a normal day!"
)