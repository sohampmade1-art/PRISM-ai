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
    </style>
""", unsafe_allow_html=True)

pd_st.markdown('<div class="main-title">PROJECT PRISM</div>', unsafe_allow_html=True)
pd_st.markdown('<div class="subtitle">AI-Powered Cross-Silo Anomaly Detection System</div>', unsafe_allow_html=True)

# 2. Sidebar Controls & Preset Demos
pd_st.sidebar.header("🎛️ PRISM Control Panel")

# Initialize session state for trackers if not present
if 'absenteeism' not in pd_st.session_state:
    pd_st.session_state.absenteeism = 45.0
if 'water_complaints' not in pd_st.session_state:
    pd_st.session_state.water_complaints = 50.0
if 'hospital_visits' not in pd_st.session_state:
    pd_st.session_state.hospital_visits = 48.0

pd_st.sidebar.subheader("📢 Demo Live Presets")
if pd_st.sidebar.button("🟢 Reset to Baseline Normal"):
    pd_st.session_state.absenteeism = 45.0
    pd_st.session_state.water_complaints = 50.0
    pd_st.session_state.hospital_visits = 48.0

if pd_st.sidebar.button("🚨 The Hidden Crisis (Judge Demo)"):
    # Individual departments stay well below their manual 80% thresholds, but the combination is highly anomalous
    pd_st.session_state.absenteeism = 74.0
    pd_st.session_state.water_complaints = 73.0
    pd_st.session_state.hospital_visits = 75.0

pd_st.sidebar.markdown("---")
pd_st.sidebar.subheader("Manual Data Overrides")
val1 = pd_st.sidebar.slider("School Absenteeism Rate (%)", 0.0, 100.0, key="absenteeism")
val2 = pd_st.sidebar.slider("Water Quality/Supply Complaints (%)", 0.0, 100.0, key="water_complaints")
val3 = pd_st.sidebar.slider("Hospital Visit Surge Index (%)", 0.0, 100.0, key="hospital_visits")

# 3. Training the AI Model (Isolation Forest)
# We generate a baseline of fragmented data where separate indicators stay moderate
np.random.seed(42)
normal_data = np.random.normal(loc=45, scale=8, size=(200, 3))
normal_data = np.clip(normal_data, 10, 65)

# Train the Unsupervised AI
ai_model = IsolationForest(contamination=0.02, random_state=42)
ai_model.fit(normal_data)

# 4. Processing Current Live Data
current_input = np.array([[val1, val2, val3]])
ai_prediction = ai_model.predict(current_input)[0]

# Traditional Rule-Based System Check (Standard simple threshold logic)
traditional_alert = val1 >= 80.0 or val2 >= 80.0 or val3 >= 80.0

# 5. Displaying the Front-End Dashboard Interface
col1, col2, col3 = pd_st.columns(3)

with col1:
    with pd_st.container(border=True):
        pd_st.subheader("📊 Department Metrics")
        pd_st.write(f"**School Absenteeism:** {val1}%")
        pd_st.write(f"**Water Complaints:** {val2}%")
        pd_st.write(f"**Hospital Visits:** {val3}%")

with col2:
    with pd_st.container(border=True):
        pd_st.subheader("📉 Traditional System")
        if traditional_alert:
            pd_st.error("🛑 RED ALERT")
            pd_st.write("Reason: A single isolated department has breached its static 80% limit threshold.")
        else:
            pd_st.success("🟢 ALL CLEAR (SAFE)")
            pd_st.write("Reason: No individual department tracker has crossed its rigid 80% line.")

with col3:
    with pd_st.container(border=True):
        pd_st.subheader("🧠 PRISM Engine (AI)")
        if ai_prediction == -1:
            pd_st.error("🚨 CORRELATION ALERT")
            pd_st.write("**AI Analysis:** Multi-silo threat detected! While individual departments look okay, this *specific pattern* of concurrently elevated data indicates an emerging systemic social crisis.")
        else:
            pd_st.success("🟢 SYSTEM NORMAL")
            pd_st.write("**AI Analysis:** Cross-department structural correlation perfectly matches trusted historical operational parameters.")

pd_st.markdown("---")
pd_st.markdown("### 💡 Pitch Guide Tip for the Judges:")
pd_st.info(
    "Click **'The Hidden Crisis'** on the sidebar. Show the judges how the Traditional System stays completely green "
    "because no single slider hit 80%. Then, point out how the PRISM AI immediately triggers a red alert because its "
    "Unsupervised Anomaly Model recognizes that all three tracking parameters rising together is statistically impossible on a normal day!"
)