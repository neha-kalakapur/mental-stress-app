import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Page settings
st.set_page_config(page_title="Stress Detector", layout="wide")

# Load trained model
model = joblib.load("mental_stress_detector.pkl")

# Title
st.title("🧠 Mental Health Stress Detector")
st.markdown("Enter your daily habits to predict your **stress level (1–5)** and get helpful suggestions.")

st.markdown("---")

# Mobile-friendly layout with columns
col1, col2 = st.columns(2)
with col1:
    sleep = st.slider("🛏️ Sleep Duration (hrs)", 0.0, 12.0, 7.0, step=0.5)
with col2:
    screen = st.slider("📱 Screen Time (hrs)", 0.0, 16.0, 4.0, step=0.5)

col3, col4 = st.columns(2)
with col3:
    meals = st.slider("🍽️ Meals/day", 1, 6, 3)
with col4:
    water = st.slider("💧 Water Intake (litres)", 0.0, 5.0, 2.5, step=0.5)

exercise = st.radio("🏃 Daily Exercise", ["Yes", "No"], horizontal=True)

# Prepare input for model
input_data = pd.DataFrame({
    "sleep_duration_(in_hours)": [sleep],
    "screen_time_per_day(in_hours)": [screen],
    "daily_exercise_(yes/no)": [1 if exercise == "Yes" else 0],
    "number_of_meals/day": [meals],  # ✅ Make sure this matches your model
    "water_intake_(litres)": [water]
})

st.markdown("---")

# Predict button centered
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    if st.button("🔍 Predict Stress Level"):
        result = model.predict(input_data)[0]

        st.success(f"🧠 **Predicted Stress Level (1–5):** {result}")

        # Interpret message
        if result == 1:
            st.markdown("🟢 **Very Low Stress** – You're relaxed! 😌")
            st.markdown("✔️ Tip: Keep maintaining good habits!")
        elif result == 2:
            st.markdown("🟢 **Low Stress** – All good, just watch your balance. 👍")
            st.markdown("✔️ Tip: Stay active and take screen breaks.")
        elif result == 3:
            st.markdown("🟡 **Moderate Stress** – Manageable but needs attention. ⚖️")
            st.markdown("✔️ Tip: Try meditation, reduce screen time, and rest well.")
        elif result == 4:
            st.markdown("🔴 **High Stress** – Your body may be overworked. 🧠💥")
            st.markdown("⚠️ Tip: Prioritize rest, hydrate, and talk to someone.")
        elif result == 5:
            st.markdown("🔴 **Very High Stress** – Take immediate action. 🆘")
            st.markdown("🚨 Tip: Seek help and slow down. Your health matters.")
        else:
            st.warning("Unexpected result. Please check your inputs.")

        st.info("🌟 *Even small lifestyle changes can make a big difference in your mental health.*")

# Visualizations
st.markdown("---")
st.header("📊 Visual Insights (Optional)")

try:
    @st.cache_data
    def load_data():
        return pd.read_csv("cleaned_stress_data.csv")

    data = load_data()

    # Avg Sleep vs Stress
    st.subheader("🛏️ Avg Sleep vs Stress Level")
    st.bar_chart(data.groupby("anxiety_(scale_1–5)")["sleep_duration_(in_hours)"].mean())

    # Screen Time vs Stress
    st.subheader("📱 Screen Time vs Stress Level")
    st.line_chart(data.groupby("anxiety_(scale_1–5)")["screen_time_per_day(in_hours)"].mean())

    # Exercise Pie Chart (Resized)
    st.subheader("🏃 Exercise Distribution")

    # Drop missing values for clean plot
    exercise_counts = data["daily_exercise_(yes/no)"].dropna().value_counts()

    fig, ax = plt.subplots(figsize=(3.5, 3.5))  # Smaller pie chart for mobile
    ax.pie(
        exercise_counts,
        labels=exercise_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#8fd9a8", "#ff9999"]
    )
    ax.set_title("Daily Exercise")
    ax.axis("equal")
    st.pyplot(fig)

except Exception as e:
    st.warning("📂 To enable graphs, make sure 'cleaned_stress_data.csv' exists and column names are correct.")
