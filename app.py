
import streamlit as st
import numpy as np
import pickle
from gtts import gTTS
import tempfile
import pandas as pd
import os

# Load model and encoders
with open("stroke_model.pkl", "rb") as f:
    model, encoders, feature_names = pickle.load(f)

# Title
st.title("🧠 Stroke Risk Prediction App with History")

st.markdown("Provide your details to assess stroke risk and track past predictions.")

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# Create form
with st.form("stroke_form"):
    inputs = []
    display_values = {}  # Keep original values for display
    for feature in feature_names:
        if feature in encoders:  # Categorical
            options = list(encoders[feature].classes_)[:3]  # Limit to 3 choices
            selected = st.selectbox(f"{feature}", options)
            encoded = encoders[feature].transform([selected])[0]
            inputs.append(encoded)
            display_values[feature] = selected
        else:  # Numeric
            value = st.number_input(f"{feature}", step=0.1)
            inputs.append(value)
            display_values[feature] = value
    submitted = st.form_submit_button("🔍 Predict Stroke Risk")

# Predict
if submitted:
    input_array = np.array(inputs).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    proba = model.predict_proba(input_array)[0][1]  # Probability of stroke

    result = "⚠️ High Risk of Stroke" if prediction == 1 else "✅ Low Risk of Stroke"
    percentage = round(proba * 100, 2)

    # Display
    st.success(f"Prediction: {result}")
    st.info(f"Stroke Risk Probability: {percentage}%")

    # Voice
    tts = gTTS(text=f"Your stroke risk is {percentage} percent. {result}", lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        tts.save(tmpfile.name)
        st.audio(tmpfile.name, format="audio/mp3")

    # Update history
    display_values["Risk %"] = percentage
    display_values["Result"] = result
    st.session_state.history.append(display_values)

# Show history
if st.session_state.history:
    st.subheader("📜 Prediction History")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df)

    # Optionally save to CSV
    if st.button("Download History as CSV"):
        hist_df.to_csv("stroke_prediction_history.csv", index=False)
        with open("stroke_prediction_history.csv", "rb") as f:
            st.download_button("Download", f, file_name="stroke_prediction_history.csv")
