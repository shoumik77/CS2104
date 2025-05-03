# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and label encoder
model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="NBA Game Predictor", page_icon="🏀")

# Custom title
st.markdown("<h1 style='text-align: center;'>🏀 NBA Game Predictor 2025 🏆</h1>", unsafe_allow_html=True)
st.markdown("#### Select two teams to predict the winner and see win probabilities.")

teams = le.classes_

# Column layout for team pickers
col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox("🏠 Home Team", teams, index=0)

with col2:
    away_team = st.selectbox("🛫 Away Team", teams, index=1)

# Spacing
st.markdown("---")

# Prediction button
predict_btn = st.button("🔮 Predict Outcome")

if predict_btn:
    if home_team == away_team:
        st.error("Please select **two different teams**.")
    else:
        # Encode and predict
        home_encoded = le.transform([home_team])[0]
        away_encoded = le.transform([away_team])[0]

        probabilities = model.predict_proba(np.array([[home_encoded, away_encoded]]))[0]
        winner = home_team if probabilities[1] > probabilities[0] else away_team

        st.markdown(f"### 🏆 Predicted Winner: **{winner}**")

        # Show probabilities
        prob_df = pd.DataFrame({
            "Team": [away_team, home_team],
            "Win Probability": [probabilities[0], probabilities[1]]
        })

        st.bar_chart(
            prob_df.set_index("Team"),
            use_container_width=True
        )

        # Optional: show emoji-based message
        if probabilities[1] > 0.8 or probabilities[0] > 0.8:
            st.info("Looks like a strong favorite in this matchup! 💪")
        elif 0.45 < probabilities[0] < 0.55:
            st.warning("Tight matchup — could go either way! ⚖️")
