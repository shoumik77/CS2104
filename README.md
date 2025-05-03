# CS2104
Shoumik Bisoi CS2104 Project:
This is a Streamlit-based app that predicts the outcome of NBA games based on historical data from Basketball Reference.

Used to:
Predict the Match Winner Between Any Two Teams
Predict Team Probabilities
The model is built using a Random Forest Classifier trained on 2025 season data

FILES:
Model.py
  - Uses Beautiful Soup to scrape data from basketball-reference.com
  - Uses sci-kit for Random Forest Classification

App.py
  - Uses streamlit for frontend GUI purposes.

HOW TO RUN:

git clone https://github.com/shoumik77/CS2104.git
cd CS2104
python3 -m venv .venv
source .venv/bin/activat
