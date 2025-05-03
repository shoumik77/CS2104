# model.py

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

def scrape_season(year=2025):
    base_url = f"https://www.basketball-reference.com/leagues/NBA_{year}_games"
    months = ['october', 'november', 'december', 'january', 'february', 'march', 'april']
    all_games = []
    headers = {"User-Agent": "Mozilla/5.0"}

    for month in months:
        url = f"{base_url}-{month}.html"
        print(f"Fetching: {url}")
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print(f"❌ Failed to fetch {url} (status {response.status_code})")
            continue

        soup = BeautifulSoup(response.content, "html.parser")
        table = soup.find("table", {"id": "schedule"})

        if table:
            df = pd.read_html(str(table))[0]
            df = df.dropna(subset=["PTS", "PTS.1"])
            all_games.append(df)
            print(f"✅ Added {len(df)} games from {month}")
        else:
            print(f"⚠️ No schedule table found in {month}")

    if not all_games:
        raise ValueError("No game data scraped. Check URLs or if the season has started.")

    season_df = pd.concat(all_games)
    return season_df


def preprocess_data(df):
    # Rename columns for easier access
    df = df.rename(columns={
        "PTS": "Home_PTS",
        "PTS.1": "Away_PTS",
        "Visitor/Neutral": "Away_Team",
        "Home/Neutral": "Home_Team"
    })

    # Only keep relevant columns
    df = df[["Home_Team", "Away_Team", "Home_PTS", "Away_PTS"]]

    # Create labels: 1 = home team wins, 0 = away team wins
    df["Home_Win"] = (df["Home_PTS"] > df["Away_PTS"]).astype(int)

    return df


def train_model(df):
    # Encode team names
    le = LabelEncoder()
    all_teams = pd.concat([df["Home_Team"], df["Away_Team"]])
    le.fit(all_teams)

    X = pd.DataFrame({
        "Home_Team": le.transform(df["Home_Team"]),
        "Away_Team": le.transform(df["Away_Team"])
    })
    y = df["Home_Win"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save model and encoder
    joblib.dump(model, "model.pkl")
    joblib.dump(le, "label_encoder.pkl")
    print("✅ Model and label encoder saved.")

    return model, le


if __name__ == "__main__":
    print("🔎 Scraping 2025 NBA season...")
    raw_data = scrape_season(2025)
    raw_data.to_csv("nba_2025_games.csv", index=False)
    print("📄 Data saved to nba_2025_games.csv")

    print("🧹 Preprocessing data...")
    clean_data = preprocess_data(raw_data)

    print("🎯 Training model...")
    train_model(clean_data)
