import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Analyse auto football-data", layout="wide")
st.title("⚽ Analyse automatique des championnats (Ligue 1, Premier League, La Liga)")

# URLs officielles football-data.co.uk (saison 2024–25)
urls = {
    "Ligue 1": "https://www.football-data.co.uk/mmz4281/2425/F1.csv",
    "Premier League": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "La Liga": "https://www.football-data.co.uk/mmz4281/2425/SP1.csv"
}

@st.cache_data
def load_data():
    data = {}
    for league, url in urls.items():
        try:
            df = pd.read_csv(url)
            df = df.rename(columns={
                "HomeTeam": "home_team",
                "AwayTeam": "away_team",
                "FTHG": "home_goals",
                "FTAG": "away_goals",
                "FTR": "result",
                "B365H": "cote_home",
                "B365D": "cote_draw",
                "B365A": "cote_away"
            })
            df["league"] = league
            df = df.dropna(subset=["cote_home", "cote_away"])
            data[league] = df
        except Exception as e:
            st.warning(f"⚠️ Erreur chargement {league}: {e}")
    return data

data = load_data()

if not data:
    st.error("Impossible de charger les données.")
    st.stop()

# Analyse
def analyze_league(df, league):
    df = df.copy()
    df["winner"] = np.where(df["result"] == "H", df["home_team"],
                            np.where(df["result"] == "A", df["away_team"], "Draw"))

    # Probabilité implicite à partir des cotes
    df["p_home"] = 1 / df["cote_home"]
    df["p_away"] = 1 / df["cote_away"]
    df["p_norm"] = df["p_home"] + df["p_away"]
    df["p_home"] /= df["p_norm"]
    df["p_away"] /= df["p_norm"]

    # Score de fiabilité (diff. entre proba & résultat réel)
    df["pred_correct"] = (
        (df["p_home"] > df["p_away"]) & (df["result"] == "H")
    ) | (
        (df["p_home"] < df["p_away"]) & (df["result"] == "A")
    )

    precision = round(df["pred_correct"].mean() * 100, 2)
    avg_cote = round(df["cote_home"].mean(), 2)

    st.markdown(f"### 📊 {league}")
    st.write(f"Précision bookmaker (cotes vs résultats réels) : **{precision}%**")
    st.write(f"Cote moyenne observée : **{avg_cote}**")

    # Sélection de matchs “sûrs”
    df["écart"] = abs(df["p_home"] - df["p_away"])
    safe_matches = df[df["écart"] > 0.25].tail(10)

    st.dataframe(safe_matches[["Date", "home_team", "away_team",
                               "cote_home", "cote_away", "winner", "écart"]],
                 use_container_width=True)

for league, df in data.items():
    analyze_league(df, league)
