import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os

st.set_page_config(page_title="Analyse interactive football", layout="wide")
st.title("⚽ Analyse interactive des matchs du week-end")

# ---------------- CONFIG API ----------------
API_KEY = "94ab52893fe364d9bf5362dc7b752213"  # Remplace par ta clé The Odds API
SPORTS = {
    "Ligue 1": "soccer_fra_ligue_one",
    "Premier League": "soccer_eng_premier_league",
    "La Liga": "soccer_spain_la_liga"
}

# ---------------- STOCKAGE DES FORMES ----------------
FORM_FILE = "teams_form.json"
if os.path.exists(FORM_FILE):
    with open(FORM_FILE, "r") as f:
        teams_form = json.load(f)
else:
    teams_form = {league: {} for league in SPORTS.keys()}

# ---------------- FONCTION POUR CHARGER MATCHS ----------------
@st.cache_data(ttl=600)
def get_upcoming_matches(sport_key):
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?regions=eu&markets=h2h&apiKey={API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        matches = []
        for game in data:
            if "bookmakers" in game and len(game["bookmakers"]) > 0:
                b365 = next((b for b in game["bookmakers"] if b["key"] == "bet365"), None)
                if b365:
                    odds = b365["markets"][0]["outcomes"]
                    home_odds = next((o["price"] for o in odds if o["name"] == game["home_team"]), None)
                    away_odds = next((o["price"] for o in odds if o["name"] == game["away_team"]), None)
                    matches.append({
                        "home_team": game["home_team"],
                        "away_team": game["away_team"],
                        "cote_home": home_odds,
                        "cote_away": away_odds
                    })
        return pd.DataFrame(matches)
    except Exception as e:
        st.error(f"Erreur API : {e}")
        return pd.DataFrame()

# ---------------- FONCTION FORMES ----------------
def parse_forme(sequence):
    mapping = {"v":3,"n":1,"d":0}
    seq = [mapping.get(x.strip().lower(),0) for x in sequence.split(",")]
    if len(seq)<5:
        seq += [0]*(5-len(seq))
    weights = np.array([5,4,3,2,1])
    return np.dot(seq,weights)/15  # Normalisé 0-1

# ---------------- INTERFACE STREAMLIT ----------------
st.subheader("Forme récente des équipes (derniers 5 matchs)")
for league in SPORTS.keys():
    st.markdown(f"### {league}")
    upcoming = get_upcoming_matches(SPORTS[league])
    if upcoming.empty:
        st.info("Aucun match disponible")
        continue

    # Afficher les équipes et leur forme
    teams = set(upcoming["home_team"]).union(set(upcoming["away_team"]))
    for team in teams:
        current_form = teams_form[league].get(team,"v,v,v,n,d")
        new_form = st.text_input(f"{team} (du plus récent au moins récent)", value=current_form, key=f"{league}_{team}")
        teams_form[league][team] = new_form

# Bouton pour sauvegarder les formes
if st.button("Enregistrer les formes"):
    with open(FORM_FILE,"w") as f:
        json.dump(teams_form,f)
    st.success("✅ Formes sauvegardées")

# ---------------- ANALYSE ----------------
st.subheader("Analyse des matchs et mise conseillée")
for league, sport_key in SPORTS.items():
    st.markdown(f"### {league}")
    df_matches = get_upcoming_matches(sport_key)
    if df_matches.empty:
        st.info("Aucun match disponible")
        continue

    # Ajouter la forme à partir du fichier
    df_matches["home_form"] = df_matches["home_team"].apply(lambda x: parse_forme(teams_form[league].get(x,"v,v,v,n,d")))
    df_matches["away_form"] = df_matches["away_team"].apply(lambda x: parse_forme(teams_form[league].get(x,"v,v,n,d,d")))

    # Score de sécurité
    df_matches["score_securite"] = (
        ((1/abs(df_matches["cote_home"] - df_matches["cote_away"] + 0.01))*40) +
        ((df_matches["home_form"] - df_matches["away_form"])*100*20)
    ).clip(0,100)

    # Probabilité implicite et winner
    df_matches["prob_home"] = np.exp(df_matches["score_securite"])/(np.exp(df_matches["score_securite"])+np.exp(100-df_matches["score_securite"]))
    df_matches["prob_away"] = 1 - df_matches["prob_home"]
    df_matches["Winner"] = np.where(df_matches["prob_home"]>df_matches["prob_away"], df_matches["home_team"], df_matches["away_team"])

    # Classement du plus sûr au moins sûr
    df_matches = df_matches.sort_values(by="score_securite", ascending=False)
    st.dataframe(df_matches[["home_team","away_team","cote_home","cote_away","Winner","score_securite"]])

    # Mise conseillée (Kelly simplifié)
    budget_total = st.number_input(f"Budget total (€) pour {league}", 50, 10000, 100)
    mises = []
    for _, row in df_matches.iterrows():
        cote = row["cote_home"] if row["Winner"]==row["home_team"] else row["cote_away"]
        p = row["prob_home"] if row["Winner"]==row["home_team"] else row["prob_away"]
        b = cote - 1
        q = 1-p
        f_star = max((b*p - q)/b,0)
        mises.append(round(f_star*budget_total,2))
    df_matches["Mise conseillée (€)"] = mises
    st.dataframe(df_matches[["home_team","away_team","Winner","prob_home","prob_away","Mise conseillée (€)"]])

