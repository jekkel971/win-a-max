import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os

st.set_page_config(page_title="Analyse football API robuste", layout="wide")
st.title("⚽ Analyse des matchs avec API (robuste)")

# ---------------- CONFIG API ----------------
API_KEY = "94ab52893fe364d9bf5362dc7b752213"  # Remplace par ta vraie clé
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
        if response.status_code == 401:
            st.warning("⚠️ Clé API invalide ou non autorisée.")
            return pd.DataFrame()
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
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur API : {e}")
        return pd.DataFrame()

# ---------------- FONCTION FORMES ----------------
def parse_forme(sequence):
    mapping = {"v":3,"n":1,"d":0}
    seq = [mapping.get(x.strip().lower(),0) for x in sequence.split(",")]
    if len(seq)<5:
        seq += [0]*(5-len(seq))
    weights = np.array([5,4,3,2,1])
    return np.dot(seq,weights)/15

# ---------------- SAISIE FORMES ----------------
st.subheader("Forme récente des équipes (5 derniers matchs)")
for league in SPORTS.keys():
    st.markdown(f"### {league}")
    df_matches = get_upcoming_matches(SPORTS[league])
    if df_matches.empty:
        st.info("Aucun match récupéré depuis l’API. Tu peux saisir les matchs manuellement ci-dessous.")
        continue

    teams = set(df_matches["home_team"]).union(set(df_matches["away_team"]))
    for team in teams:
        current_form = teams_form[league].get(team,"v,v,v,n,d")
        new_form = st.text_input(f"{team} (du plus récent au moins récent)", value=current_form, key=f"{league}_{team}")
        teams_form[league][team] = new_form

if st.button("Enregistrer les formes"):
    with open(FORM_FILE,"w") as f:
        json.dump(teams_form,f)
    st.success("✅ Formes sauvegardées")

# ---------------- ANALYSE ----------------
st.subheader("Analyse des matchs et mise conseillée")
for league, sport_key in SPORTS.items():
    st.markdown(f"### {league}")
    df_matches = get_upcoming_matches(sport_key)
    
    # Si pas de match depuis API, proposer saisie manuelle
    if df_matches.empty:
        st.info("Aucun match disponible via API pour ce championnat. Tu peux entrer les matchs manuellement ci-dessous.")
        num_matches = st.number_input(f"Nombre de matchs pour {league}", 1, 20, 3)
        manual_matches = []
        for i in range(num_matches):
            home = st.text_input(f"Équipe domicile {i+1}", key=f"{league}_home_{i}")
            away = st.text_input(f"Équipe extérieur {i+1}", key=f"{league}_away_{i}")
            cote_home = st.number_input(f"Cote domicile {i+1}", min_value=1.01, max_value=10.0, value=1.5, step=0.01, key=f"{league}_ch_{i}")
            cote_away = st.number_input(f"Cote extérieur {i+1}", min_value=1.01, max_value=10.0, value=2.5, step=0.01, key=f"{league}_ca_{i}")
            home_form = st.text_input(f"Forme domicile {i+1} (v,n,d,d,v)", value="v,v,n,d,d", key=f"{league}_hf_{i}")
            away_form = st.text_input(f"Forme extérieur {i+1} (v,n,d,d,v)", value="v,n,d,d,v", key=f"{league}_af_{i}")
            manual_matches.append({
                "home_team": home,
                "away_team": away,
                "cote_home": cote_home,
                "cote_away": cote_away,
                "home_form": home_form,
                "away_form": away_form
            })
        df_matches = pd.DataFrame(manual_matches)

    # Ajouter la forme si manuelle ou API
    df_matches["home_form_val"] = df_matches["home_form"].apply(lambda x: parse_forme(x))
    df_matches["away_form_val"] = df_matches["away_form"].apply(lambda x: parse_forme(x))

    df_matches["score_securite"] = (
        ((1/abs(df_matches["cote_home"] - df_matches["cote_away"] + 0.01))*40) +
        ((df_matches["home_form_val"] - df_matches["away_form_val"])*100*20)
    ).clip(0,100)

    df_matches["prob_home"] = np.exp(df_matches["score_securite"])/(np.exp(df_matches["score_securite"])+np.exp(100-df_matches["score_securite"]))
    df_matches["prob_away"] = 1 - df_matches["prob_home"]
    df_matches["Winner"] = np.where(df_matches["prob_home"]>df_matches["prob_away"], df_matches["home_team"], df_matches["away_team"])

    df_matches = df_matches.sort_values(by="score_securite", ascending=False)
    st.dataframe(df_matches[["home_team","away_team","cote_home","cote_away","Winner","score_securite"]])

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

