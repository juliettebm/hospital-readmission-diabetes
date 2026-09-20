import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =====================================================================
# CONFIGURATION DE LA PAGE
# =====================================================================
st.set_page_config(
    page_title="Prédicteur de Réadmission Diabète",
    page_icon="🧬",
    layout="centered"
)

# Titre principal et contexte clinique
st.title("🧬 Aide à la Décision Clinique : Risque de Réadmission")
st.markdown("""
Cette interface estime le risque de **réadmission (tous délais confondus)** d'un patient diabétique
avec un modèle **Random Forest** (ROC-AUC ≈ 0.65 sur le jeu de test, découpage par patient).
*Démonstration pédagogique : la performance est modeste et l'outil n'est pas destiné à un usage clinique.*
""")

# =====================================================================
# CHARGEMENT DU MODÈLE ET DES PARAMÈTRES DE FEATURE ENGINEERING
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_artifacts():
    """Modèle + paramètres (scaler, poids) produits par les notebooks 02 et 03."""
    model_path = os.path.join(BASE_DIR, "models", "readmission_model.pkl")
    params_path = os.path.join(BASE_DIR, "models", "feature_params.pkl")
    if not (os.path.exists(model_path) and os.path.exists(params_path)):
        return None, None
    return joblib.load(model_path), joblib.load(params_path)


def build_dims(raw: dict, params: dict) -> pd.DataFrame:
    """Reproduit `build_dims` du notebook 02 : MinMax (train) puis pondération par corrélation."""
    scaler = params["scaler"]
    cols = list(scaler.feature_names_in_)
    scaled = scaler.transform(pd.DataFrame([raw])[cols])
    scaled = pd.DataFrame(scaled, columns=cols).clip(0, 1)  # valeurs hors plage d'entraînement
    return pd.DataFrame([{
        "dim_terrain": scaled["number_diagnoses"].iloc[0],
        "dim_instability": sum(scaled[c].iloc[0] * w for c, w in zip(params["cols_instab"], params["w_instab"])),
        "dim_severity": sum(scaled[c].iloc[0] * w for c, w in zip(params["cols_sev"], params["w_sev"])),
    }])


model, params = load_artifacts()
if model is None:
    st.error("Modèle introuvable : exécutez les notebooks 01 à 03 (dossier `notebooks/`) "
             "pour générer `models/readmission_model.pkl` et `models/feature_params.pkl`.")
    st.stop()

# =====================================================================
# INTERFACE UTILISATEUR : SAISIE DES DONNÉES DU PATIENT
# =====================================================================
st.header("📋 Profil Clinique du Patient")

# Division en onglets ou sections pour refléter tes 3 dimensions cliniques
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Terrain Pathologique")
    number_diagnoses = st.number_input(
        "Nombre de diagnostics enregistrés (Comorbidités)", 
        min_value=1, max_value=20, value=5, step=1,
        help="Concept autonome mesurant le fardeau pathologique de fond du patient."
    )

    st.subheader("2. Sévérité de l'Épisode Actuel")
    time_in_hospital = st.slider("Durée du séjour en cours (jours)", 1, 14, 3)
    num_lab_procedures = st.number_input("Nombre d'examens de laboratoire", min_value=1, max_value=120, value=35)
    num_medications = st.number_input("Nombre de médicaments prescrits", min_value=1, max_value=100, value=15)

with col2:
    st.subheader("3. Instabilité Chronique (Passif)")
    number_inpatient = st.number_input(
        "Hospitalisations précédentes (12 derniers mois)", 
        min_value=0, max_value=20, value=0
    )
    number_emergency = st.number_input(
        "Passages aux urgences (12 derniers mois)", 
        min_value=0, max_value=20, value=0
    )
    number_outpatient = st.number_input(
        "Consultations externes (12 derniers mois)", 
        min_value=0, max_value=20, value=0
    )

# =====================================================================
# CALCUL DES 3 DIMENSIONS ET PRÉDICTION
# =====================================================================
st.markdown("---")

raw_inputs = {
    "number_diagnoses": number_diagnoses,
    "number_inpatient": number_inpatient,
    "number_emergency": number_emergency,
    "number_outpatient": number_outpatient,
    "time_in_hospital": time_in_hospital,
    "num_lab_procedures": num_lab_procedures,
    "num_medications": num_medications,
}
patient_data = build_dims(raw_inputs, params)
n_contacts = number_inpatient + number_emergency + number_outpatient

# Bouton de déclenchement de l'analyse
if st.button("🚀 Calculer le Risque de Réadmission", type="primary"):

    proba = model.predict_proba(patient_data[list(model.feature_names_in_)])[0][1] * 100

    # Affichage du résultat sous forme de jauge ou de score visuel
    st.header("📊 Résultat de l'Évaluation")
    
    # Choix de la couleur en fonction du niveau de risque
    if proba < 40:
        st.success(f"**Risque Faible : {proba:.1f}%**")
    elif proba < 65:
        st.warning(f"**Risque Modéré : {proba:.1f}%**")
    else:
        st.error(f"**Risque Élevé : {proba:.1f}%**")

    # =====================================================================
    # EXPLICABILITÉ : importances du modèle entraîné (pas de valeur codée en dur)
    # =====================================================================
    st.subheader("💡 Éléments d'explicabilité")
    importances = dict(zip(model.feature_names_in_, model.feature_importances_))
    st.markdown(
        f"Importances globales du modèle : instabilité chronique **{importances['dim_instability']:.0%}**, "
        f"sévérité de l'épisode **{importances['dim_severity']:.0%}**, terrain pathologique **{importances['dim_terrain']:.0%}**."
    )
    if n_contacts > 2:
        st.markdown(f"⚠️ Le patient présente {n_contacts} contacts récents avec le système de soins : "
                    "c'est la dimension la plus influente du modèle.")
    else:
        st.markdown("🔹 Le recours antérieur aux soins est faible, ce qui tire le risque vers le bas.")
    st.write("**Scores des dimensions (0 à 1) :** "
             f"terrain {patient_data['dim_terrain'].iloc[0]:.2f} · "
             f"instabilité {patient_data['dim_instability'].iloc[0]:.2f} · "
             f"sévérité {patient_data['dim_severity'].iloc[0]:.2f}")
    st.caption("Estimation statistique à visée pédagogique, non validée cliniquement.")
