# =====================================================
# TP2 – IMPLEMENTATION & DEPLOIEMENT DES MODELES ML
# APPLICATION STREAMLIT (VERSION FINALE)
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc,
    mean_absolute_error, mean_squared_error, r2_score
)

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor
)

# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(
    page_title="TP2 - ML & Déploiement",
    layout="wide"
)

st.title("📊 TP2 – Implémentation & Déploiement des Modèles ML")
st.write("Licence MTQ – Introduction à l’Intelligence Artificielle")


"""  
 [, "Expérimentation – Nouveau Dataset" ] : A ete add pour Partie 3
"""
menu = st.sidebar.selectbox(
    "Choisir une partie",
    ["Classification – Census Income", "Régression – Auto MPG","Expérimentation – Nouveau Dataset"]
)

# Créer dossier models si inexistant
if not os.path.exists("models"):
    os.makedirs("models")



# ====================================================
# PARTIE 1 : CLASSIFICATION - CENSUS
# =====================================================
if menu == "Classification – Census Income":

    st.header("🧠 Classification : Census Income")

    st.info("Chargez le fichier census.csv pour entraîner le modèle")

    uploaded_file = st.file_uploader(
        "📂 Charger le fichier census.csv",
        type=["csv"]
    )

    if uploaded_file is None:
        st.warning("Veuillez charger le fichier census.csv pour continuer")
        st.stop()

    data = pd.read_csv(uploaded_file)

    st.subheader("Aperçu des données")
    st.write(data.head())

    # Nettoyage des données
    data.replace(" ?", np.nan, inplace=True)
    data.dropna(inplace=True)

    X = data.drop("Income", axis=1)
    y = data["Income"]

    # Encodage des variables catégorielles
    X = pd.get_dummies(X)

    # Séparation train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Choix du modèle
    model_choice = st.selectbox(
        "Choisir un modèle",
        ["KNN", "Random Forest", "Gradient Boosting"]
    )

    if model_choice == "KNN":
        k = st.slider("Nombre de voisins (k)", 1, 15, 5)
        model = KNeighborsClassifier(n_neighbors=k)

    elif model_choice == "Random Forest":
        n = st.slider("Nombre d'arbres", 50, 300, 100)
        model = RandomForestClassifier(
            n_estimators=n,
            random_state=42
        )

    else:
        model = GradientBoostingClassifier(random_state=42)

    # Entraînement
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"Accuracy du modèle : {acc:.4f}")

    # Matrice de confusion
    st.subheader("Matrice de confusion")
    st.write(confusion_matrix(y_test, y_pred))

    # Courbe ROC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
        # fpr, tpr, _ = roc_curve(y_test, y_score)
        fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=">50K")

        roc_auc = auc(fpr, tpr)

        st.subheader("Courbe ROC")
        st.line_chart(pd.DataFrame({"FPR": fpr, "TPR": tpr}))
        st.write("AUC :", roc_auc)

    # Sauvegarde du modèle
    if st.button("💾 Sauvegarder le modèle (census.pkl)"):
        with open("models/census.pkl", "wb") as f:
            pickle.dump(model, f)
        st.success("Modèle census.pkl sauvegardé avec succès !")

# =====================================================
# PARTIE 2 : REGRESSION - AUTO MPG
# =====================================================
if menu == "Régression – Auto MPG":

    st.header("📈 Régression : Auto MPG")

    st.info("Chargez le fichier auto-mpg.data pour entraîner le modèle")

    uploaded_file = st.file_uploader(
        "📂 Charger le fichier auto-mpg.data",
        type=["data", "txt"]
    )

    if uploaded_file is None:
        st.warning("Veuillez charger le fichier auto-mpg.data pour continuer")
        st.stop()

    columns = [
        "mpg", "cylinders", "displacement", "horsepower",
        "weight", "acceleration", "model_year", "origin", "name"
    ]

    data = pd.read_csv(
        uploaded_file,
        delim_whitespace=True,
        names=columns,
        na_values="?"
    )

    data.dropna(inplace=True)
    data.drop("name", axis=1, inplace=True)

    st.subheader("Aperçu des données")
    st.write(data.head())

    X = data.drop("mpg", axis=1)
    y = data["mpg"]

    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Séparation train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Choix du modèle
    model_choice = st.selectbox(
        "Choisir un modèle",
        ["KNN Regressor", "Random Forest Regressor"]
    )

    if model_choice == "KNN Regressor":
        k = st.slider("Nombre de voisins (k)", 1, 15, 5)
        model = KNeighborsRegressor(n_neighbors=k)
    else:
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )

    # Entraînement
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)

    # Évaluation
    st.subheader("Résultats du modèle")
    st.write("MAE :", mean_absolute_error(y_test, y_pred))
    st.write("MSE :", mean_squared_error(y_test, y_pred))
    st.write("R² :", r2_score(y_test, y_pred))

    # Sauvegarde du modèle
    if st.button("💾 Sauvegarder le modèle (auto-mpg.pkl)"):
        with open("models/auto-mpg.pkl", "wb") as f:
            pickle.dump(model, f)
        st.success("Modèle auto-mpg.pkl sauvegardé avec succès !")




# =====================================================
# PARTIE 3 : NOUVEAU -- DATASET (OPTIMISÉE)
# =====================================================
if menu == "Expérimentation – Nouveau Dataset":
    st.header("🆕 Partie 3 : Expérimentation sur un nouveau dataset")
    st.info("Chargez votre fichier (CSV, TXT, XLSX). Si c'est creditcard.csv, cochez 'Classification'.")

    uploaded_file = st.file_uploader("📂 Charger votre dataset", type=["csv", "txt", "xlsx"])

    if uploaded_file is None:
        st.warning("Veuillez charger votre dataset pour continuer")
        st.stop()

    # Lecture automatique
    ext = os.path.splitext(uploaded_file.name)[1]
    if ext == ".csv":
        data = pd.read_csv(uploaded_file)
    elif ext in [".txt", ".data"]:
        data = pd.read_csv(uploaded_file, delim_whitespace=True)
    elif ext == ".xlsx":
        data = pd.read_excel(uploaded_file)
    
    st.subheader("Aperçu des données")
    st.write(data.head())

    # Sélection de la cible
    target_col = st.selectbox("Sélectionner la colonne cible (target)", data.columns)
    y = data[target_col]
    X = data.drop(target_col, axis=1)

    # --- PRÉTRAITEMENT ---
    # 1. Encodage
    X = pd.get_dummies(X)
    
    # 2. Normalisation (Ajout crucial pour le rapport)
    if st.checkbox("Appliquer la Normalisation (Recommandé)", value=True):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)

    # Séparation train / test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choix de la tâche
    task_type = st.radio("Type de tâche", ["Classification", "Régression"])

    if task_type == "Classification":
        model_choice = st.selectbox("Choisir un modèle", ["KNN", "Random Forest", "Gradient Boosting"])
        
        if model_choice == "KNN":
            k = st.slider("Nombre de voisins (k)", 1, 15, 5)
            model = KNeighborsClassifier(n_neighbors=k)
        elif model_choice == "Random Forest":
            n = st.slider("Nombre d'arbres", 50, 300, 100)
            model = RandomForestClassifier(n_estimators=n, random_state=42)
        else:
            model = GradientBoostingClassifier(random_state=42)

        # Entraînement
        if st.button("Lancer l'entraînement"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            st.success(f"Modèle {model_choice} entraîné avec succès !")
            
            # Métriques détaillées (indispensable pour le rapport sur la fraude)
            from sklearn.metrics import classification_report
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                st.write("**Rapport détaillé :**")
                st.text(classification_report(y_test, y_pred))
            
            with col2:
                st.write("**Matrice de confusion :**")
                st.write(confusion_matrix(y_test, y_pred))

            # Courbe ROC
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
                # On s'assure que y est binaire pour la courbe ROC
                if len(np.unique(y_test)) == 2:
                    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=np.unique(y_test)[1])
                    st.subheader("📈 Courbe ROC")
                    st.line_chart(pd.DataFrame({"FPR": fpr, "TPR": tpr}))
                    st.write("AUC :", auc(fpr, tpr))

    else:  # Régression
        # ... (Garder ton code de régression ici, il est très bien)
        model_choice = st.selectbox("Choisir un modèle", ["KNN Regressor", "Random Forest Regressor"])
        if model_choice == "KNN Regressor":
            k = st.slider("Nombre de voisins (k)", 1, 15, 5)
            model = KNeighborsRegressor(n_neighbors=k)
        else:
            model = RandomForestRegressor(n_estimators=200, random_state=42)