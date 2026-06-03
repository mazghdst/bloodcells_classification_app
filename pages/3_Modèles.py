import streamlit as st
from PIL import Image
import pandas as pd
import json
import numpy as np
import cv2
import matplotlib.patches as patches
from collections import defaultdict
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import plotly.express as px
from utils.config import vspace
from utils.config import (COMMON_CSS, CLASSES_FR, RESULTS_DIR, CLASSIFIER_MAP, MODEL_MAP,)

st.set_page_config(page_title="Modèles",
                   page_icon="🧠", layout="wide")
st.markdown(COMMON_CSS, unsafe_allow_html=True)


@st.cache_data
def load_report(path):
    rows_classes = []
    rows_global  = []
 
    with open(path) as f:
        for line in f:
            parts = line.split()

            if parts[0].isdigit():
                rows_classes.append({
                    "Classe"   : CLASSES_FR[int(parts[0])],
                    "Précision": f"{float(parts[1]):.3f}",
                    "Rappel"   : f"{float(parts[2]):.3f}",
                    "F1-score" : f"{float(parts[3]):.3f}",
                })
            # Ligne accuracy
            elif parts[0] in ("accuracy", "macro", "weighted"):
                label = "Accuracy" if parts[0] == "accuracy" else ("Macro avg" if parts[0] == "macro" else "Weighted avg")
                rows_global.append({
                    "Métrique" : label,
                    "Précision": f"{float(parts[2]):.3f}",
                    "Rappel"   : f"{float(parts[3]):.3f}",
                    "F1-score" : f"{float(parts[4]):.3f}",
                })
 
    return pd.DataFrame(rows_classes), pd.DataFrame(rows_global)


@st.cache_data
def load_kmeans_features():
    df = pd.read_csv(f"./files/kmeans_features.csv")
    clusters = pd.read_csv(f"./files/kmeans_cluster_centers.csv")

    return df, clusters

def plot_hist():

    df, clusters = load_kmeans_features()
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5))

    y = df["target"].astype("int").to_numpy()

    historams = df[[col for col in df.columns if col.startswith("Hist")]]
    class_names = ["BA", "EO", "ERB", "IG", "LY", "MO", "SNE", "PLT"]

    X = historams.to_numpy()
    classes = np.unique(y)
    n_clusters = X.shape[1]
    X = X**2

    class_means = np.array([X[y == c].mean(axis=0) for c in classes])
    i = 0
    for k in range(n_clusters):

        lab = clusters.iloc[k].to_numpy().astype(np.uint8)

        lab_img = lab.reshape(1, 1, 3)
        rgb = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB).reshape(
            3,
        )
        rgb = rgb / 255.0 

        ax = axes[i // 4, i % 4]
        i += 1

        values = class_means[:, k]
        ax.bar(
            classes, values, color="#A6A6A6", alpha=0.6, edgecolor="#6E6E6E", linewidth=0.8
        )

        ax.set_title(f"Cluster {k+1}")
        ax.set_xticks(classes)
        ax.set_xticklabels(class_names, rotation=45, fontsize=9)

        if k < 4:
            rect = patches.Rectangle(
                (0.8, 0.8), 0.15, 0.15, transform=ax.transAxes, color=rgb, clip_on=False
            )
        else:
            rect = patches.Rectangle(
                (0.05, 0.8), 0.15, 0.15, transform=ax.transAxes, color=rgb, clip_on=False
            )

        ax.add_patch(rect)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.tight_layout()
    return fig


def plot_history(history_dict):

    train_color = "#1B365D"
    val_color   = "#B23A48"
    
    loss = history_dict["loss"]
    val_loss = history_dict["val_loss"]
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    epochs = range(1, len(loss) + 1)

    loss_fig, ax1 = plt.subplots(figsize=(4, 3.2), dpi=300)

    ax1.plot(epochs, loss, label="Entraînement",linewidth=1.5, color=train_color)
    ax1.plot(epochs, val_loss, label="Validation",linewidth=1.5,
         marker="o",
         markersize=2, color=val_color)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_xlabel("Époque", fontsize=8.5)
    ax1.tick_params(labelsize=8)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.3, color="#cccccc", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8, frameon=False)
    ax1.set_title(
        "Fontion de perte",
        fontsize=11,
        fontweight="semibold",
        fontfamily="Arial",
        pad=15,
    )
    plt.tight_layout()

    # Accuracy
    acc_fig, ax2 = plt.subplots(figsize=(4, 3.2), dpi=300)
    ax2.plot(epochs, acc, label="Entraînement",linewidth=1.5, color=train_color)
    ax2.plot(epochs, val_acc, label="Validation", linewidth=1.5,
         marker="o",
         markersize=2, color=val_color)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_xlabel("Époque", fontsize=8.5)
    ax2.tick_params(labelsize=8)
    ax2.grid(True, which="both", linestyle="--", linewidth=0.3, color="#cccccc", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=8, frameon=False)
    ax2.set_title(
        "Accuracy",
        fontsize=11,
        fontweight="semibold",
        fontfamily="Arial",
        pad=15,
    )
    plt.tight_layout()
    return loss_fig, acc_fig

@st.cache_data
def read(history_path):

    with open(history_path) as f:
        history = [json.loads(line) for line in f]

    history_dict = defaultdict(list)

    for entry in history:
        for k, v in entry.items():
            history_dict[k].append(v)

    return history_dict


st.title("🧠 Modèles")
st.subheader("Présentation des modèles et résultats associés")

vspace(40)

tab1,tab2, tab3 = st.tabs([
    "Approche ML",
    "Approche DL",
    "Comparaison",
])

with tab1:

    st.subheader("Classifieurs")


    st.markdown("""
    Les classifieurs reçoivent en entrée le **vecteur de 99 features** extrait par le pipeline K-Means.
    """)

    st.dataframe(
        pd.DataFrame({
            "Classifieur"     : ["SVM", "XGBoost", "LGBM", "Voting Classifier"],
            "Nom complet": ["Support Vector Machine", "Extreme Gradient Boosting", "Light Gradient Boosting Machine", "Ensemble par vote"],
            "Détail"     : ["Kernel RBF", "Arbres en séquence", "Arbres en séquence", "Vote soft"],
            "Particularité": [
                "Hyperplan à marge maximale",
                "Croissance niveau par niveau (level-wise)",
                "Croissance feuille par feuille (leaf-wise)",
                "Agrégation des prédictions",
            ],
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("Performances")

    st.markdown("""
    <div style="display:flex; gap:16px; align-items:stretch; margin-bottom:1rem;">
        <div style="background:#f8f8f8; border-radius:10px; padding:16px 24px; flex:2;">
            <div style="font-size:13px; color:#888;">🏆 Meilleur modèle ML</div>
            <div style="font-size:28px; font-weight:500; margin:8px 0;">Voting Classifier</div>
            <div style="font-size:13px; color:#888;">SVM + XGBoost</div>
        </div>
        <div style="background:#f8f8f8; border-radius:10px; padding:16px 24px; flex:1;">
            <div style="font-size:13px; color:#888;">Accuracy</div>
            <div style="font-size:28px; font-weight:500; margin:8px 0;">98.39%</div>
        </div>
        <div style="background:#f8f8f8; border-radius:10px; padding:16px 24px; flex:1;">
            <div style="font-size:13px; color:#888;">F1 macro</div>
            <div style="font-size:28px; font-weight:500; margin:8px 0;">98.51%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


    classifiers_name = ["SVM", "XGBoost", "LGBM", "Voting Classifier"]

    vspace(10)
    selected_ml = st.selectbox("Sélectionner un classifieur", classifiers_name,
                                key="classifier_select")
    
    vspace(20)
    col_left, col_right = st.columns([1, 1.5])
    
    with col_left:
        st.markdown("<div style='text-align: center;'><b>Rapport de classification</b></div>", unsafe_allow_html=True)
        path_rp = f"{RESULTS_DIR}/reports/classification_report_{CLASSIFIER_MAP.get(selected_ml)}.txt"
        df_classes, df_global = load_report(path_rp)

        vspace(20)
        st.dataframe(df_classes, use_container_width=True, hide_index=True)
        st.dataframe(df_global, use_container_width=True, hide_index=True)

    
    with col_right:
        st.markdown("<div style='text-align: center;'><b>Matrice de confusion</b></div>", unsafe_allow_html=True)
        vspace(5)
        path_cm = f"{RESULTS_DIR}/cm/confusion_matrix_{CLASSIFIER_MAP.get(selected_ml)}.png"
        st.image(Image.open(path_cm), width="stretch")

    st.caption("Métriques obtenues sur la base de test (base d'entraînement : 80%, test 20%).")
    with st.expander("Résultats Validation Croisée 5-Fold (CV5)"):
        data_cv5 = {
            "Modèle": ["SVM", "XGBoost", "LGBM", "Voting Classifier"],
            "Accuracy (%)": ["98.12 ± 0.35", "97.66 ± 0.40", "97.93 ± 0.32", "98.29 ± 0.36"],
            "F1 macro (%)": ["98.23 ± 0.37", "97.69 ± 0.46", "98.02 ± 0.37", "98.42 ± 0.37"],
        }
        st.dataframe(data_cv5)
        st.caption("La CV5 n'a pas été appliquée aux modèles DL en raison du coût computationnel.")
    st.divider()

    st.subheader("Interprétabilité - K-Means")

    st.write("La figure ci-dessous donne un aperçu des proportions moyennes obtenues pour chaque cluster. Chaque type de cellule a une signature colorimétrique spécifique à sa morphologie. La couleur de chaque carré correspond à la couleur du centre du cluster, convertie en RGB.")

    vspace(10)

    col1, col2 = st.columns([1, 0.05])
    with col1:
        fig = plot_hist()
        st.pyplot(fig)
    st.caption("BA : basophile, EO : éosinophile,  ERB : érythroblaste, IG : granulocyte immature, LY : lymphocyte, MO : monocyte, SNE : neutrophile, PLT : plaquette.")
 
    vspace(20)

    st.subheader("Interprétabilité - SHAP par classe")
    st.write("Top 10 des features contribuant à la classification des images pour les modèles XGBoost et LGBM.")
    classe = st.selectbox("Classe", list(CLASSES_FR), key="shap")
    cols = st.columns([0.25, 2, 0.4])
    with cols[1]:
        st.image(f"images/shap/shap_{classe}.png")

    st.caption("Lire : 'Hist 1' -> proportion du cluster 1, 'Intra mean 1 L' -> moyenne intra-cluster 1 sur la composante L*, 'r1 hist 1' -> proportion du cluster 1 dans le premier anneau concentrique.")


with tab2:

    st.subheader("Architectures")

    st.markdown("""
    Tous les modèles DL utilisent le **Transfer Learning** : pré-entraînés sur ImageNet,
    puis fine-tunés sur le dataset PBC.
    """)

    st.dataframe(
        pd.DataFrame({
            "Architecture" : ["EfficientNetV2S", "EfficientNetV2M", "ResNet50V2", "DenseNet121", "VGG19", "Xception"],
            "Nom complet"  : [
                "EfficientNet V2 Small",
                "EfficientNet V2 Medium",
                "Residual Network 50 V2",
                "Dense Network 121",
                "Very Deep CNN 19",
                "Extreme Inception",
            ],
            "Paramètres"   : ["21M", "54M", "25M", "8M", "143M", "22M"],
            "Particularité": [
                "Optimisée pour le compromis précision/coût de calcul",
                "Version plus profonde d'EfficientNetV2S",
                "Connexions résiduelles ",
                "Connexions denses",
                "Référence historique",
                "Convolutions séparables en profondeur",
            ],
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("Performances")

    st.markdown("""
    <div style="display:flex; gap:16px; align-items:stretch; margin-bottom:1rem;">
        <div style="background:#f8f8f8; border-radius:10px; padding:16px 24px; flex:2;">
            <div style="font-size:13px; color:#888;">🏆 Meilleur modèle DL</div>
            <div style="font-size:28px; font-weight:500; margin:8px 0;">Ensemble</div>
            <div style="font-size:13px; color:#888;">EfficientNetV2S + VGG19 + ResNet50V2 + Xception</div>
        </div>
        <div style="background:#f8f8f8; border-radius:10px; padding:16px 24px; flex:1;">
            <div style="font-size:13px; color:#888;">Accuracy</div>
            <div style="font-size:28px; font-weight:500; margin:8px 0;">99.21%</div>
        </div>
        <div style="background:#f8f8f8; border-radius:10px; padding:16px 24px; flex:1;">
            <div style="font-size:13px; color:#888;">F1 macro</div>
            <div style="font-size:28px; font-weight:500; margin:8px 0;">99.24%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    model_name = ["EfficientNetV2S", "EfficientNetV2M", "ResNet50V2", "DenseNet121", "VGG19", "Xception", "Ensemble (EfficientNetV2S + VGG19 + ResNet50V2 + Xception)"]

    vspace(10)

    selected_dl = st.selectbox("Sélectionner un modèle", model_name, key="model_select")
    
    vspace(20)

    col_left, col_right = st.columns([1, 1.5])
    
    with col_left:

        st.markdown("<div style='text-align: center;'><b>Rapport de classification</b></div>", unsafe_allow_html=True)
        path_rp = f"{RESULTS_DIR}/reports/classification_report_{MODEL_MAP.get(selected_dl)}.txt"
        df_classes, df_global = load_report(path_rp)

        vspace(20)
        st.dataframe(df_classes, use_container_width=True, hide_index=True)
        st.dataframe(df_global, use_container_width=True, hide_index=True)

    with col_right:
        st.markdown("<div style='text-align: center;'><b>Matrice de confusion</b></div>", unsafe_allow_html=True)
        vspace(5)
        path_cm = f"{RESULTS_DIR}/cm/confusion_matrix_{MODEL_MAP.get(selected_dl)}.png"
        st.image(Image.open(path_cm), width="stretch")

    st.caption("Résultats obtenus sur la même base de test que l'approche ML (base d'entraînement : 60%, validation : 20%, test : 20%).")

    st.divider()

    st.subheader("Courbes d'apprentissage")

    if selected_dl == "Ensemble (EfficientNetV2S + VGG19 + ResNet50V2 + Xception)":
        st.write("Le modèle Ensemble ne dispose pas de courbes d'apprentissage propre, car ses prédictions résultent d'un vote par moyenne pondérée des probabilités (soft voting) des modèles qui le composent, et non d'un entraînement indépendant.")
    if selected_dl != "Ensemble (EfficientNetV2S + VGG19 + ResNet50V2 + Xception)":
        history_path = f"{RESULTS_DIR}/history/history_{MODEL_MAP.get(selected_dl)}.jsonl"
        history_dict = read(history_path)
        loss_fig, acc_fig = plot_history(history_dict)

        col_loss, col_acc, space = st.columns([1, 1, 0.05])
        with col_loss:
            vspace(10) 
            st.pyplot(loss_fig, use_container_width=True)

        with col_acc:
            vspace(10)
            st.pyplot(acc_fig, use_container_width=True)


with tab3:

    st.subheader("Comparaison des performances")


    col1, col2 = st.columns([1,1.5])

    with col1:
        df_scores = pd.DataFrame({
            "Approche": ["ML", "ML", "ML", "ML", "DL", "DL", "DL", "DL", "DL", "DL", "DL"],
            "Modèle": ["SVM", "XGBoost", "LGBM", "Voting Classifier", 
                        "EfficientNetV2S", "EfficientNetV2M", "DenseNet121", 
                        "ResNet50V2", "VGG19", "Xception", "Ensemble"],
            "Accuracy (%)": [98.33, 97.72, 97.83, 98.39, 98.97, 98.92, 98.89, 98.77, 98.97, 98.45, 99.21],  
            "F1 macro (%)": [98.42, 97.72, 97.82, 98.51, 99.02, 98.95, 98.90, 98.87, 99.03, 98.56, 99.24],      
        })

        st.dataframe(df_scores[["Modèle", "Accuracy (%)", "F1 macro (%)"]], use_container_width=True, hide_index=True)

    with col2:

        df_sorted = df_scores.sort_values(["Accuracy (%)", "F1 macro (%)"], ascending=False).reset_index(drop=True)

        fig = px.scatter(
            df_sorted,
            x="Accuracy (%)",
            y="Modèle",
            color="Approche",
            color_discrete_map={"DL": "#1D9E75", "ML": "#888780"},
            hover_data={"Approche": False, "Accuracy (%)": ":.2f", "F1 macro (%)": ":.2f"},
            category_orders={"Modèle": df_sorted["Modèle"].tolist()},
        )

        for _, row in df_sorted.iterrows():
            fig.add_shape(type="line",
                x0=96.5, x1=row["Accuracy (%)"],
                y0=row["Modèle"], y1=row["Modèle"],
                line=dict(color="#9FE1CB" if row["Approche"] == "DL" else "#D3D1C7", width=2),
                layer="below")
            
        fig.update_layout(
            xaxis=dict(range=[96.5, 99.6], ticksuffix="%", gridcolor="#CCCCCC", showline=False, fixedrange=True),
            yaxis=dict(title=None, showgrid=False, showline=False, fixedrange=True),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=400,
            margin=dict(l=140, r=20, t=20, b=50),
            legend=dict(x=0.95, y=0.99, xanchor="left", yanchor="top"),
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.caption("Ensemble : EfficientNetV2S + VGG19 + ResNet50V2 + Xception")
    st.divider()

    st.subheader("Temps computationnels")

    df_temps = pd.DataFrame({
        "": ["Training", "Inférence/image"],
        "ML (CPU)": ["~ 45 s", "~ 2.6 ms"], 
        "DL (GPU)": ["1h – 3h", "10 - 25  ms (batch 32)"],
    })

    st.dataframe(df_temps, use_container_width=True, hide_index=True)
    st.caption("Les deux meilleurs modèles DL (EfficientNetV2S et VGG19) offrent le meilleur compromis : ~2h de training et ~16 ms/image en inférence (batch 32).")