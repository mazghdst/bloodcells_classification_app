import streamlit as st
from PIL import Image
import pandas as pd
import json
from collections import defaultdict
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

def plot_history(history_dict):

    loss = history_dict["loss"]
    val_loss = history_dict["val_loss"]
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    epochs = range(1, len(loss) + 1)

    loss_fig, ax1 = plt.subplots(figsize=(4.5, 3.5), dpi=300)

    ax1.plot(epochs, loss, label="Train Loss")
    ax1.plot(epochs, val_loss, label="Val Loss")
    ax1.set_xlabel("Époque")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.3, color="#cccccc", alpha=0.5)
    ax1.legend()
    plt.tight_layout()

    # Accuracy
    acc_fig, ax2 = plt.subplots(figsize=(4.5, 3.5), dpi=300)
    ax2.plot(epochs, acc, label="Train Acc")
    ax2.plot(epochs, val_acc, label="Val Acc")
    ax2.set_xlabel("Époque")
    ax2.grid(True, which="both", linestyle="--", linewidth=0.3, color="#cccccc", alpha=0.5)
    ax2.legend()
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
            "Détail"     : ["Kernel RBF", "Arbres en séquence", "Arbres en parallèle", "Vote soft ou hard"],
            "Particularité": [
                "Séparation par hyperplan en grande dimension",
                "Chaque arbre corrige les erreurs du précédent",
                "Plus rapide que XGBoost, optimisé feuille par feuille",
                "Réduit la variance par combinaison des modèles",
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
            <div style="font-size:13px; color:#888;">SVM + XGBoost · vote soft</div>
        </div>
        <div style="background:#f8f8f8; border-radius:10px; padding:16px 24px; flex:1;">
            <div style="font-size:13px; color:#888;">Accuracy</div>
            <div style="font-size:28px; font-weight:500; margin:8px 0;">98.45%</div>
        </div>
        <div style="background:#f8f8f8; border-radius:10px; padding:16px 24px; flex:1;">
            <div style="font-size:13px; color:#888;">F1 macro</div>
            <div style="font-size:28px; font-weight:500; margin:8px 0;">98.52%</div>
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
            "Accuracy (%)": ["97.87 ± 0.39", "97.66 ± 0.22", "97.83 ± 0.24", "98.13 ± 0.29"],
            "F1 macro (%)": ["97.62 ± 0.37", "97.66 ± 0.30", "97.89 ± 0.29", "98.24 ± 0.31"],
        }
        st.dataframe(data_cv5)
        st.caption("La CV5 n'a pas été appliquée aux modèles DL en raison du coût computationnel.")
    st.divider()

    st.subheader("Interprétabilité — SHAP par classe")
    st.caption("Top 10 des features contribuant à la classification de chaque type cellulaire.")

    st.write("La figure ci-dessous représente les couleurs des centres des clusters K-Means triés par luminosité décroissante.")
    cols = st.columns([0.8, 2, 1])
    with cols[1]:
        st.image("images/figures/cluster_colors.png", width=700)

               
    captions = {
    "Basophile": "Reconnu principalement par la faible luminosité de son noyau "
    "(Intra mean 2 L bas) — le cluster 2 correspondant au violet foncé caractéristique."
    "Une luminosité élevée dans ce cluster ainsi que la présence de teintes roses du cluster 5, poussent fortement contre la classe.",
    
        
    "Éosinophile": "Reconnu par sa grande taille — la faible proportion de fond (cluster 9) "
    "indique un cytoplasme volumineux. La teinte rose du cytoplasme (cluster 5) "
    "est le signal colorimétrique distinctif. C'est la cellule la plus reconnaissable visuellement.",
    
    "Érythroblaste": "Reconnu par la texture homogène et dense de son noyau (GLCM homogeneity élevée) — "
    "contrairement aux autres cellules à noyau sombre, il est compact et sans granularité. "
    "La teinte violette concentrée (cluster 2) et l'absence de pixels très sombres au centre (cluster 1) "
    "le distinguent du lymphocyte.",
    
    "IG": "Signal SHAP plus diffus que les autres classes, reflétant l'hétérogénéité biologique de la catégorie "
    "qui regroupe plusieurs stades de maturation. Reconnu par sa grande taille (cluster 9 faible) et la variabilité "
    "de luminosité dans son noyau irrégulier (Intra std 1 L), absent chez les cellules matures.",
    
    "Lymphocyte": "L'absence de pixels violet du cluster 2 au centre de l'image (r1 hist 2 bas) le distingue du basophile. L'absence de teintes orangées (cluster 7) "
    "le distingue de l'éosinophile.",
    
    "Monocyte": "Reconnu par sa très grande taille avec la plus faible proportion de fond du dataset (cluster 9 très bas). "
    "Son cytoplasme violet s'étend jusqu'en périphérie (cluster 3 en r2 et r3). "
    "La variabilité colorimétrique globale (Global std B) reflète la présence de vacuoles caractéristiques.",
    
    "Neutrophile": "Reconnu principalement par la distribution spatiale de son noyau multilobé — "
    "les histogrammes radiaux dominent le signal (r1, r2 hist). Les lobes du noyau s'étendent dans l'anneau r2 "
    "(r2 hist 1 élevé)."
    "C'est le seul type où la structure spatiale prime sur la couleur.",
    
    "Plaquette": "Reconnu par l'absence totale de cytoplasme — là où les autres cellules ont des pixels foncés en r2, la plaquette n'a que du fond beige (r2 hist 1 très bas, r2 hist 13 élevé). L'image est quasi homogène (Global std L et A faibles). Les valeurs SHAP les plus extrêmes du dataset confirment que c'est la classe la plus facile à identifier.",
    }

    vspace(10)
    classe = st.selectbox("Classe", list(CLASSES_FR), key="shap")
    cols = st.columns([0.25, 2, 0.25])
    with cols[1]:
        st.image(f"images/shap/shap_{classe}.png")
    vspace(10)
    st.write(captions[classe])



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
                "Compound scaling · léger et rapide",
                "Plus profond que S · meilleure précision",
                "Connexions résiduelles · très stable",
                "Connexions denses · réutilisation des features",
                "Architecture classique · simple et interprétable",
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

        col_loss, col_acc = st.columns(2)
        with col_loss:
            vspace(10) 
            st.markdown("<div style='text-align: center;'><b>Fonction de perte (loss)</b></div>", unsafe_allow_html=True)
            st.pyplot(loss_fig, use_container_width=True)

        with col_acc:
            vspace(10)
            st.markdown("<div style='text-align: center;'><b>Accuracy</b></div>", unsafe_allow_html=True)
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
            "Accuracy (%)": [97.57, 97.72, 98.21, 98.45, 98.97, 98.92, 98.89, 98.77, 98.97, 98.45, 99.21],  
            "F1 macro (%)": [97.67, 97.72, 98.31, 98.52, 99.02, 98.95, 98.90, 98.87, 99.03, 98.56, 99.24],      
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

    st.caption("Ensemble : EfficientNetV2S + VGG19 + ResNet50V2 + Xception · vote soft")
    st.divider()

    st.subheader("Temps computationnels")

    df_temps = pd.DataFrame({
        "": ["Training", "Inférence/image"],
        "ML (CPU)": ["~ 1 min", "~ 4 ms"], 
        "DL (GPU)": ["1h – 3h", "10 - 25  ms (batch 32)"],
    })

    st.dataframe(df_temps, use_container_width=True, hide_index=True)
    st.caption("Les deux meilleurs modèles DL (EfficientNetV2S et VGG19) offrent le meilleur compromis : ~2h de training et ~16 ms/image en inférence (batch 32)")