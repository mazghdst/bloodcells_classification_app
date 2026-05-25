import streamlit as st
from utils.config import COMMON_CSS

st.markdown(COMMON_CSS, unsafe_allow_html=True)
st.title("Classification automatique de cellules sanguines")

st.markdown("""
### Analyse et classification d’images du dataset **PBC (Peripheral Blood Cells)**

Cette application présente un système de classification automatique basé sur **images microscopiques de cellules sanguines**.
""")

st.divider()

# ======================
# KPI
# ======================

 
st.header("Chiffres clés")
 

st.markdown("""
<div style="display:flex; gap:10px; align-items:stretch; margin-bottom:1rem;">
    <div class="metric-card"><div class="metric-label">Images</div><div class="metric-value">17 095</div></div>
    <div class="metric-card"><div class="metric-label">Classes</div><div class="metric-value">8</div></div>
    <div class="metric-card"><div class="metric-label">Modèles testés</div><div class="metric-value">10+</div></div>
    <div class="metric-card"><div class="metric-label">Accuracy max</div><div class="metric-value">99,18%</div></div>
</div>
""", unsafe_allow_html=True)

st.divider()



 
st.header("Deux approches étudiées")
 
col1, col2 = st.columns(2, gap="medium")
 
with col1:
    st.markdown("""
    <div class="card">
        <h3>⚙️ Machine Learning</h3><br>
        <div class="section-label">Extraction de caractéristiques</div>
        <div style="font-size:16px; color:#555;">✓ K-Means espace L*a*b*</div>
        <div style="font-size:16px; color:#555;">✓ Statistiques intra-clusters</div>
        <div style="font-size:16px; color:#555;">✓ GLCM, Sobel</div>
        <div style="font-size:16px; color:#555;">✓ Réduction de dimensions</div>
        <br>
        <div class="section-label">Modèles</div>
        <div style="font-size:16px; color:#555;">SVM · XGBoost · Voting Classifier</div>    </div>
    """, unsafe_allow_html=True)
 
with col2:
    st.markdown("""
    <div class="card">
        <h3>🧠 Deep Learning</h3><br>
        <div class="section-label">Méthodes</div>
        <div style="font-size:16px; color:#555;">✓ Data augmentation</div>
        <div style="font-size:16px; color:#555;">✓ Transfer learning</div>
        <div style="font-size:16px; color:#555;">✓ Fine tuning</div>
        <br>
        <div class="section-label">Architectures</div>
        <div style="font-size:16px; color:#555;">EfficientNetV2S · EfficientNetV2M · DenseNet121 · 
                ResNet50V2 · VGG19 · Xception</div>   
        </div>
    """, unsafe_allow_html=True)


st.divider()

# ======================
# NAVIGATION
# ======================

st.header("Sommaire")

st.markdown("""

#### 📊 Données
Description du dataset PBC, distribution des classes et exemples.

#### ⚙️ Prétraitement
Préparation des données, extraction de caractéristiques et transformations appliquées aux images.

#### 🧠 Modèles
Présentation des modèles, performances et interprétabilité.

#### 🔍 Démonstation
Classification interactive, Grad-CAM et analyse des erreurs.

#### 💡 Conclusion
Synthèse, limites et perspectives.
""")
