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


st.markdown("""
<style>
.card {
    border-radius: 12px;
    border: 1px solid #e8e8e8;
    padding: 20px 22px;
    min-height: 360px;
}
.card-ml { border-top: 4px solid #7F77DD; }
.card-dl { border-top: 4px solid #1D9E75; }
 
.tag {
    display: inline-block;
    font-size: 12px;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid #d0d0d0;
    background: #f0f0f0;
    color: #333;
    margin: 3px;
}
.section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 10px;
    margin-top: 4px;
}
.tag-ml { background: #EEEDFE; color: #3C3489; }
.tag-dl { background: #E1F5EE; color: #085041; }
</style>
""", unsafe_allow_html=True)


 
st.header("Deux approches étudiées")
 
col1, col2 = st.columns(2, gap="medium")
 
with col1:
    st.markdown("""
    <div class="card">
        <h3>🧮 Machine Learning</h3><br>
        <div class="section-label">Extraction de features</div>
        ✓ K-Means espace LAB<br>
        ✓ Statistiques intra-clusters<br>
        ✓ GLCM, Sobel<br>
        ✓ Réduction de dimensions<br><br>
        <div class="section-label">Modèles</div>
        <span class="tag">SVM</span>
        <span class="tag">XGBoost</span>
        <span class="tag">Voting Classifier</span>
    </div>
    """, unsafe_allow_html=True)
 
with col2:
    st.markdown("""
    <div class="card">
        <h3>🧠 Deep Learning</h3><br>
        <div class="section-label">Méthode</div>
        ✓ Transfer learning <br>
        ✓ Fine tuning<br><br>
        <div class="section-label">Architectures</div>
        <span class="tag">EfficientNetV2S</span>
        <span class="tag">EfficientNetV2M</span>
        <span class="tag">DenseNet121</span>
        <span class="tag">ResNet50V2</span>
        <span class="tag">VGG19</span>
        <span class="tag">Xception</span>
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
