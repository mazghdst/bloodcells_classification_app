import streamlit as st
from utils.config import COMMON_CSS, vspace

st.set_page_config(page_title="Conclusion",
                   page_icon="💡", layout="wide")

st.markdown(COMMON_CSS, unsafe_allow_html=True)

st.title("💡 Conclusion")

st.divider()

st.markdown("### Principaux résultats")
st.markdown("""
- Le pipeline ML — extraction de features via un clustering K-Means - offre une classification **rapide sur CPU** dont chaque prédiction peut être **expliquée** et **justifiée** via SHAP
- Le Voting Classifier atteint **98.29% d'accuracy** en validation croisée sur 5 folds, confirmant la pertinence des **features colorimétriques et texturales** extraites pour cette tâche de classification
- Les modèles DL par Transfer Learning atteignent **99,21% d'accuracy**, soit la meilleure performance observée
- Les **Grad-CAM** confirment que les modèles DL focalisent leur attention sur les zones biologiquement pertinentes — noyau et cytoplasme — ce qui valide leur comportement au-delà du simple score
""")

vspace(20)

st.markdown("### Limites")
st.markdown("""
- Les classes morphologiquement proches restent les plus difficiles à séparer — IG, Monocyte et Neutrophile — visible sur l'UMAP et les matrices de confusion
- Les deux approches sont entraînées et évaluées sur le dataset PBC uniquement, ce qui limite la généralisation à d'autres protocoles de coloration ou équipements
- Les performances du pipeline ML ont été validées par validation croisée 5 folds, contrairement aux modèles DL évalués sur un unique split train/test —           
- Les modèles DL sont coûteux en mémoire et dépendants d'un GPU pour un temps d'inférence raisonnable
""")

vspace(20)

st.markdown("### Perspectives")
st.markdown("""
- Tester la robustesse des deux approches sur d'autres datasets de microscopie hématologique, notamment avec des variations de coloration ou d'équipement d'acquisition
- Évaluer la robustesse des modèles de deep learning sur différentes partitions des données via une validation croisée
- Étendre le pipeline ML à d'autres types de features morphologiques pour réduire davantage l'écart avec le DL
- Intégrer une étape de normalisation colorimétrique en amont du pipeline ML pour améliorer la robustesse à la variabilité d'acquisition inter-datasets
  """)

# st.caption("### 📦 Stack technique")
# st.caption("""
#     | Composant | Outil |
#     |---|---|
#     | Langage | Python 3.x |
#     | DL | TensorFlow / Keras |
#     | ML | scikit-learn, XGBoost |
#     | Features | NumPy, scikit-image, OpenCV |
#     | Visualisation | Matplotlib, Plotly |
#     | UMAP | umap-learn |
#     | App | Streamlit |
#     """)

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:.9rem; margin-top:2rem">
 Projet académique de classification de cellules sanguines<br>
    Dataset : PBC (Peripheral Blood Cells) · Barcelona Hospital Clinic
</div>
""", unsafe_allow_html=True)
