import streamlit as st
import pandas as pd
from glob import glob
from PIL import Image
import random
import plotly.express as px
from utils.config import vspace
from utils.config import (COMMON_CSS, SAMPLE_DIR, CLASSES_FR, IMG_SIZE_DL, QUALITY_IMAGES)
 
st.set_page_config(page_title="Données", page_icon="🔬", layout="wide")
st.markdown(COMMON_CSS, unsafe_allow_html=True)
 
 
 
def distribution_show():
    df = pd.read_csv("files/cells_distribution.csv")
    fig = px.bar(
        df,
        x="proportion",
        y="category",
        orientation="h",
        text="proportion",
        color="category",
        color_discrete_sequence=["#8b8fa8"]
    )
    fig.update_traces(texttemplate=" %{text:.1f}%", textposition="outside")
    fig.update_layout(
        showlegend=False,
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis_title="",
        yaxis={'categoryorder': 'total ascending'},
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="Proportion (%)"
    )
    fig.update_xaxes(range=[0, 22])
    return fig
 
 
def get_images_of_class(cls, n=5):
    paths = glob(f"{SAMPLE_DIR}/{cls}/*")
    imgs = []
    for path in random.sample(paths, min(n, len(paths))):
        img = Image.open(path).convert("RGB")
        img = img.resize(IMG_SIZE_DL)
        imgs.append(img)
    return imgs
 
 
# ── Titre ────────────────────────────────────────────────────────────────────
 
st.title("🔬 Données")
st.divider()
 
# ── Dataset ──────────────────────────────────────────────────────────────────
 
st.subheader("Dataset PBC — Peripheral Blood Cells")
st.write("""
Le dataset PBC (Peripheral Blood Cells) est un dataset public de microscopie
hématologique composé d'images de **cellules sanguines** périphériques acquises
avec le CellaVision DM96 au laboratoire de l'Hospital Clinic of Barcelona.
""")
 
vspace(20)

st.markdown("""
<div style="display:flex; gap:10px; align-items:stretch; margin-bottom:1rem;">
    <div class="metric-card"><div class="metric-label">Images</div><div class="metric-value">17 095</div></div>
    <div class="metric-card"><div class="metric-label">Classes</div><div class="metric-value">8</div></div>
    <div class="metric-card"><div class="metric-label">Résolution</div><div class="metric-value">360×363</div></div>
    <div class="metric-card"><div class="metric-label">Format</div><div class="metric-value">JPG</div></div>
</div>
""", unsafe_allow_html=True)

vspace(10)
st.divider()
 
# ── Classes cellulaires ───────────────────────────────────────────────────────
 
st.subheader("Classes cellulaires")
st.write(
    "Ce jeu de données regroupe 8 types de cellules sanguines observées sur des frottis microscopiques. "
    "Chaque catégorie possède des caractéristiques morphologiques spécifiques que les modèles devront apprendre à distinguer."
)
 
vspace(30)
col1, col2 = st.columns([1.2, 2])
 
with col1:
    vspace(10)
    classes = [
        "Basophile", "Éosinophile",
        "Érythroblaste", "IG",
        "Lymphocyte", "Monocyte",
        "Neutrophile", "Plaquette"
    ]
 
    for i in range(0, len(classes), 2):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="class-card">{classes[i]}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="class-card">{classes[i+1]}</div>', unsafe_allow_html=True)
 
    st.caption("IG : Granulocyte immature")
 
with col2:
    fig = distribution_show()
    st.plotly_chart(fig, use_container_width=True)
 
st.divider()
 
# ── Exemples d'images ─────────────────────────────────────────────────────────
 
st.subheader("Exemples d'images")
 
selected = st.selectbox("Catégorie", CLASSES_FR)
imgs = get_images_of_class(selected, 5)
 
if st.button("🎲 Afficher d'autres exemples"):
    imgs = get_images_of_class(selected, 5)
 
cols = st.columns(5)
for col, img in zip(cols, imgs):
    with col:
        st.image(img)
 
st.divider()
 
# ── Qualité ───────────────────────────────────────────────────────────────────
 
st.subheader("Qualité des images et limites")
st.write("""
Les images du dataset PBC présentent globalement une bonne qualité :
- images nettes
- contraste homogène
- cellules centrées
 
Cependant certaines images contiennent plusieurs cellules nucléées simultanément.
Les analyses Grad-CAM ont montré que certains modèles focalisent parfois leur attention sur une cellule secondaire.
 
Il est également à noter qu'une faible proportion d'images présentent un fond rose (2.6 %) .
""")
 
captions = ["Cellules doubles", "Cellules multiples", "Variation de coloration"]
 
vspace(20)
cols = st.columns(5)
for i, (key, path) in enumerate(QUALITY_IMAGES.items()):
    with cols[i+1]:
        st.image(path)
        st.caption(key) 
    i += 1

st.divider()
 
# ── Source ────────────────────────────────────────────────────────────────────
 
st.markdown("""
<div style="padding:18px; border-radius:8px; background:#f7f7f5; font-size:15px;">
    📄 <b>Source du dataset</b><br><br>
    PBC (Peripheral Blood Cells) Dataset — Images acquises avec le <b>CellaVision DM96</b>
    à l'Hospital Clinic of Barcelona.<br><br>
    🔗 <a href="https://data.mendeley.com/datasets/snkd93bnjr/1" target="_blank">
    PBC Dataset - Mendeley Data</a>
</div>
""", unsafe_allow_html=True)