import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px

from utils.config import vspace
from utils.config import (COMMON_CSS, PALETTE_FR, UMAP_CSV, FIGURES_DIR)

st.set_page_config(page_title="Preprocessing",
                   page_icon="⚙️", layout="wide")

st.markdown(COMMON_CSS, unsafe_allow_html=True)

@st.cache_data
def load_umap(path):
    return pd.read_csv(path)


st.title("⚙️ Prétraitement")

vspace(40)

tab1,tab2 = st.tabs([
    "Pipeline ML",
    "Pipeline DL",
])

with tab1:
    st.header("Extraction de caractéristiques")

    st.markdown("Le pipeline ML transforme chaque image brute en un **vecteur de 156 features**.")

    col1, col2, col3 = st.columns([0.05,1,0.05])

    with col2:
        
        st.markdown("""
        <div style="display:flex;
            align-items:center;
            justify-content:space-between;
            font-size:15.5px;
            text-align:center;
            border-radius:8px;
            background-color:#f8f8f8;
            padding:20px 30px;">
            <div><b>Images RGB</b><br><span style="color:#888;font-size:13px;">128 × 128</span></div>
            <div style="font-size:24px; color:#888;">→</div>
            <div><b>Espace LAB</b><br><span style="color:#888;font-size:13px;">conversion</span></div>
            <div style="font-size:24px; color:#888;">→</div>
            <div><b>K-Means</b><br><span style="color:#888;font-size:13px;">k = 13 clusters</span></div>
            <div style="font-size:24px; color:#888;">→</div>
            <div><b>Extraction</b><br><span style="color:#888;font-size:13px;">features</span></div>
        </div>
        """, unsafe_allow_html=True)


    st.divider()

    st.subheader("Composition du vecteur de features")

    df = pd.DataFrame({
        "Catégorie" : ["K-Means", "K-Means", "K-Means", "K-Means",
                        "Global", "Global",
                        "Texture", "Texture"],
            "Feature"   : [
                "Histogrammes clusters",
                "LAB intra — moyennes",
                "LAB intra — écart-types",
                "Histogrammes radiaux",
                "Contraste",
                "Statistiques globales",
                "GLCM",
                "Sobel",
            ],
            "Description": [
                "Distribution globale des 13 clusters",
                "Moyenne L, A, B par cluster",
                "Écart-type L, A, B par cluster",
                "4 anneaux concentriques × 13 clusters",
                "Moyenne + écart-type",
                "Moyenne + écart-type sur L, A, B",
                "Contraste, homogénéité, énergie",
                "Moyenne + écart-type magnitude du gradient ",
            ],
        "Dimension":[13, 39, 39, 52, 2, 6, 3, 2]
    })

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    st.metric("Dimension totale", "156")

    st.divider()

    col1, col2 = st.columns([1.5, 0.8], vertical_alignment="center")

    with col1:

        st.subheader("Histogrammes radiaux")
        st.write(
            """
            Chaque image est divisée en **4 anneaux concentriques**
            centrés sur l'image.

            Dans chaque anneau, on calcule l'histogramme
            des labels des clusters K-Means

            ➡️ **Capture la distribution spatiale : noyau vs cytoplasme vs fond**

            """
        )

    with col2:

        img = Image.open(f"{FIGURES_DIR}/kmeans_radii.png").convert("RGB")
        st.image(img, width="stretch")


    st.divider()


    st.header("UMAP — projection des features")

    st.write("Chaque image est représentée par un point dans cet espace 2D, obtenu par réduction des 156 features extraites. La couleur indique la classe cellulaire.")

    df_umap = load_umap(UMAP_CSV)
    classes = sorted(df_umap["label"].unique())

    def reset():
        st.session_state.selected_classes = classes
        st.session_state.point_size = 2.5
        st.session_state.opacity = 0.5
        #st.rerun()

    if "selected_classes" not in st.session_state:
        st.session_state.selected_classes = classes

    if "point_size" not in st.session_state:
        st.session_state.point_size = 2.5

    if "opacity" not in st.session_state:
        st.session_state.opacity = 0.5

    left, middle, right = st.columns([2.,0.2, 1], vertical_alignment="top")

    with right:
        
        vspace(60)
        selected_classes = st.multiselect(
            "Classes",
            classes,
            key="selected_classes",
        # label_visibility="collapsed"
        )

        point_size = st.slider(
            min_value=0.5,
            max_value=5.0,
            value=2.5,
            step=0.5,
            key="point_size",
            label="Taille des points"
        )

        opacity = st.slider(
            "Opacité",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key="opacity",
                    )

        st.button("Réinitialiser", on_click=reset)

    df_plot = df_umap[
        df_umap["label"].isin(selected_classes)
    ]


    with left:
        
        fig_umap = px.scatter(
            df_plot,
            x="umap_1",
            y="umap_2",
            color="label",
            color_discrete_map=PALETTE_FR,
            hover_name="label",
            labels={
                "umap_1":"UMAP dim 1",
                "umap_2":"UMAP dim 2",
                "label":"Classe"
            },
            title=f"UMAP — {len(df_plot):,} cellules · {len(selected_classes)} classes",
            opacity=opacity
        )

        fig_umap.update_traces(
            marker=dict(size=point_size)
        )

        fig_umap.update_layout(
            height=600,
            legend_title="Classe cellulaire",
            legend_title_font=dict(size=15),
            plot_bgcolor="white",
            title_font=dict(size=20, color="#1a1a1a"),
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
            legend=dict(orientation="h", y=-0.2, font=dict(size=14), itemsizing="constant")
        )

        st.plotly_chart(fig_umap)

    vspace(5)
    st.caption("➡️ Clusters bien séparés = features discriminantes")

    
with tab2:

    st.header("Images d'entrée")
    c1, c2 = st.columns(2)
    c1.metric("Résolution", "360 × 360")
    c2.metric("Format", "RGB · float32")

    st.divider()

    st.header("🔄 Data Augmentation")
    st.caption("Appliquée uniquement sur le jeu d'entraînement "
                "pour améliorer la généralisation.")
    st.dataframe(
        pd.DataFrame({
            "Transformation": ["RandomFlip", "RandomRotation"],
            "Paramètre"     : ["Horizontal + Vertical", "factor = 0.2 (± 72°)"],
        }),
        use_container_width=True, hide_index=True,
    )

    st.divider()

    st.header("📐 Normalisation des images")
    st.caption("Normalisation spécifique à chaque architecture, "
                "reproduisant celle appliquée lors du pré-entraînement sur ImageNet.")
    st.dataframe(
        pd.DataFrame({
            "Architecture" : ["EfficientNetV2S/M", "ResNet50V2", "DenseNet121", "VGG19", "Xception"],
            "Normalisation": ["→ [-1, 1]", "Soustraction moyenne ImageNet",
                                "→ [0, 1] puis centrage",
                                "Soustraction moyenne pixel par pixel", "→ [-1, 1]"],
        }),
        use_container_width=True, hide_index=True,
    )



