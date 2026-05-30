import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px

from utils.config import vspace
from utils.config import (
    COMMON_CSS, PALETTE_FR, UMAP_CSV,
)

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
    st.subheader("Extraction de caractéristiques")

    st.markdown("Le pipeline de machine learning transforme chaque image en un **vecteur de 99 features**.")

    st.markdown("""
    <div style="display:flex; align-items:center; gap:8px; margin:20px 0;">
        <div style="background:#f8f8f8; border-radius:10px; padding:14px 20px; text-align:center; flex:1;">
            <div style="font-size:20px;">🔬</div>
            <div style="font-weight:600; margin:4px 0;">Images RGB</div>
            <div style="font-size:12px; color:#888;">360 × 363</div>
        </div>
        <div style="font-size:20px; color:#ccc;">→</div>
        <div style="background:#f8f8f8; border-radius:10px; padding:14px 20px; text-align:center; flex:1;">
            <div style="font-size:20px;">✂️</div>
            <div style="font-weight:600; margin:4px 0;">Resize + Crop</div>
            <div style="font-size:12px; color:#888;">90 × 90</div>
        </div>
        <div style="font-size:20px; color:#ccc;">→</div>
        <div style="background:#f8f8f8; border-radius:10px; padding:14px 20px; text-align:center; flex:1;">
            <div style="font-size:20px;">🎨</div>
            <div style="font-weight:600; margin:4px 0;">Espace L*a*b*</div>
            <div style="font-size:12px; color:#888;">conversion</div>
        </div>
        <div style="font-size:20px; color:#ccc;">→</div>
        <div style="background:#f8f8f8; border-radius:10px; padding:14px 20px; text-align:center; flex:1;">
            <div style="font-size:15px;">🟠</div>
            <div style="font-weight:600; margin:4px 0;">K-Means</div>
            <div style="font-size:12px; color:#888;">k = 8 clusters</div>
        </div>
        <div style="font-size:20px; color:#ccc;">→</div>
        <div style="background:#f8f8f8; border-radius:10px; padding:14px 20px; text-align:center; flex:1;">
            <div style="font-size:20px;">📐</div>
            <div style="font-weight:600; margin:4px 0;">Extraction</div>
            <div style="font-size:12px; color:#888;">99 features</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.subheader("Composition du vecteur de features")


    df = pd.DataFrame({
            "Catégorie": [
                "K-Means",
                "K-Means",
                "K-Means",
                "K-Means",
                "Global",
                "Texture",
                "Texture",
            ],
            "Feature"   : [
                "Histogrammes clusters",
                "Moyennes intra-clusters",
                "Écart-types intra-clusters",
                "Histogrammes radiaux",
                "Statistiques globales",
                "GLCM",
                "Sobel",
            ],
            "Description": [
                "Distribution globale des 8 clusters",
                "Moyenne par cluster sur L*, a* et b*",
                "Écart-type par cluster sur L*, a* et b*",
                "4 anneaux concentriques × 8 clusters",
                "Moyenne et écart-type sur L*, a* et b*",
                "Contraste, homogénéité et énergie",
                "Moyenne et écart-type de la magnitude du gradient ",
            ],
        "Dimension":[8, 24, 24, 32, 6, 3, 2]
    })

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )
    st.metric("Dimension totale", "99")
    # with col2:
    #     img = Image.open(f"{FIGURES_DIR}/kmeans_radii.png").convert("RGB")
    #     st.image(img, width="stretch")
        
    #     col11, col12, col13 = st.columns([0.1, 1, 0.1])
    #     with col12:
    #         st.caption("Segmentation par K-Means. Les couleurs correspondent aux centres des 8 clusters.")

    
    #st.caption("Dimension initiale : 90 × 90 × 3 = 24 300, rendant la classification directe difficile sur CPU.")

    vspace(10)
    with st.expander("Voir un exemple de segmentation K-Means par type de cellule"):
        classes = [
            ("basophile", "Basophile"),
            ("eosinophile", "Éosinophile"),
            ("erythroblaste", "Érythroblaste"),
            ("ig", "IG"),
            ("lymphocyte", "Lymphocyte"),
            ("monocyte", "Monocyte"),
            ("neutrophile", "Neutrophile"),
            ("plaquette", "Plaquette"),
        ]
        
        row1 = st.columns(4)
        row2 = st.columns(4)
        
        for i, (filename, label) in enumerate(classes):
            col = row1[i] if i < 4 else row2[i - 4]
            with col:
                img = Image.open(f"images/kmeans/kmeans_{filename}.png")
                st.image(img, width="stretch")
                st.markdown(f"<p style='text-align:center; font-size:0.875rem; color:grey; margin-top:-1.5rem'>{label}</p>", unsafe_allow_html=True)

        st.caption("Les couleurs correspondent aux couleurs des centres des clusters, converties en RGB.")
    st.divider()


    st.subheader("UMAP — projection des features")

    st.write("Projection UMAP des images à partir des 99 features extraites par le pipeline ML, réduite à 2 composantes. Chaque point représente une image, coloré selon sa classe cellulaire.")

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
            title=f"{len(df_plot):,} cellules · {len(selected_classes)} classes",
            opacity=opacity,
            render_mode="svg"
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

    vspace(20)

    st.caption("Cette projection montre que les features extraites sont globalement discriminantes, avec des clusters bien séparés pour la majorité des classes. Cependant, certaines classes présentent un chevauchement partiel, correspondant aux catégories de cellules les plus proches morphologiquement.")
    
with tab2:

    st.subheader("Format des images")

    st.markdown("""
    <div style="display:flex; gap:10px; align-items:stretch; margin-bottom:1rem;">
        <div class="metric-card"><div class="metric-label">Résolution</div><div class="metric-value">360 × 360</div></div>
        <div class="metric-card"><div class="metric-label">Espace colorimétrique</div><div class="metric-value">RGB</div></div>
        <div class="metric-card"><div class="metric-label">Type</div><div class="metric-value">Float32</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.subheader("Data Augmentation")
    st.caption("Augmentation des images du jeu d'entraînement pour améliorer la capacité de généralisation des modèles")
    st.dataframe(
        pd.DataFrame({
            "Transformation": ["RandomFlip", "RandomRotation"],
            "Paramètre"     : ["Horizontal + Vertical", "factor = 0.2 (± 72°)"],
        }),
        use_container_width=True, hide_index=True,
    )

    st.divider()

    st.subheader("Normalisation des images")
    st.caption("Normalisation des images spécifique à chaque architecture, correspondant à celle appliquée lors du pré-entraînement sur ImageNet.")
    st.dataframe(
        pd.DataFrame({
            "Architecture" : ["EfficientNetV2S/M", "ResNet50V2", "DenseNet121", "VGG19", "Xception"],
            "Normalisation": ["Mise à l’échelle dans [-1, 1]", "Soustraction de la moyenne ImageNet par canal (RGB)",
                                "Mise à l’échelle dans [0, 1] puis centrage par la moyenne et l'écart-type ImageNet ",
                                "Soustraction de la moyenne ImageNet pixel par pixel (BGR)", "Mise à l’échelle dans [-1, 1]"],
        }),
        use_container_width=True, hide_index=True,
    )



