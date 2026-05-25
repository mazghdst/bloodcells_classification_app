
import streamlit as st

def vspace(height):
    st.markdown(
        f"<div style='height:{height}px;'></div>",
        unsafe_allow_html=True
    )

def hspace(width):
    st.markdown(
        f"<div style='display:inline-block; width:{width}px;'></div>",
        unsafe_allow_html=True
    )

CLASSES_EN = [
    "Basophil", "Eosinophil", "Erythroblast",
    "IG", "Lymphocyte",
    "Monocyte", "Neutrophil", "Platelet"
]

CLASSES_FR = [
    "Basophile", "Éosinophile", "Érythroblaste",
    "IG", "Lymphocyte", "Monocyte", "Neutrophile", "Plaquette"
]

CLASSES_EN_TO_FR = dict(zip(CLASSES_EN, CLASSES_FR))
CLASSES_FR_TO_EN = dict(zip(CLASSES_FR, CLASSES_EN))

CLASSES_MAP = {
    0: "Basophile",
    1: "Éosinophile",
    2: "Érythroblaste",
    3: "IG",
    4: "Lymphocyte",
    5: "Monocyte",
    6: "Neutrophile",
    7: "Plaquette"
}

IMG_SIZE_DL = (360, 360)
IMG_SIZE_ML = (128, 128)

CONF_THRESHOLD = 0.70

RESULTS_DIR  = "./results"

FIGURES_DIR  = "./images/figures"
SAMPLE_DIR  = "./images/sample_images"  
QUALITY_DIR  = "./images/quality"
ERRORS_DIR = "./images/errors" 

DATASET_DIR = "../Projet/data/dataset" 

UMAP_CSV    = "./files/umap_coords.csv"

DL_MODELS_DIR = "./models/dl"
ML_MODELS_DIR = "./models/ml"

CLASSIFIER_MAP = {
    "SVM"              : "svm",
    "XGBoost"          : "xgb",
    "Voting Classifier" : "voting",
}

MODEL_MAP = {
    "EfficientNetV2S" : "efficientnetv2s",
    "EfficientNetV2M" : "efficientnetv2m",
    "ResNet50V2"     : "resnet50v2",
    "DenseNet121"    : "densenet121",
    "VGG19"          : "vgg19",
    "Xception"       : "xception",
    "Ensemble (EffV2S + VGG19 + Xception)" : "ensemble"
}

ML_PATH_MAP = {
    "SVM": f"{ML_MODELS_DIR}/svm.pkl",
    "XGBoost": f"{ML_MODELS_DIR}/xgb.pkl",
    "Voting Classifier": f"{ML_MODELS_DIR}/voting.pkl",
}

DL_PATH_MAP = {
    "EfficientNetV2S": f"{DL_MODELS_DIR}/efficientnetv2s.keras",
    "EfficientNetV2M": f"{DL_MODELS_DIR}/efficientnetv2m.keras",
    "VGG19": f"{DL_MODELS_DIR}/vgg19.keras",
    "DenseNet121": f"{DL_MODELS_DIR}/densenet121.keras",
    "ResNet50V2": f"{DL_MODELS_DIR}/resnet50v2.keras",
    "Xception": f"{DL_MODELS_DIR}/xception.keras",
}

# Palette UMAP cohérente avec le plot original
PALETTE_FR = {
    "Basophile"     : "#555555",
    "Éosinophile"   : "#E74C3C",
    "Érythroblaste" : "#9B59B6",
    "IG"            : "#FF69B4",
    "Lymphocyte"    : "#27AE60",
    "Monocyte"      : "#A0522D",
    "Neutrophile"   : "#3498DB",
    "Plaquette"     : "#F39C12",
}

# CSS commun injecté sur chaque page
COMMON_CSS = """
<style>
    .main-title { font-size:2.2rem; font-weight:800; color:#C0392B; }
    .sub-title  { font-size:1rem; color:#7f8c8d; margin-top:-10px; }
    .section    { font-size:1.2rem; font-weight:700; margin-top:1.5rem; }
    .pred-box   { background:#f8f9fa; border-left:5px solid #C0392B;
                  padding:1rem; border-radius:8px; margin-top:.5rem; }
    .badge      { display:inline-block; padding:4px 14px; border-radius:20px;
                  font-size:.9rem; font-weight:700;
                  background:#C0392B; color:white; margin-top:6px; }
    .conf-ok    { color:#27ae60; font-weight:bold; }
    .conf-low   { color:#e67e22; font-weight:bold; }
    .conf-bad   { color:#e74c3c; font-weight:bold; }
    .tag-ml     { background:#3498DB; color:white; padding:2px 10px;
                  border-radius:12px; font-size:.8rem; font-weight:600; }
    .tag-dl     { background:#E74C3C; color:white; padding:2px 10px;
                  border-radius:12px; font-size:.8rem; font-weight:600; }
    button[data-baseweb="tab"] {
        font-size: 28px !important;
        font-weight: 700 !important;
        padding: 14px 24px !important;
    }
    .stTabs [data-baseweb="tab-highlight"]{
        display:none;
    }
    .stTabs [data-baseweb="tab-list"]{
        gap: 12px;
        border-bottom: none;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"]{
        height: 48px;
        padding: 0 24px;
        border-radius: 14px;

        background:#fafafa;
        border:1px solid #ececec;

        box-shadow:
            0 2px 6px rgba(0,0,0,0.04);

        font-size: 1.1rem;
        font-weight: 600;

        transition: all .2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover{
        transform: translateY(-2px);
        box-shadow:
            0 6px 14px rgba(0,0,0,0.08);
    }
    .stTabs [aria-selected="true"]{
        background:white;
        border:1px solid #e5e7eb;

        box-shadow:
            0 6px 16px rgba(0,0,0,.06);

        font-weight:700;
    }

    .approach-card{
        background:white;
        border:1px solid #ececec;
        border-radius:20px;
        padding:30px;

        box-shadow:0 4px 14px rgba(0,0,0,.05);
    }
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

    .metric-card {
        background: #f8f8f8;
        border-radius: 10px;
        padding: 12px 20px;
        flex: 1;
    }
    .metric-label { font-size: 13px; color: #888; }
    .metric-value { font-size: 28px; font-weight: 500; margin: 8px 0; }
    .class-card {
        background: #f8f8f8;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    

</style>
"""


