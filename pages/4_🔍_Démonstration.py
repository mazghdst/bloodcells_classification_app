import streamlit as st
import numpy as np
import os
import joblib
from PIL import Image
from models.kmeans import K_Means
from huggingface_hub import hf_hub_download
import tensorflow as tf
import pandas as pd
import random
from glob import glob

from utils.config import (
    COMMON_CSS, CLASSES_MAP, CLASSES_FR, SAMPLE_DIR,
    IMG_SIZE_DL, IMG_SIZE_ML, ERRORS_DIR, MODEL_MAP, 
    ML_PATH_MAP, DL_PATH_MAP, REPO_ID
)

from utils.config import vspace
from utils.gradcam import grad_cam

st.set_page_config(page_title="Démonstration",
                   page_icon="🔍", layout="wide")

st.markdown(COMMON_CSS, unsafe_allow_html=True)

st.title("🔍 Démonstration")
st.subheader("Classification intéractive d'images de cellule sanguine")

vspace(40)

ERRORS_MAP = {
    "Basophil": "Basophile",
    "Basophil": "Basophile", 
    "Eosinophil": "Éosinophile", 
    "Erythroblast": "Érythroblaste",
    "Ig": "IG",
    "Lymphocyte": "Lymphocyte",
    "Monocyte": "Monocyte", 
    "Neutrophil": "Neutrophile", 
    "Platelet": "Plaquette",
}

def resolve_path(path):

    if os.path.exists(path):
        return path

    filename = os.path.basename(path)

    return hf_hub_download(
        repo_id=REPO_ID,
        filename=filename
    )

@st.cache_resource
def load_dl_model(model_name):

    path = resolve_path(DL_PATH_MAP[model_name])
    return tf.keras.models.load_model(path, compile=False)


@st.cache_resource
def load_ml_model(model_name):

    path = resolve_path(ML_PATH_MAP[model_name])
    return joblib.load(path)


@st.cache_resource
def get_missclassified_img(model_name):
    paths = glob(f"{ERRORS_DIR}/errors_{model_name}/*")
    imgs = []
    imgs_classe = []

    for path in paths :
        img = Image.open(path).convert("RGB")
        img = img.resize(IMG_SIZE_DL)

        filename = os.path.basename(path) 
        img_classe = ERRORS_MAP.get(filename.split("_")[0])

        imgs.append(img)
        imgs_classe.append(img_classe )

    return imgs, imgs_classe 


def get_images_of_class(cls, n=5):
    paths = glob(f"{SAMPLE_DIR}/{cls}/*")
    imgs = []
    for path in random.sample(paths, min(n, len(paths))):
        img = Image.open(path).convert("RGB")
        img = img.resize(IMG_SIZE_DL)
        imgs.append(img)
    return imgs


def predict_dl(model, img):
    img = np.expand_dims(img, axis=0)

    probs = model(img, training=False)[0].numpy()
    y_pred   = int(np.argmax(probs))
    label = CLASSES_MAP.get(y_pred)

    return probs, label

def predict_ml(model, img):
    img = img.resize(IMG_SIZE_ML)
    img = np.array(img).reshape(1, -1)

    y_pred = model.predict(img)[0]
    probs = model.predict_proba(img)[0]
    label = CLASSES_MAP.get(y_pred)

    return probs, label
 
defaults = {

    "demo_imgs":[],
    "demo_selected_idx":None,
    "demo_selected_image":None,
    "demo_categorie":None,
    "demo_imgs_categorie": [],

    "compare_imgs":[],
    "compare_selected_idx":None,
    "compare_selected_image":None,
    "compare_categorie":None,
    "compare_imgs_categorie": [],

    "error_selected_idx":None,
    "error_selected_image":None,
    "errors_model":None,
    "true_class":None,
}

for k,v in defaults.items():

    if k not in st.session_state:
        st.session_state[k]=v

tab1,tab2,tab3 = st.tabs([
    "Prédiction",
    "Comparaison des modèles",
    "Analyse des erreurs"
])     


# =========================================================================================================
with tab1:  
    st.header("⚗️ Évaluation à partir d'un modèle")
    
    demo_approche = st.radio(
        "Approche",
        ["Machine Learning", "Deep Learning"],
        horizontal=True,
        key="demo_approche"
    )

    if demo_approche == "Machine Learning":
        demo_model = st.selectbox("Modèle", ["SVM", "XGBoost", "Voting Classifier"],
        index=None,
        placeholder="Choisir une option")
    else:
        demo_model = st.selectbox("Modèle", [
            "EfficientNetV2S", "EfficientNetV2M", "ResNet50V2",
            "DenseNet121", "VGG19", "Xception"
        ],
        index=None,
        placeholder="Choisir une option")


    demo_source = st.radio(
        "Sélection de l'image",
        ["Importer une image", "Base de données"], 
        horizontal=True, 
        key="demo_source"
    )

    demo_image = None
    demo_categorie = None

    if demo_source == "Base de données":
        demo_categorie = st.selectbox("Catégorie", [
            "Basophile", "Éosinophile", "Érythroblaste", "IG",
            "Lymphocyte", "Monocyte", "Neutrophile", "Plaquette"
        ], 
        key="demo_categorie",
        index=None,
        placeholder="Choisir une option")

        if (
            st.session_state.get("demo_categorie_courante") != demo_categorie
            or "demo_imgs_categorie" not in st.session_state
        ):
            st.session_state.demo_imgs_categorie = get_images_of_class(demo_categorie, n=6)
            st.session_state.demo_categorie_courante = demo_categorie
            st.session_state.demo_selected_image = None
            st.session_state.demo_selected_idx = None
            st.session_state.demo_label_pred = None
            st.session_state.demo_probs = None
            st.session_state.demo_gradcam_im = None

        if demo_categorie is not None:
            if st.button("🎲 Nouveaux exemples", key="demo_random"):
                st.session_state.demo_imgs_categorie = get_images_of_class(demo_categorie, n=6)
                st.session_state.demo_selected_image = None
                st.session_state.demo_selected_idx = None


        demo_imgs = st.session_state.demo_imgs_categorie

        cols_per_row = 6
        for i in range(0, len(demo_imgs), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(demo_imgs):
                    idx = i + j
                    demo_img = demo_imgs[idx]
                    with col:
                        if st.session_state.demo_selected_idx == idx:
                            st.image(demo_img, use_container_width=True)
                            st.button("✅", key=f"button_{idx}")
                            
                        else:
                            st.image(demo_img, use_container_width=True)
                            if st.button("✔", key=f"button_{idx}"):
                                st.session_state.demo_selected_idx = idx
                                st.session_state.demo_selected_image = demo_img
                                st.rerun()

        demo_image = st.session_state.demo_selected_image

    else:
        uploaded = st.file_uploader("Choisir une image", type=["png", "jpg", "jpeg"])
        if uploaded:
            demo_image = Image.open(uploaded)

    st.divider()

    if demo_image and st.button("🔮 Prédire", disabled=demo_image is None):
        with st.spinner("Analyse en cours..."):

            if demo_approche == "Machine Learning":

                model = load_ml_model(demo_model)
                probs, label = predict_ml(model, demo_image)

                col1, col2, col3, col4 , col5 = st.columns([0.4, 1, 0.2, 1, 0.4])

                with col2:
                    st.image(demo_image, width="stretch")

                with col4:

                    if demo_categorie is not None:
                        if label == demo_categorie:
                            st.success(f"**Classe prédite : {label}**")
                        else:
                            st.error(f"**Classe prédite : {label}**")
                    else: 
                        st.success(f"**Classe prédite : {label}**")

                    df_probas = pd.DataFrame({
                        "Classe": CLASSES_FR,
                        "Probabilité": [round(prob, 6) for prob in probs]
                    }).sort_values("Probabilité", ascending=False)
                
                    st.dataframe(
                        df_probas[df_probas["Probabilité"] > 1e-6], 
                        use_container_width=True, 
                        hide_index=True
                    )

            else :
                model = load_dl_model(demo_model)

                probs, label = predict_dl(model, demo_image)

                col1, col2, col3= st.columns([1, 1, 1])

                with col1:

                    st.image(demo_image, width="stretch")
                    st.caption("Image originale")

                with col2:

                    gradcam_im, class_idx = grad_cam(demo_image, model, MODEL_MAP.get(demo_model))
                    st.image(gradcam_im, width="stretch")
                    if demo_model in ["Xception", "ResNet50V2"]:
                        st.caption("⚠️ Grad-CAM moins fiable")
                    else:
                        st.caption("Grad-CAM")

                with col3:
                    if demo_categorie is not None:
                        if label == demo_categorie:
                            st.success(f"**Classe prédite : {label}**")
                        else:
                            st.error(f"**Classe prédite : {label}**")
                    else: 
                        st.success(f"**Classe prédite : {label}**")
                
                    df_probas = pd.DataFrame({
                        "Classe": CLASSES_FR,
                        "Probabilité": [round(prob, 6) for prob in probs]
                    }).sort_values("Probabilité", ascending=False)
                
                    st.dataframe(
                        df_probas[df_probas["Probabilité"] > 1e-6], 
                        use_container_width=True, 
                        hide_index=True
                    )


# =========================================================================================================
with tab2:  

    st.header("⚡ Comparaison entre modèles")     


    compare_approche = st.radio(
        "Approche",
        ["Machine Learning", "Deep Learning"],
        horizontal=True,
        key="compare_mode"
    )

    compare_categorie = st.selectbox("Catégorie", [
            "Basophile", "Éosinophile", "Érythroblaste", "IG",
            "Lymphocyte", "Monocyte", "Neutrophile", "Plaquette"
        ], 
        key="compare_categorie",
        index=None,
        placeholder="Choisir une option")

    if (
        st.session_state.get("compare_categorie_courante") != compare_categorie
        or "compare_imgs_categorie" not in st.session_state
    ):
        st.session_state.compare_imgs_categorie = get_images_of_class(compare_categorie, n=6)
        st.session_state.compare_categorie_courante = compare_categorie
        st.session_state.compare_selected_image = None
        st.session_state.compare_selected_idx = None
        st.session_state.compare_label_pred = None
        st.session_state.compare_probs = None
        st.session_state.compare_gradcam_im = None

    if compare_categorie is not None:
        if st.button("🎲 Nouveaux exemples", key="compare_random"):
            st.session_state.compare_imgs_categorie = get_images_of_class(compare_categorie, n=6)
            st.session_state.compare_selected_image = None
            st.session_state.compare_selected_idx = None

    compare_imgs = st.session_state.compare_imgs_categorie

    cols_per_row = 6
    for i in range(0, len(compare_imgs), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(compare_imgs):
                idx = i + j
                compare_img = compare_imgs[idx]
                with col:
                    if st.session_state.compare_selected_idx == idx:
                        st.image(compare_img, use_container_width=True)
                        st.button("✅", key=f"compare_button_{idx}")
                    else:
                        st.image(compare_img, use_container_width=True)
                        if st.button("✔", key=f"compare_button_{idx}"):
                            st.session_state.compare_selected_idx = idx
                            st.session_state.compare_selected_image = compare_img
                            st.rerun()

    compare_image = st.session_state.compare_selected_image

    st.divider()

    if compare_image and st.button("⚡ Comparer", disabled=compare_image is None):
        with st.spinner("Analyse en cours..."):
            if compare_approche == "Machine Learning":

                models = {
                    name: load_ml_model(name) for name in ML_PATH_MAP
                }

                cols = st.columns(3)
                model_names = list(models.keys())

                for i in range(3):
                    with cols[i]:
                        name = model_names[i]
                        model = models[name]

                        probs, label = predict_ml(model, compare_image)

                        st.subheader(name)

                        if label == compare_categorie:
                            st.success(f"**Classe prédite : {label}**")
                        else:
                            st.error(f"**Classe prédite : {label}**")


                        df_probas = pd.DataFrame({
                            "Classe": CLASSES_FR,
                            "Probabilité": [round(prob, 4) for prob in probs]
                        }).sort_values("Probabilité", ascending=False)
                            
                        st.dataframe(
                            df_probas[df_probas["Probabilité"] != 0], 
                            use_container_width=True, 
                            hide_index=True
                        )

            else :

                models = {
                    name: load_dl_model(name) for name in DL_PATH_MAP
                }

                model_names = list(models.keys())

                cols_per_row = 3

                for i in range(0, len(model_names), cols_per_row):
                    cols = st.columns(cols_per_row)

                    for j, col in enumerate(cols):
                        if i+j < len(model_names):

                            name = model_names[i+j]
                            model = models[name]

                            probs,label = predict_dl(model,compare_image)

                            gradcam_im,_ = grad_cam(compare_image,model,MODEL_MAP.get(name))

                            with col:

                                with st.container(border=True):

                                    st.subheader(name)

                                    st.image(gradcam_im,use_container_width=True)

                                    if name in ["ResNet50V2", "Xception"]:

                                        st.caption("⚠️ Grad-CAM peu interprétable")

                                    if label == compare_categorie:
                                        st.success(f"**Classe prédite : {label}**")
                                    else:
                                        st.error(f"**Classe prédite : {label}**")

                                    df=pd.DataFrame({
                                        "Classe":CLASSES_FR,
                                        "Probabilité":
                                        np.round(probs,4)
                                    })

                                    st.dataframe(
                                        df.sort_values(
                                            "Probabilité",
                                            ascending=False
                                        ).head(3),
                                        hide_index=True,
                                        use_container_width=True
                                    )
        



# =========================================================================================================
with tab3:  

    st.header("🔬 Visualisation des erreurs")  
    st.caption("Analyse des erreurs des 3 modèles les plus performants : EfficientNetV2S, EfficientNetV2M et VGG19. Nombre d'erreurs relevé sur la base de test comptabilisant 3 415 images.")   
    errors_model = st.selectbox("Erreurs modèle", [
            "EfficientNetV2S", "EfficientNetV2M", "VGG19"
        ], 
        key="errors_models",
        index=None,
        placeholder="Choisir une option")

        

    gradcam_models = ["EfficientNetV2S", "EfficientNetV2M", "VGG19", "DenseNet121"]

    imgs_gradcam, imgs_classe  = get_missclassified_img(MODEL_MAP.get(errors_model))

    if (
        st.session_state.get("errors_model_courant") != errors_model
        or "errors_imgs" not in st.session_state
    ):

        imgs, classes = get_missclassified_img(MODEL_MAP.get(errors_model))

        st.session_state.errors_imgs = imgs
        st.session_state.errors_classes = classes
        st.session_state.errors_model_courant = errors_model

        st.session_state.errors_selected_idx = None
        st.session_state.errors_selected_image = None
        st.session_state.true_class = None


    imgs_gradcam = st.session_state.errors_imgs
    imgs_classe = st.session_state.errors_classes

    if errors_model:
        with st.expander("Sélectionner une image"):
            # grille d'images ici
            cols_per_row = 6
            for i in range(0, len(imgs_gradcam), cols_per_row):
                cols = st.columns(cols_per_row)

                for j, col in enumerate(cols):
                    if i + j < len(imgs_gradcam):
                        idx = i + j
                        img_gradcam = imgs_gradcam[idx]
                        img_classe  = imgs_classe[idx] 
                
                        with col:

                            if st.session_state.errors_selected_idx == idx:
                                st.image(img_gradcam, use_container_width=True)
                                st.caption(f"Classe : {img_classe}")
                                st.button("✅", key=f"errors_button_{idx}")
                            else:
                                st.image(img_gradcam, use_container_width=True)
                                st.caption(f"Classe : {img_classe}")
                                if st.button("✔", key=f"errors_button_{idx}"):
                                    st.session_state.errors_selected_idx = idx
                                    st.session_state.errors_selected_image = img_gradcam
                                    st.session_state.true_class = img_classe
                                    st.rerun()

            nb_errors_dic = {
                "EfficientNetV2S": 35,
                "EfficientNetV2M": 37,
                "VGG19": 35,
            }
            st.caption(f"Nombre total d'erreurs : {nb_errors_dic[errors_model]}")
            gradcam_image = st.session_state.errors_selected_image   
            image_classe = st.session_state.true_class                   

        if gradcam_image and st.button("🔥 Grad-CAM", disabled=gradcam_image is None):
            with st.spinner("Analyse en cours..."):

                models = {
                    name: load_dl_model(name) for name in DL_PATH_MAP
                }

                model_names = list(models.keys())

                cols_per_row = 3

                for i in range(0, len(model_names), cols_per_row):
                    cols = st.columns(cols_per_row)

                    for j, col in enumerate(cols):
                        if i+j < len(model_names):

                            name = model_names[i+j]
                            model = models[name]

                            probs,label = predict_dl(model, gradcam_image)

                            gradcam_im, _ = grad_cam(gradcam_image, model,MODEL_MAP.get(name))

                            with col:

                                with st.container(border=True):

                                    st.subheader(name)
                                    st.image(gradcam_im,use_container_width=True)
                                    if name in ["ResNet50V2", "Xception"]:
                                        st.caption("⚠️ Grad-CAM peu interprétable")

                                    if label == image_classe:
                                        st.success(f"**Classe prédite : {label}**")
                                    else:
                                        st.error(f"**Classe prédite : {label}**")

                                    df=pd.DataFrame({
                                        "Classe":CLASSES_FR,
                                        "Probabilité":
                                        np.round(probs,4)
                                    })

                                    st.dataframe(
                                        df.sort_values(
                                            "Probabilité",
                                            ascending=False
                                        ).head(3),
                                        hide_index=True,
                                        use_container_width=True
                                    )   
                
            # if modele == "XGBoost":
            #         xgb_model = model.named_steps['xgb']
            #         kmeans = model.named_steps['kmeans']

            #         feature_names = kmeans.get_feature_names()

            #         importances = xgb_model.feature_importances_
            #         indices = np.argsort(importances)[::-1][:15]  # top 15

            #         df_imp = pd.DataFrame({
            #             "Feature": [feature_names[i] for i in indices],
            #             "Importance": importances[indices]
            #         })

            #         st.bar_chart(df_imp.set_index("Feature"))
