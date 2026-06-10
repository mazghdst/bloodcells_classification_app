import streamlit as st

pg = st.navigation([
    st.Page("pages/0_Accueil.py", title="🏠 Accueil"),
    st.Page("pages/1_Données.py", title="🔬 Données"),
    st.Page("pages/2_Prétraitement.py", title="⚙️ Prétraitement"),
    st.Page("pages/3_Modèles.py", title="🧠 Modèles"),
    st.Page("pages/4_Classification_interactive.py", title="🔍 Classification interactive"),
    #st.Page("pages/5_Conclusion.py", title="💡 Conclusion"),
])

pg.run()
