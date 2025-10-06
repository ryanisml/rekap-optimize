import streamlit as st
from streamlit_option_menu import option_menu
import home
import history

st.set_page_config(layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebar"] {display: block}
    [data-testid="collapsedControl"] { display: block }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["Metode K-Means", "Metode Agglomerative"],
        icons=["house", "clock-history"],
        menu_icon="cast",
        default_index=0,
    )

if selected == "Metode K-Means":
    home.main()
elif selected == "Metode Agglomerative":
    history.main()