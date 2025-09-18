import streamlit as st
st.set_page_config(layout="wide")
import home
import history

st.sidebar.title("Navigation")

nav = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "ℹ️ History"],
    label_visibility="collapsed"  # Hides the "Go to" label
)

if nav == "🏠 Home":
    home.show()
elif nav == "ℹ️ History":
    history.show()