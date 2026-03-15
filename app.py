import streamlit as st
import pandas as pd
import altair as alt


st.set_page_config(page_title="Tender Market Analytics", layout="centered")

st.title("Tender Market Analytics", text_alignment="center")

query = st.text_area("Enter search query", height=200)

button = st.button("Search")

placeholder = st.empty()

if button and query.strip():
    print(query)
    # with st.spinner("Loading visualization..."):
