import streamlit as st
import pandas as pd
import altair as alt

# Sample dataset for demonstration
SAMPLE_DATA = pd.DataFrame({
    "item": ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"],
    "value": [5, 3, 6, 2, 4, 7, 5],
})


def generate_chart(query: str):
    """Return an Altair bar chart filtered by `query` (substring match).

    If `query` is empty, return chart for all items.
    """
    q = (query or "").lower().strip()
    if q:
        df = SAMPLE_DATA[SAMPLE_DATA["item"].str.contains(q)]
    else:
        df = SAMPLE_DATA

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("item:N", sort="-y", title="Item"),
            y=alt.Y("value:Q", title="Value"),
            tooltip=["item", "value"],
        )
        .properties(title=(f"Results for '{query}'" if q else "All items"))
    )

    return chart

st.set_page_config(page_title="Search Visualization", layout="centered")

st.title("Search Visualization Demo", text_alignment="center")

query = st.text_area("Enter search query", height=200)

button = st.button("Search")

placeholder = st.empty()

if button or query:
    with st.spinner("Loading visualization..."):
        chart = generate_chart(query)
        placeholder.altair_chart(chart, use_container_width=True)
