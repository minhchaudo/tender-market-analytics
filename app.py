import streamlit as st
import altair as alt
import pandas as pd
from datetime import datetime
import numpy as np
from query_logic import query, SQLCompileError

# def query_db(query: str):
#     print(query)
#     return pd.read_csv("kit_test_df_clean.csv")

class colnames:
    contractor_name = "contractor_name"
    product_name = "product_name"
    product_code = "product_code"
    product = "product"
    raw_manufacturer = "manufacturer_raw"
    raw_origin = "origin_raw"
    country_origin = "country_of_origin"
    region_origin = "region_of_origin"
    quantity = "quantity"
    unit = "unit"
    unit_price = "unit_price"
    total_price = "total_price"
    itb_code = "itb_code"
    url = "URL"
    bid_package_name = "bid_package_name"
    investor = "investor"
    location = "location"
    posting_date = "posting_date"
    closing_date = "closing_date"
    origin = "origin"
    manufacturer = "manufacturer"
    province = "province"

def handle_search_click():
    query_str = st.session_state["text: query"].strip()
    if query_str == "":
        st.session_state["query_error"] = 1
        return
    else:
        try:
            df = query(query_str).dropna()
        except Exception as e:
            st.session_state["query_error"] = e
            return
    st.session_state["query_error"] = None
    df[colnames.posting_date] = pd.to_datetime(df[colnames.posting_date], format="%Y-%m-%d %H:%M:%S")
    df[colnames.closing_date] = pd.to_datetime(df[colnames.closing_date], format="%Y-%m-%d %H:%M:%S")

    st.session_state["data"] = df
    st.session_state[f"Filter: {colnames.investor}"] = set()
    st.session_state[f"Filter: {colnames.contractor_name}"] = set()
    st.session_state[f"Filter: {colnames.manufacturer}"] = set()
    st.session_state[f"Filter: {colnames.country_origin}"] = set()
    st.session_state[f"Filter: {colnames.region_origin}"] = set()
    st.session_state[f"Filter: {colnames.quantity}"] = (df[colnames.quantity].min(), df[colnames.quantity].max())
    st.session_state[f"Filter: {colnames.unit_price}"] = (df[colnames.unit_price].min(), df[colnames.unit_price].max())
    st.session_state[f"Filter: {colnames.total_price}"] = (df[colnames.total_price].min(), df[colnames.total_price].max())
    st.session_state[f"Filter: {colnames.posting_date}"] = (df[colnames.posting_date].min(), df[colnames.posting_date].max())
    st.session_state[f"Filter: {colnames.closing_date}"] = (df[colnames.closing_date].min(), df[colnames.closing_date].max())

def generate_label_filter(cname: str):
    label_col, button_col = st.columns([5, 1], gap="small", vertical_alignment="center")
    with label_col:
        st.markdown(f"**filter {cname.replace("_", " ")}**")
    with button_col:
        st.button("🗑️", key=f"button: erase {cname}", help="Clear this filter", on_click=lambda: st.session_state[f"Filter: {cname}"].clear(), type="tertiary", use_container_width=True)
    
    def handle_selectbox_change():
        st.session_state[f"Filter: {cname}"].add(st.session_state[f"selectbox: {cname}"])
        st.session_state[f"selectbox: {cname}"] = None
    st.selectbox(f"Filter {cname.replace("_", " ")}", sorted(set(st.session_state["data"][cname])), key=f"selectbox: {cname}", index=None, placeholder=f"Select {cname} to filter", label_visibility="collapsed", on_change=handle_selectbox_change)

    st.pills("Current filters", sorted({"❌ " + s for s in st.session_state[f"Filter: {cname}"]}), key=f"pills: {cname}", label_visibility="collapsed", on_change=lambda: st.session_state[f"Filter: {cname}"].discard(st.session_state[f"pills: {cname}"][2:]))


def generate_checkboxes(cname: str):
    df = st.session_state["data"]
    elems = set(df[cname])

    st.markdown(f"**{cname.replace("_", " ")}**")
    for e in sorted(elems):
        st.checkbox(e.replace("_", " "), key=f"checkbox: {cname}: {e}", value=e in st.session_state[f"Filter: {cname}"], on_change=lambda e=e: st.session_state[f"Filter: {cname}"].add(e) if st.session_state.get(f"checkbox: {cname}: {e}", False) else st.session_state[f"Filter: {cname}"].discard(e))

def generate_slider(cname: str):
    df = st.session_state["data"]
    
    st.slider(f"{cname.replace("_", " ")}", min_value=df[cname].min(), max_value=df[cname].max(), key=f"Filter: {cname}")

def generate_date_range(cname: str):
    df = st.session_state["data"]

    st.date_input(f"{cname.replace("_", " ")}", min_value=df[cname].min(), max_value=df[cname].max(), key=f"Filter: {cname}")

st.set_page_config(page_title="Tender Market Analytics", layout="wide")
st.title("Tender Market Analytics", text_alignment="center")
left_col, right_col = st.columns([2, 8], gap="large")
with left_col:
    with st.container(height=650, border=True):
        with st.form("Query", border=False):
            st.text_area("Search query", height=100, key="text: query")
            if "query_error" in st.session_state and st.session_state["query_error"] != None:
                if st.session_state["query_error"] == 1:
                    st.error("Please enter your query first!")
                elif isinstance(st.session_state["query_error"], SQLCompileError):
                    st.error(str(st.session_state["query_error"]))
                else:
                    st.error(str(st.session_state["query_error"]))
                    # st.error("Unknown error occured! Please check your query and try again.")
            st.form_submit_button("Search", on_click=handle_search_click, width="stretch")
        if "data" in st.session_state and not st.session_state["data"].empty:
            generate_label_filter(colnames.investor)
            generate_label_filter(colnames.contractor_name)
            generate_label_filter(colnames.manufacturer)
            generate_label_filter(colnames.country_origin)
            generate_label_filter(colnames.region_origin)
            generate_slider(colnames.quantity)
            generate_slider(colnames.unit_price)
            generate_slider(colnames.total_price)
            generate_date_range(colnames.posting_date)
            generate_date_range(colnames.closing_date)
with right_col:
    tab1, tab2 = st.tabs(["Analyze", "Predict"])
    with tab1:
        with st.container(height=600, border=True):
            if not "data" in st.session_state:
                st.info("Query something, insight will be drawn here.")
            elif st.session_state["data"].empty:
                st.info("No result. Try another query.")
            else:
                raw_df = st.session_state["data"]
                try:
                    s_post, e_post = st.session_state[f"Filter: {colnames.posting_date}"]
                    s_bid, e_bid = st.session_state[f"Filter: {colnames.closing_date}"]
                except Exception:
                    s_post = st.session_state[f"Filter: {colnames.posting_date}"][0]
                    e_post = datetime.now()
                    s_bid = st.session_state[f"Filter: {colnames.closing_date}"][0]
                    e_bid = datetime.now()
                df = raw_df[
                    (raw_df[colnames.investor].isin(st.session_state[f"Filter: {colnames.investor}"]) if st.session_state[f"Filter: {colnames.investor}"] else True) &
                    (raw_df[colnames.contractor_name].isin(st.session_state[f"Filter: {colnames.contractor_name}"]) if st.session_state[f"Filter: {colnames.contractor_name}"] else True) &
                    (raw_df[colnames.manufacturer].isin(st.session_state[f"Filter: {colnames.manufacturer}"]) if st.session_state[f"Filter: {colnames.manufacturer}"] else True) &
                    (raw_df[colnames.country_origin].isin(st.session_state[f"Filter: {colnames.country_origin}"]) if st.session_state[f"Filter: {colnames.country_origin}"] else True) &
                    (raw_df[colnames.region_origin].isin(st.session_state[f"Filter: {colnames.region_origin}"]) if st.session_state[f"Filter: {colnames.region_origin}"] else True) &
                    (raw_df[colnames.quantity].between(*st.session_state[f"Filter: {colnames.quantity}"])) &
                    (raw_df[colnames.unit_price].between(*st.session_state[f"Filter: {colnames.unit_price}"])) &
                    (raw_df[colnames.total_price].between(*st.session_state[f"Filter: {colnames.total_price}"])) &
                    (raw_df[colnames.posting_date].between(pd.Timestamp(s_post), pd.Timestamp(e_post) + pd.Timedelta(days=1))) &
                    (raw_df[colnames.closing_date].between(pd.Timestamp(s_bid), pd.Timestamp(e_bid) + pd.Timedelta(days=1)))
                ]
                if df.empty:
                    st.info("No result. Try adjusting your filters.")
                else:
                    with st.container():
                        st.subheader("**Distribution of unit prices**")

                        s = df[colnames.unit_price]
                        bins = 100
                        counts, _ = np.histogram(s, bins=bins)
                        stats_df = pd.DataFrame([{
                            "x0": 0, "x1": float(s.max()), "y0": 0, "y1": int(counts.max()) if counts.size else 0,
                            "mean": s.mean(), "std": s.std(), "min": s.min(), "q1": s.quantile(0.25), "q3": s.quantile(0.75), "max": s.max()
                        }])

                        bars = alt.Chart(df).mark_bar().encode(
                            x=alt.X(colnames.unit_price, bin=alt.Bin(maxbins=bins), title="Unit price (million VND)",
                                    axis=alt.Axis(labelExpr="datum.value / 1e6"), scale=alt.Scale(domainMin=0)),
                            y=alt.Y("count()", title="Count"),
                        )

                        hover = alt.Chart(stats_df).mark_rect(opacity=0).encode(
                            x="x0", x2="x1", y="y0", y2="y1",
                            tooltip=[
                                alt.Tooltip("mean", title="Mean", format=",.2f"),
                                alt.Tooltip("std", title="Standard deviation", format=",.2f"),
                                alt.Tooltip("min", title="Min", format=",.2f"),
                                alt.Tooltip("q1", title="Q1", format=",.2f"),
                                alt.Tooltip("q3", title="Q3", format=",.2f"),
                                alt.Tooltip("max", title="Max", format=",.2f")
                            ],
                        )

                        st.altair_chart(
                            (bars + hover).properties(height=alt.Step(36)).configure_view(stroke=None).configure_axis(
                                labelColor="black", titleColor="black", labelFontSize=16, titleFontSize=16
                            ),
                            width="stretch",
                        )
                    with st.container():
                        st.subheader("**Top contractors**")

                        data = df.groupby(colnames.contractor_name, as_index=False)[colnames.total_price].sum().sort_values(by=colnames.total_price, ascending=False).head(5)
                        st.altair_chart(alt.Chart(data).mark_bar().encode(
                                x=alt.X(colnames.total_price, title="Total price (billion VND)", axis=alt.Axis(labelExpr="datum.value / 1e9")),
                                y=alt.Y(colnames.contractor_name, sort="-x", axis=alt.Axis(labelLimit=400, labelOverlap=False), title=None)
                            ).properties(height=alt.Step(36)).configure_view(stroke=None).configure_axis(labelColor="black", titleColor="black", labelFontSize=16, titleFontSize=16), width="stretch")
                    with st.container():
                        st.subheader("**Top customers**")

                        data = df.groupby(colnames.investor, as_index=False)[colnames.total_price].sum().sort_values(by=colnames.total_price, ascending=False).head(5)
                        st.altair_chart(alt.Chart(data).mark_bar().encode(
                                x=alt.X(colnames.total_price, title="Total price (billion VND)", axis=alt.Axis(labelExpr="datum.value / 1e9")),
                                y=alt.Y(colnames.investor, sort="-x", axis=alt.Axis(labelLimit=400, labelOverlap=False), title=None)
                            ).properties(height=alt.Step(36)).configure_view(stroke=None).configure_axis(labelColor="black", titleColor="black", labelFontSize=16, titleFontSize=16), width="stretch")
                    with st.container():
                        st.subheader("**Unit price by contractor**")

                        top_bidder = set(df.groupby(colnames.contractor_name, as_index=False)[colnames.total_price].sum().sort_values(by=colnames.total_price, ascending=False).head(10)[colnames.contractor_name])
                        data = df[df[colnames.contractor_name].isin(top_bidder)]
                        st.altair_chart(alt.Chart(data).mark_boxplot().encode(
                                x=alt.X(colnames.unit_price, title="Unit price (million VND)", axis=alt.Axis(labelExpr="datum.value / 1e6")),
                                y=alt.Y(colnames.contractor_name, sort=alt.EncodingSortField(field=colnames.unit_price, op="median", order="descending"), axis=alt.Axis(labelLimit=400, labelOverlap=False), title=None)
                            ).properties(height=alt.Step(36)).configure_view(stroke=None).configure_axis(labelColor="black", titleColor="black", labelFontSize=16, titleFontSize=16), width="stretch")
                    with st.container():
                        st.subheader("**Unit price by origin**")

                        top_origin = set(df.groupby(colnames.country_origin, as_index=False)[colnames.total_price].sum().sort_values(by=colnames.total_price, ascending=False).head(10)[colnames.country_origin])
                        data = df[df[colnames.country_origin].isin(top_origin)]
                        st.altair_chart(alt.Chart(data).mark_boxplot().encode(
                                x=alt.X(colnames.unit_price, title="Unit price (million VND)", axis=alt.Axis(labelExpr="datum.value / 1e6")),
                                y=alt.Y(colnames.country_origin, sort=alt.EncodingSortField(field=colnames.unit_price, op="median", order="descending"), axis=alt.Axis(labelLimit=400, labelOverlap=False), title=None)
                            ).properties(height=alt.Step(36)).configure_view(stroke=None).configure_axis(labelColor="black", titleColor="black", labelFontSize=16, titleFontSize=16), width="stretch")
                    with st.container():
                        st.subheader("**Unit price by manufacturer**")

                        top_origin = set(df.groupby(colnames.manufacturer, as_index=False)[colnames.total_price].sum().sort_values(by=colnames.total_price, ascending=False).head(10)[colnames.manufacturer])
                        data = df[df[colnames.manufacturer].isin(top_origin)]
                        st.altair_chart(alt.Chart(data).mark_boxplot().encode(
                                x=alt.X(colnames.unit_price, title="Unit price (million VND)", axis=alt.Axis(labelExpr="datum.value / 1e6")),
                                y=alt.Y(colnames.manufacturer, sort=alt.EncodingSortField(field=colnames.unit_price, op="median", order="descending"), axis=alt.Axis(labelLimit=400, labelOverlap=False), title=None)
                            ).properties(height=alt.Step(36)).configure_view(stroke=None).configure_axis(labelColor="black", titleColor="black", labelFontSize=16, titleFontSize=16), width="stretch")
                    with st.container():
                        st.subheader("**Total value by country of origin**")

                        data_by_origin = df.groupby(colnames.country_origin, as_index=False)[colnames.total_price].sum().sort_values(by=colnames.total_price, ascending=False)
                        top9 = data_by_origin.head(9)
                        rest = data_by_origin.iloc[9:][colnames.total_price].sum()

                        data = pd.concat([top9, pd.DataFrame([{colnames.country_origin: "Others", colnames.total_price: rest}])], ignore_index=True) if rest > 0 else top9
                        st.altair_chart(alt.Chart(data).mark_arc().encode(
                            theta=alt.Theta(colnames.total_price, title="Total price", sort="x"),
                            color=alt.Color(colnames.country_origin, legend=alt.Legend(title=colnames.country_origin, labelColor="black", titleColor="black", labelFontSize=16, titleFontSize=16)),
                            order=alt.Order(colnames.total_price, sort="ascending"),
                            tooltip=[
                                alt.Tooltip(colnames.country_origin, title=colnames.country_origin),
                                alt.Tooltip(colnames.total_price, title="Total price", format=",.0f")
                            ]), width="stretch")
                    with st.container():
                        st.subheader("**Total value by region of origin**")

                        data_by_origin = df.groupby(colnames.region_origin, as_index=False)[colnames.total_price].sum().sort_values(by=colnames.total_price, ascending=False)
                        top9 = data_by_origin.head(9)
                        rest = data_by_origin.iloc[9:][colnames.total_price].sum()

                        data = pd.concat([top9, pd.DataFrame([{colnames.region_origin: "Others", colnames.total_price: rest}])], ignore_index=True) if rest > 0 else top9
                        st.altair_chart(alt.Chart(data).mark_arc().encode(
                            theta=alt.Theta(colnames.total_price, title="Total price", sort="x"),
                            color=alt.Color(colnames.region_origin, legend=alt.Legend(title=colnames.region_origin, labelColor="black", titleColor="black", labelFontSize=16, titleFontSize=16)),
                            order=alt.Order(colnames.total_price, sort="ascending"),
                            tooltip=[
                                alt.Tooltip(colnames.region_origin, title=colnames.region_origin),
                                alt.Tooltip(colnames.total_price, title="Total price", format=",.0f")
                            ]), width="stretch")
                    with st.container():
                        st.subheader("**Total value by manufacturer**")

                        data_by_origin = df.groupby(colnames.manufacturer, as_index=False)[colnames.total_price].sum().sort_values(by=colnames.total_price, ascending=False)
                        top9 = data_by_origin.head(9)
                        rest = data_by_origin.iloc[9:][colnames.total_price].sum()

                        data = pd.concat([top9, pd.DataFrame([{colnames.manufacturer: "Others", colnames.total_price: rest}])], ignore_index=True) if rest > 0 else top9
                        st.altair_chart(alt.Chart(data).mark_arc().encode(
                            theta=alt.Theta(colnames.total_price, title="Total price", sort="x"),
                            color=alt.Color(colnames.manufacturer, legend=alt.Legend(title=colnames.manufacturer, labelColor="black", titleColor="black", labelFontSize=16, titleFontSize=16)),
                            order=alt.Order(colnames.total_price, sort="ascending"),
                            tooltip=[
                                alt.Tooltip(colnames.manufacturer, title=colnames.manufacturer),
                                alt.Tooltip(colnames.total_price, title="Total price", format=",.0f")
                            ]), width="stretch")
    with tab2:
        with st.container(height=600, border=True):
            if not "data" in st.session_state:
                st.info("Query something before predicting.")
            elif st.session_state["data"].empty:
                st.info("No result. Try another query before predicting.")
            else:
                with st.form("Parameters"):
                    st.subheader("Parameters for the prediction")
                    banner = st.container()
                    left_col, right_col = st.columns([1, 1], gap="large")
                    with left_col:
                        st.selectbox(colnames.investor, sorted(set(st.session_state["data"][colnames.investor]) | {"Other"}), index=None, key=f"Predict: {colnames.investor}")
                        st.selectbox("Province", sorted(set(st.session_state["data"][colnames.province]) | {"Other"}), index=None, key=f"Predict: {colnames.province}")
                        inner_left_col, inner_right_col = st.columns([1, 1], gap="medium")
                        with inner_left_col:
                            st.number_input(colnames.quantity, key=f"Predict: {colnames.quantity}", min_value=1)
                        with inner_right_col:
                            st.date_input("Bidding date", key=f"Predict: {colnames.posting_date}")
                    with right_col:
                        st.selectbox(colnames.manufacturer, sorted(set(st.session_state["data"][colnames.manufacturer]) | {"Other"}), index=None, key=f"Predict: {colnames.manufacturer}")
                        st.selectbox("Country of origin", sorted(set(st.session_state["data"][colnames.country_origin]) | {"Other"}), index=None, key=f"Predict: {colnames.country_origin}")
                    _, button_space, _ = st.columns([1, 1, 1], gap="large")
                    with button_space:
                        if st.form_submit_button("Search", width="stretch"):
                            filled = (st.session_state[f"Predict: {colnames.investor}"] is not None) and (st.session_state[f"Predict: {colnames.province}"] is not None) and (st.session_state[f"Predict: {colnames.manufacturer}"] is not None) and (st.session_state[f"Predict: {colnames.country_origin}"] is not None)
                            if not filled:
                                with banner:
                                    st.error("Please fill all the required fields!")
