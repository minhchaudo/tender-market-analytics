import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from query_logic import query, SQLCompileError
from prob_modeling import train_model, predict
from tooltips import *
from llm import llm
from get_summary import get_info
from price_rec import recommend_price
import time

class colnames:
    contractor_name = "contractor_name"
    product = "product"
    country_origin = "country_of_origin"
    region_origin = "region_of_origin"
    quantity = "quantity"
    unit = "unit"
    unit_price = "unit_price"
    total_price = "total_price"
    investor = "investor"
    posting_date = "posting_date"
    closing_date = "closing_date"
    manufacturer = "manufacturer"
    province = "province"

class EmptyQueryError(Exception):
    pass
class PredictFieldMissingError(Exception):
    pass
class TrainAndPredictError(Exception):
    pass

def handle_search_click():
    banner_search.empty()
    query_str = st.session_state["text: query"].strip()
    if query_str == "":
        st.session_state["query_error"] = EmptyQueryError()
        return
    else:
        try:
            df = query(query_str)
        except Exception as e:
            st.session_state["query_error"] = e
            return
    st.session_state["query_error"] = None
    st.session_state["query_str"] = query_str
    df[colnames.posting_date] = pd.to_datetime(df[colnames.posting_date], format="%Y-%m-%d %H:%M:%S")
    df[colnames.closing_date] = pd.to_datetime(df[colnames.closing_date], format="%Y-%m-%d %H:%M:%S")

    st.session_state["data"] = df
    st.session_state["filtered_data"] = None
    st.session_state[f"Filter: {colnames.investor}"] = set()
    st.session_state[f"Filter: {colnames.contractor_name}"] = set()
    st.session_state[f"Filter: {colnames.manufacturer}"] = set()
    st.session_state[f"Filter: {colnames.country_origin}"] = set()
    st.session_state[f"Filter: {colnames.region_origin}"] = set()
    st.session_state[f"Filter: {colnames.province}"] = set()
    st.session_state[f"Filter: {colnames.quantity}"] = (
        df[colnames.quantity].min(),
        df[colnames.quantity].max(),
    )
    st.session_state[f"Filter: {colnames.unit_price}"] = (
        df[colnames.unit_price].min(),
        df[colnames.unit_price].max(),
    )
    st.session_state[f"Filter: {colnames.total_price}"] = (
        df[colnames.total_price].min(),
        df[colnames.total_price].max(),
    )
    st.session_state[f"Filter: min {colnames.posting_date}"] = df[colnames.posting_date].min()
    st.session_state[f"Filter: max {colnames.posting_date}"] = df[colnames.posting_date].max()
    st.session_state[f"Filter: min {colnames.closing_date}"] = df[colnames.closing_date].min()
    st.session_state[f"Filter: max {colnames.closing_date}"] = df[colnames.closing_date].max()

    st.session_state[f"Predict: {colnames.investor}"] = None
    st.session_state[f"Predict: {colnames.province}"] = None
    st.session_state[f"Predict: {colnames.quantity}"] = None
    st.session_state[f"Predict: {colnames.closing_date}"] = None
    st.session_state[f"Predict: {colnames.manufacturer}"] = None
    st.session_state[f"Predict: {colnames.country_origin}"] = None
    st.session_state[f"Predict: cost"] = None

    st.session_state["predict_error"] = None
    st.session_state["predict it"] = False
    st.session_state["model_default_all"] = None
    st.session_state["model_default_filtered"] = None 
    st.session_state["model_auto_all"] = None
    st.session_state["model_auto_filtered"] = None
    st.session_state["model_filtered_training_data"] = None
    st.session_state["latest_pred"] = None
    
def handle_predict_click():
    banner_predict.empty()
    if not (
        (st.session_state[f"Predict: {colnames.investor}"] is not None)
        and (st.session_state[f"Predict: {colnames.province}"] is not None)
        and (st.session_state[f"Predict: {colnames.quantity}"] is not None)
        and (st.session_state[f"Predict: {colnames.closing_date}"] is not None)
        and (st.session_state[f"Predict: {colnames.manufacturer}"] is not None)
        and (st.session_state[f"Predict: {colnames.country_origin}"] is not None)
        and (st.session_state[f"Predict: model_class"] is not None)
    ):
        st.session_state["predict_error"] = PredictFieldMissingError()
    else:
        st.session_state["predict_error"] = None
        st.session_state["predict it"] = True


def generate_label_filter(cname: str):
    label_col, button_col = st.columns([10, 1], gap="small", vertical_alignment="center")
    with label_col:
        st.markdown(f"**{cname.replace("_", " ").capitalize()}**", help=tooltip_filter_map[cname])
    with button_col:
        st.button(
            "🗑️",
            key=f"button: erase {cname}",
            help="Clear this filter",
            on_click=lambda: st.session_state[f"Filter: {cname}"].clear(),
            type="tertiary",
            use_container_width=True,
        )

    def handle_selectbox_change():
        st.session_state[f"Filter: {cname}"].add(st.session_state[f"selectbox: {cname}"])
        st.session_state[f"selectbox: {cname}"] = None

    st.selectbox(
        f"{cname.replace("_", " ").capitalize()}",
        sorted(set(st.session_state["data"][cname])),
        key=f"selectbox: {cname}",
        index=None,
        placeholder=f"Select {cname.replace("_", " ")} to filter",
        label_visibility="collapsed",
        on_change=handle_selectbox_change,
    )
                                                     
    st.pills(
        "Current filters",
        sorted({"❌ " + s for s in st.session_state[f"Filter: {cname}"]}),
        key=f"pills: {cname}",
        label_visibility="collapsed",
        on_change=lambda: st.session_state[f"Filter: {cname}"].discard(st.session_state[f"pills: {cname}"][2:]),
    )


def generate_checkboxes(cname: str):
    df = st.session_state["data"]
    elems = set(df[cname])

    st.markdown(f"**{cname.replace("_", " ")}**")
    for e in sorted(elems):
        st.checkbox(
            e.replace("_", " "),
            key=f"checkbox: {cname}: {e}",
            value=e in st.session_state[f"Filter: {cname}"],
            on_change=lambda e=e: (
                st.session_state[f"Filter: {cname}"].add(e)
                if st.session_state.get(f"checkbox: {cname}: {e}", False)
                else st.session_state[f"Filter: {cname}"].discard(e)
            ),
        )


def generate_slider(cname: str):
    df = st.session_state["data"]

    st.markdown(f"**{cname.replace("_", " ").capitalize()}**", help=tooltip_filter_map[cname])
    st.slider(
        f"**{cname.replace("_", " ").capitalize()}**",
        label_visibility="collapsed",
        min_value=df[cname].min(),
        max_value=df[cname].max(),
        key=f"Filter: {cname}",
    )


def generate_date_range(cname: str):
    df = st.session_state["data"]
    min_date = df[cname].min()
    max_date = df[cname].max()

    left_col, right_col = st.columns([5, 1], gap="xxlarge", vertical_alignment="center")
    with left_col:
        st.markdown(f"**{cname.replace("_", " ").capitalize()}**", help=tooltip_filter_map[cname])

    def reset_date(cname, min_date, max_date):
        st.session_state[f"Filter: min {cname}"] = min_date
        st.session_state[f"Filter: max {cname}"] = max_date

    with right_col:
        st.button(
            "🗑️",
            key=f"button: erase {cname}",
            help="Reset this filter",
            on_click=lambda: reset_date(cname, min_date, max_date),
            type="tertiary",
            use_container_width=True,
        )

    left_col, right_col = st.columns([1, 1], gap="small", vertical_alignment="center")
    with left_col:
        st.date_input(
            "From",
            min_value=min_date,
            max_value=max_date,
            format="DD-MM-YYYY",
            key=f"Filter: min {cname}",
        )
    with right_col:
        st.date_input(
            "To",
            min_value=min_date,
            max_value=max_date,
            format="DD-MM-YYYY",
            key=f"Filter: max {cname}",
        )


st.set_page_config(page_title="Tender Market Analytics", layout="wide")
st.markdown("""
<style>
    [data-testid="stHeader"] {
        display: none !important;
    }
    [data-testid="stMainBlockContainer"] {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
    }
</style>
""", unsafe_allow_html=True)
global_error_banner = st.empty()
global_error_banner.empty()

st.title("Tender Market Analytics", text_alignment="center")
left_col, right_col = st.columns([3, 7], gap="large")
ALL_CONTENT_HEIGHT = 700
with left_col:
    st.markdown("<div style='height: 58px;'></div>", unsafe_allow_html=True)
    with st.container(height=ALL_CONTENT_HEIGHT, border=True):
        with st.form("Query", border=False):
            st.markdown("### Search query", help=help_query_box)
            raw_df_rows_count = st.markdown(f"Retrieved **0** rows.")
            st.text_area("Search query", height=200, key="text: query", label_visibility="collapsed")
            st.form_submit_button("Search", on_click=handle_search_click, width="stretch")
            banner_search = st.empty()
            with banner_search:
                if "query_error" in st.session_state and st.session_state["query_error"] is not None:
                    print("=========== Exception ===========")
                    print(st.session_state["query_error"])
                    if isinstance(st.session_state["query_error"], EmptyQueryError):
                        st.error("Please enter your query first.")
                    elif isinstance(st.session_state["query_error"], SQLCompileError):
                        st.error(str(st.session_state["query_error"]))
                    else:
                        st.error("An error has occurred! Please check your query and try again. Hint: if your query involves multiple fields, wrap field-specific conditions in parentheses.")
            with raw_df_rows_count:
                if "data" in st.session_state and not st.session_state["data"].empty:
                    count = len(st.session_state["data"])
                else:
                    count = 0
                st.markdown(f"Retrieved **{count}** rows.")
        if "data" in st.session_state and not st.session_state["data"].empty:
            st.markdown(f"### Filter")
            filtered_df_rows_count = st.markdown(f"Filtered **{len(st.session_state["data"])}** rows.")
            generate_label_filter(colnames.investor)
            generate_label_filter(colnames.contractor_name)
            generate_label_filter(colnames.manufacturer)
            generate_label_filter(colnames.country_origin)
            generate_label_filter(colnames.region_origin)
            generate_label_filter(colnames.province)
            generate_slider(colnames.quantity)
            generate_slider(colnames.unit_price)
            generate_slider(colnames.total_price)
            generate_date_range(colnames.posting_date)
            generate_date_range(colnames.closing_date)
with right_col:
    tab1, tab2 = st.tabs(["Analyze", "Predict"])
    with tab1:
        with st.container(height=ALL_CONTENT_HEIGHT, border=True):
            if not "data" in st.session_state:
                st.info("Query something, insights will be drawn here.")
            elif st.session_state["data"].empty:
                st.info("No result. Try another query.")
            else:
                raw_df = st.session_state["data"]
                s_post = (
                    st.session_state[f"Filter: min {colnames.posting_date}"]
                    if st.session_state[f"Filter: min {colnames.posting_date}"] is not None
                    else raw_df[colnames.posting_date].min()
                )
                e_post = (
                    st.session_state[f"Filter: max {colnames.posting_date}"]
                    if st.session_state[f"Filter: max {colnames.posting_date}"] is not None
                    else raw_df[colnames.posting_date].max()
                )
                s_bid = (
                    st.session_state[f"Filter: min {colnames.closing_date}"]
                    if st.session_state[f"Filter: min {colnames.closing_date}"] is not None
                    else raw_df[colnames.closing_date].min()
                )
                e_bid = (
                    st.session_state[f"Filter: max {colnames.closing_date}"]
                    if st.session_state[f"Filter: max {colnames.closing_date}"] is not None
                    else raw_df[colnames.closing_date].max()
                )
                df = raw_df[
                    (
                        raw_df[colnames.investor].isin(st.session_state[f"Filter: {colnames.investor}"])
                        if st.session_state[f"Filter: {colnames.investor}"]
                        else True
                    )
                    & (
                        raw_df[colnames.contractor_name].isin(st.session_state[f"Filter: {colnames.contractor_name}"])
                        if st.session_state[f"Filter: {colnames.contractor_name}"]
                        else True
                    )
                    & (
                        raw_df[colnames.manufacturer].isin(st.session_state[f"Filter: {colnames.manufacturer}"])
                        if st.session_state[f"Filter: {colnames.manufacturer}"]
                        else True
                    )
                    & (
                        raw_df[colnames.country_origin].isin(st.session_state[f"Filter: {colnames.country_origin}"])
                        if st.session_state[f"Filter: {colnames.country_origin}"]
                        else True
                    )
                    & (
                        raw_df[colnames.region_origin].isin(st.session_state[f"Filter: {colnames.region_origin}"])
                        if st.session_state[f"Filter: {colnames.region_origin}"]
                        else True
                    )
                    & (
                        raw_df[colnames.province].isin(st.session_state[f"Filter: {colnames.province}"])
                        if st.session_state[f"Filter: {colnames.province}"]
                        else True
                    )
                    & (raw_df[colnames.quantity].between(*st.session_state[f"Filter: {colnames.quantity}"]))
                    & (raw_df[colnames.unit_price].between(*st.session_state[f"Filter: {colnames.unit_price}"]))
                    & (raw_df[colnames.total_price].between(*st.session_state[f"Filter: {colnames.total_price}"]))
                    & (
                        raw_df[colnames.posting_date].between(
                            pd.Timestamp(s_post),
                            pd.Timestamp(e_post) + pd.Timedelta(seconds=1),
                        )
                    )
                    & (
                        raw_df[colnames.closing_date].between(
                            pd.Timestamp(s_bid),
                            pd.Timestamp(e_bid) + pd.Timedelta(seconds=1),
                        )
                    )
                ]
                with filtered_df_rows_count:
                    if len(df) == len(raw_df):
                        st.markdown(f"No filter applied.")
                    else:
                        st.markdown(f"Filtered **{len(df)}** rows.")
                if df.empty:
                    st.info("No result. Try adjusting your filters.")
                else:
                    st.session_state["filtered_data"] = df

                    all_cols = [
                        v for k, v in colnames.__dict__.items()
                        if not k.startswith("__") and not callable(v)
                    ]
                    df = df[all_cols]
                    df = df.copy()
                    if df[colnames.unit_price].max() < 1e3:
                        unit_price_factor = 1
                        unit_unit_price = "VND"
                    elif df[colnames.unit_price].max() < 1e6:
                        unit_price_factor = 1e3
                        df[colnames.unit_price] /= 1e3
                        unit_unit_price = "thousand VND"
                    elif df[colnames.unit_price].max() < 1e9:
                        unit_price_factor = 1e6
                        df[colnames.unit_price] /= 1e6
                        unit_unit_price = "million VND"
                    else:
                        unit_price_factor = 1e9
                        df[colnames.unit_price] /= 1e9
                        unit_unit_price = "billion VND"

                    if df[colnames.total_price].max() < 1e3:
                        total_price_factor = 1
                        unit_total_price = "VND"
                    elif df[colnames.total_price].max() < 1e6:
                        total_price_factor = 1e3
                        df[colnames.total_price] /= 1e3
                        unit_total_price = "thousand VND"
                    elif df[colnames.total_price].max() < 1e9:
                        total_price_factor = 1e6
                        df[colnames.total_price] /= 1e6
                        unit_total_price = "million VND"
                    else:
                        total_price_factor = 1e9
                        df[colnames.total_price] /= 1e9
                        unit_total_price = "billion VND"

                    with st.container():
                        st.subheader("Top investors", help=help_top_investors)

                        data = (
                            df.groupby(colnames.investor, as_index=False)[colnames.total_price]
                            .sum()
                            .sort_values(by=colnames.total_price, ascending=False)
                            .head(10)
                        )
                        st.altair_chart(
                            alt.Chart(data)
                            .transform_calculate(
                                total_price_vnd=f"datum[{colnames.total_price!r}] * {total_price_factor}"
                            )
                            .mark_bar()
                            .encode(
                                x=alt.X(
                                    colnames.total_price,
                                    title=f"Total value ({unit_total_price})",
                                ),
                                y=alt.Y(
                                    colnames.investor,
                                    title="Investor",
                                    sort="-x",
                                    axis=alt.Axis(labelLimit=400, labelOverlap=False, title=None),
                                ),
                                tooltip=[
                                    alt.Tooltip(
                                        colnames.investor,
                                        title="Investor",
                                    ),
                                    alt.Tooltip(
                                        "total_price_vnd:Q",
                                        title="Total value (VND)",
                                        format=",.0f",
                                    ),
                                ],
                            )
                            .properties(height=alt.Step(36))
                            .configure_view(stroke=None)
                            .configure_axis(
                                labelColor="black",
                                titleColor="black",
                                labelFontSize=16,
                                titleFontSize=16,
                            ),
                            width="stretch",
                        )
                    with st.container():
                        st.subheader("Top contractors", help=help_top_contractors)

                        data = (
                            df.groupby(colnames.contractor_name, as_index=False)[colnames.total_price]
                            .sum()
                            .sort_values(by=colnames.total_price, ascending=False)
                            .head(10)
                        )
                        st.altair_chart(
                            alt.Chart(data)
                            .transform_calculate(
                                total_price_vnd=f"datum['{colnames.total_price}'] * {total_price_factor}"
                            )
                            .mark_bar()
                            .encode(
                                x=alt.X(
                                    colnames.total_price,
                                    title=f"Total value ({unit_total_price})",
                                ),
                                y=alt.Y(
                                    colnames.contractor_name,
                                    title="Contractor",
                                    sort="-x",
                                    axis=alt.Axis(labelLimit=400, labelOverlap=False, title=None),
                                ),
                                tooltip=[
                                    alt.Tooltip(
                                        colnames.contractor_name,
                                        title="Contractor",
                                    ),
                                    alt.Tooltip(
                                        "total_price_vnd:Q",
                                        title="Total value (VND)",
                                        format=",.0f",
                                    ),
                                ],
                            )
                            .properties(height=alt.Step(36))
                            .configure_view(stroke=None)
                            .configure_axis(
                                labelColor="black",
                                titleColor="black",
                                labelFontSize=16,
                                titleFontSize=16,
                            ),
                            width="stretch",
                        ) 
                    with st.container():
                        st.subheader("Unit price distribution", help=help_unit_price)

                        bins = 100

                        bars = (
                            alt.Chart(df)
                            .transform_bin(
                                ["bin_start", "bin_end"],
                                field=colnames.unit_price,
                                bin=alt.Bin(maxbins=bins),
                            )
                            .transform_calculate(
                                bin_start_vnd=f"datum.bin_start * {unit_price_factor}",
                                bin_end_vnd=f"datum.bin_end * {unit_price_factor}",
                            )
                            .mark_bar()
                            .encode(
                                x=alt.X(
                                    "bin_start:Q",
                                    title=f"Unit price ({unit_unit_price})",
                                    scale=alt.Scale(domainMin=0),
                                ),
                                x2="bin_end:Q",
                                y=alt.Y("count()", title="Count"),
                                tooltip=[
                                    alt.Tooltip("bin_start_vnd:Q", title="Bin start (VND)", format=",.0f"),
                                    alt.Tooltip("bin_end_vnd:Q", title="Bin end (VND)", format=",.0f"),
                                    alt.Tooltip("count()", title="Count"),
                                ],
                            )
                        )

                        background = (
                            alt.Chart(pd.DataFrame({"dummy": [0]}))
                            .mark_rect(opacity=0.001)
                            .transform_calculate(
                                mean_vnd=f"{df[colnames.unit_price].mean() * unit_price_factor:.6f}",
                                std_vnd=f"{df[colnames.unit_price].std() * unit_price_factor:.6f}",
                                min_vnd=f"{df[colnames.unit_price].min() * unit_price_factor:.6f}",
                                q1_vnd=f"{df[colnames.unit_price].quantile(0.25) * unit_price_factor:.6f}",
                                q3_vnd=f"{df[colnames.unit_price].quantile(0.75) * unit_price_factor:.6f}",
                                max_vnd=f"{df[colnames.unit_price].max() * unit_price_factor:.6f}",
                            )
                            .encode(
                                x=alt.value(0),
                                x2=alt.value("width"),
                                y=alt.value(0),
                                y2=alt.value("height"),
                                tooltip=[
                                    alt.Tooltip("mean_vnd:Q", title="Mean (VND)", format=",.0f"),
                                    alt.Tooltip("std_vnd:Q", title="Standard deviation (VND)", format=",.0f"),
                                    alt.Tooltip("min_vnd:Q", title="Min (VND)", format=",.0f"),
                                    alt.Tooltip("q1_vnd:Q", title="Q1 (VND)", format=",.0f"),
                                    alt.Tooltip("q3_vnd:Q", title="Q3 (VND)", format=",.0f"),
                                    alt.Tooltip("max_vnd:Q", title="Max (VND)", format=",.0f"),
                                ],
                            )
                        )

                        st.altair_chart(
                            (background + bars)
                            .properties(height=alt.Step(36))
                            .configure_view(stroke=None)
                            .configure_axis(
                                labelColor="black",
                                titleColor="black",
                                labelFontSize=16,
                                titleFontSize=16,
                            ),
                            width="stretch",
                        )

                    with st.container():
                        st.subheader("Unit price by contractor", help=help_unit_price_by_contractor)

                        top_bidder = (
                            df.groupby(colnames.contractor_name, as_index=False)[colnames.total_price]
                            .sum()
                            .sort_values(by=colnames.total_price, ascending=False)
                            .head(10)[colnames.contractor_name]
                            .to_list()
                        )
                        data = df[df[colnames.contractor_name].isin(top_bidder)]

                        y = alt.Y(
                            colnames.contractor_name,
                            title="Contractor",
                            sort=top_bidder,
                            axis=alt.Axis(labelLimit=400, labelOverlap=False, title=None),
                        )

                        x_title = f"Unit price ({unit_unit_price})"

                        # Keep chart geometry in original unit price
                        base = alt.Chart(data).transform_joinaggregate(
                            q1=f"q1({colnames.unit_price})",
                            median=f"median({colnames.unit_price})",
                            q3=f"q3({colnames.unit_price})",
                            groupby=[colnames.contractor_name],
                        ).transform_calculate(
                            iqr="datum.q3 - datum.q1",
                            lower_fence="datum.q1 - 1.5 * datum.iqr",
                            upper_fence="datum.q3 + 1.5 * datum.iqr",
                            # Converted values for tooltip only
                            unit_price_vnd=f"datum['{colnames.unit_price}'] * {unit_price_factor}",
                            q1_vnd=f"datum.q1 * {unit_price_factor}",
                            median_vnd=f"datum.median * {unit_price_factor}",
                            q3_vnd=f"datum.q3 * {unit_price_factor}",
                            lower_fence_vnd=f"datum.lower_fence * {unit_price_factor}",
                            upper_fence_vnd=f"datum.upper_fence * {unit_price_factor}",
                        )

                        whisker_stats = (
                            base.transform_filter(
                                f"(datum['{colnames.unit_price}'] >= datum.lower_fence) && "
                                f"(datum['{colnames.unit_price}'] <= datum.upper_fence)"
                            )
                            .transform_aggregate(
                                whisker_min=f"min({colnames.unit_price})",
                                whisker_max=f"max({colnames.unit_price})",
                                q1="min(q1)",
                                median="min(median)",
                                q3="min(q3)",
                                lower_fence="min(lower_fence)",
                                upper_fence="min(upper_fence)",
                                groupby=[colnames.contractor_name],
                            )
                            .transform_calculate(
                                whisker_min_vnd=f"datum.whisker_min * {unit_price_factor}",
                                whisker_max_vnd=f"datum.whisker_max * {unit_price_factor}",
                                q1_vnd=f"datum.q1 * {unit_price_factor}",
                                median_vnd=f"datum.median * {unit_price_factor}",
                                q3_vnd=f"datum.q3 * {unit_price_factor}",
                                lower_fence_vnd=f"datum.lower_fence * {unit_price_factor}",
                                upper_fence_vnd=f"datum.upper_fence * {unit_price_factor}",
                            )
                        )

                        whiskers = whisker_stats.mark_rule().encode(
                            x=alt.X("whisker_min:Q", title=x_title, axis=alt.Axis(format=",.2f")),
                            x2="whisker_max",
                            y=y,
                            tooltip=[
                                alt.Tooltip(colnames.contractor_name, title="Contractor"),
                                alt.Tooltip("whisker_min_vnd:Q", title="Whisker min (VND)", format=",.0f"),
                                alt.Tooltip("q1_vnd:Q", title="Q1 (VND)", format=",.0f"),
                                alt.Tooltip("median_vnd:Q", title="Median (VND)", format=",.0f"),
                                alt.Tooltip("q3_vnd:Q", title="Q3 (VND)", format=",.0f"),
                                alt.Tooltip("whisker_max_vnd:Q", title="Whisker max (VND)", format=",.0f"),
                            ],
                        )

                        box = whisker_stats.mark_bar(size=18, stroke="#0169CA").encode(
                            x=alt.X("q1:Q", title=x_title, axis=alt.Axis(format=",.2f")),
                            x2="q3",
                            y=y,
                            tooltip=[
                                alt.Tooltip(colnames.contractor_name, title="Contractor"),
                                alt.Tooltip("q1_vnd:Q", title="Q1 (VND)", format=",.0f"),
                                alt.Tooltip("median_vnd:Q", title="Median (VND)", format=",.0f"),
                                alt.Tooltip("q3_vnd:Q", title="Q3 (VND)", format=",.0f"),
                            ],
                        )

                        median_tick = whisker_stats.mark_tick(
                            color="#BCD3F0",
                            size=18,
                            thickness=1,
                        ).encode(
                            x=alt.X("median:Q", title=x_title, axis=alt.Axis(format=",.2f")),
                            y=y,
                            tooltip=[
                                alt.Tooltip(colnames.contractor_name, title="Contractor"),
                                alt.Tooltip("median_vnd:Q", title="Median (VND)", format=",.0f"),
                            ],
                        )

                        outliers = base.transform_filter(
                            f"(datum['{colnames.unit_price}'] < datum.lower_fence) || "
                            f"(datum['{colnames.unit_price}'] > datum.upper_fence)"
                        ).mark_point(
                            shape="circle",
                            filled=True,
                            fill="white",
                            stroke="#2E77D0",
                            strokeWidth=1.2,
                            size=35,
                        ).encode(
                            x=alt.X(f"{colnames.unit_price}:Q", title=x_title, axis=alt.Axis(format=",.2f")),
                            y=y,
                            tooltip=[
                                alt.Tooltip(colnames.contractor_name, title="Contractor"),
                                alt.Tooltip("unit_price_vnd:Q", title="Unit price (VND)", format=",.0f"),
                            ],
                        )

                        st.altair_chart(
                            (whiskers + box + median_tick + outliers)
                            .properties(height=alt.Step(36))
                            .configure_view(stroke=None)
                            .configure_axis(
                                labelColor="black",
                                titleColor="black",
                                labelFontSize=16,
                                titleFontSize=16,
                            ),
                            width="stretch",
                        )

                    with st.container():
                        st.subheader("Unit price by country of origin", help=help_unit_price_by_country_origin)

                        top_origin = (
                            df.groupby(colnames.country_origin, as_index=False)[colnames.total_price]
                            .sum()
                            .sort_values(by=colnames.total_price, ascending=False)
                            .head(10)[colnames.country_origin]
                            .to_list()
                        )
                        data = df[df[colnames.country_origin].isin(top_origin)]

                        y = alt.Y(
                            colnames.country_origin,
                            title="Origin",
                            sort=top_origin,
                            axis=alt.Axis(labelLimit=400, labelOverlap=False, title=None),
                        )

                        x_title = f"Unit price ({unit_unit_price})"

                        base = (
                            alt.Chart(data)
                            .transform_joinaggregate(
                                q1=f"q1({colnames.unit_price})",
                                median=f"median({colnames.unit_price})",
                                q3=f"q3({colnames.unit_price})",
                                groupby=[colnames.country_origin],
                            )
                            .transform_calculate(
                                iqr="datum.q3 - datum.q1",
                                lower_fence="datum.q1 - 1.5 * datum.iqr",
                                upper_fence="datum.q3 + 1.5 * datum.iqr",
                                q1_vnd=f"datum.q1 * {unit_price_factor}",
                                median_vnd=f"datum.median * {unit_price_factor}",
                                q3_vnd=f"datum.q3 * {unit_price_factor}",
                                unit_price_vnd=f"datum['{colnames.unit_price}'] * {unit_price_factor}",
                            )
                        )

                        whisker_stats = (
                            base.transform_filter(
                                f"(datum['{colnames.unit_price}'] >= datum.lower_fence) && "
                                f"(datum['{colnames.unit_price}'] <= datum.upper_fence)"
                            )
                            .transform_aggregate(
                                whisker_min=f"min({colnames.unit_price})",
                                whisker_max=f"max({colnames.unit_price})",
                                q1="min(q1)",
                                median="min(median)",
                                q3="min(q3)",
                                groupby=[colnames.country_origin],
                            )
                            .transform_calculate(
                                whisker_min_vnd=f"datum.whisker_min * {unit_price_factor}",
                                whisker_max_vnd=f"datum.whisker_max * {unit_price_factor}",
                                q1_vnd=f"datum.q1 * {unit_price_factor}",
                                median_vnd=f"datum.median * {unit_price_factor}",
                                q3_vnd=f"datum.q3 * {unit_price_factor}",
                            )
                        )

                        whiskers = whisker_stats.mark_rule().encode(
                            x=alt.X("whisker_min:Q", title=x_title, axis=alt.Axis(format=",.2f")),
                            x2="whisker_max",
                            y=y,
                            tooltip=[
                                alt.Tooltip(colnames.country_origin, title="Origin"),
                                alt.Tooltip("whisker_min_vnd:Q", title="Whisker min (VND)", format=",.0f"),
                                alt.Tooltip("q1_vnd:Q", title="Q1 (VND)", format=",.0f"),
                                alt.Tooltip("median_vnd:Q", title="Median (VND)", format=",.0f"),
                                alt.Tooltip("q3_vnd:Q", title="Q3 (VND)", format=",.0f"),
                                alt.Tooltip("whisker_max_vnd:Q", title="Whisker max (VND)", format=",.0f"),
                            ],
                        )

                        box = whisker_stats.mark_bar(size=18, stroke="#0169CA").encode(
                            x=alt.X("q1:Q", title=x_title, axis=alt.Axis(format=",.2f")),
                            x2="q3",
                            y=y,
                            tooltip=[
                                alt.Tooltip(colnames.country_origin, title="Origin"),
                                alt.Tooltip("q1_vnd:Q", title="Q1 (VND)", format=",.0f"),
                                alt.Tooltip("median_vnd:Q", title="Median (VND)", format=",.0f"),
                                alt.Tooltip("q3_vnd:Q", title="Q3 (VND)", format=",.0f"),
                            ],
                        )

                        median_tick = whisker_stats.mark_tick(
                            color="#BCD3F0",
                            size=18,
                            thickness=1,
                        ).encode(
                            x=alt.X("median:Q", title=x_title, axis=alt.Axis(format=",.2f")),
                            y=y,
                            tooltip=[
                                alt.Tooltip(colnames.country_origin, title="Origin"),
                                alt.Tooltip("median_vnd:Q", title="Median (VND)", format=",.0f"),
                            ],
                        )

                        outliers = base.transform_filter(
                            f"(datum['{colnames.unit_price}'] < datum.lower_fence) || "
                            f"(datum['{colnames.unit_price}'] > datum.upper_fence)"
                        ).mark_point(
                            shape="circle",
                            filled=True,
                            fill="white",
                            stroke="#2E77D0",
                            strokeWidth=1.2,
                            size=35,
                        ).encode(
                            x=alt.X(f"{colnames.unit_price}:Q", title=x_title, axis=alt.Axis(format=",.2f")),
                            y=y,
                            tooltip=[
                                alt.Tooltip(colnames.country_origin, title="Origin"),
                                alt.Tooltip("unit_price_vnd:Q", title="Unit price (VND)", format=",.0f"),
                            ],
                        )

                        st.altair_chart(
                            (whiskers + box + median_tick + outliers)
                            .properties(height=alt.Step(36))
                            .configure_view(stroke=None)
                            .configure_axis(
                                labelColor="black",
                                titleColor="black",
                                labelFontSize=16,
                                titleFontSize=16,
                            ),
                            width="stretch",
                        )


                    with st.container():
                        st.subheader("Unit price by manufacturer", help=help_unit_price_by_manufacturer)

                        top_manufacturer = (
                            df.groupby(colnames.manufacturer, as_index=False)[colnames.total_price]
                            .sum()
                            .sort_values(by=colnames.total_price, ascending=False)
                            .head(10)[colnames.manufacturer]
                            .to_list()
                        )
                        data = df[df[colnames.manufacturer].isin(top_manufacturer)]

                        y = alt.Y(
                            colnames.manufacturer,
                            title="Manufacturer",
                            sort=top_manufacturer,
                            axis=alt.Axis(labelLimit=400, labelOverlap=False, title=None),
                        )

                        x_title = f"Unit price ({unit_unit_price})"

                        base = (
                            alt.Chart(data)
                            .transform_joinaggregate(
                                q1=f"q1({colnames.unit_price})",
                                median=f"median({colnames.unit_price})",
                                q3=f"q3({colnames.unit_price})",
                                groupby=[colnames.manufacturer],
                            )
                            .transform_calculate(
                                iqr="datum.q3 - datum.q1",
                                lower_fence="datum.q1 - 1.5 * datum.iqr",
                                upper_fence="datum.q3 + 1.5 * datum.iqr",
                                q1_vnd=f"datum.q1 * {unit_price_factor}",
                                median_vnd=f"datum.median * {unit_price_factor}",
                                q3_vnd=f"datum.q3 * {unit_price_factor}",
                                unit_price_vnd=f"datum['{colnames.unit_price}'] * {unit_price_factor}",
                            )
                        )

                        whisker_stats = (
                            base.transform_filter(
                                f"(datum['{colnames.unit_price}'] >= datum.lower_fence) && "
                                f"(datum['{colnames.unit_price}'] <= datum.upper_fence)"
                            )
                            .transform_aggregate(
                                whisker_min=f"min({colnames.unit_price})",
                                whisker_max=f"max({colnames.unit_price})",
                                q1="min(q1)",
                                median="min(median)",
                                q3="min(q3)",
                                groupby=[colnames.manufacturer],
                            )
                            .transform_calculate(
                                whisker_min_vnd=f"datum.whisker_min * {unit_price_factor}",
                                whisker_max_vnd=f"datum.whisker_max * {unit_price_factor}",
                                q1_vnd=f"datum.q1 * {unit_price_factor}",
                                median_vnd=f"datum.median * {unit_price_factor}",
                                q3_vnd=f"datum.q3 * {unit_price_factor}",
                            )
                        )

                        whiskers = whisker_stats.mark_rule().encode(
                            x=alt.X("whisker_min:Q", title=x_title, axis=alt.Axis(format=",.2f")),
                            x2="whisker_max",
                            y=y,
                            tooltip=[
                                alt.Tooltip(colnames.manufacturer, title="Manufacturer"),
                                alt.Tooltip("whisker_min_vnd:Q", title="Whisker min (VND)", format=",.0f"),
                                alt.Tooltip("q1_vnd:Q", title="Q1 (VND)", format=",.0f"),
                                alt.Tooltip("median_vnd:Q", title="Median (VND)", format=",.0f"),
                                alt.Tooltip("q3_vnd:Q", title="Q3 (VND)", format=",.0f"),
                                alt.Tooltip("whisker_max_vnd:Q", title="Whisker max (VND)", format=",.0f"),
                            ],
                        )

                        box = whisker_stats.mark_bar(size=18, stroke="#0169CA").encode(
                            x=alt.X("q1:Q", title=x_title, axis=alt.Axis(format=",.2f")),
                            x2="q3",
                            y=y,
                            tooltip=[
                                alt.Tooltip(colnames.manufacturer, title="Manufacturer"),
                                alt.Tooltip("q1_vnd:Q", title="Q1 (VND)", format=",.0f"),
                                alt.Tooltip("median_vnd:Q", title="Median (VND)", format=",.0f"),
                                alt.Tooltip("q3_vnd:Q", title="Q3 (VND)", format=",.0f"),
                            ],
                        )

                        median_tick = whisker_stats.mark_tick(
                            color="#BCD3F0",
                            size=18,
                            thickness=1,
                        ).encode(
                            x=alt.X("median:Q", title=x_title, axis=alt.Axis(format=",.2f")),
                            y=y,
                            tooltip=[
                                alt.Tooltip(colnames.manufacturer, title="Manufacturer"),
                                alt.Tooltip("median_vnd:Q", title="Median (VND)", format=",.0f"),
                            ],
                        )

                        outliers = base.transform_filter(
                            f"(datum['{colnames.unit_price}'] < datum.lower_fence) || "
                            f"(datum['{colnames.unit_price}'] > datum.upper_fence)"
                        ).mark_point(
                            shape="circle",
                            filled=True,
                            fill="white",
                            stroke="#2E77D0",
                            strokeWidth=1.2,
                            size=35,
                        ).encode(
                            x=alt.X(f"{colnames.unit_price}:Q", title=x_title, axis=alt.Axis(format=",.2f")),
                            y=y,
                            tooltip=[
                                alt.Tooltip(colnames.manufacturer, title="Manufacturer"),
                                alt.Tooltip("unit_price_vnd:Q", title="Unit price (VND)", format=",.0f"),
                            ],
                        )

                        st.altair_chart(
                            (whiskers + box + median_tick + outliers)
                            .properties(height=alt.Step(36))
                            .configure_view(stroke=None)
                            .configure_axis(
                                labelColor="black",
                                titleColor="black",
                                labelFontSize=16,
                                titleFontSize=16,
                            ),
                            width="stretch",
                        )

                    with st.container():
                        st.subheader("Total value by country of origin", help=help_total_value_by_country_origin)

                        data_by_origin = (
                            df.groupby(colnames.country_origin, as_index=False)[colnames.total_price]
                            .sum()
                            .sort_values(by=colnames.total_price, ascending=False)
                        )
                        top9 = data_by_origin.head(9)
                        rest = data_by_origin.iloc[9:][colnames.total_price].sum()

                        data = (
                            pd.concat(
                                [
                                    top9,
                                    pd.DataFrame(
                                        [
                                            {
                                                colnames.country_origin: "Others",
                                                colnames.total_price: rest,
                                            }
                                        ]
                                    ),
                                ],
                                ignore_index=True,
                            )
                            if rest > 0
                            else top9
                        )
                        st.altair_chart(
                            alt.Chart(data)
                            .mark_arc()
                            .encode(
                                theta=alt.Theta(colnames.total_price, title=f"Total price ({unit_total_price})", sort="x"),
                                color=alt.Color(
                                    colnames.country_origin,
                                    title="Country of origin",
                                    legend=alt.Legend(
                                        title="Country of origin",
                                        labelColor="black",
                                        titleColor="black",
                                        labelFontSize=16,
                                        titleFontSize=16,
                                    ),
                                ),
                                order=alt.Order(colnames.total_price, sort="ascending"),
                                tooltip=[
                                    alt.Tooltip(
                                        colnames.country_origin,
                                        title="Country of origin",
                                    ),
                                    alt.Tooltip(
                                        colnames.total_price,
                                        title=f"Total price ({unit_total_price})",
                                        format=",.3f",
                                    ),
                                ],
                            ),
                            width="stretch",
                        )
                    with st.container():
                        st.subheader("Total value by region of origin", help=help_total_value_by_region_origin)

                        data_by_origin = (
                            df.groupby(colnames.region_origin, as_index=False)[colnames.total_price].sum().sort_values(by=colnames.total_price, ascending=False)
                        )
                        top9 = data_by_origin.head(9)
                        rest = data_by_origin.iloc[9:][colnames.total_price].sum()

                        data = (
                            pd.concat(
                                [
                                    top9,
                                    pd.DataFrame(
                                        [
                                            {
                                                colnames.region_origin: "Others",
                                                colnames.total_price: rest,
                                            }
                                        ]
                                    ),
                                ],
                                ignore_index=True,
                            )
                            if rest > 0
                            else top9
                        )
                        st.altair_chart(
                            alt.Chart(data)
                            .mark_arc()
                            .encode(
                                theta=alt.Theta(colnames.total_price, title=f"Total price ({unit_total_price})", sort="x"),
                                color=alt.Color(
                                    colnames.region_origin,
                                    legend=alt.Legend(
                                        title="Region of origin",
                                        labelColor="black",
                                        titleColor="black",
                                        labelFontSize=16,
                                        titleFontSize=16,
                                    ),
                                ),
                                order=alt.Order(colnames.total_price, sort="ascending"),
                                tooltip=[
                                    alt.Tooltip(colnames.region_origin, title="Region of origin"),
                                    alt.Tooltip(
                                        colnames.total_price,
                                        title=f"Total price ({unit_total_price})",
                                        format=",.3f",
                                    ),
                                ],
                            ),
                            width="stretch",
                        )
                    
                    with st.container():
                        st.subheader("Total value by manufacturer", help=help_total_value_by_manufacturer)

                        data_by_origin = (
                            df.groupby(colnames.manufacturer, as_index=False)[colnames.total_price].sum().sort_values(by=colnames.total_price, ascending=False)
                        )
                        top9 = data_by_origin.head(9)
                        rest = data_by_origin.iloc[9:][colnames.total_price].sum()

                        data = (
                            pd.concat(
                                [
                                    top9,
                                    pd.DataFrame(
                                        [
                                            {
                                                colnames.manufacturer: "Others",
                                                colnames.total_price: rest,
                                            }
                                        ]
                                    ),
                                ],
                                ignore_index=True,
                            )
                            if rest > 0
                            else top9
                        )
                        manu_order = (
                            data.groupby(colnames.manufacturer)[colnames.total_price]
                            .sum()
                            .sort_values(ascending=False) 
                            .index.tolist()
                        )
                        st.altair_chart(
                            alt.Chart(data)
                            .transform_calculate(
                                total_price_vnd=f"datum['{colnames.total_price}'] * {total_price_factor}"
                            )
                            .mark_arc()
                            .encode(
                                theta=alt.Theta(colnames.total_price, title=f"Total price ({unit_total_price})", sort="x"),
                                color=alt.Color(
                                    colnames.manufacturer,
                                    sort=manu_order,
                                    legend=alt.Legend(
                                        title="Manufacturer",
                                        labelColor="black",
                                        titleColor="black",
                                        labelFontSize=16,
                                        titleFontSize=16,
                                    ),
                                ),
                                order=alt.Order(colnames.total_price, sort="descending"),
                                tooltip=[
                                    alt.Tooltip(colnames.manufacturer, title="Manufacturer"),
                                    alt.Tooltip(
                                        "total_price_vnd:Q",
                                        title=f"Total price (VND)",
                                        format=",.0f",
                                    ),
                                ],
                            ),
                            width="stretch",
                        )
    with tab2:
        with st.container(height=ALL_CONTENT_HEIGHT, border=True):
            if not "data" in st.session_state:
                st.info("Query something before predicting pricing strategies.")
            elif st.session_state["data"].empty:
                st.info("No result. Try another query before predicting pricing strategies.")
            else:
                with st.form("Parameters"):
                    st.subheader("Information for winning bid price prediction")
                    left_col, right_col = st.columns([1, 1], gap="large")
                    with left_col:
                        st.selectbox(
                            colnames.investor.replace("_", " ").capitalize(),
                            sorted(set(st.session_state["data"][colnames.investor]) | {"Other"}),
                            index=None,
                            key=f"Predict: {colnames.investor}",
                            help=tooltip_predict_form_map[colnames.investor]
                        )
                        st.selectbox(
                            colnames.province.replace("_", " ").capitalize(),
                            sorted(set(st.session_state["data"][colnames.province]) | {"Other"}),
                            index=None,
                            key=f"Predict: {colnames.province}",
                            help=tooltip_predict_form_map[colnames.province]
                        )
                        inner_left_col, inner_right_col = st.columns([1, 1], gap="medium")
                        with inner_left_col:
                            st.number_input(
                                colnames.quantity.replace("_", " ").capitalize(),
                                key=f"Predict: {colnames.quantity}",
                                min_value=1,
                                help=tooltip_predict_form_map[colnames.quantity]
                            )
                        with inner_right_col:
                            st.date_input(colnames.closing_date.replace("_", " ").capitalize(), key=f"Predict: {colnames.closing_date}", format="DD-MM-YYYY", help=tooltip_predict_form_map[colnames.closing_date])
                    with right_col:
                        st.selectbox(
                            colnames.manufacturer.replace("_", " ").capitalize(),
                            sorted(set(st.session_state["data"][colnames.manufacturer]) | {"Other"}),
                            index=None,
                            key=f"Predict: {colnames.manufacturer}",
                            help=tooltip_predict_form_map[colnames.manufacturer]
                        )
                        st.selectbox(
                            colnames.country_origin.replace("_", " ").capitalize(),
                            sorted(set(st.session_state["data"][colnames.country_origin]) | {"Other"}),
                            index=None,
                            key=f"Predict: {colnames.country_origin}",
                            help=tooltip_predict_form_map[colnames.country_origin]
                        )
                        st.number_input(
                                "Cost (VND)",
                                key=f"Predict: cost",
                                min_value=0,
                                help=tooltip_predict_form_map["cost"]
                            )
                    _, data_space, class_space, button_space, _ = st.columns([1, 3.5, 1.5, 2, 1], gap="small", vertical_alignment="bottom")
                    options = ["All queried data (recommended)"]
                    if len(st.session_state["filtered_data"]) < len(st.session_state["data"]):
                        options.append("Filtered data (advanced)")
                    with data_space:
                        if len(st.session_state["filtered_data"]) == len(st.session_state["data"]):
                            st.session_state["Predict: training_data"] = "All queried data (recommended)"
                        st.selectbox("Training data", options, key="Predict: training_data", help=help_training_data)

                    with class_space:
                        st.selectbox("Model class", ["Default", "Auto"], key=f"Predict: model_class", help=help_model_class)
                    with button_space:
                        st.form_submit_button("Fit & predict", on_click=handle_predict_click, width="stretch")

                    banner_predict = st.empty()
                    banner_predict.empty()

                
                model_class = "default" if st.session_state["Predict: model_class"] == "Default" else "auto"
                training_data = "all" if st.session_state["Predict: training_data"] == "All queried data (recommended)" else "filtered"

                progress_space = st.empty()
                left_col, right_col = st.columns([3, 7])
                left_placeholder = left_col.empty()
                right_placeholder = right_col.empty()

                if "predict it" in st.session_state and st.session_state["predict it"]:
                    st.session_state["predict it"] = False
                    left_placeholder.empty()
                    right_placeholder.empty()
                    def handle_progress_update(trained_models, total_models):
                        def update_pbar_and_caption(tqdm):
                            with progress_space:
                                st.progress(tqdm.n/tqdm.total, f"Fitting model {trained_models}/{total_models}")
                        return update_pbar_and_caption
                    
                    try:
                        if training_data == "filtered":
                            old_ids = st.session_state["model_filtered_training_data"]
                            curr_ids = st.session_state["filtered_data"]["id"].values
                            if st.session_state[f"model_{model_class}_filtered"] is None or (len(old_ids) != len(curr_ids)) or not all(old_ids == curr_ids):
                                model, _, _ = train_model(st.session_state["filtered_data"], model_class, handle_progress_update)
                                progress_space.empty()
                                st.session_state["model_filtered_training_data"] = curr_ids
                                st.session_state[f"model_{model_class}_filtered"] = model
                            else:
                                model = st.session_state[f"model_{model_class}_filtered"]
                        else:
                            if st.session_state[f"model_{model_class}_all"] is None:
                                model, _, _ = train_model(st.session_state["data"], model_class, handle_progress_update)
                                progress_space.empty()
                                st.session_state[f"model_{model_class}_all"] = model
                            else:
                                model = st.session_state[f"model_{model_class}_all"]

                        user_config = {
                            colnames.investor: st.session_state[f"Predict: {colnames.investor}"],
                            colnames.province: st.session_state[f"Predict: {colnames.province}"],
                            colnames.quantity: st.session_state[f"Predict: {colnames.quantity}"],
                            colnames.closing_date: st.session_state[f"Predict: {colnames.closing_date}"],
                            colnames.manufacturer: st.session_state[f"Predict: {colnames.manufacturer}"],
                            colnames.country_origin: st.session_state[f"Predict: {colnames.country_origin}"],
                            "cost": st.session_state[f"Predict: cost"] * 1e-3 if st.session_state[f"Predict: cost"] is not None else None
                        }
                        pred_dist = predict(
                            user_config,
                            model,
                        )
                        st.session_state["latest_pred"] = {"pred_dist": pred_dist, "metrics": model["metrics"], "training_data": training_data, "training_df": st.session_state["data"] if training_data == "all" else st.session_state["filtered_data"], "user_config": user_config, "summary": None}
                        st.session_state["predict_error"] = None
                    except Exception as e:
                        progress_space.empty()
                        st.session_state["predict_error"] = TrainAndPredictError()
                        st.session_state["latest_pred"] = None
                
                with banner_predict:
                    if st.session_state["predict_error"] != None:
                        e = st.session_state["predict_error"]
                        print("=========== Exception ===========")
                        print(e)
                        if isinstance(e, PredictFieldMissingError):
                            st.error("Please fill all required fields.")
                        elif isinstance(e, TrainAndPredictError):
                            st.error("An error occured in the training and predicting process. Please try again.")

                if st.session_state["latest_pred"] is not None:
                    pred_dist = st.session_state["latest_pred"]["pred_dist"]
                    metrics = st.session_state["latest_pred"]["metrics"]
                    st.subheader("Model performance on historical data")
                    _, content, _ = st.columns([3, 4, 3])
                    with content:
                        # Header
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown("**Metric**")
                        with col2:
                            st.markdown("**Value**")

                        # Row 1: MAE
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown("Mean absolute error", help=tooltip_metrics["mae"])
                        with col2:
                            st.markdown(f"{metrics['MAE']*1000:,.0f} VND")

                        # Row 2: Coverage 50
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown("Coverage (50%)", help=tooltip_metrics["coverage_50"])
                        with col2:
                            st.markdown(f"{metrics['Coverage_50']*100:.1f}%")

                        # Row 3: Coverage 90
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown("Coverage (90%)", help=tooltip_metrics["coverage_90"])
                        with col2:
                            st.markdown(f"{metrics['Coverage_90']*100:.1f}%")
                            
                    def to_scalar(value):
                        arr = np.asarray(value)
                        if arr.size == 0:
                            return np.nan
                        return float(arr.reshape(-1)[0])

                    quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
                    quantiles_log = {q: to_scalar(pred_dist.ppf(q)) for q in quantile_levels}

                    def exp_safe(v):
                        return float(np.exp(np.clip(v, -700, 700)))

                    quantiles = {q: exp_safe(quantiles_log[q]) for q in quantile_levels}
                    mu_log = to_scalar(getattr(pred_dist, "mu", np.nan))
                    sigma_log = to_scalar(getattr(pred_dist, "sigma", np.nan))
                    mean_original = np.exp(mu_log + 0.5 * (sigma_log ** 2)) if np.isfinite(mu_log) and np.isfinite(sigma_log) else np.nan
                    median_original = np.exp(mu_log) if np.isfinite(mu_log) else np.nan
                    std_original = (
                        np.sqrt((np.exp(sigma_log ** 2) - 1.0) * np.exp(2.0 * mu_log + sigma_log ** 2))
                        if np.isfinite(mu_log) and np.isfinite(sigma_log)
                        else np.nan
                    )

                    y_low = to_scalar(pred_dist.ppf(0.001))
                    y_high = to_scalar(pred_dist.ppf(0.999))
                    if not (np.isfinite(y_low) and np.isfinite(y_high)) or y_low >= y_high:
                        y_low = quantiles_log[0.1]
                        y_high = quantiles_log[0.9]
                        if not (np.isfinite(y_low) and np.isfinite(y_high)) or y_low >= y_high:
                            y_low, y_high = -1.0, 1.0

                    y_grid = np.linspace(y_low, y_high, 400)
                    x_grid = np.exp(np.clip(y_grid, -700, 700))

                    try:
                        pdf_y = np.asarray(pred_dist.pdf(y_grid), dtype=float).reshape(-1)
                        if pdf_y.size != y_grid.size:
                            raise ValueError("Unexpected pdf output shape")
                    except Exception:
                        pdf_y = np.array([to_scalar(pred_dist.pdf(y)) for y in y_grid], dtype=float)

                    with np.errstate(divide="ignore", invalid="ignore"):
                        pdf_values = pdf_y / x_grid

                    valid = np.isfinite(pdf_values)
                    if not valid.any():
                        st.warning("Could not evaluate the distribution PDF for plotting.")
                    else:
                        density_df = pd.DataFrame({"x": x_grid[valid], "pdf": pdf_values[valid]})
                        y_max = float(density_df["pdf"].max())
                        y_offset = -0.06
                        y_axis_max = y_max * 1.25 if y_max > 0 else 1.2

                        x_low = 0.0
                        x_high = max(float(density_df["x"].max()), quantiles[0.9], quantiles[0.5], 1.0)

                        q_df = pd.DataFrame(
                            {
                                "quantile": quantile_levels,
                                "x": [quantiles[q] for q in quantile_levels],
                            }
                        )
                        q_df["label"] = q_df["quantile"].map(lambda q: f"{q:.2f}")
                        q_df["pdf"] = np.interp(q_df["x"], density_df["x"], density_df["pdf"])

                        summary_bg = (
                            alt.Chart(pd.DataFrame({"dummy": [0]}))
                            .mark_rect(opacity=0.001)
                            .transform_calculate(
                                mean_vnd=f"{mean_original * 1000:.6f}" if np.isfinite(mean_original) else "null",
                                std_vnd=f"{std_original * 1000:.6f}" if np.isfinite(std_original) else "null",
                                q10_vnd=f"{quantiles[0.1] * 1000:.6f}" if np.isfinite(quantiles[0.1]) else "null",
                                q25_vnd=f"{quantiles[0.25] * 1000:.6f}" if np.isfinite(quantiles[0.25]) else "null",
                                q50_vnd=f"{quantiles[0.5] * 1000:.6f}" if np.isfinite(quantiles[0.5]) else "null",
                                q75_vnd=f"{quantiles[0.75] * 1000:.6f}" if np.isfinite(quantiles[0.75]) else "null",
                                q90_vnd=f"{quantiles[0.9] * 1000:.6f}" if np.isfinite(quantiles[0.9]) else "null",
                            )
                            .encode(
                                x=alt.value(0),
                                x2=alt.value("width"),
                                y=alt.value(0),
                                y2=alt.value("height"),
                                tooltip=[
                                    alt.Tooltip("mean_vnd:Q", title="Mean (VND)", format=",.0f"),
                                    alt.Tooltip("q50_vnd:Q", title="Median (VND)", format=",.0f"),
                                    alt.Tooltip("std_vnd:Q", title="Standard deviation (VND)", format=",.0f"),
                                ]
                            )
                        )

                        density_chart = (
                            alt.Chart(density_df)
                            .transform_calculate(
                                x_vnd="datum.x * 1000"
                            )
                            .mark_area(opacity=0.45, color="#9db7d5")
                            .encode(
                                x=alt.X("x:Q", title="Unit price (thousand VND)"),
                                y=alt.Y("pdf:Q", title="Density", scale=alt.Scale(domain=[0, y_axis_max])),
                                tooltip=[
                                    alt.Tooltip("x_vnd:Q", title="Unit price (VND)", format=",.0f"),
                                    alt.Tooltip("pdf:Q", title="Density", format=".6f"),
                                ],
                            )
                        )

                        density_outline = (
                            alt.Chart(density_df)
                            .transform_calculate(
                                x_vnd="datum.x * 1000"
                            )
                            .mark_line(color="#0b3c6f", strokeWidth=2)
                            .encode(
                                x=alt.X("x:Q", title="Unit price (thousand VND)"),
                                y=alt.Y("pdf:Q", title="Density", scale=alt.Scale(domain=[0, y_axis_max])),
                                tooltip=[
                                    alt.Tooltip("x_vnd:Q", title="Unit price (VND)", format=",.0f"),
                                    alt.Tooltip("pdf:Q", title="Density", format=".6f"),
                                ],
                            )
                        )

                        q_df["y0"] = 0.0

                        q_rules = (
                            alt.Chart(q_df)
                            .transform_calculate(
                                x_vnd="datum.x * 1000"
                            )
                            .mark_rule(strokeWidth=2, strokeDash=[6, 6], color="#1f2937")
                            .encode(
                                x=alt.X("x:Q"),
                                y=alt.Y("y0:Q"),
                                y2=alt.Y2("pdf:Q"),
                                tooltip=[
                                    alt.Tooltip("label:N", title="Quantile"),
                                    alt.Tooltip("x_vnd:Q", title="Unit price (VND)", format=",.0f"),
                                ],
                            )
                        )

                        q_points = (
                            alt.Chart(q_df)
                            .transform_calculate(
                                x_vnd="datum.x * 1000"
                            )
                            .mark_point(size=80, filled=True, color="#1f2937")
                            .encode(
                                x=alt.X("x:Q"),
                                y=alt.Y("pdf:Q"),
                                tooltip=[
                                    alt.Tooltip("label:N", title="Quantile"),
                                    alt.Tooltip("x_vnd:Q", title="Unit price (VND)", format=",.0f"),
                                    alt.Tooltip("pdf:Q", title="Density", format=".6f"),
                                ],
                            )
                        )

                        st.subheader("Predicted winning bid price distribution")
                        _, plot_space, metrics_space = st.columns([1, 8, 1])
                        with plot_space:
                            st.altair_chart(
                                (summary_bg + density_chart + density_outline + q_rules + q_points)
                                .properties(height=320)
                                .configure_view(stroke=None)
                                .configure_axis(
                                    labelColor="black",
                                    titleColor="black",
                                    labelFontSize=14,
                                    titleFontSize=14,
                                ),
                                width="stretch"
                            )

                        quantity = st.session_state["latest_pred"]["user_config"][colnames.quantity]
                        cost = st.session_state["latest_pred"]["user_config"]["cost"]
                        if cost is not None:
                            st.subheader("Proxy for expected profit", help=help_profit_proxy)
                            _, plot_space, _ = st.columns([1, 8, 1])
                            x_vals = density_df["x"].to_numpy(dtype=float)
                            log_x_vals = np.log(np.clip(x_vals, 1e-12, None))

                            try:
                                cdf_vals = np.asarray(pred_dist.cdf(log_x_vals), dtype=float).reshape(-1)
                                if cdf_vals.size != x_vals.size:
                                    raise ValueError("Unexpected cdf output shape")
                            except Exception:
                                cdf_vals = np.array([to_scalar(pred_dist.cdf(np.log(max(x, 1e-12)))) for x in x_vals], dtype=float)

                            survival_vals = np.clip(1.0 - cdf_vals, 0.0, 1.0)
                            expected_profit = (x_vals * quantity - cost) * survival_vals
                            expected_profit = np.maximum(expected_profit, 0.0)

                            profit_df = pd.DataFrame({"x": x_vals, "expected_profit": expected_profit})
                            valid_profit_df = profit_df[np.isfinite(profit_df["expected_profit"])].copy()

                            profit_line = (
                                alt.Chart(profit_df)
                                .transform_calculate(
                                    x_vnd="datum.x * 1000",
                                    expected_profit_vnd="datum.expected_profit * 1000",
                                )
                                .mark_line(color="#0b3c6f", strokeWidth=2)
                                .encode(
                                    x=alt.X("x:Q", title="Unit price (thousand VND)"),
                                    y=alt.Y(
                                        "expected_profit:Q",
                                        title="Expected profit (thousand VND)",
                                        scale=alt.Scale(domainMin=0),
                                    ),
                                    tooltip=[
                                        alt.Tooltip("x_vnd:Q", title="Unit price (VND)", format=",.0f"),
                                        alt.Tooltip("expected_profit_vnd:Q", title="Expected profit (VND)", format=",.0f"),
                                    ],
                                )
                            )

                            max_profit_line = alt.Chart(pd.DataFrame({"x": []})).mark_rule()
                            max_profit_point = alt.Chart(pd.DataFrame({"x": [], "expected_profit": []})).mark_point()

                            if not valid_profit_df.empty:
                                max_idx = valid_profit_df["expected_profit"].idxmax()
                                max_row = valid_profit_df.loc[[max_idx], ["x", "expected_profit"]]

                                max_profit_line = (
                                    alt.Chart(max_row)
                                    .transform_calculate(
                                        x_vnd="datum.x * 1000",
                                        expected_profit_vnd="datum.expected_profit * 1000",
                                    )
                                    .mark_rule(color="#1f2937", strokeDash=[6, 6], strokeWidth=2)
                                    .encode(
                                        x=alt.X("x:Q"),
                                        tooltip=[
                                            alt.Tooltip("x_vnd:Q", title="Optimal unit price (VND)", format=",.0f"),
                                            alt.Tooltip("expected_profit_vnd:Q", title="Maximum expected profit (VND)", format=",.0f"),
                                        ],
                                    )
                                )

                                max_profit_point = (
                                    alt.Chart(max_row)
                                    .transform_calculate(
                                        x_vnd="datum.x * 1000",
                                        expected_profit_vnd="datum.expected_profit * 1000",
                                    )
                                    .mark_point(color="#1f2937", size=90, filled=True)
                                    .encode(
                                        x=alt.X("x:Q"),
                                        y=alt.Y("expected_profit:Q"),
                                        tooltip=[
                                            alt.Tooltip("x_vnd:Q", title="Optimal unit price (VND)", format=",.0f"),
                                            alt.Tooltip("expected_profit_vnd:Q", title="Maximum expected profit (VND)", format=",.0f"),
                                        ],
                                    )
                                )

                            with plot_space:
                                st.altair_chart(
                                    (profit_line + max_profit_line + max_profit_point)
                                    .properties(height=280)
                                    .configure_view(stroke=None)
                                    .configure_axis(
                                        labelColor="black",
                                        titleColor="black",
                                        labelFontSize=14,
                                        titleFontSize=14,
                                    ),
                                    width="stretch",
                                )
                    st.subheader("Pricing strategy recommendation")
                    rec = recommend_price(
                        df=st.session_state["latest_pred"]["training_df"],
                        user_config=st.session_state["latest_pred"]["user_config"],
                        pred_dist=st.session_state["latest_pred"]["pred_dist"],
                        pred_metrics=st.session_state["latest_pred"]["metrics"]
                    )["recommendation_text"]
                    st.write(rec)
                    # llm_contain = st.chat_message(name="assistant")
                    # content = llm_contain.empty()
                    # if st.session_state["latest_pred"]["summary"] is None:
                    #     prompt = get_info(
                    #         df=st.session_state["latest_pred"]["training_df"],
                    #         query=st.session_state["query_str"],
                    #         user_config=st.session_state["latest_pred"]["user_config"],
                    #         pred_metrics=st.session_state["latest_pred"]["metrics"],
                    #         pred_dist=st.session_state["latest_pred"]["pred_dist"],
                    #         filtered=st.session_state["latest_pred"]["training_data"] == "filtered",
                    #         )
                    #     print(prompt)
                    #     for t in range(3):
                    #         try:
                    #             with content.container():
                    #                 with st.spinner("Thinking..." if t == 0 else f"Retry thinking {t}/2..."):
                    #                     llm_res = st.write_stream(llm(prompt))
                    #                     st.session_state["latest_pred"]["summary"] = llm_res
                    #                     break
                    #         except Exception as e:
                    #             print("=========== Query error ===========")
                    #             print(str(e))
                    #             st.session_state["latest_pred"]["summary"] = None
                    #             if t < 2:
                    #                 time.sleep(2)
                    #             else:
                    #                 with content.container():
                    #                     st.write("Connection error! Please check your network connection and API key and try again.")
                    # else:
                    #     with content.container():
                    #         st.write(st.session_state["latest_pred"]["summary"])

                            

                    
