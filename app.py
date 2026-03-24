import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from query_logic import query, SQLCompileError
from prob_modeling import train_model, predict


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
            df = query(query_str)
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

    st.session_state["model_default"] = None
    st.session_state["model_auto"] = None
    st.session_state["prediction"] = None


def handle_predict_click():
    if not (
        (st.session_state[f"Predict: {colnames.investor}"] is not None)
        and (st.session_state[f"Predict: {colnames.province}"] is not None)
        and (st.session_state[f"Predict: {colnames.quantity}"] is not None)
        and (st.session_state[f"Predict: {colnames.closing_date}"] is not None)
        and (st.session_state[f"Predict: {colnames.manufacturer}"] is not None)
        and (st.session_state[f"Predict: {colnames.country_origin}"] is not None)
        and (st.session_state[f"Predict: model_class"] is not None)
    ):
        with banner:
            st.error("Please fill all the required fields!")
    else:
        if st.session_state[f"model_{st.session_state["Predict: model_class"].lower()}"] == None:
            model, _, _ = train_model(st.session_state["data"], st.session_state["Predict: model_class"].lower())
            st.session_state[f"model_{st.session_state["Predict: model_class"].lower()}"] = model
        else:
            model = st.session_state[f"model_{st.session_state["Predict: model_class"].lower()}"]
        st.session_state["prediction"] = predict(
            {
                colnames.investor: st.session_state[f"Predict: {colnames.investor}"],
                colnames.province: st.session_state[f"Predict: {colnames.province}"],
                colnames.quantity: st.session_state[f"Predict: {colnames.quantity}"],
                colnames.closing_date: st.session_state[f"Predict: {colnames.closing_date}"],
                colnames.manufacturer: st.session_state[f"Predict: {colnames.manufacturer}"],
                colnames.country_origin: st.session_state[f"Predict: {colnames.country_origin}"],
            },
            model,
        )
        print("=============================================")
        print(st.session_state["prediction"])
        print("=============================================")


def generate_label_filter(cname: str):
    label_col, button_col = st.columns([5, 1], gap="xxlarge", vertical_alignment="center")
    with label_col:
        st.markdown(f"**Filter {cname.replace("_", " ")}**")
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
        f"Filter {cname.replace("_", " ")}",
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

    st.markdown(f"**{cname.replace("_", " ").capitalize()}**")
    st.slider(
        f"{cname.replace("_", " ").capitalize()}",
        label_visibility="collapsed",
        min_value=df[cname].min(),
        max_value=df[cname].max(),
        key=f"Filter: {cname}",
    )


def generate_date_range(cname: str):
    df = st.session_state["data"]

    left_col, right_col = st.columns([5, 1], gap="xxlarge", vertical_alignment="center")
    with left_col:
        st.markdown(f"**{cname.replace("_", " ").capitalize()}**")
    with right_col:
        st.button(
            "🗑️",
            key=f"button: erase {cname}",
            help="Clear this filter",
            on_click=lambda: (st.session_state[f"Filter: min {cname}"].clear(), st.session_state[f"Filter: max {cname}"].clear()),
            type="tertiary",
            use_container_width=True,
        )

    left_col, right_col = st.columns([1, 1], gap="small", vertical_alignment="center")
    with left_col:
        label, input = st.columns([1, 3], gap="xxsmall", vertical_alignment="center")
        with label:
            st.markdown(f"From:")
        with input:
            st.date_input(
                "From",
                label_visibility="collapsed",
                min_value=df[cname].min(),
                max_value=df[cname].max(),
                format="DD-MM-YYYY",
                key=f"Filter: min {cname}",
            )
    with right_col:
        label, input = st.columns([1, 3], gap="xxsmall", vertical_alignment="center")
        with label:
            st.markdown(f"To:")
        with input:
            st.date_input(
                "To",
                label_visibility="collapsed",
                min_value=df[cname].min(),
                max_value=df[cname].max(),
                format="DD-MM-YYYY",
                key=f"Filter: max {cname}",
            )


st.set_page_config(page_title="Tender Market Analytics", layout="wide")
st.title("Tender Market Analytics", text_alignment="center")
left_col, right_col = st.columns([3, 7], gap="large")
with left_col:
    with st.container(height=650, border=True):
        with st.form("Query", border=False):
            st.text_area("Search query", height=200, key="text: query")
            if "query_error" in st.session_state and st.session_state["query_error"] is not None:
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
                if df.empty:
                    st.info("No result. Try adjusting your filters.")
                else:
                    if df[colnames.unit_price].max() < 1e3:
                        unit_unit_price = "VND"
                    elif df[colnames.unit_price].max() < 1e6:
                        df[colnames.unit_price] /= 1e3
                        unit_unit_price = "thousand VND"
                    elif df[colnames.unit_price].max() < 1e9:
                        df[colnames.unit_price] /= 1e6
                        unit_unit_price = "million VND"
                    else:
                        df[colnames.unit_price] /= 1e9
                        unit_unit_price = "billion VND"

                    if df[colnames.total_price].max() < 1e3:
                        unit_total_price = "VND"
                    elif df[colnames.total_price].max() < 1e6:
                        df[colnames.total_price] /= 1e3
                        unit_total_price = "thousand VND"
                    elif df[colnames.total_price].max() < 1e9:
                        df[colnames.total_price] /= 1e6
                        unit_total_price = "million VND"
                    else:
                        df[colnames.total_price] /= 1e9
                        unit_total_price = "billion VND"

                    with st.container():
                        st.subheader("Distribution of unit prices")

                        bins = 100
                        bars = (
                            alt.Chart(df)
                            .mark_bar()
                            .encode(
                                x=alt.X(colnames.unit_price, bin=alt.Bin(maxbins=bins), title=f"Unit price ({unit_unit_price})"),
                                y=alt.Y("count()", title="Count"),
                            )
                        )
                        background = (
                            bars.encode(
                                tooltip=[
                                    alt.Tooltip("mean:Q", title=f"Mean ({unit_unit_price})"),
                                    alt.Tooltip(
                                        "std:Q",
                                        title=f"Standard deviation ({unit_unit_price})",
                                    ),
                                    alt.Tooltip("min:Q", title=f"Min ({unit_unit_price})"),
                                    alt.Tooltip("q1:Q", title=f"Q1 ({unit_unit_price})"),
                                    alt.Tooltip("q3:Q", title=f"Q3 ({unit_unit_price})"),
                                    alt.Tooltip("max:Q", title=f"Max ({unit_unit_price})"),
                                ]
                            )
                            .transform_calculate(
                                mean=f"'{df[colnames.unit_price].mean():.3f}'",
                                std=f"'{df[colnames.unit_price].std():.3f}'",
                                min=f"'{df[colnames.unit_price].min():.3f}'",
                                q1=f"'{df[colnames.unit_price].quantile(0.25):.3f}'",
                                q3=f"'{df[colnames.unit_price].quantile(0.75):.3f}'",
                                max=f"'{df[colnames.unit_price].max():.3f}'",
                            )
                            .add_params(alt.selection_point(nearest=True))
                        )
                        bars = bars.add_params(alt.selection_point(on="mouseover", empty="none"))

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
                        st.subheader("Top contractors")

                        data = (
                            df.groupby(colnames.contractor_name, as_index=False)[colnames.total_price]
                            .sum()
                            .sort_values(by=colnames.total_price, ascending=False)
                            .head(5)
                        )
                        st.altair_chart(
                            alt.Chart(data)
                            .mark_bar()
                            .encode(
                                x=alt.X(
                                    colnames.total_price,
                                    title=f"Total price ({unit_total_price})",
                                ),
                                y=alt.Y(
                                    colnames.contractor_name,
                                    title="Contractor",
                                    sort="-x",
                                    axis=alt.Axis(labelLimit=400, labelOverlap=False, title=None),
                                ),
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
                        st.subheader("Top investor")

                        data = (
                            df.groupby(colnames.investor, as_index=False)[colnames.total_price]
                            .sum()
                            .sort_values(by=colnames.total_price, ascending=False)
                            .head(5)
                        )
                        st.altair_chart(
                            alt.Chart(data)
                            .mark_bar()
                            .encode(
                                x=alt.X(
                                    colnames.total_price,
                                    title=f"Total price ({unit_total_price})",
                                ),
                                y=alt.Y(
                                    colnames.investor,
                                    title="Investor",
                                    sort="-x",
                                    axis=alt.Axis(labelLimit=400, labelOverlap=False, title=None),
                                ),
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
                        st.subheader("Unit price by contractor")

                        top_bidder = set(
                            df.groupby(colnames.contractor_name, as_index=False)[colnames.total_price]
                            .sum()
                            .sort_values(by=colnames.total_price, ascending=False)
                            .head(10)[colnames.contractor_name]
                        )
                        data = df[df[colnames.contractor_name].isin(top_bidder)]
                        st.altair_chart(
                            alt.Chart(data)
                            .mark_boxplot()
                            .encode(
                                x=alt.X(
                                    colnames.unit_price,
                                    title=f"Unit price ({unit_unit_price})",
                                ),
                                y=alt.Y(
                                    colnames.contractor_name,
                                    title="Contractor",
                                    sort=alt.EncodingSortField(
                                        field=colnames.unit_price,
                                        op="median",
                                        order="descending",
                                    ),
                                    axis=alt.Axis(labelLimit=400, labelOverlap=False, title=None),
                                ),
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
                        st.subheader("Unit price by origin")

                        top_origin = set(
                            df.groupby(colnames.country_origin, as_index=False)[colnames.total_price]
                            .sum()
                            .sort_values(by=colnames.total_price, ascending=False)
                            .head(10)[colnames.country_origin]
                        )
                        data = df[df[colnames.country_origin].isin(top_origin)]
                        st.altair_chart(
                            alt.Chart(data)
                            .mark_boxplot()
                            .encode(
                                x=alt.X(
                                    colnames.unit_price,
                                    title=f"Unit price ({unit_unit_price})",
                                ),
                                y=alt.Y(
                                    colnames.country_origin,
                                    title="Origin",
                                    sort=alt.EncodingSortField(
                                        field=colnames.unit_price,
                                        op="median",
                                        order="descending",
                                    ),
                                    axis=alt.Axis(labelLimit=400, labelOverlap=False, title=None),
                                ),
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
                        st.subheader("Unit price by manufacturer")

                        top_origin = set(
                            df.groupby(colnames.manufacturer, as_index=False)[colnames.total_price]
                            .sum()
                            .sort_values(by=colnames.total_price, ascending=False)
                            .head(10)[colnames.manufacturer]
                        )
                        data = df[df[colnames.manufacturer].isin(top_origin)]
                        st.altair_chart(
                            alt.Chart(data)
                            .mark_boxplot()
                            .encode(
                                x=alt.X(
                                    colnames.unit_price,
                                    title=f"Unit price ({unit_unit_price})",
                                ),
                                y=alt.Y(
                                    colnames.manufacturer,
                                    title="Manufacturer",
                                    sort=alt.EncodingSortField(
                                        field=colnames.unit_price,
                                        op="median",
                                        order="descending",
                                    ),
                                    axis=alt.Axis(labelLimit=400, labelOverlap=False, title=None),
                                ),
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
                        st.subheader("Total value by country of origin")

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
                        st.subheader("Total value by region of origin")

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
                        st.subheader("Total value by manufacturer")

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
                        st.altair_chart(
                            alt.Chart(data)
                            .mark_arc()
                            .encode(
                                theta=alt.Theta(colnames.total_price, title=f"Total price ({unit_total_price})", sort="x"),
                                color=alt.Color(
                                    colnames.manufacturer,
                                    legend=alt.Legend(
                                        title="Manufacturer",
                                        labelColor="black",
                                        titleColor="black",
                                        labelFontSize=16,
                                        titleFontSize=16,
                                    ),
                                ),
                                order=alt.Order(colnames.total_price, sort="ascending"),
                                tooltip=[
                                    alt.Tooltip(colnames.manufacturer, title="Manufacturer"),
                                    alt.Tooltip(
                                        colnames.total_price,
                                        title=f"Total price ({unit_total_price})",
                                        format=",.3f",
                                    ),
                                ],
                            ),
                            width="stretch",
                        )
    with tab2:
        with st.container(height=600, border=True):
            if not "data" in st.session_state:
                st.info("Query something before predicting.")
            elif st.session_state["data"].empty:
                st.info("No result. Try another query before predicting.")
            else:
                with st.form("Parameters"):
                    st.subheader("Parameters for the prediction")
                    left_col, right_col = st.columns([1, 1], gap="large")
                    with left_col:
                        st.selectbox(
                            colnames.investor.replace("_", " ").capitalize(),
                            sorted(set(st.session_state["data"][colnames.investor]) | {"Other"}),
                            index=None,
                            key=f"Predict: {colnames.investor}",
                        )
                        st.selectbox(
                            colnames.province.replace("_", " ").capitalize(),
                            sorted(set(st.session_state["data"][colnames.province]) | {"Other"}),
                            index=None,
                            key=f"Predict: {colnames.province}",
                        )
                        inner_left_col, inner_right_col = st.columns([1, 1], gap="medium")
                        with inner_left_col:
                            st.number_input(
                                colnames.quantity.replace("_", " ").capitalize(),
                                key=f"Predict: {colnames.quantity}",
                                min_value=1,
                            )
                        with inner_right_col:
                            st.date_input(colnames.closing_date.replace("_", " ").capitalize(), key=f"Predict: {colnames.closing_date}", format="DD-MM-YYYY")
                    with right_col:
                        st.selectbox(
                            colnames.manufacturer.replace("_", " ").capitalize(),
                            sorted(set(st.session_state["data"][colnames.manufacturer]) | {"Other"}),
                            index=None,
                            key=f"Predict: {colnames.manufacturer}",
                        )
                        st.selectbox(
                            colnames.country_origin.replace("_", " ").capitalize(),
                            sorted(set(st.session_state["data"][colnames.country_origin]) | {"Other"}),
                            index=None,
                            key=f"Predict: {colnames.country_origin}",
                        )
                    banner = st.container()
                    _, class_space, button_space, _ = st.columns([2, 1, 2, 3], gap="small", vertical_alignment="bottom")
                    with class_space:
                        st.selectbox("Model class", ["Default", "Auto"], key=f"Predict: model_class")
                    with button_space:
                        st.form_submit_button("Train & predict", on_click=handle_predict_click, width="stretch")
                left_col, right_col = st.columns([1, 1])
                if "prediction" in st.session_state and st.session_state["prediction"] != None:
                    with left_col:
                        st.subheader("Results on test dataset")
                        metrics = st.session_state[f"model_{st.session_state["Predict: model_class"].lower()}"]["metrics"]
                        st.markdown(f"**Mean absolute error (VND): {metrics["MAE"]:.3f}**")
                        st.markdown(f"**Negative log likelihood: {metrics["NLL"]:.3f}**")
                    with right_col:
                        dist = st.session_state["prediction"]

                        def to_scalar(value):
                            arr = np.asarray(value)
                            if arr.size == 0:
                                return np.nan
                            return float(arr.reshape(-1)[0])

                        quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
                        quantiles_log = {q: to_scalar(dist.ppf(q)) for q in quantile_levels}

                        def exp_safe(v):
                            return float(np.exp(np.clip(v, -700, 700)))

                        quantiles = {q: exp_safe(quantiles_log[q]) for q in quantile_levels}
                        mu_log = to_scalar(getattr(dist, "mu", np.nan))
                        sigma_log = to_scalar(getattr(dist, "sigma", np.nan))
                        mean_original = np.exp(mu_log + 0.5 * (sigma_log ** 2)) if np.isfinite(mu_log) and np.isfinite(sigma_log) else np.nan

                        y_low = to_scalar(dist.ppf(0.001))
                        y_high = to_scalar(dist.ppf(0.999))
                        if not (np.isfinite(y_low) and np.isfinite(y_high)) or y_low >= y_high:
                            y_low = quantiles_log[0.1]
                            y_high = quantiles_log[0.9]
                            if not (np.isfinite(y_low) and np.isfinite(y_high)) or y_low >= y_high:
                                y_low, y_high = -1.0, 1.0

                        y_grid = np.linspace(y_low, y_high, 400)
                        x_grid = np.exp(np.clip(y_grid, -700, 700))

                        try:
                            pdf_y = np.asarray(dist.pdf(y_grid), dtype=float).reshape(-1)
                            if pdf_y.size != y_grid.size:
                                raise ValueError("Unexpected pdf output shape")
                        except Exception:
                            pdf_y = np.array([to_scalar(dist.pdf(y)) for y in y_grid], dtype=float)

                        with np.errstate(divide="ignore", invalid="ignore"):
                            pdf_values = pdf_y / x_grid

                        valid = np.isfinite(pdf_values)
                        if not valid.any():
                            st.warning("Could not evaluate the distribution PDF for plotting.")
                        else:
                            density_df = pd.DataFrame({"x": x_grid[valid], "pdf": pdf_values[valid]})

                            x_low = 0.0
                            x_high = max(float(density_df["x"].max()), quantiles[0.9], quantiles[0.5], 1.0)

                            q_df = pd.DataFrame(
                                {
                                    "quantile": quantile_levels,
                                    "x": [quantiles[q] for q in quantile_levels],
                                }
                            )
                            q_df["label"] = q_df["quantile"].map(lambda q: f"q={q:.2f}")
                            q_df["pdf"] = np.interp(q_df["x"], density_df["x"], density_df["pdf"])

                            density_chart = (
                                alt.Chart(density_df)
                                .mark_area(opacity=0.45, color="#9db7d5")
                                .encode(
                                    x=alt.X("x:Q", title="Unit price (thousand VND)"),
                                    y=alt.Y("pdf:Q", title="Density"),
                                    tooltip=[
                                        alt.Tooltip("x:Q", title="Unit price", format=",.3f"),
                                        alt.Tooltip("pdf:Q", title="Density", format=".6f"),
                                    ],
                                )
                            )

                            density_outline = (
                                alt.Chart(density_df)
                                .mark_line(color="#0b3c6f", strokeWidth=2)
                                .encode(
                                    x=alt.X("x:Q", title="Unit price (thousand VND)"),
                                    y=alt.Y("pdf:Q", title="Density"),
                                )
                            )

                            q_df["y0"] = 0.0

                            q_rules = (
                                alt.Chart(q_df)
                                .mark_rule(strokeWidth=2, strokeDash=[6, 6], color="#1f2937")
                                .encode(
                                    x=alt.X("x:Q"),
                                    y=alt.Y("y0:Q"),
                                    y2=alt.Y2("pdf:Q"),
                                    tooltip=[
                                        alt.Tooltip("label:N", title="Quantile"),
                                        alt.Tooltip("x:Q", title="Value", format=",.3f"),
                                    ],
                                )
                            )

                            q_points = (
                                alt.Chart(q_df)
                                .mark_point(size=80, filled=True, color="#1f2937")
                                .encode(
                                    x=alt.X("x:Q"),
                                    y=alt.Y("pdf:Q"),
                                    tooltip=[
                                        alt.Tooltip("label:N", title="Quantile"),
                                        alt.Tooltip("x:Q", title="Value", format=",.3f"),
                                        alt.Tooltip("pdf:Q", title="Density", format=".6f"),
                                    ],
                                )
                            )

                            q_labels = (
                                alt.Chart(q_df)
                                .mark_text(align="left", baseline="middle", dx=8, color="#1f2937", fontSize=12)
                                .encode(
                                    x=alt.X("x:Q"),
                                    y=alt.Y("pdf:Q"),
                                    text=alt.Text("label:N"),
                                )
                            )

                            st.subheader("Predicted winning bid price distribution")
                            if np.isfinite(mean_original):
                                st.markdown(f"**Mean (original scale): {mean_original:,.3f}**")
                            st.altair_chart(
                                (density_chart + density_outline + q_rules + q_points + q_labels)
                                .properties(height=320)
                                .configure_view(stroke=None)
                                .configure_axis(labelColor="black", titleColor="black", labelFontSize=14, titleFontSize=14),
                                width="stretch",
                            )
