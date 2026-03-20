import streamlit as st
import altair as alt
import pandas as pd
from datetime import datetime

def query_db(query: str):
    print(query)
    return pd.read_excel("Market_survey.xlsx")

def handle_search_click():
    df = query_db(st.session_state["text: query"]).dropna()
    df["Thời điểm đăng tải"] = pd.to_datetime(df["Thời điểm đăng tải"], format="%d-%m-%Y %H:%M")
    df["Thời điểm đóng thầu"] = pd.to_datetime(df["Thời điểm đóng thầu"], format="%d-%m-%Y %H:%M")
    # Temp fix
    df["Mã hàng hoá"] = df["Mã hàng hoá"].astype(str)
    df["Mặt hàng"] = df["Mặt hàng"].astype(str)

    st.session_state["data"] = df
    st.session_state["Filter: Truy vấn"] = set()
    st.session_state["Filter: Chủ đầu tư"] = set()
    st.session_state["Filter: Nhà thầu trúng thầu"] = set()
    st.session_state["Filter: Nhà sản xuất"] = set()
    st.session_state["Filter: Xuất xứ"] = set()
    st.session_state["Filter: Số lượng"] = (df["Số lượng"].min(), df["Số lượng"].max())
    st.session_state["Filter: Đơn giá trúng thầu"] = (df["Đơn giá trúng thầu"].min(), df["Đơn giá trúng thầu"].max())
    st.session_state["Filter: Thành tiền"] = (df["Thành tiền"].min(), df["Thành tiền"].max())
    st.session_state["Filter: Thời điểm đăng tải"] = (df["Thời điểm đăng tải"].min(), df["Thời điểm đăng tải"].max())
    st.session_state["Filter: Thời điểm đóng thầu"] = (df["Thời điểm đóng thầu"].min(), df["Thời điểm đóng thầu"].max())

def generate_label_filter(cname: str):
    label_col, button_col = st.columns([5, 1], gap="small", vertical_alignment="center")
    with label_col:
        st.markdown(f"**Filter {cname}**")
    with button_col:
        st.button("🗑️", key=f"button: erase {cname}", help="Clear this filter", on_click=lambda: st.session_state[f"Filter: {cname}"].clear(), type="tertiary", use_container_width=True)
    
    def handle_selectbox_change():
        st.session_state[f"Filter: {cname}"].add(st.session_state[f"selectbox: {cname}"])
        st.session_state[f"selectbox: {cname}"] = None
    st.selectbox(f"Filter {cname}", sorted(set(st.session_state["data"][cname])), key=f"selectbox: {cname}", index=None, placeholder=f"Select {cname} to filter", label_visibility="collapsed", on_change=handle_selectbox_change)

    st.pills("Current filters", sorted({"❌ " + s for s in st.session_state[f"Filter: {cname}"]}), key=f"pills: {cname}", label_visibility="collapsed", on_change=lambda: st.session_state[f"Filter: {cname}"].discard(st.session_state[f"pills: {cname}"][2:]))


def generate_checkboxes(cname: str):
    df = st.session_state["data"]
    elems = set(df[cname])

    st.markdown(f"**{cname}**")
    for e in sorted(elems):
        st.checkbox(e, key=f"checkbox: {cname}: {e}", value=e in st.session_state[f"Filter: {cname}"], on_change=lambda e=e: st.session_state[f"Filter: {cname}"].add(e) if st.session_state.get(f"checkbox: {cname}: {e}", False) else st.session_state[f"Filter: {cname}"].discard(e))

def generate_slider(cname: str):
    df = st.session_state["data"]
    
    st.slider(f"{cname}", min_value=df[cname].min(), max_value=df[cname].max(), key=f"Filter: {cname}")

def generate_date_range(cname: str):
    df = st.session_state["data"]

    st.date_input(f"{cname}", min_value=df[cname].min(), max_value=df[cname].max(), key=f"Filter: {cname}")

st.set_page_config(page_title="Tender Market Analytics", layout="wide")
st.title("Tender Market Analytics", text_alignment="center")
left_col, right_col = st.columns([2, 8], gap="large")
with left_col:
    with st.container(height=650, border=True):
        with st.form("Query", border=False):
            st.text_area("Search query", height=100, key="text: query")
            st.form_submit_button("Search", on_click=handle_search_click, width="stretch")
        if "data" in st.session_state and not st.session_state["data"].empty:
            generate_checkboxes("Truy vấn")
            generate_label_filter("Chủ đầu tư")
            generate_label_filter("Nhà thầu trúng thầu")
            generate_label_filter("Nhà sản xuất")
            generate_label_filter("Xuất xứ")
            generate_slider("Số lượng")
            generate_slider("Đơn giá trúng thầu")
            generate_slider("Thành tiền")
            generate_date_range("Thời điểm đăng tải")
            generate_date_range("Thời điểm đóng thầu")
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
                    s_post, e_post = st.session_state["Filter: Thời điểm đăng tải"]
                    s_bid, e_bid = st.session_state["Filter: Thời điểm đóng thầu"]
                except Exception:
                    s_post = st.session_state["Filter: Thời điểm đăng tải"][0]
                    e_post = datetime.now()
                    s_bid = st.session_state["Filter: Thời điểm đóng thầu"][0]
                    e_bid = datetime.now()
                df = raw_df[
                    (raw_df["Truy vấn"].isin(st.session_state["Filter: Truy vấn"]) if st.session_state["Filter: Truy vấn"] else True) &
                    (raw_df["Chủ đầu tư"].isin(st.session_state["Filter: Chủ đầu tư"]) if st.session_state["Filter: Chủ đầu tư"] else True) &
                    (raw_df["Nhà thầu trúng thầu"].isin(st.session_state["Filter: Nhà thầu trúng thầu"]) if st.session_state["Filter: Nhà thầu trúng thầu"] else True) &
                    (raw_df["Nhà sản xuất"].isin(st.session_state["Filter: Nhà sản xuất"]) if st.session_state["Filter: Nhà sản xuất"] else True) &
                    (raw_df["Xuất xứ"].isin(st.session_state["Filter: Xuất xứ"]) if st.session_state["Filter: Xuất xứ"] else True) &
                    (raw_df["Số lượng"].between(*st.session_state["Filter: Số lượng"])) &
                    (raw_df["Đơn giá trúng thầu"].between(*st.session_state["Filter: Đơn giá trúng thầu"])) &
                    (raw_df["Thành tiền"].between(*st.session_state["Filter: Thành tiền"])) &
                    (raw_df["Thời điểm đăng tải"].between(pd.Timestamp(s_post), pd.Timestamp(e_post) + pd.Timedelta(days=1))) &
                    (raw_df["Thời điểm đóng thầu"].between(pd.Timestamp(s_bid), pd.Timestamp(e_bid) + pd.Timedelta(days=1)))
                ]
                if df.empty:
                    st.info("No result. Try adjusting your filters.")
                else:
                    with st.container():
                        st.subheader("**Distribution of unit prices**")

                        st.altair_chart(alt.Chart(df).mark_bar().encode(
                                x=alt.X("Đơn giá trúng thầu", title="Đơn giá trúng thầu (triệu đồng)", axis=alt.Axis(labelExpr="datum.value / 1e6"), scale=alt.Scale(domainMin=0)),
                                y=alt.Y("count()", title="Count")
                            ).properties(height=alt.Step(36)).configure_view(stroke=None).configure_axis(labelColor="black", titleColor="black", labelFontSize=16, titleFontSize=16), width="stretch")
                    with st.container():
                        st.subheader("**Top contractors**")

                        data = df.groupby("Nhà thầu trúng thầu", as_index=False)["Thành tiền"].sum().sort_values(by="Thành tiền", ascending=False).head(5)
                        st.altair_chart(alt.Chart(data).mark_bar().encode(
                                x=alt.X("Thành tiền", title="Thành tiền (tỷ đồng)", axis=alt.Axis(labelExpr="datum.value / 1e9")),
                                y=alt.Y("Nhà thầu trúng thầu", sort="-x", axis=alt.Axis(labelLimit=400, labelOverlap=False), title=None)
                            ).properties(height=alt.Step(36)).configure_view(stroke=None).configure_axis(labelColor="black", titleColor="black", labelFontSize=16, titleFontSize=16), width="stretch")
                    with st.container():
                        st.subheader("**Top customers**")

                        data = df.groupby("Chủ đầu tư", as_index=False)["Thành tiền"].sum().sort_values(by="Thành tiền", ascending=False).head(5)
                        st.altair_chart(alt.Chart(data).mark_bar().encode(
                                x=alt.X("Thành tiền", title="Thành tiền (tỷ đồng)", axis=alt.Axis(labelExpr="datum.value / 1e9")),
                                y=alt.Y("Chủ đầu tư", sort="-x", axis=alt.Axis(labelLimit=400, labelOverlap=False), title=None)
                            ).properties(height=alt.Step(36)).configure_view(stroke=None).configure_axis(labelColor="black", titleColor="black", labelFontSize=16, titleFontSize=16), width="stretch")
                    with st.container():
                        st.subheader("**Unit price by contractor**")

                        top_bidder = set(df.groupby("Nhà thầu trúng thầu", as_index=False)["Thành tiền"].sum().sort_values(by="Thành tiền", ascending=False).head(10)["Nhà thầu trúng thầu"])
                        data = df[df["Nhà thầu trúng thầu"].isin(top_bidder)]
                        st.altair_chart(alt.Chart(data).mark_boxplot().encode(
                                x=alt.X("Đơn giá trúng thầu", title="Đơn giá trúng thầu (triệu đồng)", axis=alt.Axis(labelExpr="datum.value / 1e6")),
                                y=alt.Y("Nhà thầu trúng thầu", sort=alt.EncodingSortField(field="Đơn giá trúng thầu", op="median", order="descending"), axis=alt.Axis(labelLimit=400, labelOverlap=False), title=None)
                            ).properties(height=alt.Step(36)).configure_view(stroke=None).configure_axis(labelColor="black", titleColor="black", labelFontSize=16, titleFontSize=16), width="stretch")
                    with st.container():
                        st.subheader("**Bid price by origin**")

                        data_by_origin = df.groupby("Xuất xứ", as_index=False)["Thành tiền"].sum().sort_values(by="Thành tiền", ascending=False)
                        top9 = data_by_origin.head(9)
                        rest = data_by_origin.iloc[9:]["Thành tiền"].sum()

                        data = pd.concat([top9, pd.DataFrame([{"Xuất xứ": "Khác", "Thành tiền": rest}])], ignore_index=True) if rest > 0 else top9
                        st.altair_chart(alt.Chart(data).mark_arc().encode(
                            theta=alt.Theta("Thành tiền", title="Thành tiền", sort="x"),
                            color=alt.Color("Xuất xứ", legend=alt.Legend(title="Xuất xứ", labelColor="black", titleColor="black", labelFontSize=16, titleFontSize=16)),
                            order=alt.Order("Thành tiền", sort="ascending"),
                            tooltip=[
                                alt.Tooltip("Xuất xứ", title="Xuất xứ"),
                                alt.Tooltip("Thành tiền", title="Thành tiền", format=",.0f")
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
                    left_col, right_col = st.columns([1, 1], gap="large")
                    with left_col:
                        st.selectbox("Investor", sorted(set(st.session_state["data"]["Chủ đầu tư"]) | {"Other"}), index=None, key="Predict: investor")
                        st.selectbox("Province", sorted(set(st.session_state["data"]["Địa điểm"]) | {"Other"}), index=None, key="Predict: province")
                        inner_left_col, inner_right_col = st.columns([1, 1], gap="medium")
                        with inner_left_col:
                            st.number_input("Quantity", key="Predict: quantity", min_value=1)
                        with inner_right_col:
                            st.date_input("Bidding date", key="Predict: date")
                    with right_col:
                        st.selectbox("Manufacturer", sorted(set(st.session_state["data"]["Nhà sản xuất"]) | {"Other"}), index=None, key="Predict: manufacturer")
                        st.selectbox("Country of origin", sorted(set(st.session_state["data"]["Xuất xứ"]) | {"Other"}), index=None, key="Predict: origin")
                    _, button_space, _ = st.columns([1, 1, 1], gap="large")
                    with button_space:
                        st.form_submit_button("Search", on_click=handle_search_click, width="stretch")
