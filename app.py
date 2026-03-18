import streamlit as st
import altair as alt
import pandas as pd
import streamlit.components.v1 as components
from datetime import datetime

def query_db(query: str):
    return pd.read_excel("Market_survey.xlsx")

st.set_page_config(page_title="Tender Market Analytics", layout="wide")
components.html(
        """
        <script>
        (function () {
            const doc = window.parent.document;
            if (doc.__ctrlWheelZoomInstalled) return;
            doc.__ctrlWheelZoomInstalled = true;

            doc.addEventListener(
                "wheel",
                function (event) {
                    const chartTarget = event.target.closest(
                        '[data-testid="stVegaLiteChart"], [data-testid="stPlotlyChart"], .js-plotly-plot'
                    );
                    if (!chartTarget) return;
                    if (!event.ctrlKey) event.stopPropagation();
                },
                { capture: true, passive: true }
            );
        })();
        </script>
        """,
        height=0,
    )
_, content, _ = st.columns([1, 6, 1], gap="small")
panel_height = 600

if "data" not in st.session_state:
    st.session_state["data"] = query_db("hehe").iloc[0:0]
if "query" not in st.session_state:
    st.session_state["query"] = "to be or not to be"
if "Filter: Truy vấn" not in st.session_state:
    st.session_state["Filter: Truy vấn"] = set()
if "Filter: Chủ đầu tư" not in st.session_state:
    st.session_state["Filter: Chủ đầu tư"] = set()
if "Filter: Nhà thầu trúng thầu" not in st.session_state:
    st.session_state["Filter: Nhà thầu trúng thầu"] = set()
if "Filter: Nhà sản xuất" not in st.session_state:
    st.session_state["Filter: Nhà sản xuất"] = set()
if "Filter: Xuất xứ" not in st.session_state:
    st.session_state["Filter: Xuất xứ"] = set()
if "Filter: Số lượng" not in st.session_state:
    st.session_state["Filter: Số lượng"] = (0, 0)
if "Filter: Đơn giá trúng thầu" not in st.session_state:
    st.session_state["Filter: Đơn giá trúng thầu"] = (0, 0)
if "Filter: Thành tiền" not in st.session_state:
    st.session_state["Filter: Thành tiền"] = (0, 0)
if "Filter: Thời điểm đăng tải" not in st.session_state:
    st.session_state["Filter: Thời điểm đăng tải"] = (datetime.now(), datetime.now())
if "Filter: Thời điểm đóng thầu" not in st.session_state:
    st.session_state["Filter: Thời điểm đóng thầu"] = (datetime.now(), datetime.now())

def handle_search_click():
    query = st.session_state.get("query", "")
    df = query_db(query).dropna()
    df["Thời điểm đăng tải"] = pd.to_datetime(df["Thời điểm đăng tải"], format="%d-%m-%Y %H:%M")
    df["Thời điểm đóng thầu"] = pd.to_datetime(df["Thời điểm đóng thầu"], format="%d-%m-%Y %H:%M")
    st.session_state["data"] = df

    st.session_state["Filter: Truy vấn"] = set(df["Truy vấn"].dropna())
    st.session_state["Filter: Chủ đầu tư"] = set(df["Chủ đầu tư"].dropna())
    st.session_state["Filter: Nhà thầu trúng thầu"] = set(df["Nhà thầu trúng thầu"].dropna())
    st.session_state["Filter: Nhà sản xuất"] = set(df["Nhà sản xuất"].dropna())
    st.session_state["Filter: Xuất xứ"] = set(df["Xuất xứ"].dropna())
    st.session_state["Filter: Số lượng"] = (df["Số lượng"].min(), df["Số lượng"].max())
    st.session_state["Filter: Đơn giá trúng thầu"] = (df["Đơn giá trúng thầu"].min(), df["Đơn giá trúng thầu"].max())
    st.session_state["Filter: Thành tiền"] = (df["Thành tiền"].min(), df["Thành tiền"].max())
    st.session_state["Filter: Thời điểm đăng tải"] = (df["Thời điểm đăng tải"].min(), df["Thời điểm đăng tải"].max())
    st.session_state["Filter: Thời điểm đóng thầu"] = (df["Thời điểm đóng thầu"].min(), df["Thời điểm đóng thầu"].max())

def generate_checkboxes(cname: str):
    df = st.session_state.get("data", pd.DataFrame())
    elems = set(df[cname].dropna())

    return (st.markdown(f"**{cname}**"), [st.checkbox(e, key=f"checkbox: {cname}: {e}", value=e in st.session_state[f"Filter: {cname}"], on_change=lambda e=e: st.session_state[f"Filter: {cname}"].add(e) if st.session_state.get(f"checkbox: {cname}: {e}", False) else st.session_state[f"Filter: {cname}"].discard(e)) for e in sorted(elems)])

def generate_slider(cname: str):
    df = st.session_state.get("data", pd.DataFrame())
    
    return st.slider(f"{cname}", min_value=df[cname].min(), max_value=df[cname].max(), key=f"Filter: {cname}")

def generate_date_range(cname: str):
    df = st.session_state.get("data", pd.DataFrame())

    if not df.empty:
        return st.date_input(f"{cname}", min_value=df[cname].min(), max_value=df[cname].max(), key=f"Filter: {cname}")

with content:
    st.markdown("<h1 align='center'>Tender Market Analytics</h1>", unsafe_allow_html=True)
    left_col, right_col = st.columns([3, 7], gap="large")
    
    with left_col:
        with st.container(height="stretch"):
            st.text_area(
                "Search query",
                height=100,
                key="query"
            )
            st.button(
                "Search",
                key="search_button",
                on_click=handle_search_click,
                width="stretch",
            )

            generate_checkboxes("Truy vấn")
            # generate_checkboxes("Chủ đầu tư")
            # generate_checkboxes("Nhà thầu trúng thầu")
            # generate_checkboxes("Nhà sản xuất")
            # generate_checkboxes("Xuất xứ")
            generate_slider("Số lượng")
            generate_slider("Đơn giá trúng thầu")
            generate_slider("Thành tiền")
            generate_date_range("Thời điểm đăng tải")
            generate_date_range("Thời điểm đóng thầu")

    with right_col:
        with st.container(height="stretch"):
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
                raw_df["Truy vấn"].isin(st.session_state["Filter: Truy vấn"]) &
                raw_df["Chủ đầu tư"].isin(st.session_state["Filter: Chủ đầu tư"]) &
                raw_df["Nhà thầu trúng thầu"].isin(st.session_state["Filter: Nhà thầu trúng thầu"]) &
                raw_df["Nhà sản xuất"].isin(st.session_state["Filter: Nhà sản xuất"]) &
                raw_df["Xuất xứ"].isin(st.session_state["Filter: Xuất xứ"]) &
                raw_df["Số lượng"].between(*st.session_state["Filter: Số lượng"]) &
                raw_df["Đơn giá trúng thầu"].between(*st.session_state["Filter: Đơn giá trúng thầu"]) &
                raw_df["Thành tiền"].between(*st.session_state["Filter: Thành tiền"]) &
                raw_df["Thời điểm đăng tải"].between(pd.Timestamp(s_post), pd.Timestamp(e_post) + pd.Timedelta(days=1)) &
                raw_df["Thời điểm đóng thầu"].between(pd.Timestamp(s_bid), pd.Timestamp(e_bid) + pd.Timedelta(days=1))
            ]
            if df.empty:
                st.info("Query something, insight will be drawn here.")
            else:
                with st.container():
                    st.markdown("**Top contractors**")

                    data = df.groupby("Nhà thầu trúng thầu", as_index=False)["Thành tiền"].sum().sort_values(by="Thành tiền", ascending=False).head(5)
                    st.altair_chart(alt.Chart(data).mark_bar().encode(
                            x="Thành tiền",
                            y=alt.Y("Nhà thầu trúng thầu", sort="-x", axis=alt.Axis(labelLimit=500), title=None)
                        ).configure_view(stroke=None), width="stretch")
                
                with st.container():
                    st.markdown("**Top customers**")

                    data = df.groupby("Chủ đầu tư", as_index=False)["Thành tiền"].sum().sort_values(by="Thành tiền", ascending=False).head(5)
                    st.altair_chart(alt.Chart(data).mark_bar().encode(
                            x="Thành tiền",
                            y=alt.Y("Chủ đầu tư", sort="-x", axis=alt.Axis(labelLimit=500), title=None)
                        ).configure_view(stroke=None), width="stretch")

                with st.container():
                    st.markdown("**Unit price by contractor**")

                    top_bidder = set(df.groupby("Nhà thầu trúng thầu", as_index=False)["Thành tiền"].sum().sort_values(by="Thành tiền", ascending=False).head(10)["Nhà thầu trúng thầu"])
                    data = df[df["Nhà thầu trúng thầu"].isin(top_bidder)]
                    st.altair_chart(alt.Chart(data).mark_boxplot().encode(
                            x=alt.X("Đơn giá trúng thầu", title="Đơn giá trúng thầu"),
                            y=alt.Y("Nhà thầu trúng thầu", sort=alt.EncodingSortField(field="Đơn giá trúng thầu", op="median", order="descending"), axis=alt.Axis(labelLimit=500), title=None)
                        ).configure_view(stroke=None), width="stretch")

                with st.container():
                    st.markdown("**Distribution of bid prices**")

                    # l = df["Thành tiền"].quantile(0.05)
                    # h = df["Thành tiền"].quantile(0.95)
                    # chart_df = df[df["Thành tiền"].between(l, h)]

                    st.altair_chart(alt.Chart(df).mark_bar().encode(
                            x=alt.X("Thành tiền", title="Thành tiền"),
                            y=alt.Y("count()", title="Count")
                        ).configure_view(stroke=None), width="stretch")
                
                with st.container():
                    st.markdown("**Bid price by origin**")

                    data_by_origin = df.groupby("Xuất xứ", as_index=False)["Thành tiền"].sum().sort_values(by="Thành tiền", ascending=False)
                    top9 = data_by_origin.head(9)
                    rest = data_by_origin.iloc[9:]["Thành tiền"].sum()

                    data = pd.concat([top9, pd.DataFrame([{"Xuất xứ": "Khác", "Thành tiền": rest}])], ignore_index=True) if rest > 0 else top9
                    st.altair_chart(alt.Chart(data).mark_arc().encode(
                        theta=alt.Theta("Thành tiền", title="Thành tiền", sort="x"),
                        color=alt.Color("Xuất xứ", legend=alt.Legend(title="Xuất xứ")),
                        order=alt.Order("Thành tiền", sort="ascending"),
                        tooltip=[
                            alt.Tooltip("Xuất xứ", title="Xuất xứ"),
                            alt.Tooltip("Thành tiền", title="Thành tiền", format=",.0f")
                        ]).configure_view(stroke=None),width="stretch")
