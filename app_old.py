"""Political Misinformation Risk Monitor — end-user interface.

A simple, non-technical front end over the existing detection backend.
Two pages: Overview (the current situation) and Check News (analyse one item).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import DATASET_SOURCES
from src.data_loader import load_prepared
from src.predict import Predictor
from src.train import has_trained_model

st.set_page_config(
    page_title="Political Misinformation Risk Monitor",
    page_icon=":shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

RISK_STYLE = {
    "Low": {"color": "#16a34a", "emoji": "🟢", "label": "LOW RISK"},
    "Medium": {"color": "#d97706", "emoji": "🟡", "label": "MEDIUM RISK"},
    "High": {"color": "#dc2626", "emoji": "🔴", "label": "HIGH RISK"},
}

RISK_ORDER = ["Low", "Medium", "High"]
RISK_COLOR_MAP = {k: v["color"] for k, v in RISK_STYLE.items()}


# --------------------------------------------------------------------------- #
# Cached backend access
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner="Loading data…")
def load_data() -> pd.DataFrame:
    return load_prepared(sorted(set(DATASET_SOURCES)))


@st.cache_resource(show_spinner="Preparing…")
def get_predictor() -> Predictor:
    df = load_data() if not has_trained_model() else None
    return Predictor.load(df)


@st.cache_data(show_spinner="Working…")
def get_catalog() -> pd.DataFrame:
    predictor = get_predictor()
    df = load_data()
    return predictor.score_df(df)


# --------------------------------------------------------------------------- #
# Presentation helpers
# --------------------------------------------------------------------------- #

def _risk_text_for(level: str) -> str:
    return {
        "Low": "Lower concern. This news looks unlikely to be misinformation.",
        "Medium": "Moderate concern. This news may need some monitoring.",
        "High": "High concern. This news should be reviewed carefully.",
    }[level]


def _level_for(score: float) -> str:
    if score >= 70:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def _verdict_sentence(is_misinfo: bool) -> str:
    return (
        "This news looks like misinformation."
        if is_misinfo
        else "This news looks reliable."
    )


def _risk_card(title: str, value: str, subtitle: str, color: str) -> None:
    st.markdown(
        f"""
        <div style="border-left:5px solid {color}; background:#ffffff;
                    border:1px solid #e2e8f0; padding:16px 18px; border-radius:12px;
                    margin-bottom:8px;">
            <div style="color:#475569; font-size:13px; letter-spacing:.04em;
                        text-transform:uppercase;">{title}</div>
            <div style="font-size:34px; font-weight:800; color:{color}; line-height:1.2;">
                {value}</div>
            <div style="color:#64748b; font-size:13px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _component_card(title: str, score: float, note: str) -> None:
    color = RISK_STYLE[_level_for(score)]["color"]
    st.markdown(
        f"<div style='border:1px solid #e2e8f0; border-radius:12px; "
        f"padding:16px 18px;'>"
        f"<div style='color:#64748b; font-size:13px; letter-spacing:.04em; "
        f"text-transform:uppercase;'>{title}</div>"
        f"<div style='font-size:30px; font-weight:800; color:{color};'>"
        f"{score:.0f} / 100</div>"
        f"<div style='font-size:13px; color:#64748b; margin-top:6px;'>{note}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def overall_risk_gauge(score: float) -> go.Figure:
    """Semicircular speedometer showing where the risk lands (Low/Medium/High)."""
    value = float(np.clip(score, 0.0, 100.0))
    fig = go.Figure(
        go.Indicator(
            mode="gauge",
            value=value,
            gauge={
                "shape": "angular",
                "axis": {
                    "range": [0, 100],
                    "tickvals": [0, 20, 55, 85, 100],
                    "ticktext": ["0", "LOW", "MEDIUM", "HIGH", "100"],
                    "tickfont": {"size": 13, "color": "#475569"},
                    "ticks": "",
                },
                "bar": {"color": "rgba(0,0,0,0)", "thickness": 0},
                "steps": [
                    {"range": [0, 40], "color": "#16a34a"},
                    {"range": [40, 70], "color": "#f59e0b"},
                    {"range": [70, 100], "color": "#dc2626"},
                ],
                "threshold": {
                    "line": {"color": "#111827", "width": 6},
                    "thickness": 0.9,
                    "value": value,
                },
            },
        )
    )
    fig.update_layout(height=250, margin=dict(l=30, r=30, t=20, b=10))
    return fig


# --------------------------------------------------------------------------- #
# Shared article detail panel
# --------------------------------------------------------------------------- #

def render_article_detail(row: pd.Series) -> None:
    level = row["risk_level"]
    is_misinfo = row["pred_label"] == 1
    st.markdown(f"#### {row['title']}")
    st.markdown(
        f"<div style='padding:16px;border-radius:12px;background:{RISK_STYLE[level]['color']};"
        f"color:#fff;font-size:20px;font-weight:800;'>"
        f"{RISK_STYLE[level]['emoji']} {RISK_STYLE[level]['label']} — "
        f"{row['risk_score']:.0f} / 100</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Risk", f"{row['risk_score']:.0f} / 100")
    c2.metric("Content Risk", f"{row['content_risk']:.0f} / 100")
    c3.metric("Spread Risk", f"{row['propagation_risk']:.0f} / 100")

    st.markdown(
        f"**{_verdict_sentence(is_misinfo)}**  ·  "
        f"Shared {int(row['tweet_count']):,} times"
    )


# --------------------------------------------------------------------------- #
# Page 1 — Overview
# --------------------------------------------------------------------------- #

def page_overview() -> None:
    st.title("Risk Overview")
    st.caption("The current situation across the political news being monitored.")

    catalog = get_catalog()
    total = len(catalog)
    counts = catalog["risk_level"].value_counts().to_dict()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _risk_card("High Risk", str(counts.get("High", 0)), "Needs attention",
                   RISK_STYLE["High"]["color"])
    with c2:
        _risk_card("Medium Risk", str(counts.get("Medium", 0)), "Watch closely",
                   RISK_STYLE["Medium"]["color"])
    with c3:
        _risk_card("Low Risk", str(counts.get("Low", 0)), "Lower concern",
                   RISK_STYLE["Low"]["color"])
    with c4:
        _risk_card("Total Checked", f"{total:,}", "News items monitored", "#334155")

    st.write("")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("##### How risk is spread")
        dist = pd.DataFrame(
            {"Risk Level": RISK_ORDER,
             "Count": [counts.get(k, 0) for k in RISK_ORDER]}
        )
        fig = px.pie(
            dist, names="Risk Level", values="Count", color="Risk Level",
            color_discrete_map=RISK_COLOR_MAP, hole=0.5,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320,
                          showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with right:
        st.markdown("##### Needs attention")
        high = catalog[catalog["risk_level"] == "High"].sort_values(
            "risk_score", ascending=False
        )
        if high.empty:
            st.info("Nothing is currently flagged as high risk.")
        else:
            st.dataframe(
                high[["title", "risk_score", "tweet_count"]]
                .rename(columns={
                    "title": "Headline",
                    "risk_score": "Risk Score",
                    "tweet_count": "Shares",
                })
                .head(10),
                width="stretch", height=340,
                column_config={
                    "Risk Score": st.column_config.NumberColumn(format="%d / 100"),
                    "Shares": st.column_config.NumberColumn(format="%d"),
                },
            )

    st.markdown("##### Browse monitored news")
    fc1, fc2 = st.columns([1, 2])
    with fc1:
        level_filter = st.selectbox(
            "Risk level", ["All", "High", "Medium", "Low"]
        )
    with fc2:
        search = st.text_input("Search", placeholder="Type a headline…")

    view = catalog.copy()
    if level_filter != "All":
        view = view[view["risk_level"] == level_filter]
    if search.strip():
        view = view[view["title"].str.contains(search.strip(), case=False, na=False)]
    view = view.sort_values("risk_score", ascending=False)

    display = view[["title", "risk_level", "risk_score", "tweet_count"]].copy()
    display["risk_level"] = display["risk_level"].map(
        lambda lvl: f"{RISK_STYLE[lvl]['emoji']} {lvl.title()}"
    )
    st.dataframe(
        display.rename(columns={
            "title": "Headline",
            "risk_level": "Risk",
            "risk_score": "Risk Score",
            "tweet_count": "Shares",
        }).head(500),
        width="stretch", height=420,
        column_config={
            "Risk Score": st.column_config.NumberColumn(format="%d / 100"),
            "Shares": st.column_config.NumberColumn(format="%d"),
        },
    )

    st.markdown("##### View an item")
    if view.empty:
        st.info("No news matches the current filter.")
        return
    options = view["title"].head(300).tolist()
    chosen = st.selectbox("Choose a headline", options)
    if chosen:
        render_article_detail(view[view["title"] == chosen].iloc[0])


# --------------------------------------------------------------------------- #
# Page 2 — Check News
# --------------------------------------------------------------------------- #

def page_check_news() -> None:
    st.title("Check News")
    st.caption("Paste a piece of news to see whether it may be misinformation.")

    headline = st.text_input(
        "News headline",
        placeholder="Enter or paste the news headline…",
    )
    body = st.text_area(
        "Article text (optional)", height=140, placeholder="Paste the article here…"
    )
    shares = st.number_input(
        "How many times has it been shared?",
        min_value=0, max_value=1000000, value=0, step=1,
        help="Approx. number of shares/likes on social media.",
    )

    if st.button("Check this news", type="primary", width="stretch"):
        if not headline.strip():
            st.error("Please enter a news headline.")
            return
        with st.spinner("Checking…"):
            try:
                predictor = get_predictor()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not start the checker: {exc}")
                return
            label, _label_name, p_fake, details = predictor.predict(
                headline, body if body.strip() else None, int(shares)
            )
        st.session_state["analysis"] = {
            "risk_score": details["risk_score"],
            "risk_level": details["risk_level"],
            "content_risk": details["content_risk"],
            "propagation_risk": details["propagation_risk"],
            "label": label,
        }

    if "analysis" in st.session_state:
        render_analysis_result(st.session_state["analysis"])


def render_analysis_result(a: dict) -> None:
    level = a["risk_level"]
    style = RISK_STYLE[level]
    is_misinfo = a["label"] == 1
    v_color = "#b91c1c" if is_misinfo else "#166534"
    v_emoji = "⚠️" if is_misinfo else "✅"

    st.divider()
    st.markdown("### Result")

    st.plotly_chart(overall_risk_gauge(a["risk_score"]), width="stretch")

    st.markdown(
        f"<div style='text-align:center;'>"
        f"<div style='font-size:40px; font-weight:800; color:{style['color']}; "
        f"line-height:1.1;'>{a['risk_score']:.0f} / 100</div>"
        f"<div style='font-size:22px; font-weight:800; color:{style['color']}; "
        f"margin-top:4px;'>{style['emoji']} {style['label']}</div>"
        f"<div style='font-size:15px; color:#475569; margin-top:8px; "
        f"max-width:560px; margin-left:auto; margin-right:auto;'>"
        f"{_risk_text_for(level)}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='padding:14px 18px; border-radius:10px; margin-top:12px; "
        f"font-size:17px; font-weight:700; color:{v_color}; text-align:center;'>"
        f"{v_emoji} {_verdict_sentence(is_misinfo)}</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Why this result?"):
        c1, c2 = st.columns(2)
        with c1:
            _component_card(
                "Content Risk", a["content_risk"],
                "How much the text looks like misinformation.",
            )
        with c2:
            _component_card(
                "Spread Risk", a["propagation_risk"],
                "How widely this news is being shared.",
            )


# --------------------------------------------------------------------------- #
# Sidebar + routing
# --------------------------------------------------------------------------- #

def main() -> None:
    with st.sidebar:
        st.markdown(
            "<div style='font-size:16px;font-weight:800;line-height:1.25;'>"
            "POLITICAL MISINFORMATION<br>RISK MONITOR</div>",
            unsafe_allow_html=True,
        )
        st.write("")
        page = st.radio("Menu", ["🏠 Overview", "🔍 Check News"])

    if page == "🏠 Overview":
        page_overview()
    else:
        page_check_news()


if __name__ == "__main__":
    main()