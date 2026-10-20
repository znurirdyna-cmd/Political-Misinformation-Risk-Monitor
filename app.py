from __future__ import annotations

import html

import pandas as pd
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

# Weights of the composite score (Interim report, Section 3.6.3).
# Keep these in sync with the values the backend uses.
CONTENT_WEIGHT = 0.6
SPREAD_WEIGHT = 0.4

PAGE_SIZE = 10

# --------------------------------------------------------------------------- #
# Look and feel
# --------------------------------------------------------------------------- #

ICON_ALERT = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>'
    '<path d="M12 9v4"/><path d="M12 17h.01"/></svg>'
)
ICON_EYE = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/>'
    '<circle cx="12" cy="12" r="3"/></svg>'
)
ICON_CHECK = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<circle cx="12" cy="12" r="10"/><path d="m8 12 3 3 5-6"/></svg>'
)

# Colours differ in lightness (dark / mid / light), and every level also has an
# icon and a word, so the levels can be told apart without relying on colour.
LEVELS = {
    "High": {
        "bg": "#9A2A1F", "fg": "#FFFFFF", "border": "#9A2A1F", "icon": ICON_ALERT,
        "blurb": "Likely to be false and spreading widely.",
        "action": "Review immediately.",
    },
    "Medium": {
        "bg": "#FBEACB", "fg": "#5E3803", "border": "#EBCB91", "icon": ICON_EYE,
        "blurb": "Some warning indicators are present.",
        "action": "Monitor regularly.",
    },
    "Low": {
        "bg": "#E3ECF5", "fg": "#1E3F5E", "border": "#C3D4E6", "icon": ICON_CHECK,
        "blurb": "Likely to be accurate and not spreading widely.",
        "action": "No immediate action required.",
    },
}
LEVEL_ORDER = ["High", "Medium", "Low"]

VERDICT = {
    1: {"word": "Likely misinformation", "short": "Misinformation", "emoji": "⚠️",
        "bg": "#FBE4E4", "fg": "#7A1F1F", "border": "#E0B4B4"},
    0: {"word": "Likely reliable", "short": "Reliable", "emoji": "✅",
        "bg": "#E4F4EA", "fg": "#1E5E3A", "border": "#B8DCC4"},
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap');
.stApp { background:#F4F2EC; color:#1C1E23; }
.stApp, .stApp p, .stApp label, .stApp input, .stApp textarea, .stApp button {
  font-family:'IBM Plex Sans', system-ui, sans-serif; }
.stApp h1, .stApp h2, .stApp h3 {
  font-family:'Source Serif 4', Georgia, serif !important; font-weight:600 !important; color:#1C1E23; }
section[data-testid="stSidebar"] { background:#ECE9E0; }

.rm-eyebrow { font-size:13px; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:#5B5E66; }
.rm-lead { font-family:'Source Serif 4', Georgia, serif; font-size:26px; line-height:1.35; margin:4px 0 2px; }
.rm-muted { color:#4A4D55; font-size:15px; }
.rm-small { color:#6B6E75; font-size:13px; }

.rm-tile { border-radius:14px; padding:22px 24px; display:flex; flex-direction:column; gap:12px; height:100%; }
.rm-tile-head { display:flex; align-items:center; gap:10px; font-weight:600; font-size:16px; }
.rm-tile-num { font-family:'Source Serif 4', Georgia, serif; font-size:52px; font-weight:600; line-height:1; }

.rm-card { background:#FFFFFF; border:1px solid #E2DED3; border-radius:14px; padding:24px 28px; height:100%; }
.rm-card h3 { margin:0 0 6px; font-size:22px; }

.rm-badge { display:inline-block; padding:5px 10px; border-radius:6px; font-weight:600; font-size:14px; white-space:nowrap; }

.rm-meter { height:12px; border-radius:6px; background:#EFEBE1; margin:6px 0; }
.rm-meter > div { height:12px; border-radius:6px; }

.rm-scale { display:flex; height:26px; border-radius:6px; overflow:hidden; font-size:13px; font-weight:600; }
.rm-scale > div { display:flex; align-items:center; justify-content:center; }

.rm-split { display:flex; height:34px; border-radius:8px; overflow:hidden; font-size:14px; font-weight:600; }
.rm-split > div { display:flex; align-items:center; justify-content:center; white-space:nowrap; overflow:hidden; }

.rm-colhead { font-size:13px; font-weight:600; letter-spacing:.04em; text-transform:uppercase; color:#4A4D55; }
.rm-row-title { font-size:16px; font-weight:500; line-height:1.35; }
.rm-rule { border:none; border-top:1px solid #E2DED3; margin:4px 0; }

.rm-scorebox { border-radius:14px; padding:20px 24px; text-align:center; }
.rm-scorebox .num { font-family:'Source Serif 4', Georgia, serif; font-size:60px; font-weight:600; line-height:1; margin:6px 0; }

.rm-note { background:#F7F5EF; border-radius:10px; padding:14px 16px; font-size:14px; color:#3E4148; line-height:1.5; }
</style>
"""


# --------------------------------------------------------------------------- #
# Backend access (unchanged)
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner="Loading articles…")
def load_data() -> pd.DataFrame:
    return load_prepared(sorted(set(DATASET_SOURCES)))


@st.cache_resource(show_spinner="Preparing the assessment service…")
def get_predictor() -> Predictor:
    df = load_data() if not has_trained_model() else None
    return Predictor.load(df)


@st.cache_data(show_spinner="Assessing all articles (first load only)…")
def get_catalog() -> pd.DataFrame:
    predictor = get_predictor()
    return predictor.score_df(load_data())


# --------------------------------------------------------------------------- #
# Plain-language helpers
# --------------------------------------------------------------------------- #

def looks_false_word(content_risk: float) -> str:
    if content_risk >= 80:
        return "Very likely"
    if content_risk >= 60:
        return "Likely"
    if content_risk >= 30:
        return "Unsure"
    return "Unlikely"


def spreading_word(propagation_risk: float) -> str:
    if propagation_risk >= 80:
        return "Very fast"
    if propagation_risk >= 60:
        return "Fast"
    if propagation_risk >= 30:
        return "Moderate"
    return "Slow"


def in_short(content_risk: float, propagation_risk: float, level: str) -> str:
    if content_risk >= 80:
        wording = "strongly resembles"
    elif content_risk >= 60:
        wording = "closely resembles"
    elif content_risk >= 30:
        wording = "partially resembles"
    else:
        wording = "does not closely resemble"

    if propagation_risk >= 80:
        spread = "it is being shared far more widely than most articles"
    elif propagation_risk >= 60:
        spread = "it is being shared more widely than most articles"
    elif propagation_risk >= 30:
        spread = "it is being shared at a level similar to other articles"
    else:
        spread = "it is not being shared widely"

    action = {
        "High": "This article should be reviewed by a person today.",
        "Medium": "This article should be monitored in case it spreads further.",
        "Low": "No immediate action is required.",
    }[level]
    return (
        f"The wording of this article {wording} content that fact-checkers have "
        f"previously found to be false, and {spread}. <strong>{action}</strong>"
    )


def shared_more_than(shares: int, all_shares: pd.Series) -> str:
    if all_shares.empty:
        return ""
    frac = float((all_shares < shares).mean())
    if frac >= 0.9:
        return "more than 9 out of 10 other articles"
    if frac >= 0.75:
        return "more than 3 out of 4 other articles"
    if frac >= 0.5:
        return "more than half of the other articles"
    if frac >= 0.25:
        return "more than 1 out of 4 other articles"
    return "less than most other articles"


def badge(level: str, score: float) -> str:
    s = LEVELS[level]
    return (
        f"<span class='rm-badge' style='background:{s['bg']};color:{s['fg']};"
        f"border:1px solid {s['border']};'>{level} · {score:.0f}</span>"
    )


def verdict_badge(pred_label: int) -> str:
    v = VERDICT[int(pred_label)]
    return (
        f"<span class='rm-badge' style='background:{v['bg']};color:{v['fg']};"
        f"border:1px solid {v['border']};'>{v['emoji']} {v['short']}</span>"
    )


def scale_bar(marker: float | None = None) -> str:
    marker_html = ""
    if marker is not None:
        pos = max(0.0, min(100.0, float(marker)))
        marker_html = (
            f"<div style='position:absolute;left:calc({pos}% - 4px);top:-6px;width:8px;"
            f"height:38px;border-radius:4px;background:#1C1E23;border:2px solid #fff;'></div>"
        )
    return (
        "<div style='position:relative;max-width:600px;margin-top:10px;'>"
        "<div class='rm-scale'>"
        f"<div style='width:40%;background:{LEVELS['Low']['bg']};color:{LEVELS['Low']['fg']};'>Low 0–39</div>"
        f"<div style='width:30%;background:{LEVELS['Medium']['bg']};color:{LEVELS['Medium']['fg']};'>Medium 40–69</div>"
        f"<div style='width:30%;background:{LEVELS['High']['bg']};color:{LEVELS['High']['fg']};'>High 70–100</div>"
        f"</div>{marker_html}</div>"
    )


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Navigation state
# --------------------------------------------------------------------------- #

def open_story(idx) -> None:
    st.session_state["story"] = idx


def close_story() -> None:
    st.session_state.pop("story", None)


def show_more() -> None:
    st.session_state["shown"] = st.session_state.get("shown", PAGE_SIZE) + PAGE_SIZE


def reset_list() -> None:
    st.session_state["shown"] = PAGE_SIZE


# --------------------------------------------------------------------------- #
# Article detail (used by both pages)
# --------------------------------------------------------------------------- #

def render_story(story: dict, all_shares: pd.Series, show_title: bool = True) -> None:
    level = story["risk_level"]
    s = LEVELS[level]
    score = float(story["risk_score"])
    content = float(story["content_risk"])
    spread = float(story["propagation_risk"])
    shares = int(story.get("tweet_count", 0) or 0)
    pred_label = int(story.get("pred_label", 0))

    # Points each part adds. Rounding can leave the two parts 1 point off the
    # total, so the second part takes up that difference.
    total_pts = round(score)
    content_pts = round(content * CONTENT_WEIGHT)
    spread_pts = round(spread * SPREAD_WEIGHT)
    if abs(content_pts + spread_pts - total_pts) <= 1:
        spread_pts = total_pts - content_pts

    head_l, head_r = st.columns([4, 1], vertical_alignment="top")
    with head_l:
        html_block("<div class='rm-eyebrow'>Assessment result</div>")
        if show_title:
            st.markdown(f"## {html.escape(str(story['title']))}")
        html_block(f"<div class='rm-muted'>Shared {shares:,} times</div>")
    with head_r:
        html_block(
            f"<div class='rm-scorebox' style='background:{s['bg']};color:{s['fg']};"
            f"border:1px solid {s['border']};'>"
            f"<div class='rm-tile-head' style='justify-content:center;'>{s['icon']} {level} risk</div>"
            f"<div class='num'>{score:.0f}</div><div>out of 100</div></div>"
        )

    st.write("")
    v = VERDICT[pred_label]
    html_block(
        f"<div style='padding:16px 20px;border-radius:10px;background:{v['bg']};"
        f"color:{v['fg']};border:1px solid {v['border']};font-size:18px;font-weight:700;'>"
        f"{v['emoji']} {v['word']}</div>"
    )

    st.write("")
    html_block(
        "<div class='rm-card'><h3>Summary</h3>"
        f"<p style='font-size:18px;line-height:1.55;margin:0;'>{in_short(content, spread, level)}</p>"
        f"{scale_bar(score)}</div>"
    )

    st.write("")
    left, right = st.columns(2)
    with left:
        html_block(
            f"<div class='rm-card'><h3>How the score is calculated</h3>"
            "<div style='display:flex;justify-content:space-between;font-size:16px;margin-top:14px;'>"
            f"<strong>Content risk</strong><span><strong>{looks_false_word(content)}</strong> · {content:.0f} / 100</span></div>"
            f"<div class='rm-meter'><div style='width:{content:.0f}%;background:#3B3F47;'></div></div>"
            f"<div class='rm-muted' style='font-size:14px;'>Accounts for {CONTENT_WEIGHT:.0%} of the score ({content_pts} points)</div>"
            "<div style='display:flex;justify-content:space-between;font-size:16px;margin-top:18px;'>"
            f"<strong>Spread risk</strong><span><strong>{spreading_word(spread)}</strong> · {spread:.0f} / 100</span></div>"
            f"<div class='rm-meter'><div style='width:{spread:.0f}%;background:#8A8E97;'></div></div>"
            f"<div class='rm-muted' style='font-size:14px;'>Accounts for {SPREAD_WEIGHT:.0%} of the score ({spread_pts} points)</div>"
            "<div style='display:flex;justify-content:space-between;align-items:center;margin-top:18px;"
            "padding:12px 16px;border-radius:10px;background:#F7F5EF;font-size:16px;'>"
            f"<span>{content_pts} + {spread_pts} =</span><strong style='font-size:19px;'>{total_pts} · {level} risk</strong></div>"
            "</div>"
        )
    with right:
        compare = shared_more_than(shares, all_shares)
        compare_html = (
            f"<p style='font-size:17px;line-height:1.5;margin:12px 0 18px;'>"
            f"This article has been shared <strong>{shares:,} times</strong>, which is {compare}.</p>"
            if compare else ""
        )
        advice = {
            "High": "Submit it to a fact-checker or verify it against an official source today.",
            "Medium": "No action is required yet. Review it again if it begins to spread more widely.",
            "Low": "No action is required at this time.",
        }[level]
        html_block(
            "<div class='rm-card'><h3>Sharing activity</h3>"
            f"{compare_html}"
            "<h3 style='margin-top:8px;'>Recommended action</h3>"
            f"<p style='font-size:16px;line-height:1.5;margin:6px 0 16px;'>{advice}</p>"
            "<div class='rm-note'>This is an indication of risk, not a definitive verdict. "
            "The system may be incorrect, so please verify against an official source "
            "before responding publicly.</div>"
            "</div>"
        )


# --------------------------------------------------------------------------- #
# Page 1 — Articles for Review
# --------------------------------------------------------------------------- #

def page_stories() -> None:
    catalog = get_catalog()

    if "story" in st.session_state and st.session_state["story"] in catalog.index:
        st.button("← Back to all articles", on_click=close_story)
        row = catalog.loc[st.session_state["story"]]
        render_story(row.to_dict(), catalog["tweet_count"])
        return

    total = len(catalog)
    counts = catalog["risk_level"].value_counts().to_dict()
    high = counts.get("High", 0)

    html_block("<div class='rm-eyebrow'>Political misinformation · Risk monitor</div>")
    st.title("Political Misinformation Risk Overview")
    if high:
        lead = (f"The system has assessed <strong>{total:,} articles</strong>. "
                f"<strong style='color:{LEVELS['High']['bg']};'>{high:,} require review.</strong>")
    else:
        lead = (f"The system has assessed <strong>{total:,} articles</strong>. "
                f"None require urgent review at this time.")
    html_block(f"<p class='rm-lead'>{lead}</p>"
               "<p class='rm-muted'>The list below prioritises articles by risk level.</p>")

    # Three risk tiles
    cols = st.columns(3)
    for col, level in zip(cols, LEVEL_ORDER):
        s = LEVELS[level]
        n = counts.get(level, 0)
        share = f"{n / total:.0%} of all articles" if total else ""
        col.markdown(
            f"<div class='rm-tile' style='background:{s['bg']};color:{s['fg']};border:1px solid {s['border']};'>"
            f"<div class='rm-tile-head'>{s['icon']} {level} risk</div>"
            f"<div style='display:flex;align-items:baseline;gap:10px;'>"
            f"<span class='rm-tile-num'>{n:,}</span><span>articles · {share}</span></div>"
            f"<div style='font-size:16px;line-height:1.45;'>{s['blurb']} <strong>{s['action']}</strong></div>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.write("")
    left, right = st.columns([1.3, 1])
    with left:
        parts = []
        for level in ["Low", "Medium", "High"]:
            n = counts.get(level, 0)
            if total and n:
                s = LEVELS[level]
                parts.append(
                    f"<div style='width:{n / total * 100:.2f}%;background:{s['bg']};color:{s['fg']};'>"
                    f"{level} {n / total:.0%}</div>"
                )
        html_block(
            "<div class='rm-card'><h3>Risk distribution</h3>"
            "<p class='rm-muted' style='margin:0 0 14px;'>Each colour represents one risk level. "
            "A longer section indicates more articles.</p>"
            f"<div class='rm-split'>{''.join(parts)}</div>"
            "</div>"
        )
    with right:
        html_block(
            "<div class='rm-card'><h3>How the risk score is calculated</h3>"
            "<p class='rm-muted' style='margin:0 0 12px;'>Each article receives a score from 0 to 100, "
            "based on two factors:</p>"
            f"<div style='display:flex;justify-content:space-between;'><strong>1. Content risk</strong>"
            f"<span class='rm-muted'>counts {CONTENT_WEIGHT:.0%}</span></div>"
            f"<div class='rm-meter'><div style='width:{CONTENT_WEIGHT * 100:.0f}%;background:#3B3F47;'></div></div>"
            "<div class='rm-muted' style='font-size:14px;margin-bottom:12px;'>The wording is compared against "
            "articles that fact-checkers have previously verified.</div>"
            f"<div style='display:flex;justify-content:space-between;'><strong>2. Spread risk</strong>"
            f"<span class='rm-muted'>counts {SPREAD_WEIGHT:.0%}</span></div>"
            f"<div class='rm-meter'><div style='width:{SPREAD_WEIGHT * 100:.0f}%;background:#8A8E97;'></div></div>"
            "<div class='rm-muted' style='font-size:14px;'>How widely the article is shared relative to other articles.</div>"
            f"{scale_bar()}</div>"
        )

    # Article list
    st.write("")
    st.subheader("Articles for review, ordered by risk")
    f1, f2 = st.columns([1.2, 1], vertical_alignment="bottom")
    with f1:
        level_filter = st.segmented_control(
            "Filter",
            ["All", "High", "Medium", "Low"],
            default="All",
            key="level_filter",
            on_change=reset_list,
        )
    with f2:
        search = st.text_input("Search", placeholder="Type words from the headline…",
                               on_change=reset_list)

    view = catalog
    if level_filter and level_filter != "All":
        view = view[view["risk_level"] == level_filter]
    if search.strip():
        view = view[view["title"].astype(str).str.contains(search.strip(), case=False, na=False, regex=False)]
    view = view.sort_values("risk_score", ascending=False)

    if view.empty:
        st.info("No articles match the selected filters. Please try another level or different search terms.")
        return

    widths = [5, 2, 1.4, 1.4, 1.4, 1.0, 1.6]
    head = st.columns(widths)
    for col, label in zip(head, ["Article", "Verdict", "Risk level", "Content risk", "Spread risk", "Shares", ""]):
        col.markdown(f"<div class='rm-colhead'>{label}</div>", unsafe_allow_html=True)
    html_block("<hr class='rm-rule'>")

    shown = st.session_state.get("shown", PAGE_SIZE)
    for idx, row in view.head(shown).iterrows():
        c = st.columns(widths, vertical_alignment="center")
        c[0].markdown(f"<div class='rm-row-title'>{html.escape(str(row['title']))}</div>",
                      unsafe_allow_html=True)
        c[1].markdown(verdict_badge(row["pred_label"]), unsafe_allow_html=True)
        c[2].markdown(badge(row["risk_level"], row["risk_score"]), unsafe_allow_html=True)
        c[3].markdown(looks_false_word(row["content_risk"]))
        c[4].markdown(spreading_word(row["propagation_risk"]))
        c[5].markdown(f"{int(row['tweet_count']):,}")
        c[6].button("View details", key=f"view_{idx}", on_click=open_story, args=(idx,),
                    type="primary" if row["risk_level"] == "High" else "secondary",
                    width="stretch")
        html_block("<hr class='rm-rule'>")

    html_block(f"<div class='rm-small'>Showing {min(shown, len(view)):,} of {len(view):,} articles</div>")
    if shown < len(view):
        st.button(f"Show {PAGE_SIZE} more", on_click=show_more)


# --------------------------------------------------------------------------- #
# Page 2 — Article Assessment
# --------------------------------------------------------------------------- #

def page_check() -> None:
    html_block("<div class='rm-eyebrow'>Political misinformation · Risk monitor</div>")
    st.title("Article Assessment")
    html_block("<p class='rm-muted'>Enter a political headline to assess its risk. "
               "The result uses the same assessment as the articles on the main page.</p>")

    with st.form("check_form"):
        headline = st.text_input("Headline", placeholder="Paste the headline here…")
        body = st.text_area("Full article (optional — improves accuracy)", height=140,
                            placeholder="Paste the rest of the article here…")
        shares = st.number_input(
            "Number of shares (enter 0 if unknown)",
            min_value=0, max_value=10_000_000, value=0, step=1,
        )
        submitted = st.form_submit_button("Assess this article", type="primary", width="stretch")

    if submitted:
        if not headline.strip():
            st.error("Please enter a headline before proceeding.")
            return
        with st.spinner("Assessing the article…"):
            try:
                predictor = get_predictor()
            except Exception as exc:  # noqa: BLE001
                st.error(f"The assessment service could not start: {exc}")
                return
            label, _name, _p_fake, details = predictor.predict(
                headline, body if body.strip() else None, int(shares)
            )
        st.session_state["checked"] = {
            "title": headline.strip(),
            "risk_score": details["risk_score"],
            "risk_level": details["risk_level"],
            "content_risk": details["content_risk"],
            "propagation_risk": details["propagation_risk"],
            "tweet_count": int(shares),
            "pred_label": label,
        }

    if "checked" in st.session_state:
        st.divider()
        render_story(st.session_state["checked"], get_catalog()["tweet_count"])


# --------------------------------------------------------------------------- #
# Sidebar + routing
# --------------------------------------------------------------------------- #

def main() -> None:
    html_block(CSS)
    with st.sidebar:
        html_block("<div style='font-family:\"Source Serif 4\",Georgia,serif;font-size:20px;"
                   "font-weight:600;line-height:1.25;'>Political Misinformation<br>Risk Monitor</div>")
        st.write("")
        page = st.radio("Navigation", ["Articles for Review", "Article Assessment"], on_change=close_story)
        st.write("")
        html_block("<div class='rm-note'>This system identifies articles that may contain misinformation. "
                   "It does not determine the truth; please verify against an official source "
                   "before taking action.</div>")

    if page == "Articles for Review":
        page_stories()
    else:
        page_check()


if __name__ == "__main__":
    main()
