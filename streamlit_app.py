import altair as alt
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"  # our FastAPI server

# Same rows the model was trained on (see train_model.py), shown on the chart
TRAINING_DATA = pd.DataFrame(
    {
        "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 1],
        "prev_score": [40, 45, 50, 60, 65, 70, 75, 85, 80, 30],
        "label": ["Fail", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Fail"],
    }
)

# Palette (matches .streamlit/config.toml)
TEXT = "#FAFAFA"
MUTED = "#8E8E96"
BORDER = "#232327"
SURFACE = "#111113"
PASS_COLOR = "#3DDC97"
FAIL_COLOR = "#FF6B6B"

st.set_page_config(page_title="Examine", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500&display=swap');

    :root {{
        --text: {TEXT}; --muted: {MUTED}; --border: {BORDER}; --surface: {SURFACE};
        --pass: {PASS_COLOR}; --fail: {FAIL_COLOR};
        --font: 'Geist', system-ui, sans-serif;
        --mono: 'Geist Mono', ui-monospace, monospace;
    }}

    html, body, [data-testid="stAppViewContainer"], .stMarkdown, p, label, button, input {{
        font-family: var(--font) !important;
    }}

    /* Hide Streamlit chrome */
    header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stSidebarCollapsedControl"],
    footer, #MainMenu {{ display: none !important; }}
    .block-container, [data-testid="stMainBlockContainer"] {{ padding: 1.5rem 2.5rem 3rem; max-width: 1240px; }}

    /* Nav */
    .nav {{ display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0 1.4rem; }}
    .brand {{ display: flex; align-items: center; gap: 0.6rem; font-weight: 600; font-size: 0.95rem; color: var(--text); }}
    .mark {{
        width: 22px; height: 22px; border-radius: 6px; background: var(--text);
        display: grid; place-items: center;
    }}
    .mark::after {{ content: ""; width: 8px; height: 8px; border-radius: 50%; background: #09090B; }}
    .pill {{
        display: inline-flex; align-items: center; gap: 0.45rem; padding: 0.3rem 0.75rem;
        border: 1px solid var(--border); border-radius: 999px; font-size: 0.8rem; color: var(--muted);
    }}
    .pill .dot {{ width: 6px; height: 6px; border-radius: 50%; background: var(--tone); box-shadow: 0 0 0 3px color-mix(in srgb, var(--tone) 25%, transparent); }}

    /* Hero */
    .hero {{ padding: 2.2rem 0 2rem; }}
    .hero h1 {{
        font-family: var(--font); font-weight: 600; font-size: 2.9rem; line-height: 1.05;
        letter-spacing: -0.035em; color: var(--text) !important; margin: 0; padding: 0;
    }}
    .hero h1 .dim {{ color: var(--muted) !important; }}
    .hero [data-testid="stHeaderActionElements"] {{ display: none; }}
    .notice {{ display: inline-flex; align-items: center; gap: 0.6rem; margin-top: 1rem; color: var(--muted); font-size: 0.85rem; }}
    .notice span {{
        padding: 0.15rem 0.55rem; border-radius: 6px; border: 1px solid var(--border);
        color: var(--text); font-size: 0.75rem; font-weight: 500;
    }}

    /* Cards (keyed containers) */
    .st-key-inputs, .st-key-result, .st-key-chart, .st-key-activity {{
        background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
        padding: 1.4rem 1.5rem;
    }}
    .card-title {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.1rem; }}
    .card-title h3 {{
        font-family: var(--font); font-size: 0.95rem; font-weight: 500; color: var(--text); margin: 0; padding: 0;
        letter-spacing: -0.01em;
    }}
    .card-title small {{ color: var(--muted); font-size: 0.8rem; }}

    /* Inputs */
    .field {{ display: flex; justify-content: space-between; align-items: baseline; }}
    .field-name {{ color: var(--muted); font-size: 0.88rem; }}
    .field-value {{ font-family: var(--mono); color: var(--text); font-size: 0.9rem; font-variant-numeric: tabular-nums; }}
    .field-value span {{ color: var(--muted); }}
    [data-testid="stSliderThumbValue"], [data-testid="stSliderTickBar"] {{ display: none !important; }}
    [data-testid="stSlider"] {{ margin-bottom: 0.6rem; }}

    /* Result */
    .badge {{
        display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.25rem 0.65rem; border-radius: 999px;
        font-size: 0.8rem; font-weight: 500; color: var(--tone);
        background: color-mix(in srgb, var(--tone) 12%, transparent);
        border: 1px solid color-mix(in srgb, var(--tone) 30%, transparent);
    }}
    .badge::before {{ content: ""; width: 6px; height: 6px; border-radius: 50%; background: var(--tone); }}
    .big {{
        font-size: 3.6rem; font-weight: 600; letter-spacing: -0.045em; line-height: 1; color: var(--text);
        font-variant-numeric: tabular-nums;
    }}
    .big-caption {{ color: var(--muted); font-size: 0.88rem; margin-top: 0.4rem; }}
    .meter {{ height: 6px; border-radius: 999px; background: var(--border); margin: 1.2rem 0 1.3rem; overflow: hidden; }}
    .meter > div {{ height: 100%; border-radius: 999px; background: var(--tone); }}
    .stats {{ display: grid; grid-template-columns: repeat(2, 1fr); border-top: 1px solid var(--border); }}
    .stats > div {{ padding: 0.9rem 0 0.2rem; }}
    .stats > div + div {{ border-left: 1px solid var(--border); padding-left: 1rem; }}
    .stat-value {{ font-family: var(--mono); font-size: 1.05rem; color: var(--text); font-variant-numeric: tabular-nums; }}
    .stat-label {{ color: var(--muted); font-size: 0.78rem; margin-top: 0.15rem; }}
    .note {{ color: var(--muted); font-size: 0.88rem; line-height: 1.6; margin-top: 1.1rem; }}
    .note b {{ color: var(--text); font-weight: 500; }}
    .error {{ color: var(--fail); font-size: 0.9rem; line-height: 1.6; }}
    .error code {{ color: var(--text); background: var(--border); }}

    /* Chart legend */
    .legend {{ display: flex; gap: 1.2rem; flex-wrap: wrap; font-size: 0.8rem; color: var(--muted); }}
    .legend span {{ display: inline-flex; align-items: center; gap: 0.4rem; }}
    .legend i {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}

    /* Tabs */
    [data-baseweb="tab-list"] {{ gap: 0.3rem; border-bottom: 1px solid var(--border); }}
    [data-baseweb="tab"] {{ padding: 0.5rem 0.2rem; margin-right: 1rem; }}
    [data-baseweb="tab"] p {{ font-size: 0.88rem; font-weight: 500; }}
    [data-baseweb="tab-highlight"] {{ background: var(--text); height: 2px; }}
    [data-baseweb="tab-border"] {{ display: none; }}

    /* History table */
    .history {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; border: none; margin-top: 0.4rem; }}
    .history tr, .history th, .history td {{ border: none; background: transparent; }}
    .history th {{
        text-align: left; color: var(--muted); font-weight: 400; font-size: 0.8rem;
        padding: 0.7rem 0; border-bottom: 1px solid var(--border);
    }}
    .history td {{ padding: 0.7rem 0; border-bottom: 1px solid var(--border); color: var(--text); }}
    .history td.num {{ font-family: var(--mono); font-variant-numeric: tabular-nums; }}
    .history td .badge {{ font-size: 0.75rem; padding: 0.1rem 0.55rem; }}

    .footer {{
        margin-top: 2.5rem; display: flex; justify-content: space-between; gap: 1rem; flex-wrap: wrap;
        color: var(--muted); font-size: 0.8rem;
    }}
    .footer a {{ color: var(--muted); text-decoration: underline; text-underline-offset: 3px; }}

    @media (max-width: 760px) {{
        .block-container, [data-testid="stMainBlockContainer"] {{ padding: 1rem 1rem 2rem; }}
        .hero h1 {{ font-size: 2.1rem; }}
        .big {{ font-size: 2.8rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Helpers ---
@st.cache_resource
def load_local_model():
    # Only used to draw the chart and find the pass line; predictions come from the API
    try:
        return joblib.load("model.pkl")
    except FileNotFoundError:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def call_api(study_hours: float, prev_score: float):
    response = requests.post(
        f"{API_URL}/predict",
        json={"study_hours": study_hours, "prev_score": prev_score},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def html(markup: str):
    st.markdown(markup, unsafe_allow_html=True)


def card_title(title: str, aside: str = ""):
    html(f'<div class="card-title"><h3>{title}</h3><small>{aside}</small></div>')


def badge(label: str):
    tone = PASS_COLOR if label == "Pass" else FAIL_COLOR
    return f'<span class="badge" style="--tone:{tone}">{label}</span>'


def field(name: str, key: str, min_value: float, max_value: float, default: float, step: float, unit: str):
    # Custom label row with the live value, then a bare slider underneath
    value = st.session_state.get(key, default)
    html(
        f'<div class="field"><div class="field-name">{name}</div>'
        f'<div class="field-value">{value:g}<span> {unit}</span></div></div>'
    )
    return st.slider(name, min_value, max_value, default, step, key=key, label_visibility="collapsed")


def boundary_chart(model, study_hours: float, prev_score: float):
    w_hours, w_score = model.coef_[0]
    bias = model.intercept_[0]

    # The pass line: every point where the model is exactly 50/50
    hours = np.linspace(0, 10, 50)
    line = pd.DataFrame({"study_hours": hours, "boundary": np.clip(-(bias + w_hours * hours) / w_score, 0, 100)})
    line["top"], line["bottom"] = 100, 0

    x = alt.X("study_hours:Q", title="Study hours per day", scale=alt.Scale(domain=[0, 10], nice=False))
    y = alt.Y("prev_score:Q", title="Previous exam score", scale=alt.Scale(domain=[0, 100], nice=False))

    base = alt.Chart(line).encode(x=x)
    pass_zone = base.mark_area(color=PASS_COLOR, opacity=0.06).encode(y="boundary:Q", y2="top:Q")
    fail_zone = base.mark_area(color=FAIL_COLOR, opacity=0.06).encode(y="bottom:Q", y2="boundary:Q")
    boundary = base.mark_line(color=MUTED, strokeDash=[3, 3], strokeWidth=1).encode(y="boundary:Q")

    training_points = alt.Chart(TRAINING_DATA).mark_circle(size=55, opacity=0.9).encode(
        x=x, y=y,
        color=alt.Color("label:N", scale=alt.Scale(domain=["Pass", "Fail"], range=[PASS_COLOR, FAIL_COLOR]), legend=None),
        tooltip=[
            alt.Tooltip("study_hours", title="Study hours"),
            alt.Tooltip("prev_score", title="Previous score"),
            alt.Tooltip("label", title="Actual result"),
        ],
    )

    student = pd.DataFrame({"study_hours": [study_hours], "prev_score": [prev_score]})
    guide_x = alt.Chart(student).mark_rule(color=TEXT, opacity=0.18, strokeWidth=1).encode(x=x)
    guide_y = alt.Chart(student).mark_rule(color=TEXT, opacity=0.18, strokeWidth=1).encode(y=y)
    ring = alt.Chart(student).mark_point(size=380, color=TEXT, strokeWidth=1.2, opacity=0.5).encode(x=x, y=y)
    student_point = alt.Chart(student).mark_circle(size=90, color=TEXT, opacity=1).encode(
        x=x, y=y,
        tooltip=[alt.Tooltip("study_hours", title="Study hours"), alt.Tooltip("prev_score", title="Previous score")],
    )

    return (
        alt.layer(pass_zone, fail_zone, boundary, training_points, guide_x, guide_y, ring, student_point)
        .properties(height=400, background="transparent")
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=MUTED, titleColor=MUTED, labelFont="Geist Mono", titleFont="Geist",
            labelFontSize=11, titleFontSize=12, titleFontWeight=400, titlePadding=12,
            gridColor=BORDER, domain=False, ticks=False, labelPadding=8, tickCount=5,
        )
    )


def pass_line(model, study_hours: float):
    # The previous score where the model flips between Pass and Fail (probability is 50%)
    w_hours, w_score = model.coef_[0]
    bias = model.intercept_[0]
    return -(bias + w_hours * study_hours) / w_score


# --- Prediction ---
model = load_local_model()

# Slider values live in session_state, so we can predict before drawing the page
study_hours = st.session_state.get("study_hours", 4.0)
prev_score = st.session_state.get("prev_score", 60.0)

result, error = None, None
try:
    result = call_api(study_hours, prev_score)
except requests.exceptions.HTTPError as exc:
    error = f"The API returned an error ({exc.response.status_code})."
except requests.exceptions.RequestException:
    error = "API offline. Run <code>uvicorn main:app</code>."
api_online = error is None

html(
    f"""
    <div class="nav">
      <div class="brand"><div class="mark"></div>Examine</div>
      <div class="pill" style="--tone:{PASS_COLOR if api_online else FAIL_COLOR}">
        <span class="dot"></span>{'API connected' if api_online else 'API offline'}
      </div>
    </div>
    <div class="hero">
      <h1>Will they pass? <span class="dim">Find out.</span></h1>
      <div class="notice"><span>Demo</span>Trained on a small sample. Not a perfect model, so treat results as rough estimates.</div>
    </div>
    """
)

left, right = st.columns([1, 1.55], gap="medium")

with left:
    with st.container(key="inputs"):
        card_title("Student")
        study_hours = field("Study hours / day", "study_hours", 0.0, 10.0, 4.0, 0.5, "h")
        prev_score = field("Last exam score", "prev_score", 0.0, 100.0, 60.0, 1.0, "/ 100")

    with st.container(key="result"):
        if error:
            card_title("Prediction")
            html(f'<div class="error">{error}</div>')
        else:
            passed = result["prediction"] == 1
            tone = PASS_COLOR if passed else FAIL_COLOR
            confidence = result["confidence"] * 100
            pass_probability = confidence if passed else 100 - confidence

            card_title("Prediction", badge(result["result_label"]))

            line_score = f"{pass_line(model, study_hours):.0f}" if model is not None else "&ndash;"

            html(
                f"""
                <div style="--tone:{tone}">
                  <div class="big">{pass_probability:.1f}%</div>
                  <div class="big-caption">Chance of passing</div>
                  <div class="meter"><div style="width:{pass_probability:.1f}%"></div></div>
                  <div class="stats">
                    <div><div class="stat-value">{confidence:.1f}%</div><div class="stat-label">Confidence</div></div>
                    <div><div class="stat-value">{line_score}</div><div class="stat-label">Score needed</div></div>
                  </div>
                </div>
                """
            )

            # Remember this prediction (skip repeats of the same inputs)
            history = st.session_state.setdefault("history", [])
            entry = (study_hours, prev_score, result["result_label"], confidence)
            if not history or history[-1] != entry:
                history.append(entry)

with right:
    with st.container(key="chart"):
        card_title("Decision boundary")
        if model is not None:
            st.altair_chart(boundary_chart(model, study_hours, prev_score), width="stretch")
            html(
                f"""
                <div class="legend">
                  <span><i style="background:{TEXT}"></i>This student</span>
                  <span><i style="background:{PASS_COLOR}"></i>Passed</span>
                  <span><i style="background:{FAIL_COLOR}"></i>Failed</span>
                  <span>- - Pass line</span>
                </div>
                """
            )
        else:
            html('<div class="note">No model.pkl found. Run <code>python train_model.py</code>.</div>')

# --- History & developer view ---
st.write("")
with st.container(key="activity"):
    history_tab, dev_tab = st.tabs(["History", "API"])
    with history_tab:
        history = st.session_state.get("history", [])
        if history:
            rows = "".join(
                f'<tr><td class="num">{h:g} h</td><td class="num">{s:g}</td>'
                f'<td>{badge(label)}</td><td class="num">{c:.1f}%</td></tr>'
                for h, s, label, c in history[::-1][:8]
            )
            html(
                '<table class="history"><tr><th>Study hours</th><th>Previous score</th>'
                f"<th>Result</th><th>Confidence</th></tr>{rows}</table>"
            )
            if st.button("Clear history", type="tertiary"):
                st.session_state.history = []
                st.rerun()
        else:
            html('<div class="note">No predictions yet.</div>')
    with dev_tab:
        request_col, response_col = st.columns(2)
        with request_col:
            html('<div class="field-name">Request &middot; POST /predict</div>')
            st.json({"study_hours": study_hours, "prev_score": prev_score})
        with response_col:
            html('<div class="field-name">Response</div>')
            st.json(result if result else {"error": error})

html(
    f"""
    <div class="footer">
      <div>Examine</div>
      <div><a href="{API_URL}/docs">API docs</a></div>
    </div>
    """
)
