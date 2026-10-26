import json
import os

import altair as alt
import gradio as gr
import joblib
import numpy as np
import pandas as pd
import uvicorn

from main import app as api  # FastAPI app: /predict, /health, /docs
from theme import BORDER, CSS_PATH, FAIL_COLOR, HEAD, MUTED, PASS_COLOR, TEXT, THEME

# Same rows the model was trained on (see train_model.py), shown on the chart
TRAINING_DATA = pd.DataFrame(
    {
        "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 1],
        "prev_score": [40, 45, 50, 60, 65, 70, 75, 85, 80, 30],
        "label": [
            "Fail",
            "Fail",
            "Fail",
            "Pass",
            "Pass",
            "Pass",
            "Pass",
            "Pass",
            "Pass",
            "Fail",
        ],
    }
)


# --- Model ---
def load_model():
    try:
        return joblib.load("model.pkl")
    except FileNotFoundError:
        return None


model = load_model()


def predict(study_hours: float, prev_score: float):
    # Same logic as POST /predict in main.py
    input_data = np.array([[study_hours, prev_score]])
    prediction = int(model.predict(input_data)[0])
    confidence = round(float(model.predict_proba(input_data)[0][prediction]), 3)
    return {
        "prediction": prediction,
        "result_label": "Pass" if prediction == 1 else "Fail",
        "confidence": confidence,
    }


def pass_line(study_hours: float):
    # The previous score where the model flips between Pass and Fail (probability is 50%)
    w_hours, w_score = model.coef_[0]
    bias = model.intercept_[0]
    return -(bias + w_hours * study_hours) / w_score


# --- HTML pieces ---
def card_title(title: str, aside: str = ""):
    return f'<div class="card-title"><h3>{title}</h3><small>{aside}</small></div>'


def badge(label: str):
    tone = PASS_COLOR if label == "Pass" else FAIL_COLOR
    return f'<span class="badge" style="--tone:{tone}">{label}</span>'


def field(name: str, value: float, unit: str):
    return (
        f'<div class="field"><div class="field-name">{name}</div>'
        f'<div class="field-value">{value:g}<span> {unit}</span></div></div>'
    )


def result_card(result, study_hours: float):
    if result is None:
        return (
            card_title("Prediction")
            + '<div class="note">No model.pkl found. Run <code>python train_model.py</code>.</div>'
        )

    passed = result["prediction"] == 1
    tone = PASS_COLOR if passed else FAIL_COLOR
    confidence = result["confidence"] * 100
    pass_probability = confidence if passed else 100 - confidence
    return card_title("Prediction", badge(result["result_label"])) + f"""
        <div style="--tone:{tone}">
          <div class="big">{pass_probability:.1f}%</div>
          <div class="big-caption">Chance of passing</div>
          <div class="meter"><div style="width:{pass_probability:.1f}%"></div></div>
          <div class="stats">
            <div><div class="stat-value">{confidence:.1f}%</div><div class="stat-label">Confidence</div></div>
            <div><div class="stat-value">{pass_line(study_hours):.0f}</div><div class="stat-label">Score needed</div></div>
          </div>
        </div>
    """


def history_table(history):
    if not history:
        return '<div class="note">No predictions yet.</div>'
    rows = "".join(
        f'<tr><td class="num">{h:g} h</td><td class="num">{s:g}</td>'
        f'<td>{badge(label)}</td><td class="num">{c:.1f}%</td></tr>'
        for h, s, label, c in history[::-1][:8]
    )
    return (
        '<table class="history"><tr><th>Study hours</th><th>Previous score</th>'
        f"<th>Result</th><th>Confidence</th></tr>{rows}</table>"
    )


def boundary_chart(study_hours: float, prev_score: float):
    w_hours, w_score = model.coef_[0]
    bias = model.intercept_[0]

    # The pass line: every point where the model is exactly 50/50
    hours = np.linspace(0, 10, 50)
    line = pd.DataFrame(
        {
            "study_hours": hours,
            "boundary": np.clip(-(bias + w_hours * hours) / w_score, 0, 100),
        }
    )
    line["top"], line["bottom"] = 100, 0

    x = alt.X(
        "study_hours:Q",
        title="Study hours per day",
        scale=alt.Scale(domain=[0, 10], nice=False),
    )
    y = alt.Y(
        "prev_score:Q",
        title="Previous exam score",
        scale=alt.Scale(domain=[0, 100], nice=False),
    )

    base = alt.Chart(line).encode(x=x)
    pass_zone = base.mark_area(color=PASS_COLOR, opacity=0.06).encode(
        y="boundary:Q", y2="top:Q"
    )
    fail_zone = base.mark_area(color=FAIL_COLOR, opacity=0.06).encode(
        y="bottom:Q", y2="boundary:Q"
    )
    boundary = base.mark_line(color=MUTED, strokeDash=[3, 3], strokeWidth=1).encode(
        y="boundary:Q"
    )

    training_points = (
        alt.Chart(TRAINING_DATA)
        .mark_circle(size=55, opacity=0.9)
        .encode(
            x=x,
            y=y,
            color=alt.Color(
                "label:N",
                scale=alt.Scale(
                    domain=["Pass", "Fail"], range=[PASS_COLOR, FAIL_COLOR]
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("study_hours", title="Study hours"),
                alt.Tooltip("prev_score", title="Previous score"),
                alt.Tooltip("label", title="Actual result"),
            ],
        )
    )

    student = pd.DataFrame({"study_hours": [study_hours], "prev_score": [prev_score]})
    guide_x = (
        alt.Chart(student)
        .mark_rule(color=TEXT, opacity=0.18, strokeWidth=1)
        .encode(x=x)
    )
    guide_y = (
        alt.Chart(student)
        .mark_rule(color=TEXT, opacity=0.18, strokeWidth=1)
        .encode(y=y)
    )
    ring = (
        alt.Chart(student)
        .mark_point(size=380, color=TEXT, strokeWidth=1.2, opacity=0.5)
        .encode(x=x, y=y)
    )
    student_point = (
        alt.Chart(student)
        .mark_circle(size=90, color=TEXT, opacity=1)
        .encode(
            x=x,
            y=y,
            tooltip=[
                alt.Tooltip("study_hours", title="Study hours"),
                alt.Tooltip("prev_score", title="Previous score"),
            ],
        )
    )

    return (
        alt.layer(
            pass_zone,
            fail_zone,
            boundary,
            training_points,
            guide_x,
            guide_y,
            ring,
            student_point,
        )
        .properties(width="container", height=400, background="transparent")
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=MUTED,
            titleColor=MUTED,
            labelFont="Geist Mono",
            titleFont="Geist",
            labelFontSize=11,
            titleFontSize=12,
            titleFontWeight=400,
            titlePadding=12,
            gridColor=BORDER,
            domain=False,
            ticks=False,
            labelPadding=8,
            tickCount=5,
        )
    )


# --- Event handlers ---
def update(study_hours: float, prev_score: float, history: list):
    result = predict(study_hours, prev_score) if model is not None else None
    chart = boundary_chart(study_hours, prev_score) if model is not None else None

    # Remember this prediction (skip repeats of the same inputs)
    if result is not None:
        entry = (
            study_hours,
            prev_score,
            result["result_label"],
            result["confidence"] * 100,
        )
        if not history or history[-1] != entry:
            history = history + [entry]

    request = {"study_hours": study_hours, "prev_score": prev_score}
    response = result if result else {"error": "Model not loaded"}
    return (
        field("Study hours / day", study_hours, "h"),
        field("Last exam score", prev_score, "/ 100"),
        result_card(result, study_hours),
        chart,
        history,
        history_table(history),
        json.dumps(request, indent=2),
        json.dumps(response, indent=2),
    )


def clear_history():
    return [], history_table([])


# --- Layout ---
model_ready = model is not None

with gr.Blocks(title="Examine", fill_width=True) as demo:
    history = gr.State([])

    gr.HTML(f"""
        <div class="nav">
          <div class="brand"><div class="mark"></div>Examine</div>
          <div class="pill" style="--tone:{PASS_COLOR if model_ready else FAIL_COLOR}">
            <span class="dot"></span>{'Model loaded' if model_ready else 'Model missing'}
          </div>
        </div>
        <div class="hero">
          <h1>Will they pass? <span class="dim">Find out.</span></h1>
          <div class="notice"><span>Demo</span>Trained on a small sample. Not a perfect model, so treat results as rough estimates.</div>
        </div>
        """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=100, min_width=300):
            with gr.Column(elem_classes="card"):
                gr.HTML(card_title("Student"))
                hours_label = gr.HTML()
                hours = gr.Slider(
                    0,
                    10,
                    value=4,
                    step=0.5,
                    show_label=False,
                    container=False,
                    elem_classes="bare-slider",
                )
                score_label = gr.HTML()
                score = gr.Slider(
                    0,
                    100,
                    value=60,
                    step=1,
                    show_label=False,
                    container=False,
                    elem_classes="bare-slider",
                )

            with gr.Column(elem_classes="card"):
                result_html = gr.HTML()

        with gr.Column(scale=155, min_width=340):
            with gr.Column(elem_classes="card"):
                gr.HTML(card_title("Decision boundary"))
                chart = gr.Plot(show_label=False, container=False, elem_classes="chart")
                gr.HTML(f"""
                    <div class="legend">
                      <span><i style="background:{TEXT}"></i>This student</span>
                      <span><i style="background:{PASS_COLOR}"></i>Passed</span>
                      <span><i style="background:{FAIL_COLOR}"></i>Failed</span>
                      <span>- - Pass line</span>
                    </div>
                    """)

    # --- History & developer view ---
    with gr.Column(elem_classes=["card", "activity"]):
        with gr.Tabs():
            with gr.Tab("History"):
                history_html = gr.HTML()
                clear = gr.Button("Clear history", size="sm", elem_classes="clear-btn")
            with gr.Tab("API"):
                with gr.Row():
                    with gr.Column():
                        gr.HTML(
                            '<div class="field-name">Request &middot; POST /predict</div>'
                        )
                        request_code = gr.Code(
                            language="json", show_label=False, interactive=False
                        )
                    with gr.Column():
                        gr.HTML('<div class="field-name">Response</div>')
                        response_code = gr.Code(
                            language="json", show_label=False, interactive=False
                        )

    gr.HTML("""
        <div class="footer">
          <div>Examine</div>
          <div><a href="/docs" target="_blank">API docs</a></div>
        </div>
        """)

    outputs = [
        hours_label,
        score_label,
        result_html,
        chart,
        history,
        history_html,
        request_code,
        response_code,
    ]
    # always_last: while sliding, skip stale updates but always run the final value
    gr.on(
        [demo.load, hours.change, score.change],
        update,
        inputs=[hours, score, history],
        outputs=outputs,
        trigger_mode="always_last",
    )
    clear.click(clear_history, outputs=[history, history_html])


# Serve the Gradio UI at / alongside the FastAPI routes (/predict, /health, /docs)
app = gr.mount_gradio_app(
    api, demo, path="/", theme=THEME, css_paths=CSS_PATH, head=HEAD, footer_links=[]
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
