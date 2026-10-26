import inspect
from pathlib import Path

import gradio as gr

# Palette (the same colours are used in style.css)
BG = "#09090B"
TEXT = "#FAFAFA"
MUTED = "#8E8E96"
BORDER = "#232327"
SURFACE = "#111113"
PASS_COLOR = "#3DDC97"
FAIL_COLOR = "#FF6B6B"

CSS_PATH = Path(__file__).parent / "style.css"

# Geist fonts from Google Fonts
HEAD = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700'
    '&family=Geist+Mono:wght@400;500&display=swap" rel="stylesheet">'
)


def dark(theme, **values):
    # Set each theme variable for both light and dark mode, so the page is always dark
    params = inspect.signature(theme.set).parameters
    return theme.set(**values, **{f"{k}_dark": v for k, v in values.items() if f"{k}_dark" in params})


# Colours for Gradio's own components (sliders, tabs, code boxes)
THEME = dark(
    gr.themes.Base(font=["Geist", "system-ui", "sans-serif"], font_mono=["Geist Mono", "ui-monospace", "monospace"]),
    body_background_fill=BG,
    body_text_color=TEXT,
    body_text_color_subdued=MUTED,
    background_fill_primary=SURFACE,
    background_fill_secondary=SURFACE,
    block_background_fill=SURFACE,
    block_border_color=BORDER,
    border_color_primary=BORDER,
    color_accent=TEXT,
    color_accent_soft=BORDER,
    slider_color=TEXT,
    code_background_fill=BG,
    button_secondary_background_fill=SURFACE,
    button_secondary_text_color=MUTED,
)
