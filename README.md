---
title: Examine
emoji: 🎓
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# Examine

Predicts whether a student will pass from their daily study hours and previous exam score, using a small logistic regression model.

- `/`: Gradio interface
- `/predict`: JSON API (`POST {"study_hours": 4, "prev_score": 60}`)
- `/docs`: interactive API docs

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:7860.
