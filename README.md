# Examine

> A demo project. The model is trained on a tiny hand-made dataset, so treat the predictions as rough estimates, not real advice.

A small ML app that predicts whether a student will pass, based on how many hours they study per day and their last exam score. Move two sliders and it tells you the chance of passing, how confident the model is, and what score you'd need to be on the safe side.

The model itself is tiny on purpose. The point was to build the full path end to end: train a model, save it, serve it through an API, put a UI on top, and deploy it.

## Live Demo
https://examine.up.railway.app/

API docs: https://examine.up.railway.app/docs

## Core Features
- Logistic regression model trained with scikit-learn, saved to `model.pkl` with joblib so the app never retrains on startup
- FastAPI backend with a `POST /predict` endpoint, inputs validated with Pydantic before they ever reach the model
- Live predictions, every slider move updates the result, no submit button
- Decision boundary chart that shows the training data, the pass and fail zones, and where the current student sits
- One server for everything, the Gradio UI is mounted on the same FastAPI app, so `/`, `/predict` and `/docs` all run from one process

## How It Works

### Training
`train_model.py` takes 10 hand-made rows of `[study_hours, prev_score]` with a Pass or Fail label, fits a `LogisticRegression`, and saves it to `model.pkl`.

### Prediction
The model outputs a probability for each class. The app picks the more likely one (Pass or Fail) and reports its probability as the confidence.

## Tech Stack
- Python
- scikit-learn + joblib (model training and saving)
- FastAPI + Pydantic (API and validation)
- Gradio (UI)
- Altair (decision boundary chart)
- NumPy + pandas
- Deployed on Railway

I wanted one project where I actually took a model all the way from a training script to a live URL, instead of stopping at a notebook.
