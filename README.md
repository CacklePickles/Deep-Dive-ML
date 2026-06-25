# Deep Dive ML — Phishing Detection

Web app for detecting phishing in email text and URLs using an ensemble of ML models (Logistic Regression, Naive Bayes, Random Forest, SVM, XGBoost).

## Features

- Multi-model ensemble with weighted voting
- Web UI for text/URL analysis
- User feedback loop that adjusts model weights
- SQLite storage for analysis history (see [DATABASE.md](DATABASE.md))

## Requirements

- Python 3.10+

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python backend/main.py
```

Open http://localhost:5000 in your browser.

## Project structure

```
backend/          Flask API and prediction logic
frontend/         Web UI
models/           Trained model files (.pkl)
preprocessing/    TF-IDF vectorizer and URL scaler
feedback_data/    Model weights and stats from user feedback
data/             SQLite database (created at runtime)
```
