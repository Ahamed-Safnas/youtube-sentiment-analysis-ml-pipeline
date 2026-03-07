# 🎯 YouTube Comment Sentiment Analysis — MLOps Pipeline

> End-to-end MLOps pipeline for YouTube comment sentiment analysis — LightGBM + TF-IDF, DVC-orchestrated stages, MLflow experiment tracking on AWS EC2, S3 artifact storage, and a Flask REST API with live sentiment charts and word cloud generation.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Pipeline Stages](#pipeline-stages)
- [Experiment Tracking](#experiment-tracking)
- [Model Registry](#model-registry)
- [Flask API](#flask-api)
- [API Endpoints](#api-endpoints)

---

## Overview

This project implements a **production-style ML pipeline** that classifies YouTube comments into three sentiment categories:

| Label | Sentiment |
|-------|-----------|
| `1`   | Positive  |
| `0`   | Neutral   |
| `-1`  | Negative  |

The pipeline is fully reproducible via **DVC**, experiments are tracked on **MLflow hosted on AWS EC2**, artifacts are stored in **AWS S3**, and the final model is served through a **Flask REST API** that supports batch prediction, sentiment charts, word clouds, and trend graphs over time.

---

## Architecture

```
Raw Data (CSV)
     │
     ▼
┌─────────────────┐
│  Data Ingestion  │  ← Train/test split (params.yaml)
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│  Data Preprocessing   │  ← Lowercase, lemmatize, stopword removal
└────────┬─────────────┘
         │
         ▼
┌──────────────────┐
│  Model Building   │  ← TF-IDF (1-3 ngrams) + LightGBM
└────────┬─────────┘
         │
         ▼
┌───────────────────┐
│  Model Evaluation  │  ← Accuracy, classification report, confusion matrix
└────────┬──────────┘
         │
         ▼
┌────────────────────────┐
│  Model Registration     │  ← MLflow Model Registry (AWS EC2 + S3)
└────────┬───────────────┘
         │
         ▼
┌──────────────────┐
│   Flask REST API  │  ← /predict, /generate_chart, /generate_wordcloud
└──────────────────┘
```

---

## Tech Stack

| Category | Tools |
|---|---|
| **ML / Modeling** | LightGBM, scikit-learn, TF-IDF |
| **NLP** | NLTK (lemmatization, stopwords) |
| **Pipeline Orchestration** | DVC |
| **Experiment Tracking** | MLflow |
| **Cloud Infrastructure** | AWS EC2, AWS S3 |
| **API Serving** | Flask, Flask-CORS |
| **Visualization** | Matplotlib, WordCloud, Seaborn |
| **Environment** | Python 3.11, Conda |

---

## Project Structure

```
youtube-sentiment-mlops-pipeline/
│
├── src/
│   ├── data/
│   │   ├── data_ingestion.py        # Load and split raw data
│   │   └── data_preprocessing.py   # Clean and lemmatize comments
│   └── model/
│       ├── model_building.py        # TF-IDF + LightGBM training
│       ├── model_evaluation.py      # Metrics and MLflow logging
│       └── register_model.py        # Push best model to MLflow registry
│
├── flask-api/
│   └── main.py                      # REST API with prediction & visualization endpoints
│
├── notebook/
│   └── 2_experiment_1_baseline_model.ipynb   # Baseline RF experiment
│
├── data/
│   ├── raw/                         # train.csv, test.csv (DVC tracked)
│   └── interim/                     # Preprocessed CSVs (DVC tracked)
│
├── dvc.yaml                         # Pipeline stage definitions
├── dvc.lock                         # Reproducibility lock file
├── params.yaml                      # Tunable hyperparameters
├── requirements.txt
├── setup.py
└── README.md
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Ahamed-Safnas/youtube-sentiment-mlops-pipeline.git
cd youtube-sentiment-mlops-pipeline
```

### 2. Create and Activate the Conda Environment

```bash
conda create -n youtube python=3.11 -y
conda activate youtube
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> This also installs the `src/` folder as a local package via `-e .` in `requirements.txt`.

### 4. Configure AWS Credentials

```bash
aws configure
```

You'll need an IAM user with access to the S3 bucket used for MLflow artifact storage.

---

## Pipeline Stages

The full pipeline is defined in `dvc.yaml` and orchestrated by DVC across five stages:

```
data_ingestion → data_preprocessing → model_building → model_evaluation → model_registration
```

### Run the Full Pipeline

```bash
dvc repro
```

DVC will only re-execute stages where inputs or code have changed.

### Visualize the DAG

```bash
dvc dag
```

### Hyperparameters

All tunable parameters live in `params.yaml`:

```yaml
data_ingestion:
  test_size: 0.20

model_building:
  ngram_range: [1, 3]
  max_features: 1000
  learning_rate: 0.09
  max_depth: 20
  n_estimators: 367
```

---

## Experiment Tracking

MLflow is hosted on an **AWS EC2 instance** with artifacts stored in **S3**.

### Connect to the MLflow Tracking Server

```python
import mlflow
mlflow.set_tracking_uri("http://<your-ec2-public-dns>:5000/")
```

All training runs log the following automatically:
- Hyperparameters (learning rate, max depth, n_estimators, etc.)
- Evaluation metrics (accuracy, F1 per class)
- Confusion matrix as a plot artifact
- The trained model artifact

---

## Model Registry

After evaluation, the best model is automatically pushed to the **MLflow Model Registry** via `register_model.py`. The registered model is versioned and can be promoted through stages (`Staging` → `Production`).

The Flask API loads the production model directly from the registry at startup:

```python
mlflow.set_tracking_uri("http://<your-ec2-public-dns>:5000/")
model = mlflow.sklearn.load_model(source)  # Loads from S3 via MLflow
```

---

## Flask API

### Start the Server

```bash
cd flask-api
python main.py
```

The server runs on `http://localhost:5005` by default.

---

## API Endpoints

### `POST /predict`

Classify a batch of raw comments.

**Request:**
```json
{
  "comments": [
    "This video is awesome! I loved it a lot",
    "Very bad explanation. Poor video"
  ]
}
```

**Response:**
```json
[
  {"comment": "This video is awesome! I loved it a lot", "sentiment": 1},
  {"comment": "Very bad explanation. Poor video", "sentiment": -1}
]
```

---

### `POST /predict_with_timestamps`

Classify comments and return results with their original timestamps (used for trend analysis).

**Request:**
```json
{
  "comments": [
    {"text": "Great content!", "timestamp": "2024-03-15T10:30:00"}
  ]
}
```

---

### `POST /generate_chart`

Returns a **PNG pie chart** of sentiment distribution.

**Request:**
```json
{
  "sentiment_counts": {"1": 120, "0": 45, "-1": 35}
}
```

---

### `POST /generate_wordcloud`

Returns a **PNG word cloud** generated from the provided comments.

**Request:**
```json
{
  "comments": ["great video", "loved the explanation", "very helpful"]
}
```

---

### `POST /generate_trend_graph`

Returns a **PNG line chart** of monthly sentiment percentages over time.

**Request:**
```json
{
  "sentiment_data": [
    {"text": "Great video!", "sentiment": "1", "timestamp": "2024-01-10T08:00:00"},
    {"text": "Not helpful", "sentiment": "-1", "timestamp": "2024-02-05T12:00:00"}
  ]
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">Built by <a href="https://github.com/Ahamed-Safnas">Ahamed Safnas</a></p>
