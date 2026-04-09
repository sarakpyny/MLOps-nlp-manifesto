# Party Influence in French Electoral Manifestos

[![CI Tests](https://github.com/sarakpyny/MLOps-nlp-manifesto/actions/workflows/test.yml/badge.svg)](https://github.com/sarakpyny/MLOps-nlp-manifesto/actions/workflows/test.yml)
[![Docker Image](https://github.com/sarakpyny/MLOps-nlp-manifesto/actions/workflows/docker.yml/badge.svg)](https://github.com/sarakpyny/MLOps-nlp-manifesto/actions/workflows/docker.yml)
[![Docker Hub](https://img.shields.io/docker/pulls/nysarakpy/mlops-manifestos-api)](https://hub.docker.com/r/nysarakpy/mlops-manifestos-api)

## Project Overview

This project analyzes whether political discourse in French electoral manifestos is primarily structured by **party ideology** or by **candidates' socio-professional characteristics**.

Using manifesto texts from the Archelec archive (Sciences Po / CEVIPOF), the project applies **topic modeling (LDA)** to extract themes and quantify how they vary across parties, professions, and time.

This project is adapted from the original NLP work:

<https://github.com/sarakpyny/NLP-Course-Ensae/tree/main/Project>

---

## MLOps Objective

The goal of this project is to transform a research-style NLP notebook into a **reproducible, modular, and deployable machine learning system**.

The system follows a full pipeline:

```text
data → preprocessing → feature preparation → modeling → outputs → usage
```

## System Architecture

The project follows an end-to-end MLOps pipeline:

```text
External Parquet data (S3)
        ↓
training pipeline (train.py)
        ↓
text preprocessing + feature preparation
        ↓
LDA topic model + cross-validation
        ↓
saved artifacts + MLflow tracking
        ↓
registered production model
        ↓
FastAPI service
        ↓
Kubernetes deployment via ArgoCD
        ↓
API consumers + documentation website

```

### Architecture notes

* `train.py` handles training, model selection, artifact generation, and MLflow registration.
* `app/api.py` serves predictions through FastAPI.
* `/predict_topics` uses the registered MLflow production model.
* Analytical endpoints such as `/topics` and `/stats` use saved reference artifacts from a baseline experiment.
* The API is containerized with Docker, published through GitHub Actions, and deployed on SSP Cloud through a separate GitOps repository synchronized by ArgoCD.
* A Quarto website provides a presentation layer for non-technical users.

---

## Project Structure

```text
MLOps-nlp-manifesto/
├── app/                    # FastAPI application
├── deployment/             # Kubernetes deployment manifests
├── docs/                   # Rendered Quarto website for GitHub Pages
├── src/
│   ├── data/               # Data loading and output saving
│   ├── preprocessing/      # Text and metadata cleaning
│   ├── features/           # Dataset preparation and filtering
│   ├── models/             # Topic modeling logic
│   ├── analysis/           # Analytical modules
│   ├── inference/          # Load trained artifacts and predict
│   └── utils/              # Logging and shared utilities
├── tests/                  # Unit and API tests
├── website/                # Quarto website source files
├── .github/workflows/      # GitHub Actions CI/CD workflows
├── logs/                   # Log files (ignored by Git)
├── outputs/                # Saved experiment artifacts (ignored by Git)
├── Dockerfile              # Container definition
├── train.py                # Training pipeline entry point
├── pyproject.toml          # Dependencies managed with uv
├── uv.lock                 # Locked environment
├── install.sh              # One-command local setup
├── README.md
└── .env.example            # Example configuration
```

---

## Installation

### Recommended setup

Clone the repository and rebuild the environment:

```bash
git clone https://github.com/sarakpyny/MLOps-nlp-manifesto.git
cd MLOps-nlp-manifesto

bash install.sh
source .venv/bin/activate
cp .env.example .env
```

### Manual setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv sync
source .venv/bin/activate

uv run python -m spacy download fr_core_news_md
uv run python -c "import nltk; nltk.download('stopwords')"

cp .env.example .env
```

---

## Reproducibility

This project is designed to run on another machine without hidden dependencies.

The repository includes:

* `pyproject.toml` for explicit dependency declaration
* `uv.lock` for reproducible environments
* `install.sh` for automatic setup
* `.env.example` for configuration transparency

The following are intentionally **not versioned**:

* raw data (`data/`)
* outputs (`outputs/`)
* model artifacts (`*.joblib`)
* logs (`logs/`)
* secrets (`.env`)

---

## Data

The training pipeline reads a merged Parquet dataset from external object storage S3:

```text
https://minio.lab.sspcloud.fr/sny/mlops-manifesto/processed/manifestos_raw.parquet
```

This URL is provided through the environment variable `URL_RAW`.

---

## Usage

### Run baseline (no lemmatization)

```bash
uv run python train.py \
  --n-topics 8 \
  --min-doc-length 100 \
  --start-year 1981 \
  --end-year 1993 \
  --random-seed 42 \
  --output-dir outputs \
  --experiment-name baseline_lda
```

### Run with lemmatization

```bash
uv run python train.py \
  --n-topics 8 \
  --min-doc-length 100 \
  --start-year 1981 \
  --end-year 1993 \
  --random-seed 42 \
  --output-dir outputs \
  --experiment-name baseline_lda_lemma \
  --use-lemmatization
```

### Run model selection with cross-validation

```bash
uv run python train.py \
  --experiment-name lda_cv_search \
  --cv-folds 3 \
  --candidate-n-topics 6 8 10 \
  --candidate-min-df 10 20 \
  --candidate-max-df 0.90 0.95
```

---

## Outputs

Each run creates a structured output folder:

```text
outputs/<experiment_name>/
├── data_topics.csv
├── topics_summary.csv
├── lda_model.joblib
├── vectorizer.joblib
├── run_config.json
├── topic_labels.json
├── cv_results.csv
├── cv_summary.csv
└── best_params.json
```

---

## Inference

```bash
from pathlib import Path
from src.inference.predictor import predict_topics

result = predict_topics(
    text="Nous défendons la justice sociale et les services publics.",
    experiment_dir=Path("outputs/baseline_lda"),
)

print(result)
```

---

## API

### Run locally

```bash
uv run uvicorn app.api:app --reload
```

Local docs:

```text
http://127.0.0.1:8000/docs
```

Live deployed docs:

```text
https://manifesto-api-sny.lab.sspcloud.fr/docs
```

---

## Serving design

The API uses a hybrid serving design:

* `/predict_topics` serves the registered MLflow production model
* `/topics`, `/stats`, `/party_profile/{party}`, and `/profession_profile/{profession}` rely on saved analytical artifacts from a reference experiment directory

Check:

```text
https://manifesto-api-sny.lab.sspcloud.fr/health
```

---

## Presentation Layer

A public-facing documentation website built with Quarto and deployed on GitHub Pages.
The website includes:

* project overview
* dataset description
* method summary
* API usage
* architecture page
* results summary

Live site:

```text
https://sarakpyny.github.io/MLOps-nlp-manifesto/
```

---

## Deployment

The FastAPI application is deployed on SSP Cloud with Kubernetes and synchronized through ArgoCD from a separate GitOps repository `https://github.com/sarakpyny/manifesto-application-deployment.git`.

Live endpoints:

```text
https://manifesto-api-sny.lab.sspcloud.fr/docs
```

The deployed API currently serves:

* a registry-backed prediction endpoint through MLflow
* analysis endpoints backed by saved experiment artifacts

---

## Docker

### Build locally

```bash
docker build \
  --build-arg URL_RAW="https://minio.lab.sspcloud.fr/sny/mlops-manifesto/processed/manifestos_raw.parquet" \
  -t mlops-manifestos-api .
```

### Run locally

```bash
docker run -p 8000:8000 mlops-manifestos-api
```

### Swagger UI

```text
http://127.0.0.1:8000/docs
```

### Pull published image

```bash
docker pull nysarakpy/mlops-manifestos-api:v0.3.0
```

---

## CI/CD

This project includes two GitHub Actions workflows:

* `test.yml` Recreates the environment, installs required NLP resources, builds test artifacts, runs Ruff, PyLint, and pytest.
* `docker.yml` Builds and publishes the Docker image to Docker Hub.

Required GitHub secrets

* `URL_RAW`
* `DOCKERHUB_USERNAME`
* `DOCKERHUB_TOKEN`

---

## Testing

Run the full test suite:

```bash
uv run pytest -v
```

---

## MLOps Phases

### Phase 0 — Initialization

Initialized the project from the original NLP notebook repository, preserved `main` as baseline, and used `mlops` as the development branch.

### Phase 1 — Product Definition

Defined the problem, users, inputs, and outputs, and reframed the work as a complete ML system.

### Phase 2 — Baseline Pipeline

Created `train.py` to run the end-to-end training pipeline from the command line.

### Phase 3 — Parameterization

Added CLI arguments and `.env` configuration to make experiments reproducible and configurable.

### Phase 4 — Modular Structure

Refactored the notebook logic into `src/` modules and kept `train.py` as the orchestration entry point only.

### Phase 5 — Reproducibility

Added `pyproject.toml`, `uv.lock`, `install.sh`, `.env.example`, and a stronger `.gitignore` for reproducible setup across machines.

### Phase 5bis — Externalized Data Layer

Moved the training input to an external Parquet dataset so the pipeline no longer depends on local raw files.

### Phase 6 — Logging & Tests

Added structured logging and unit tests to improve traceability and reliability before CI integration.

### Phase 7 — Separate Training and Inference

Separated training from inference, added reusable prediction code in `src/inference/`, and introduced human-readable topic labels.

### Phase 8 — API

Built a FastAPI application with prediction and analysis endpoints, API tests, and Swagger documentation.

### Phase 9 — Containerization

Packaged the application in Docker so it can run consistently across environments.

### Phase 10 — CI/CD Automation

Added GitHub Actions workflows for testing, linting, and Docker image publication.

### Phase 11 — Deployment

Deployed the FastAPI application on SSP Cloud and exposed the live API and Swagger documentation.

### Phase 12 — Presentation Layer

Added a Quarto website published on GitHub Pages to present the project, method, architecture, and API.

### Phase 13 — Advanced MLOps

Added cross-validation-based model selection, MLflow experiment tracking, model registry integration, and registry-based serving for prediction.

---

## Current Status

The project is now implemented as an end-to-end MLOps pipeline.

* reproducible Python environment with `uv`, `pyproject.toml`, and `uv.lock`
* modular training and inference code under `src/`
* automated quality checks and tests with GitHub Actions
* Docker image publishing to Docker Hub
* Kubernetes deployment on SSP Cloud through ArgoCD and a separate GitOps repository
* public FastAPI documentation exposed through `/docs`
* Quarto website published on GitHub Pages
* MLflow experiment tracking and registry-based prediction serving

The deployed API currently serves:

* `/predict_topics` from the registered MLflow production model
* analytical endpoints from saved experiment artifacts

---

## Future Improvements

Possible extensions:

* moving analytical endpoints to fully registry-based or external artifact serving
* connecting the deployment to a shared remote MLflow tracking server instead of image-local storage
* enriching the website with more visualizations and model comparison results

---

## Users

* Data science students
* Researchers in political science
* Instructors

---

## Notes

* Notebooks are for **exploration only**
* The pipeline is designed to be **fully reproducible from the command line**
* Data is not included in the repository
