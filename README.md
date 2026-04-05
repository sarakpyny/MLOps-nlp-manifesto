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

---

## Project Structure

```text
MLOps-nlp-manifesto/
├── app/                    # FastAPI application
├── deployment/             # Kubernetes deployment manifests
├── docs/                   # Rendered Quarto website for GitHub Pages
├── notebooks/              # Exploration only (not used in production)
├── src/
│   ├── data/               # Data loading and output saving
│   ├── preprocessing/      # Text and metadata cleaning
│   ├── features/           # Dataset preparation and filtering
│   ├── models/             # Topic modeling logic
│   ├── analysis/           # Analytical modules
│   ├── inference/          # Load trained artifacts and predict
│   └── utils/              # Logging and shared utilities
├── tests/                  # Unit and API tests
├── website/                # website presented layers
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

```bash
https://minio.lab.sspcloud.fr/sny/mlops-manifesto/processed/manifestos_raw.parquet

```

This URL is provided through the environment variable URL_RAW.

---

## Usage

### Run baseline (fast, no lemmatization)

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

## API

### Run API locally

```bash
uv run uvicorn app.api:app --reload

```

### Documentation

#### Local

```bash
http://127.0.0.1:8000/docs
```

#### Live deployed API

```bahs
https://manifesto-api-sny.lab.sspcloud.fr/docs
```

Example:

```bash

curl -X POST "<http://127.0.0.1:8000/predict_topics>" \
  -H "Content-Type: application/json" \
  -d '{"text": "Nous voulons défendre la justice sociale."}'

```

### API design note

The API currently uses a hybrid serving design:

* `/predict_topics` loads the prediction model from the MLflow registry
* `/topics, /stats, /party_profile/{party}, and /profession_profile/{profession}` still rely on local saved analytical files from an experiment directory

## Presentation Layer

Phase 12 adds a public-facing documentation website built with Quarto and deployed on GitHub Pages.
The website includes:

* project overview
* dataset description
* method summary
* API usage
* architecture page

results summary

Live site:

```bash
https://sarakpyny.github.io/MLOps-nlp-manifesto/
```

---

## Deployment

The FastAPI application is deployed on SSP Cloud with a Kubernetes-based setup managed through ArgoCD.

Live endpoints

```bash
Swagger UI: https://manifesto-api-sny.lab.sspcloud.fr/docs

```

## Docker

### Build locally

```bash
docker build \
  --build-arg URL_RAW="https://minio.lab.sspcloud.fr/sny/mlops-manifesto/processed/manifestos_raw.parquet" \
  -t mlops-manifestos-api .

```

Run local image:

```bash
docker run -p 8000:8000 mlops-manifestos-api
```

Pull published image:

```bash
docker pull nysarakpy/mlops-manifestos-api:latest
```

Swagger UI:

```bash
http://127.0.0.1:8000/docs

```

## CI/CD

This project includes two GitHub Actions workflows:

* `test.yml` Recreates the environment, installs required NLP resources, builds test artifacts, runs Ruff, PyLint, and pytest.
* `docker.yml` Builds and publishes the Docker image to Docker Hub.

Required GitHub secrets

* `URL_RAW`
* `DOCKERHUB_USERNAME`
* `DOCKERHUB_TOKEN`

## Testing

Run the full test suite:

```bash
uv run pytest -v

```

---

## MLOps Phases

### Phase 0 — Initialization

* Project initialized from an NLP notebook-based repository
* `main` branch preserved as baseline
* `mlops` branch used for development

### Phase 1 — Product Definition

* Defined problem, inputs, outputs, and users
* Framed as a full ML system

### Phase 2 — Baseline Pipeline

* Created `train.py` script
* End-to-end execution from terminal

### Phase 3 — Parameterization

* Added CLI arguments with `argparse`
* Introduced `.env` configuration
* Improved reproducibility and logging

### Phase 4 — Modular Structure

* Refactored code into `src/` modules
* Separated data, preprocessing, features, modeling, and outputs
* `train.py` handles orchestration only

### Phase 5 — Reproducibility

* Added `pyproject.toml` for explicit dependency management
* Added `uv.lock` for locked reproducible environments
* Added `install.sh` for one-command setup
* Added `.env.example` for transparent configuration
* Strengthened `.gitignore` to exclude data, outputs, models, logs, and secrets
* Updated installation instructions for portability

### Phase 5bis — Externalized Data Layer

* Exported the merged dataset to CSV and Parquet
* Updated training and feature preparation to rely on the external Parquet dataset instead of local raw files

---

## Phase 6 — Logging & Tests

* Added structured logging to console and file
* Replaced print-based pipeline traces with logging
* Added 11 unit tests covering core pipeline components
* Improved reliability and traceability before CI

## Phase 7 — Separate Training and Inference

* Kept `train.py` dedicated to model training and artifact generation
* Implemented `src/inference/` to load saved artifacts and predict on new text
* Added single-text preprocessing for inference
* Introduced `topic_labels.json` for human-readable topic outputs
* Added inference tests for:
  * invalid input
  * output structure
  * probability validity

## Phase 8 — API

* Built FastAPI app (app/api.py)
* Added endpoints for prediction and analysis
* Added API tests (TestClient)
* Enabled Swagger UI

## Phase 9 — Containerization

* Added a Dockerfile
* Built a runnable image for the FastAPI application
* Packaged dependencies, source code, and saved artifacts in the container

## Phase 10 — CI/CD Automation

* Added GitHub Actions workflows under `.github/workflows/`
* Automated Docker image build and publication to Docker Hub
* Made runtime dependencies explicit in CI and Docker builds

## Phase 11 — Deployment

* Deployed the FastAPI application on SSP Cloud
* Exposed a live API URL
* Made Swagger documentation reachable online at /docs

## Phase 12 — Presentation Layer

* Added a Quarto documentation website
* Published the site with GitHub Pages
* Structured the presentation for non-technical users
* Included project overview, dataset, method, API usage, architecture, and results summary

## Phase 13 — Advanced MLOps

* Added configurable model selection with cross-validation in train.py
* Introduced candidate hyperparameter search for LDA experiments
* Registered the prediction model in the MLflow Model Registry
* Updated the prediction API endpoint to load the model from the registry

## Next Steps

Promote registered versions to Staging or Production
Extend registry-based serving to more API endpoints
Add automated retraining workflows
Enrich the website with additional result visualizations

## Users

* Data science students
* Researchers in political science
* Instructors

---

## Notes

* Notebooks are for **exploration only**
* The pipeline is designed to be **fully reproducible from the command line**
* Data is not included in the repository
