# Party Influence in French Electoral Manifestos

## Project Overview

This project analyzes whether political discourse in French electoral manifestos is primarily structured by **party ideology** or by **candidates' socio-professional characteristics**.

Using manifesto texts from the Archelec archive (Sciences Po / CEVIPOF), we apply **topic modeling (LDA)** to extract themes and quantify how they vary across parties, professions, and time.

This project is adapted from the original NLP work:  

<https://github.com/sarakpyny/NLP-Course-Ensae/tree/main/Project>

---

## MLOps Objective

The goal of this project is to transform a research-style NLP notebook into a **reproducible, modular, and deployable machine learning system**.

The system follows a full pipeline:

```bash
data → preprocessing → feature preparation → modeling → outputs → usage
```

---

## Project Structure

```text

## Project Structure

```text
PROJECT/
├── app/                    # FastAPI application (Phase 8)
├── deployment/             # Deployment-related files (future)
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
├── logs/                   # Log files (ignored by Git)
├── train.py                # Training pipeline entry point
├── pyproject.toml          # Dependencies (uv)
├── uv.lock                 # Locked environment
├── install.sh              # One-command setup
├── requirements.txt        # Legacy dependency list
└── README.md
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

The pipeline expects:

* **Metadata CSV** (e.g. `data/archelect_search.csv`)
* **Text files directory** (e.g. `data/text_files/`)

These must be provided locally and are ignored by Git.

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

## Outputs

Each run creates a structured output folder:

```text
outputs/<experiment_name>/
├── data_topics.csv
├── topics_summary.csv
├── lda_model.joblib
├── vectorizer.joblib
├── run_config.json
└── topic_labels.json
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

### Run the API

```bash
uv run uvicorn app.api:app --reload

```

### Access interactive

#### Local

```bash

http://127.0.0.1:8000/docs
```

#### Remote

```bash
https://<your-instance>/proxy/8000/docs
```

Example:

```bash

curl -X POST "<http://127.0.0.1:8000/predict_topics>" \
  -H "Content-Type: application/json" \
  -d '{"text": "Nous voulons défendre la justice sociale."}'

```

---

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

## Next Steps

* Add CI/CD workflows (`.github/workflows/`)
* Introduce experiment tracking (MLflow)
* Containerize with Docker (`deployment/`)

## Users

* Data science students
* Researchers in political science
* Instructors

---

## Notes

* Notebooks are for **exploration only**
* The pipeline is designed to be **fully reproducible from the command line**
* Data is not included in the repository
