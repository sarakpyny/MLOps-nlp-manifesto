# Party Influence in French Electoral Manifestos

## Project Overview

This project analyzes whether political discourse in French electoral manifestos is primarily structured by **party ideology** or by **candidates' socio-professional characteristics**.

Using manifesto texts from the Archelec archive (Sciences Po / CEVIPOF), we apply **topic modeling (LDA)** to extract themes and quantify how they vary across parties, professions, and time.

---

## MLOps Objective

The goal of this project is to transform a research-style NLP notebook into a **reproducible, modular, and deployable machine learning system**.

The system follows a full pipeline:

```
data → preprocessing → feature preparation → modeling → outputs → usage
```

---

## Project Structure

```text
PROJECT/
├── app/                    # Future API or application layer
├── deployment/             # Deployment-related files
├── notebooks/              # Exploration only (not used in production)
├── src/
│   ├── data/               # Data loading and output saving
│   ├── preprocessing/      # Text and metadata cleaning
│   ├── features/           # Dataset preparation and filtering
│   ├── models/             # Topic modeling logic
│   ├── analysis/           # Analytical modules (future)
│   ├── inference/          # Inference pipeline (future)
│   └── utils/              # CLI config and shared utilities
├── tests/                  # Unit tests (future)
├── train.py                # Main pipeline entry point
├── requirements.txt
└── README.md
```

---

## Installation

Create and activate your environment, then install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download fr_core_news_md
python -c "import nltk; nltk.download('stopwords')"
cp .env.example .env
```

---

## Data

The pipeline expects:

* **Metadata CSV** (e.g. `data/archelect_search.csv`)
* **Text files directory** (e.g. `data/text_files/`)

These are ignored in Git (`.gitignore`) and must be provided locally.

---

## Usage

The training pipeline is executed from `train.py`, which orchestrates reusable modules inside `src/`.

### Run baseline (fast, no lemmatization)

```bash
python train.py \
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
python train.py \
  --n-topics 8 \
  --min-doc-length 100 \
  --start-year 1981 \
  --end-year 1993 \
  --random-seed 42 \
  --output-dir outputs \
  --experiment-name baseline_lda_lemma \
  --use-lemmatization
```

---

## Outputs

Each run creates a structured output folder:

```text
outputs/<experiment_name>/
├── data_topics.csv        # Document-topic distributions
├── topics_summary.csv     # Top words per topic
├── lda_model.joblib       # Trained LDA model
├── vectorizer.joblib      # CountVectorizer
└── run_config.json        # Parameters used for the run
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

### Phase 4 — Modular Structure (current)

* Refactored code into `src/` modules
* Separated:

  * data loading
  * preprocessing
  * feature preparation
  * modeling
  * output saving
* `train.py` now handles orchestration only
* Prepared project for testing, API, and deployment

---

## Next Steps

Planned improvements in future phases:

* Add unit tests (`tests/`)
* Introduce experiment tracking (MLflow)
* Build inference pipeline (`src/inference/`)
* Create API with FastAPI (`app/`)
* Add CI/CD workflows (`.github/workflows/`)
* Containerize with Docker (`deployment/`)

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

---
