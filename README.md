# Party Influence in French Electoral Manifestos

## Project Overview

This project analyzes whether political discourse in French electoral manifestos is primarily structured by **party ideology** or by **candidates' socio-professional characteristics**.

Using manifesto texts from the Archelec archive (Sciences Po / CEVIPOF), we apply **topic modeling (LDA)** and **regression analysis** to measure how much variation in political themes is explained by party, profession, and year.

---

## Phase 0 — Initialization

This repository is initialized from a previous NLP project.

* `main` preserves the initial checkpoint
* `mlops` is the working branch
* notebooks are kept for **exploration only**

---

## Project Structure

```
PROJECT/
├── data/                  # Raw input data
├── notebooks/             # Exploration only
├── outputs/               # Saved artifacts from train.py
├── train.py               # Phase 2 terminal baseline
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
python -m spacy download fr_core_news_md
python -c "import nltk; nltk.download('stopwords')"
cp .env.example .env
```

---

## Phase 1 — MLOps Product Scope

### Problem

Determine whether manifesto topics are better explained by **party affiliation** or **profession**.

### Input

* Manifesto texts
* Metadata such as:
  * political party
  * profession
  * election year
  * additional candidate attributes when available

### Processing

* Load manifesto corpus
* Merge metadata
* Clean and filter texts
* Lemmatize text
* Train a topic model
* Produce document-topic outputs

### Output

* Topic distributions by document
* Topic keyword summaries
* Cleaned corpus with metadata
* Saved model artifacts

### Users

* Instructors
* Researchers
* Political science users
* Data science students

### Usage

* Batch execution from terminal
* API / dashboard in later phases

### System

**data → preprocessing → topic modeling → outputs → usage**

---

## Phase 2 — Terminal Baseline Pipeline

The first executable baseline replaces notebook-only execution with a single script that runs end to end from the terminal.

### Goal

Build one minimal pipeline that:

* loads manifesto texts
* merges metadata
* cleans and filters texts
* trains one topic model
* generates outputs
* saves artifacts locally

### Run the baseline topic-model training

Run it fast, without lemmatization:

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

Run it with lemmatization:

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

## Phase 3 — Clean and parameterized script

This phase removes notebook-style hidden parameters and makes the pipeline configurable from the terminal.

Main improvements:

* CLI parameters with `argparse`
* `.env.example` for environment configuration
* helper functions for repeated logic
* cleaner output organization by experiment name
* linting with Ruff and PyLint
