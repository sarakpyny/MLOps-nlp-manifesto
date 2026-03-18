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
PROJECT
├── data/                # Raw and processed data
├── notebooks/           # Exploration only
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Phase 1 — MLOps Product Scope

### Problem

Determine whether manifesto topics are better explained by **party** or **profession**.

### Input

* Manifesto texts
* Metadata: party, profession, year

### Processing

* Preprocessing (cleaning, tokenization)
* Topic modeling (LDA)
* Regression / variance analysis

### Output

* Topic distributions
* Party/profession profiles
* Statistical results

### Users

* Instructors
* Researchers
* Data scientists

### Usage

* Batch execution (current)
* API / dashboard (future)

### System

**data → preprocessing → modeling → analysis → usage**

---
