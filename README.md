# Party Influence in French Electoral Manifestos

## Project Overview

This project analyzes whether political discourse in French electoral manifestos is primarily structured by **party ideology** or by **candidates' socio-professional characteristics**.

Using manifesto texts from the **Archelec archive (Sciences Po / CEVIPOF)**, we apply **topic modeling and regression analysis** to measure how much variation in political themes is explained by:

* Party affiliation
* Candidate profession
* Election year

The goal is to quantify whether manifesto discourse is driven more by **party discipline** or by **candidate-level social backgrounds**.

---

## Methodology

The analysis follows these steps:

1. **Data collection and cleaning** of manifesto texts and metadata.
2. **Exploratory data analysis** of the corpus.
3. **Topic modeling** using Latent Dirichlet Allocation (LDA).
4. **Regression analysis** estimating topic prevalence as a function of party, profession, and year.
5. **Robustness checks**, including varying the number of topics and permutation tests.

---

## Data Sources

* Metadata: <https://archelec.sciencespo.fr>
* OCR transcriptions: <https://gitlab.teklia.com/ckermorvant/arkindex_archelec>

The dataset contains French electoral manifestos from several legislative elections.

---

## Project Structure

```
PROJECT
│
├── data/                # Raw texts and processed datasets
├── notebooks/           # Analysis notebooks
│   ├── 01_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_topic_model.ipynb
│   ├── 04_regression.ipynb
│   └── 05_robustness.ipynb
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Author

NY Sarakpy  
ENSAE Paris – NLP Course Project  
GitHub: <https://github.com/sarakpyny>
