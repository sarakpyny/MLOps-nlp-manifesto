"""Text and metadata cleaning utilities."""

import re
import unicodedata

import numpy as np
import pandas as pd
import spacy


def remove_accents(text: str) -> str:
    """Remove accents from a string."""
    return "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )


def categorize_profession(profession: str) -> str:
    """Map raw profession labels into broader categories."""
    if pd.isna(profession):
        return "Unknown"

    profession = remove_accents(str(profession).lower())
    profession_parts = re.split(r"[;,/]", profession)

    for part in profession_parts:
        if any(word in part for word in ["professeur", "enseignant", "instituteur"]):
            return "Education"
        if any(word in part for word in ["avocat", "juriste"]):
            return "Law"
        if any(
            word in part
            for word in ["medecin", "docteur", "infirmier", "pharmacien", "chirurgien"]
        ):
            return "Health"
        if any(word in part for word in ["agric", "cultivat", "eleveur"]):
            return "Agriculture"
        if any(
            word in part
            for word in ["chef d", "entrepreneur", "commerc", "gerant", "industriel"]
        ):
            return "Business"
        if any(word in part for word in ["ingenieur", "technicien"]):
            return "Technical"
        if any(word in part for word in ["ouvrier", "employe"]):
            return "Worker"
        if "journaliste" in part:
            return "Media"
        if any(word in part for word in ["maire", "depute", "conseiller"]):
            return "Political"

    return "Other"


def clean_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean metadata columns and derive grouped profession and party labels."""
    cleaned = df.copy()

    cleaned["titulaire-profession"] = cleaned["titulaire-profession"].replace(
        "non mentionné", np.nan
    )
    cleaned["titulaire-liste"] = cleaned["titulaire-liste"].replace(
        "non mentionné", np.nan
    )

    cleaned = cleaned[cleaned["titulaire-profession"].notna()].copy()
    cleaned = cleaned[cleaned["titulaire-liste"].notna()].copy()

    cleaned["profession_clean"] = cleaned["titulaire-profession"].apply(
        categorize_profession
    )

    party_counts = cleaned["titulaire-liste"].value_counts()
    major_parties = party_counts[party_counts > 100].index

    cleaned["party_clean"] = cleaned["titulaire-liste"].where(
        cleaned["titulaire-liste"].isin(major_parties),
        "Other",
    )

    cleaned["party_clean"] = cleaned["party_clean"].replace(
        {
            "Liste entente populaire et nationale": (
                "Liste d'entente populaire et nationale"
            )
        }
    )

    return cleaned


def normalize_text_series(texts: pd.Series) -> pd.Series:
    """Normalize text with lowercase and basic regex cleaning."""
    return (
        texts.astype(str)
        .str.lower()
        .str.replace(r"[^a-zA-ZÀ-ÿ\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def lemmatize_series(texts: pd.Series, model_name: str) -> pd.Series:
    """Lemmatize a pandas Series of texts using spaCy."""
    nlp = spacy.load(model_name)

    def lemmatize_text(text: str) -> str:
        doc = nlp(str(text))
        return " ".join(
            token.lemma_ for token in doc if not token.is_stop and token.is_alpha
        )

    return texts.apply(lemmatize_text)


def build_processed_texts(
    texts: pd.Series,
    use_lemmatization: bool,
    spacy_model: str,
) -> pd.Series:
    """Normalize texts and optionally lemmatize them."""
    normalized_texts = normalize_text_series(texts)

    if use_lemmatization:
        print("Lemmatizing texts...")
        return lemmatize_series(normalized_texts, spacy_model)

    print("Skipping lemmatization and using normalized raw text...")
    return normalized_texts
