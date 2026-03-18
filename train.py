import argparse
import os
import re
import unicodedata
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import spacy
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def remove_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def unzip_archives(base_path: Path) -> None:
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".zip"):
                zip_path = Path(root) / file
                extract_folder = Path(root) / Path(file).stem

                if extract_folder.exists() and any(extract_folder.iterdir()):
                    continue

                extract_folder.mkdir(parents=True, exist_ok=True)

                try:
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(extract_folder)
                except zipfile.BadZipFile:
                    print(f"Bad zip file skipped: {zip_path}")


def load_text_files(base_path: Path) -> dict:
    text_dict = {}

    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = Path(root) / file
                file_id = file_path.stem

                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text_dict[file_id] = f.read()

    return text_dict


def categorize_profession(prof: str) -> str:
    if pd.isna(prof):
        return "Unknown"

    prof = remove_accents(str(prof).lower())
    prof_parts = re.split(r"[;,/]", prof)

    for part in prof_parts:
        if any(w in part for w in ["professeur", "enseignant", "instituteur"]):
            return "Education"
        if any(w in part for w in ["avocat", "juriste"]):
            return "Law"
        if any(w in part for w in ["medecin", "docteur", "infirmier", "pharmacien", "chirurgien"]):
            return "Health"
        if any(w in part for w in ["agric", "cultivat", "eleveur"]):
            return "Agriculture"
        if any(w in part for w in ["chef d", "entrepreneur", "commerc", "gerant", "industriel"]):
            return "Business"
        if any(w in part for w in ["ingenieur", "technicien"]):
            return "Technical"
        if any(w in part for w in ["ouvrier", "employe"]):
            return "Worker"
        if any(w in part for w in ["journaliste"]):
            return "Media"
        if any(w in part for w in ["maire", "depute", "conseiller"]):
            return "Political"

    return "Other"


def build_stopwords() -> list:
    french_stopwords = set(stopwords.words("french"))

    extra_stopwords = {
        "cevipof", "fonds",
        "circonscription", "elections", "législatives", "tour",
        "candidat", "candidats", "suppléant", "suppléants",
        "maire", "conseiller", "ans",
        "comme", "contre", "faire", "fait", "faut",
        "ceux", "leurs", "depuis", "tout", "tous",
        "être", "falloir", "vouloir", "mettre", "donner",
        "die", "der", "und", "für", "den", "sie", "eine", "das",
        "wir", "werden", "auf", "nicht", "einer", "dass", "gegen",
        "ihr", "auch", "mit", "von", "ist", "dem", "ein", "ich",
        "sich", "wird", "haben", "durch", "ihre", "als",
        "frankreich", "leben", "sind", "mehr", "einen", "politik",
        "mehrheit", "hat", "geben", "juni",
        "alsace", "strasbourg",
    }

    return list(french_stopwords.union(extra_stopwords))


def lemmatize_series(texts: pd.Series, model_name: str) -> pd.Series:
    nlp = spacy.load(model_name)

    def lemmatize_text(text: str) -> str:
        doc = nlp(str(text))
        return " ".join(
            token.lemma_
            for token in doc
            if not token.is_stop and token.is_alpha
        )

    return texts.apply(lemmatize_text)


def load_and_prepare_data(
    metadata_path: Path,
    text_files_path: Path,
    min_words: int,
) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path, low_memory=False)

    unzip_archives(text_files_path)
    text_dict = load_text_files(text_files_path)

    metadata["text"] = metadata["id"].astype(str).map(text_dict)
    df = metadata[metadata["text"].notna()].copy()

    df["word_count"] = df["text"].astype(str).str.split().str.len()
    df = df[df["word_count"] > min_words].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df = df[df["year"].isin([1981, 1988, 1993])].copy()

    df["titulaire-profession"] = df["titulaire-profession"].replace(
        "non mentionné", np.nan)
    df["titulaire-liste"] = df["titulaire-liste"].replace(
        "non mentionné", np.nan)

    df = df[df["titulaire-profession"].notna()].copy()
    df = df[df["titulaire-liste"].notna()].copy()

    df["profession_clean"] = df["titulaire-profession"].apply(
        categorize_profession)

    party_counts = df["titulaire-liste"].value_counts()
    major_parties = party_counts[party_counts > 100].index

    df["party_clean"] = df["titulaire-liste"].where(
        df["titulaire-liste"].isin(major_parties),
        "Other",
    )

    df["party_clean"] = df["party_clean"].replace(
        {
            "Liste entente populaire et nationale":
            "Liste d'entente populaire et nationale"
        }
    )

    selected_cols = [
        "id",
        "date",
        "year",
        "titulaire-liste",
        "titulaire-profession",
        "titulaire-age-calcule",
        "titulaire-sexe",
        "departement-insee",
        "text",
        "word_count",
        "profession_clean",
        "party_clean",
    ]

    return df[selected_cols].copy()


def train_topic_model(
    texts: pd.Series,
    n_topics: int,
    min_df: int,
    max_df: float,
    stop_words: list,
):
    vectorizer = CountVectorizer(
        stop_words=stop_words,
        token_pattern=r"\b[a-zA-ZÀ-ÿ]{3,}\b",
        max_df=max_df,
        min_df=min_df,
    )

    X = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="batch",
    )

    doc_topics = lda.fit_transform(X)
    return lda, vectorizer, X, doc_topics


def extract_topics(model, vectorizer, top_n: int = 10) -> pd.DataFrame:
    feature_names = vectorizer.get_feature_names_out()
    rows = []

    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-top_n:][::-1]
        top_words = [feature_names[i] for i in top_indices]

        rows.append(
            {
                "topic_id": topic_idx,
                "top_words": ", ".join(top_words),
            }
        )

    return pd.DataFrame(rows)


def save_outputs(
    df: pd.DataFrame,
    doc_topics: np.ndarray,
    topics_df: pd.DataFrame,
    lda,
    vectorizer,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    doc_topics_df = pd.DataFrame(
        doc_topics,
        columns=[f"Topic_{i}" for i in range(doc_topics.shape[1])],
    )

    result_df = pd.concat([df.reset_index(drop=True), doc_topics_df], axis=1)
    result_df["dominant_topic"] = doc_topics_df.idxmax(axis=1)

    result_df.to_csv(output_dir / "data_topics.csv",
                     index=False, encoding="utf-8")
    topics_df.to_csv(output_dir / "topics_summary.csv",
                     index=False, encoding="utf-8")

    joblib.dump(lda, output_dir / "lda_model.joblib")
    joblib.dump(vectorizer, output_dir / "vectorizer.joblib")


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline LDA topic model.")
    parser.add_argument("--metadata-path", type=str,
                        default="data/archelect_search.csv")
    parser.add_argument("--text-files-path", type=str,
                        default="data/text_files")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--n-topics", type=int, default=8)
    parser.add_argument("--min-words", type=int, default=100)
    parser.add_argument("--min-df", type=int, default=20)
    parser.add_argument("--max-df", type=float, default=0.95)
    parser.add_argument("--spacy-model", type=str, default="fr_core_news_md")

    args = parser.parse_args()

    print("Loading and preparing data...")
    df = load_and_prepare_data(
        metadata_path=Path(args.metadata_path),
        text_files_path=Path(args.text_files_path),
        min_words=args.min_words,
    )

    print("Lemmatizing texts...")
    df["text_lemma"] = lemmatize_series(df["text"], args.spacy_model)

    print("Training topic model...")
    stop_words = build_stopwords()
    lda, vectorizer, _, doc_topics = train_topic_model(
        texts=df["text_lemma"],
        n_topics=args.n_topics,
        min_df=args.min_df,
        max_df=args.max_df,
        stop_words=stop_words,
    )

    topics_df = extract_topics(lda, vectorizer)

    print("Saving outputs...")
    save_outputs(
        df=df,
        doc_topics=doc_topics,
        topics_df=topics_df,
        lda=lda,
        vectorizer=vectorizer,
        output_dir=Path(args.output_dir),
    )

    print("Training finished")
    print("Outputs generated")
    print("Model/artifacts saved")


if __name__ == "__main__":
    main()
