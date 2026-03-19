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
from dotenv import load_dotenv
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline LDA topic model on French electoral manifestos."
    )

    parser.add_argument(
        "--metadata-path",
        type=str,
        default="data/archelect_search.csv",
        help="Path to metadata CSV file.",
    )
    parser.add_argument(
        "--text-files-path",
        type=str,
        default="data/text_files",
        help="Path to directory containing manifesto text files and zip archives.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Base output directory.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="baseline_lda",
        help="Name of the experiment subfolder inside output-dir.",
    )

    parser.add_argument(
        "--n-topics",
        type=int,
        default=8,
        help="Number of LDA topics.",
    )
    parser.add_argument(
        "--min-doc-length",
        type=int,
        default=100,
        help="Minimum document length in words.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1981,
        help="Start year for filtering documents.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=1993,
        help="End year for filtering documents.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=20,
        help="Minimum document frequency for CountVectorizer.",
    )
    parser.add_argument(
        "--max-df",
        type=float,
        default=0.95,
        help="Maximum document frequency for CountVectorizer.",
    )
    parser.add_argument(
        "--top-n-words",
        type=int,
        default=10,
        help="Number of top words to save for each topic.",
    )
    parser.add_argument(
        "--spacy-model",
        type=str,
        default=None,
        help="spaCy model name. If omitted, uses SPACY_MODEL from .env or fr_core_news_md.",
    )
    parser.add_argument(
        "--use-lemmatization",
        action="store_true",
        help="Enable spaCy lemmatization. By default, raw text is used for faster runs.",
    )

    return parser.parse_args()


def get_spacy_model(cli_value: str | None) -> str:
    if cli_value:
        return cli_value

    env_value = os.getenv("SPACY_MODEL")
    if env_value:
        return env_value

    return "fr_core_news_md"


def remove_accents(text: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )


def unzip_archives(base_path: Path) -> None:
    for root, _, files in os.walk(base_path):
        for file_name in files:
            if not file_name.endswith(".zip"):
                continue

            zip_path = Path(root) / file_name
            extract_folder = Path(root) / Path(file_name).stem

            if extract_folder.exists() and any(extract_folder.iterdir()):
                continue

            extract_folder.mkdir(parents=True, exist_ok=True)

            try:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(extract_folder)
            except zipfile.BadZipFile:
                print(f"Bad zip file skipped: {zip_path}")


def load_text_files(base_path: Path) -> dict[str, str]:
    text_dict: dict[str, str] = {}

    for root, _, files in os.walk(base_path):
        for file_name in files:
            if not file_name.endswith(".txt"):
                continue

            file_path = Path(root) / file_name
            file_id = file_path.stem

            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                text_dict[file_id] = file.read()

    return text_dict


def categorize_profession(profession: str) -> str:
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


def build_stopwords() -> list[str]:
    french_stopwords = set(stopwords.words("french"))

    extra_stopwords = {
        "cevipof",
        "fonds",
        "circonscription",
        "elections",
        "législatives",
        "tour",
        "candidat",
        "candidats",
        "suppléant",
        "suppléants",
        "maire",
        "conseiller",
        "ans",
        "comme",
        "contre",
        "faire",
        "fait",
        "faut",
        "ceux",
        "leurs",
        "depuis",
        "tout",
        "tous",
        "être",
        "falloir",
        "vouloir",
        "mettre",
        "donner",
        "die",
        "der",
        "und",
        "für",
        "den",
        "sie",
        "eine",
        "das",
        "wir",
        "werden",
        "auf",
        "nicht",
        "einer",
        "dass",
        "gegen",
        "ihr",
        "auch",
        "mit",
        "von",
        "ist",
        "dem",
        "ein",
        "ich",
        "sich",
        "wird",
        "haben",
        "durch",
        "ihre",
        "als",
        "frankreich",
        "leben",
        "sind",
        "mehr",
        "einen",
        "politik",
        "mehrheit",
        "hat",
        "geben",
        "juni",
        "alsace",
        "strasbourg",
    }

    return list(french_stopwords.union(extra_stopwords))


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    return pd.read_csv(metadata_path, low_memory=False)


def merge_metadata_with_texts(
    metadata: pd.DataFrame,
    text_files_path: Path,
) -> pd.DataFrame:
    unzip_archives(text_files_path)
    text_dict = load_text_files(text_files_path)

    merged = metadata.copy()
    merged["text"] = merged["id"].astype(str).map(text_dict)

    return merged[merged["text"].notna()].copy()


def filter_documents(
    df: pd.DataFrame,
    min_doc_length: int,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    filtered = df.copy()

    filtered["word_count"] = filtered["text"].astype(str).str.split().str.len()
    filtered = filtered[filtered["word_count"] > min_doc_length].copy()

    filtered["date"] = pd.to_datetime(filtered["date"], errors="coerce")
    filtered["year"] = filtered["date"].dt.year
    filtered = filtered[filtered["year"].between(start_year, end_year)].copy()

    return filtered


def clean_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
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
            "Liste entente populaire et nationale": "Liste d'entente populaire et nationale"
        }
    )

    return cleaned


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
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


def lemmatize_series(texts: pd.Series, model_name: str) -> pd.Series:
    nlp = spacy.load(model_name)

    def lemmatize_text(text: str) -> str:
        doc = nlp(str(text))
        return " ".join(
            token.lemma_ for token in doc if not token.is_stop and token.is_alpha
        )

    return texts.apply(lemmatize_text)


def normalize_text_series(texts: pd.Series) -> pd.Series:
    return (
        texts.astype(str)
        .str.lower()
        .str.replace(r"[^a-zA-ZÀ-ÿ\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def build_processed_texts(
    texts: pd.Series,
    use_lemmatization: bool,
    spacy_model: str,
) -> pd.Series:
    normalized_texts = normalize_text_series(texts)

    if use_lemmatization:
        print("Lemmatizing texts...")
        return lemmatize_series(normalized_texts, spacy_model)

    print("Skipping lemmatization and using normalized raw text...")
    return normalized_texts


def load_and_prepare_data(
    metadata_path: Path,
    text_files_path: Path,
    min_doc_length: int,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    metadata = load_metadata(metadata_path)
    df = merge_metadata_with_texts(metadata, text_files_path)
    df = filter_documents(df, min_doc_length, start_year, end_year)
    df = clean_metadata_columns(df)
    return select_columns(df)


def train_topic_model(
    texts: pd.Series,
    n_topics: int,
    min_df: int,
    max_df: float,
    stop_words: list[str],
    random_seed: int,
) -> tuple[LatentDirichletAllocation, CountVectorizer, np.ndarray]:
    vectorizer = CountVectorizer(
        stop_words=stop_words,
        token_pattern=r"\b[a-zA-ZÀ-ÿ]{3,}\b",
        max_df=max_df,
        min_df=min_df,
    )

    matrix = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_seed,
        learning_method="batch",
    )

    doc_topics = lda.fit_transform(matrix)
    return lda, vectorizer, doc_topics


def extract_topics(
    model: LatentDirichletAllocation,
    vectorizer: CountVectorizer,
    top_n: int = 10,
) -> pd.DataFrame:
    feature_names = vectorizer.get_feature_names_out()
    rows = []

    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-top_n:][::-1]
        top_words = [feature_names[index] for index in top_indices]
        rows.append({"topic_id": topic_idx, "top_words": ", ".join(top_words)})

    return pd.DataFrame(rows)


def save_run_config(args: argparse.Namespace, run_dir: Path, spacy_model: str) -> None:
    config = vars(args).copy()
    config["resolved_spacy_model"] = spacy_model
    pd.Series(config).to_json(run_dir / "run_config.json", indent=2)


def save_outputs(
    df: pd.DataFrame,
    doc_topics: np.ndarray,
    topics_df: pd.DataFrame,
    lda: LatentDirichletAllocation,
    vectorizer: CountVectorizer,
    output_dir: Path,
    experiment_name: str,
    args: argparse.Namespace,
    spacy_model: str,
) -> None:
    run_dir = output_dir / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    doc_topics_df = pd.DataFrame(
        doc_topics,
        columns=[f"Topic_{index}" for index in range(doc_topics.shape[1])],
    )

    result_df = pd.concat([df.reset_index(drop=True), doc_topics_df], axis=1)
    result_df["dominant_topic"] = doc_topics_df.idxmax(axis=1)

    result_df.to_csv(run_dir / "data_topics.csv",
                     index=False, encoding="utf-8")
    topics_df.to_csv(run_dir / "topics_summary.csv",
                     index=False, encoding="utf-8")

    joblib.dump(lda, run_dir / "lda_model.joblib")
    joblib.dump(vectorizer, run_dir / "vectorizer.joblib")
    save_run_config(args, run_dir, spacy_model)


def validate_inputs(args: argparse.Namespace) -> None:
    if args.n_topics <= 0:
        raise ValueError("--n-topics must be greater than 0.")
    if args.min_doc_length < 0:
        raise ValueError("--min-doc-length must be >= 0.")
    if args.start_year > args.end_year:
        raise ValueError(
            "--start-year must be less than or equal to --end-year.")
    if args.min_df <= 0:
        raise ValueError("--min-df must be greater than 0.")
    if not 0 < args.max_df <= 1:
        raise ValueError("--max-df must be in the interval (0, 1].")
    if args.top_n_words <= 0:
        raise ValueError("--top-n-words must be greater than 0.")


def main() -> None:
    load_dotenv()
    args = parse_args()
    validate_inputs(args)

    spacy_model = get_spacy_model(args.spacy_model)

    print("Loading and preparing data...")
    df = load_and_prepare_data(
        metadata_path=Path(args.metadata_path),
        text_files_path=Path(args.text_files_path),
        min_doc_length=args.min_doc_length,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    if df.empty:
        raise ValueError(
            "No documents available after filtering. Check your parameters."
        )

    print("Preparing text data...")
    df["text_processed"] = build_processed_texts(
        texts=df["text"],
        use_lemmatization=args.use_lemmatization,
        spacy_model=spacy_model,
    )

    print("Building stopwords...")
    stop_words = build_stopwords()

    print("Training topic model...")
    lda, vectorizer, doc_topics = train_topic_model(
        texts=df["text_processed"],
        n_topics=args.n_topics,
        min_df=args.min_df,
        max_df=args.max_df,
        stop_words=stop_words,
        random_seed=args.random_seed,
    )

    print("Extracting topics...")
    topics_df = extract_topics(
        model=lda,
        vectorizer=vectorizer,
        top_n=args.top_n_words,
    )

    print("Saving outputs...")
    save_outputs(
        df=df,
        doc_topics=doc_topics,
        topics_df=topics_df,
        lda=lda,
        vectorizer=vectorizer,
        output_dir=Path(args.output_dir),
        experiment_name=args.experiment_name,
        args=args,
        spacy_model=spacy_model,
    )

    print("Training finished")
    print(f"Experiment: {args.experiment_name}")
    print(f"Lemmatization enabled: {args.use_lemmatization}")
    print("Outputs generated")
    print("Model artifacts saved")


if __name__ == "__main__":
    main()
