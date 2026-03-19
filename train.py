"""Train baseline LDA topic model on French electoral manifestos."""

from pathlib import Path

from dotenv import load_dotenv

from src.data.save_outputs import save_outputs
from src.features.prepare_data import load_and_prepare_data
from src.models.topic_model import build_stopwords, extract_topics, train_topic_model
from src.preprocessing.cleaning import build_processed_texts
from src.utils.config import get_spacy_model, parse_args, validate_inputs


def main() -> None:
    """Run the end-to-end training pipeline."""
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
