"""Train and save an LDA topic model for French electoral manifestos."""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv

from src.data.save_outputs import save_outputs
from src.features.prepare_data import load_and_prepare_data
from src.models.topic_model import (
    build_stopwords,
    extract_topics,
    generate_topic_labels,
    train_topic_model,
)
from src.preprocessing.cleaning import build_processed_texts
from src.utils.config import get_spacy_model, parse_args, validate_inputs
from src.utils.logging_config import setup_logging


def main() -> None:
    """Run the end-to-end training pipeline."""
    load_dotenv()
    setup_logging()
    logger = logging.getLogger(__name__)

    args = parse_args()
    validate_inputs(args)

    logger.info("Starting training pipeline")
    logger.info("Experiment name: %s", args.experiment_name)
    logger.info("Lemmatization enabled: %s", args.use_lemmatization)

    spacy_model = get_spacy_model(args.spacy_model)
    logger.info("Loaded spaCy model: %s", args.spacy_model)

    logger.info("Loading and preparing data")
    df = load_and_prepare_data(
        min_doc_length=args.min_doc_length,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    logger.info("Prepared dataframe with %s rows", len(df))

    if df.empty:
        logger.error("No documents available after filtering")
        raise ValueError(
            "No documents available after filtering. Check your parameters."
        )

    logger.info("Preparing text data")
    df["text_processed"] = build_processed_texts(
        texts=df["text"],
        use_lemmatization=args.use_lemmatization,
        spacy_model=spacy_model,
    )

    logger.info("Building stopwords")
    stop_words = build_stopwords()

    logger.info("Training topic model with %s topics", args.n_topics)
    lda, vectorizer, doc_topics = train_topic_model(
        texts=df["text_processed"],
        n_topics=args.n_topics,
        min_df=args.min_df,
        max_df=args.max_df,
        stop_words=stop_words,
        random_seed=args.random_seed,
    )

    logger.info("Extracting topics")
    topics_df = extract_topics(
        model=lda,
        vectorizer=vectorizer,
        top_n=args.top_n_words,
    )

    logger.info("Generating topic labels")
    topic_labels = generate_topic_labels(topics_df)

    logger.info("Saving outputs")
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
        topic_labels=topic_labels,
    )

    logger.info("Training finished successfully")
    logger.info(
        "Outputs saved under: %s/%s",
        args.output_dir,
        args.experiment_name,
    )


if __name__ == "__main__":
    main()
