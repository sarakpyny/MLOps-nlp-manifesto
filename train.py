"""Train and save an LDA topic model for French electoral manifestos."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
from dotenv import load_dotenv
from mlflow.models import infer_signature

from src.data.save_outputs import save_outputs
from src.features.prepare_data import load_and_prepare_data
from src.inference.mlflow_model import ManifestoTopicPyFuncModel
from src.models.topic_model import (
    build_candidate_configs,
    build_stopwords,
    cross_validate_topic_model,
    extract_topics,
    generate_topic_labels,
    select_best_config,
    summarize_cv_results,
    train_topic_model,
)
from src.preprocessing.cleaning import build_processed_texts
from src.utils.config import (
    get_spacy_model,
    parse_args,
    resolve_candidate_grid,
    validate_inputs,
)
from src.utils.logging_config import setup_logging


def configure_mlflow(experiment_name: str) -> str | None:
    """Configure MLflow from environment and return the tracking URI."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)
    return tracking_uri


def log_run_params(
    args,
    spacy_model: str,
    candidate_grid: dict[str, list[int | float]],
) -> None:
    """Log run parameters to MLflow."""
    mlflow.log_params(
        {
            "experiment_name": args.experiment_name,
            "min_doc_length": args.min_doc_length,
            "start_year": args.start_year,
            "end_year": args.end_year,
            "random_seed": args.random_seed,
            "top_n_words": args.top_n_words,
            "use_lemmatization": args.use_lemmatization,
            "spacy_model": spacy_model,
            "cv_folds": args.cv_folds,
            "selection_metric": args.selection_metric,
            "n_candidate_n_topics": len(candidate_grid["n_topics"]),
            "n_candidate_min_df": len(candidate_grid["min_df"]),
            "n_candidate_max_df": len(candidate_grid["max_df"]),
            "n_candidate_configs": (
                len(candidate_grid["n_topics"])
                * len(candidate_grid["min_df"])
                * len(candidate_grid["max_df"])
            ),
        }
    )

    mlflow.log_dict(candidate_grid, "candidate_grid.json")


def log_cv_outputs(
    cv_summary: pd.DataFrame,
    best_config: dict[str, int | float],
    experiment_path: Path,
) -> None:
    """Log CV metrics and artifacts to MLflow."""
    best_row = cv_summary.iloc[0]

    std_value = best_row["std_perplexity"]
    if pd.isna(std_value):
        std_value = 0.0

    mlflow.log_metrics(
        {
            "best_mean_perplexity": float(best_row["mean_perplexity"]),
            "best_std_perplexity": float(std_value),
            "best_n_folds": int(best_row["n_folds"]),
        }
    )

    mlflow.log_params(
        {
            "best_n_topics": int(best_config["n_topics"]),
            "best_min_df": int(best_config["min_df"]),
            "best_max_df": float(best_config["max_df"]),
        }
    )

    mlflow.log_artifact(str(experiment_path / "cv_results.csv"))
    mlflow.log_artifact(str(experiment_path / "cv_summary.csv"))
    mlflow.log_artifact(str(experiment_path / "best_params.json"))


def log_final_output_artifacts(experiment_path: Path) -> None:
    """Log saved training artifacts to MLflow."""
    artifact_names = [
        "data_topics.csv",
        "topics_summary.csv",
        "lda_model.joblib",
        "vectorizer.joblib",
        "run_config.json",
        "topic_labels.json",
    ]

    for name in artifact_names:
        path = experiment_path / name
        if path.exists():
            mlflow.log_artifact(str(path), artifact_path="final_outputs")


def register_prediction_model(experiment_path: Path) -> None:
    """Log and register the prediction bundle as an MLflow pyfunc model."""
    input_example = pd.DataFrame(
        {"text": ["Nous défendons la justice sociale et les services publics."]}
    )

    output_example = [
        {
            "processed_text": "nous défendons la justice sociale et les services publics",
            "top_topic_id": 0,
            "top_topic_label": "example_label",
            "top_topic_score": 0.5,
            "topic_distribution": [0.5, 0.5],
        }
    ]

    signature = infer_signature(input_example, output_example)

    mlflow.pyfunc.log_model(
        artifact_path="manifesto_topic_model",
        python_model=ManifestoTopicPyFuncModel(),
        artifacts={
            "lda_model": str(experiment_path / "lda_model.joblib"),
            "vectorizer": str(experiment_path / "vectorizer.joblib"),
            "run_config": str(experiment_path / "run_config.json"),
            "topic_labels": str(experiment_path / "topic_labels.json"),
        },
        input_example=input_example,
        signature=signature,
        registered_model_name="manifesto_topic_model",
    )


def main() -> None:
    """Run the end-to-end training pipeline."""
    load_dotenv()
    setup_logging()
    logger = logging.getLogger(__name__)

    args = parse_args()
    validate_inputs(args)

    tracking_uri = configure_mlflow(args.experiment_name)

    logger.info("Starting training pipeline")
    logger.info("Experiment name: %s", args.experiment_name)
    logger.info("MLflow tracking URI: %s", tracking_uri or "local default")
    logger.info("Lemmatization enabled: %s", args.use_lemmatization)

    spacy_model = get_spacy_model(args.spacy_model)
    logger.info("Loaded spaCy model: %s", spacy_model)

    with mlflow.start_run():
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

        mlflow.log_metric("n_documents", int(len(df)))

        logger.info("Preparing text data")
        df["text_processed"] = build_processed_texts(
            texts=df["text"],
            use_lemmatization=args.use_lemmatization,
            spacy_model=spacy_model,
        )

        logger.info("Building stopwords")
        stop_words = build_stopwords()

        candidate_grid = resolve_candidate_grid(args)
        candidate_configs = build_candidate_configs(
            candidate_n_topics=candidate_grid["n_topics"],
            candidate_min_df=candidate_grid["min_df"],
            candidate_max_df=candidate_grid["max_df"],
        )

        log_run_params(
            args=args,
            spacy_model=spacy_model,
            candidate_grid=candidate_grid,
        )

        logger.info("Number of candidate configurations: %s",
                    len(candidate_configs))
        logger.info("Running model selection with %s fold(s)", args.cv_folds)

        cv_results = cross_validate_topic_model(
            texts=df["text_processed"],
            candidate_configs=candidate_configs,
            stop_words=stop_words,
            random_seed=args.random_seed,
            cv_folds=args.cv_folds,
        )

        cv_summary = summarize_cv_results(cv_results)
        best_config = select_best_config(cv_summary)

        logger.info("Best configuration selected: %s", best_config)

        logger.info("Retraining best topic model on full dataset")
        lda, vectorizer, doc_topics = train_topic_model(
            texts=df["text_processed"],
            n_topics=int(best_config["n_topics"]),
            min_df=int(best_config["min_df"]),
            max_df=float(best_config["max_df"]),
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

        experiment_path = Path(args.output_dir) / args.experiment_name
        cv_results.to_csv(experiment_path / "cv_results.csv", index=False)
        cv_summary.to_csv(experiment_path / "cv_summary.csv", index=False)

        with (experiment_path / "best_params.json").open("w", encoding="utf-8") as file:
            json.dump(best_config, file, indent=2)

        log_cv_outputs(
            cv_summary=cv_summary,
            best_config=best_config,
            experiment_path=experiment_path,
        )
        log_final_output_artifacts(experiment_path)

        logger.info("Registering prediction model in MLflow Model Registry")
        register_prediction_model(experiment_path)

        logger.info("Training finished successfully")
        logger.info("Outputs saved under: %s/%s",
                    args.output_dir, args.experiment_name)


if __name__ == "__main__":
    main()
