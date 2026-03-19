"""Configuration helpers for the training pipeline."""

import argparse
import os


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model training."""
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
    """Resolve the spaCy model from CLI, .env, or fallback default."""
    if cli_value:
        return cli_value

    env_value = os.getenv("SPACY_MODEL")
    if env_value:
        return env_value

    return "fr_core_news_md"


def validate_inputs(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""
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
