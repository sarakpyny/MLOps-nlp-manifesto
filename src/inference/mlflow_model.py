"""MLflow pyfunc wrapper for manifesto topic prediction."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow.pyfunc
import pandas as pd

from src.preprocessing.cleaning import build_processed_text


class ManifestoTopicPyFuncModel(mlflow.pyfunc.PythonModel):
    """Custom MLflow model wrapping LDA topic prediction artifacts."""

    def load_context(self, context) -> None:
        """Load bundled artifacts when the model is initialized."""
        artifacts = context.artifacts

        self.lda = joblib.load(artifacts["lda_model"])
        self.vectorizer = joblib.load(artifacts["vectorizer"])

        with open(artifacts["run_config"], "r", encoding="utf-8") as file:
            self.run_config = json.load(file)

        with open(artifacts["topic_labels"], "r", encoding="utf-8") as file:
            self.topic_labels = json.load(file)

    def predict(self, context, model_input: pd.DataFrame) -> list[dict]:
        """Predict topic distributions from a dataframe with a 'text' column."""
        if "text" not in model_input.columns:
            raise ValueError("Input dataframe must contain a 'text' column.")

        results: list[dict] = []

        for raw_text in model_input["text"].tolist():
            if not isinstance(raw_text, str) or not raw_text.strip():
                raise ValueError("Each input text must be a non-empty string.")

            processed_text = build_processed_text(
                text=raw_text,
                use_lemmatization=self.run_config["use_lemmatization"],
                spacy_model=self.run_config["resolved_spacy_model"],
            )

            vectorized = self.vectorizer.transform([processed_text])
            topic_distribution = self.lda.transform(vectorized)[0]

            top_topic_id = int(topic_distribution.argmax())
            top_topic_score = float(topic_distribution[top_topic_id])
            top_topic_label = self.topic_labels.get(
                str(top_topic_id),
                f"topic_{top_topic_id}",
            )

            results.append(
                {
                    "processed_text": processed_text,
                    "top_topic_id": top_topic_id,
                    "top_topic_label": top_topic_label,
                    "top_topic_score": top_topic_score,
                    "topic_distribution": topic_distribution.tolist(),
                }
            )

        return results
