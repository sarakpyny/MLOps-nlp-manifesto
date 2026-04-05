#!/usr/bin/env bash
set -Eeuo pipefail

need_cmd() { command -v "$1" >/dev/null 2>&1; }

need_cmd python3 || { echo "python3 is required."; exit 1; }
need_cmd curl || { echo "curl is required."; exit 1; }

if ! need_cmd uv; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "Syncing project environment with dev dependencies..."
uv sync --dev

echo "Checking spaCy French model..."
.venv/bin/python -c "import spacy; spacy.load('fr_core_news_md'); print('fr_core_news_md OK')"

echo "Downloading NLTK stopwords..."
.venv/bin/python -c "import nltk; nltk.download('stopwords', quiet=True)"

echo "Preparing environment file..."
[ -f .env ] || cp .env.example .env

echo "Done."
echo "Activate the environment with:"
echo "source .venv/bin/activate"