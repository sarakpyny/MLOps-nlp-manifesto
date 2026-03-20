#!/usr/bin/env bash
set -Eeuo pipefail

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

if ! need_cmd python3; then
  echo "python3 is required but not installed."
  exit 1
fi

if ! need_cmd curl; then
  echo "curl is required but not installed."
  exit 1
fi

if ! need_cmd uv; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "Syncing project environment..."
uv sync

echo "Bootstrapping pip inside .venv..."
uv run python -m ensurepip --upgrade || true
uv run python -m pip install --upgrade pip

echo "Installing spaCy French model..."
uv run python -m spacy download fr_core_news_md

echo "Downloading NLTK stopwords..."
uv run python -c "import nltk; nltk.download('stopwords')"

echo "Done."
echo "source .venv/bin/activate"
echo "cp .env.example .env"