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

echo "Bootstrapping pip inside .venv..."
uv run python -m ensurepip --upgrade
uv run python -m pip install --upgrade pip

echo "Installing spaCy French model..."
uv run python -m pip install fr_core_news_md

echo "Downloading NLTK stopwords..."
uv run python -c "import nltk; nltk.download('stopwords', quiet=True)"

echo "Done."
echo "Activate the environment with:"
echo "source .venv/bin/activate"
echo "Then copy config with:"
echo "cp .env.example .env"