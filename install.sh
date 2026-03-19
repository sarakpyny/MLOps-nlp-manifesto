#!/usr/bin/env bash
set -e

echo "Updating system packages..."
sudo apt-get -y update

echo "Installing system dependencies..."
sudo apt-get install -y python3-pip python3-venv curl

echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "Syncing project environment..."
uv sync

echo "Bootstrapping pip inside .venv..."
uv run python -m ensurepip --upgrade
uv run python -m pip install --upgrade pip

echo "Installing spaCy French model..."
uv run python -m spacy download fr_core_news_md

echo "Downloading NLTK stopwords..."
uv run python -c "import nltk; nltk.download('stopwords')"

echo "Done."
echo "Activate the environment with:"
echo "source .venv/bin/activate"
echo "Then copy config with:"
echo "cp .env.example .env"