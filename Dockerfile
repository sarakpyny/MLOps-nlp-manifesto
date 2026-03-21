FROM ubuntu:22.04

WORKDIR /app

RUN apt-get -y update && \
    apt-get install -y python3-pip python3-venv curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

RUN .venv/bin/python -m ensurepip --upgrade && \
    .venv/bin/python -m pip install --upgrade pip && \
    .venv/bin/python -m pip install fr_core_news_md && \
    .venv/bin/python -c "import nltk; nltk.download('stopwords', quiet=True)"

COPY app ./app
COPY src ./src
COPY train.py ./train.py

ARG URL_RAW
ENV URL_RAW=${URL_RAW}

RUN uv run python train.py --experiment-name baseline_lda

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]