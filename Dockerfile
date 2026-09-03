FROM python:3.13-slim-trixie AS builder

WORKDIR /opt

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/pypoetry \
    --mount=type=bind,source=./pyproject.toml,target=./pyproject.toml \
    --mount=type=bind,source=./poetry.lock,target=./poetry.lock \
    pip install poetry==2.1.3 && \
    poetry self add poetry-plugin-export && \
    poetry export -f requirements.txt --without-hashes --without dev > ./requirements.txt

FROM python:3.13-slim-trixie AS runner

WORKDIR /opt/app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    FASTMCP_LOG_ENABLED=false \
    DO_NOT_TRACK=true

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
      libmagic1 \
      libgl1 \
      libgthread-2.0 && \
    apt-get satisfy -y \
      "util-linux (>=2.41.5-0+deb13u1)" \
      "openssl (>=3.5.7-1~deb13u2)" && \
    apt-get clean

RUN python -m ensurepip --version && \
    python -m pip uninstall -y pip setuptools wheel

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,from=builder,source=/opt/requirements.txt,target=./requirements.txt \
    uv pip install -r requirements.txt \
      --index-strategy unsafe-best-match \
      --system

RUN --mount=type=cache,target=/root/.cache/uv \
    python -m spacy download en_core_web_sm --system && \
    python -m spacy download uk_core_news_sm --system

# creates a non-root user with an explicit UID and adds permission to access the app folder
# for more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 1001 --disabled-password --gecos "" appuser && \
    chown -R appuser /opt/app

USER appuser

# download embedding model
ENV BGE_EMBEDDING_MODEL_PATH=/home/appuser/bge-small-en

RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('epam/bge-small-en', device='cpu').save('$BGE_EMBEDDING_MODEL_PATH')"

COPY --chown=appuser ./src ./

CMD [ "python3", "main.py" ]
