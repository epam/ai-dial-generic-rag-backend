FROM python:3.13-slim AS builder

WORKDIR /opt

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/pypoetry \
    --mount=type=bind,source=./pyproject.toml,target=./pyproject.toml \
    --mount=type=bind,source=./poetry.lock,target=./poetry.lock \
    pip install poetry==2.1.3 && \
    poetry self add poetry-plugin-export && \
    poetry export -f requirements.txt --without-hashes --without dev > ./requirements.txt

FROM python:3.13-slim AS runner

WORKDIR /opt/app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
      libmagic1 \
      libgl1 \
      libgthread-2.0 && \
    apt-get clean

# creates a non-root user with an explicit UID and adds permission to access the app folder
# for more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 1001 --disabled-password --gecos "" appuser && \
    chown -R appuser /opt/app

USER appuser

RUN mkdir ~/.cache

RUN --mount=type=cache,target=/home/appuser/.cache/pip,uid=1001 \
    --mount=type=bind,from=builder,source=/opt/requirements.txt,target=./requirements.txt \
    pip install -r requirements.txt --no-warn-script-location

# downloading sentence transformer model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('epam/bge-small-en', device='cpu')"

RUN python -m spacy download en_core_web_sm && \
    python -m spacy download uk_core_news_sm

COPY --chown=appuser ./src ./

# disable usage tracking for unstructured
ENV DO_NOT_TRACK=true

CMD [ "python3", "main.py" ]
