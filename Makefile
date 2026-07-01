.PHONY: init_venv remove_venv install download_spacy install_all lint format test up run main down cleanup

init_venv:
	poetry env use python3.13

remove_venv:
	poetry env remove --all

install: init_venv
	poetry install

# spacy cache expires after some time, thus models will be re-downloaded even if already installed.
# thus we have separate make commands to install venv packages and download spacy models.
download_spacy:
	poetry run python -m spacy download en_core_web_sm
	poetry run python -m spacy download uk_core_news_sm

install_all: install download_spacy

lint: install
	poetry check --lock
	poetry run ruff check

format: install
	poetry run ruff check --fix

test: install
	poetry run pytest tests/unit -v

up:
	docker compose -f docker-compose.yml up -d

run:
	export GENERIC_RAG_URL="http://generic-rag:5000" && \
	docker compose \
		-f docker-compose.yml \
		-f docker-compose.app.yml \
		run --build --rm -p 5000:5000 generic-rag

main: install
	poetry run python ./src/main.py

down:
	docker compose \
		-f docker-compose.yml \
		-f docker-compose.app.yml \
		down

cleanup: down
	docker compose -f docker-compose.yml -f docker-compose.app.yml down --volumes
