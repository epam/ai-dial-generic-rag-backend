# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Generic RAG is a DIAL "Application Runner": a FastAPI/aidial-sdk service that answers user questions over collections of preloaded, pre-indexed documents. One deployment of this service backs many DIAL applications; each application maps to a **channel** (its own documents, indexes, and configuration). Requires Python 3.13, Poetry, PostgreSQL with pgvector, and optionally Elasticsearch.

## Commands

```bash
make install        # create venv + install dependencies
make install_all    # install + download spacy models (en_core_web_sm, uk_core_news_sm)
make lint           # poetry check --lock, ruff check, ruff format --check
make format         # ruff check --fix && ruff format
make test           # pytest tests/unit -v
make up             # start dependency stack via docker compose (DIAL core, chat UI, redis, pgvector, elasticsearch, mcp-inspector)
make main           # run the app locally (uvicorn on port 5000)
make run            # run the app itself in docker too
make down           # stop containers; make cleanup also removes data volumes
```

Run a single test: `poetry run pytest tests/unit/test_references_parser.py -v` (or `-k <pattern>`; `pytest.ini_options` sets `pythonpath=src`, `asyncio_mode=auto`).

Local run requires a `.env` (copy `.env.template`, fill `REMOTE_DIAL_URL`/`REMOTE_DIAL_API_KEY` for the upstream DIAL env) and the docker services from `make up` — the app connects to Postgres at startup and auto-applies migrations. Chat UI: http://localhost:3000, swagger: http://localhost:5000/docs.

`scripts/tools.py` is a click CLI to export/import/reindex a channel's documents through a running DIAL deployment.

Linting is ruff with an extensive rule set (line length 110, `preview = true` for format). CI (`.github/workflows/pr.yml`) delegates to the shared `epam/ai-dial-ci` python workflow; PRs target `development`.

## Architecture

Entry point: `src/main.py` → `generic_rag/app/factory.py:create_app()` builds a `DIALApp` and in its lifespan wires everything: DB engine + migrations, REST routes, the `generic-rag` chat completion + `generic-rag-embeddings` endpoints, and the MCP server.

**Channels and scopes.** Every request arrives with a DIAL `Api-Key` and an `x-dial-application-id` header. `ChannelBindings` (`src/generic_rag/scope.py`) opens a python-injection *mapped scope* named `channel` around each request — this happens in three places: the FastAPI dependency `_setup_channel_scope` (`app/routes.py`), `ChannelCompletion.chat_completion` (`app/chat_completion.py`), and `ChannelMiddleware` for MCP (`app/mcp.py`). Inside that scope, `channel_factory` (`app/module.py`) resolves the `Channel` by fetching the application's properties from DIAL core (`ChannelService`). The channel's unique `channel_key` is persisted as a file in the application's DIAL file-storage bucket.

**Dependency injection.** Uses `python-injection` (`@singleton`, `@scoped(ScopeName.channel)`, `@asfunction`, `Inject[...]` in FastAPI signatures). Providers live in `app/module.py`; services and components self-register via decorators, which only works because `lifespan` calls `load_packages(generic_rag.components, generic_rag.services)` — a new module in those packages is picked up automatically, but code outside them must be imported explicitly.

**Component framework** (`src/generic_rag/types.py`) — the core extension mechanism. `ConfigurableComponent` subclasses are discovered at runtime via `__subclasses__` recursion (`get_implementations()`); each gets a string qualifier derived from its class name (e.g. `ClassicRetriever` → `classic`). `get_aggregated_config_model()` assembles all implementations' pydantic config models into a discriminated union on a `type` field. Component families in `components/`:

- `parsers/` — `DocumentParser`: extract text/image chunks from documents
- `indexers/` — `Indexer`: turn chunks/queries into index values (text or embedding vectors)
- `storage/` — `IndexStorageBackend`: `pgvector` and `elasticsearch` backends storing `IndexRecord`s
- `retrieval/` — `Retriever`: find relevant chunks for a query
- `generation/` — `AnswerGenerator`: produce the answer from retrieved chunks
- `search_index.py` — `Index`/`ChunkIndex`: binds an indexer to a storage backend per channel index

To add a new parser/indexer/retriever/generator/backend: subclass the base with its own config model — it appears in the channel config schema automatically, no registration call needed.

**Dynamic pydantic models.** `ChannelConfig` (`src/generic_rag/channel.py`) is built at runtime via the `get_dynamic_model()` pattern, composed from whatever component implementations exist. Its JSON schema is served at `/application-type-schema`, which DIAL core uses to validate application properties. Per-request overrides (`custom_fields.configuration` in chat requests, or retriever overrides in MCP tools) are deep-merged over channel defaults via `RequestConfig.create()`.

**Document lifecycle.** Upload via `POST /channel/documents` stores the original in DIAL file storage (`DialClient`, with optional in-memory LRU cache) and enqueues an `index_document` task (`app/tasks.py`). Taskiq uses an `InMemoryBroker` with `await_inplace=True` — tasks execute inline, because per-request DIAL API keys expire after the request. Status transitions: `created → processing → processed → indexing → ready`/`error`. Chunks are persisted in Postgres; each configured index is rebuilt from chunks by the matching indexer/storage backend.

**Database.** Async SQLAlchemy (asyncpg) + pgvector; entities in `db/entities.py`. Migrations are raw SQL files in `db/migrations/` run by yoyo automatically at startup (`db/connection.py`, via the psycopg driver).

**MCP server** (`app/mcp.py`). FastMCP mounted at `/mcp/streamable-http` (stateless HTTP), exposing `list_documents_unordered`, `get_page`, `retrieve_text_chunks`, `rag_search`. Tool arg/output schemas are rewritten at list-time by `DynamicSchemasTransform` to inject channel-specific metadata-filter models. See `MCP.md` for endpoint URL construction and client setup.

**Chat completion flow** (`app/chat_completion.py`): resolve channel → merge request config → `Retriever.create(...)` with a stage listener that mirrors retrieval progress into DIAL `[DEBUG]` stages → `AnswerGenerator.invoke(...)` streams content and reference attachments into the response choice.
