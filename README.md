# AI DIAL Generic RAG Backend

**Generic RAG** is a DIAL application that answers user questions based on data from collection of preloaded and pre-indexed documents.

- [Local Run with Docker](#local-run-with-docker)
  - [Pre-requisites, local run with Docker](#pre-requisites-local-run-with-docker)
    - [1. Install Make](#1-install-make)
    - [2. Install Docker Engine and Docker Compose suitable for your OS](#2-install-docker-engine-and-docker-compose-suitable-for-your-os)
    - [3. Environment variables](#3-environment-variables)
  - [Run the application](#run-the-application)
- [Local Development](#local-development)
  - [Pre-requisites, local development](#pre-requisites-local-development)
    - [1. Install Python 3.13](#1-install-python-313)
    - [2. Install Poetry](#2-install-poetry)
  - [Run the application](#run-the-application-1)
- [MCP Server](#mcp-server)
- [Deployment](#deployment)
  - [Pre-requisites, deployment](#pre-requisites-deployment)
  - [Configuration](#configuration)

## Local Run with Docker

### Pre-requisites, local run with Docker

#### 1. Install [Make](https://www.gnu.org/software/make/)

- MacOS - should already be installed
- [Windows](https://gnuwin32.sourceforge.net/packages/make.htm)
- [Windows, using Chocolatey](https://community.chocolatey.org/packages/make)
- Make sure that `make` is in the PATH (run `which make`).

#### 2. Install Docker Engine and Docker Compose suitable for your OS

Since Docker Desktop requires a paid license for commercial use, you can use one of the following alternatives:

- [Docker Engine and Docker Compose on Linux](https://docs.docker.com/engine/install/)
- [Rancher Desktop](https://rancherdesktop.io/) on Windows or MacOS

#### 3. Environment variables

Follow the next steps to prepare environment variable

- Create `.env` file using `.env.template`

```bash
cp .env.template .env
```

- Put correct value for `REMOTE_DIAL_URL` and `REMOTE_DIAL_API_KEY` variables for upstream DIAL
  env so that you local setup will be able to proxy model requests there.

### Run the application

- Run the application (the images will be built automatically)

```bash
make run
```

- Now you can open <http://localhost:3000> for chat and <http://localhost:5000/docs> for swagger documentation
- Use `make stop` to stop the containers that continue running in background
- Use `make cleanup` to clean data volumes created during run
- If you want to add new DIAL application of Generic RAG type (with different channel configuration), you can
  add it into `applications` section of `./dial_conf/config-template.json` file (see existing applications as references).
  Do not forget to restart `core` service after doing so.

> ⚠️ **macOS users:** `elasticsearch:8.19.6` container might not work properly on macOS.
> You can use `elasticsearch:8.15.3` instead.

## Local Development

### Pre-requisites, local development

In addition to [Pre-requisites for local run with Docker](#pre-requisites-local-run-with-docker), follow the next steps.

#### 1. Install Python 3.13

Direct installation:

- [MacOS, using Homebrew](https://formulae.brew.sh/formula/python@3.13) - `brew install python@3.13`
- [Windows or MacOS, using official repository](https://www.python.org/downloads/)
- [Windows, using Chocolatey](https://community.chocolatey.org/packages/python313)
- Make sure that `python3` or `python3.13` is in the PATH and works properly (run `python3.13 --version`).

Alternative: use [pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation):

- `pyenv` allows to manage different python versions on the same machine
- execute following from the repository root folder:

  ```bash
  pyenv install 3.13
  pyenv local 3.13  # use Python 3.13 for the current project
  ```

#### 2. Install [Poetry](https://python-poetry.org/docs/#installation)

Recommended way - system-wide, independent of any particular python venv:

- MacOS - recommended way to install poetry is to [use pipx](https://python-poetry.org/docs/#installing-with-pipx)
- Windows - recommended way to install poetry is to
  use [official installer](https://python-poetry.org/docs/#installing-with-the-official-installer)
- Make sure that `poetry` is in the PATH and works properly (run `poetry --version`).

### Run the application

- Create virtual environment for the project

```bash
make init_venv
```

- Install python dependencies required by the app

```bash
make install_all
```

- Run the services the application depends on

```bash
make up
```

> ⚠️ **macOS users:** `elasticsearch:8.19.6` container might not work properly on macOS.
> You can use `elasticsearch:8.15.3` instead.

- Now you can run the app:
  - using python:

    ```bash
    source .venv/bin/activate  # activate virtual environment
    python ./src/main.py
    ```

  - using make:

    ```bash
    make main  # handles environment automatically
    ```

  - or use you favorite IDE

Since generic-rag is Application Runner, you need to create DIAL application to work with, which can be done
either by defining the application in DIAL's `config.json` or by creating the application via [DIAL API](https://dialx.ai/dial_api#tag/Applications/operation/saveCustomApplication).

The repository already has preconfigured example application `generic-rag-example` which can be used as a reference.

- Now you can open:
    - <http://localhost:3000> for DIAL Chat UI
    - <http://localhost:5000/docs> for generic-rag swagger documentation

> ⚠️ **macOS users:** If core container is not able to reach the generic app app on your host machine, comment out the `extra_hosts` section
> for the `core` service in `docker-compose.yml`. Docker on MacOS already provides `host.docker.internal`
> natively, and the `extra_hosts: host.docker.internal:host-gateway` directive overrides it with an
> incorrect IP (`172.17.0.1`), preventing the core container from reaching the app on host machine.

- Default value of `application_id` that you see in swagger matches the `application_id` of the example app defined in DIAL config
- Use `make stop` to stop the containers that continue running in background
- Use `make cleanup` to clean data volumes created during run

## MCP Server

Generic RAG includes MCP server for coding agents (Cursor, Claude Code) available via a toolset.
See [MCP.md](MCP.md) for setup instructions and coding agent configuration.

## Deployment

### Pre-requisites, deployment

In order to deploy Generic RAG service, the following dependencies should be created:

- separate DIAL api key for indexing purposes
- PostgreSQL database with created `pqvector` extension and optionally with configured MSI for authentication

> NOTE: in order to make sure the database has `pgvector` extension run the following query:
>
> ```sql
> create extension if not exists vector;
> ```

### Configuration

Once these dependencies are set up, the following environment variables should be specified during deployment:

- `DIAL_URL`: URL of dial core
- `DB_HOST`: database host
- `DB_PORT`: database port
- `DB_NAME`: database name
- `DB_USERNAME`: database user
- `DB_PASSWORD`: the database user password if default password authentication should be used,
  OR `DB_MSI_ENABLED` (can be `1` or `true`) to use MSI authentication instead;
  if both `DB_PASSWORD` and `DB_MSI_ENABLED` are specified, the value of `DB_PASSWORD` will be ignored.
- `ELASTICSEARCH_URL` (if you plan to use Elasticsearch to store indexes)

Once the service is deployed, required channels can be configured in DIAL core's `config.json`.
