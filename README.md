<h1 align="center">
         AI DIAL Generic RAG Backend
    </h1>
    <p align="center">
        <p align="center">
        <a href="https://dialx.ai/">
          <img src="https://dialx.ai/logo/dialx_logo.svg" alt="About DIALX">
        </a>
    </p>
<h4 align="center">
    <a href="https://discord.gg/ukzj9U9tEe">
        <img src="https://img.shields.io/static/v1?label=DIALX%20Community%20on&message=Discord&color=blue&logo=Discord&style=flat-square" alt="Discord">
    </a>
</h4>

**Generic RAG** is Application Runner that allows building DIAL applications that answers user questions
based on data from collection of preloaded and pre-indexed documents, with flexible configuration of the processing pipeline.

<!-- TOC -->
* [Quick Start](#quick-start)
* [MCP Server](#mcp-server)
* [Configuration](#configuration)
  * [Prerequisites](#prerequisites)
  * [Environment Variables](#environment-variables)
  * [DIAL Core Configuration](#dial-core-configuration)
* [Local Development](#local-development)
  * [Pre-requisites](#pre-requisites)
  * [Run the application](#run-the-application)
<!-- TOC -->
# Quick Start

Install [Make](https://www.gnu.org/software/make/), Docker, and Docker Compose suitable for your OS.

* [Docker Desktop](https://docs.docker.com/desktop/)
* [Docker Engine and Docker Compose on Linux](https://docs.docker.com/engine/install/)
* [Rancher Desktop](https://rancherdesktop.io/) on Windows or MacOS

Create `.env` file using `.env.template`:

```bash
cp .env.template .env
```

Put there correct values for `REMOTE_DIAL_URL` and `REMOTE_DIAL_API_KEY` variables for
upstream DIAL env so that you local setup will be able to proxy model requests here.

Run the application (the images will be built automatically):

```bash
make run
```

> [!IMPORTANT]
> If your OS is not Linux, Docker Engine could already support `host.docker.internal` resolution,
> and defining it in the way as it's done in `docker-compose.yml` will break it. So if this is the case,
> comment out the `extra_hosts` section for the `core` service in `docker-compose.yml`.

* Now you can open <http://localhost:3000> for chat and <http://localhost:5000/docs> for swagger documentation
* Use `make down` to stop the containers that continue running in background
* Use `make cleanup` to clean data volumes created during run

Since Generic RAG is Application Runner, you need DIAL application to work with.
Applications can be created either by defining it in `applications` section of DIAL configuration,
or by using [DIAL API](https://dialx.ai/dial_api#tag/Applications/operation/saveCustomApplication).

The repository already has preconfigured example application `generic-rag-example` which can be used as a reference.

# MCP Server

Generic RAG includes MCP server for coding agents (Cursor, Claude Code).
See [MCP.md](MCP.md) for setup instructions and coding agent configuration.

# Configuration

## Prerequisites

* DIAL installation with DIAL core `0.44.0` or higher
* Postgresql database with `pgvector` extension
* (optional) Elasticsearch installation

## Environment Variables

|Variable|Required|Description|Available Values|Default Values|
|---|---|---|---|---|
| `DIAL_URL` | Yes | URL to the DIAL core. |  |  |
| `DIAL_API_KEY` | No | Optional api-key for background jobs execution. |  |
| `DIAL_PUBLIC_URL` | No | URL where DIAL core is publicly accessible (used to generate interactive documentation). |  |  |
| `IN_MEMORY_CACHE_ENABLED` | No | Whether in-memory file cache is enabled.  | `yes`/`true`/`1`, `no`/`false`/`0` | `yes` |
| `IN_MEMORY_CACHE_CAPACITY` | No | In-memoty cache capacity (examples: `128MiB`, `1GiB`, `2.5GiB`) |  | `128MiB` |
| `DB_HOST` | Yes | Postgresql database host |  |  |
| `DB_PORT` | No | Postgresql database port |  | `5432` |
| `DB_NAME` | Yes | Postgresql database name |  |  |
| `DB_USERNAME` | Yes | Postgresql database username |  |  |
| `DB_PASSWORD` | No | Database password, if you plan to use password authentication |  |  |
| `DB_MSI_ENABLED` | No | Use MSI authentication for database access | `yes`/`true`/`1`, `no`/`false`/`0` | `no` |
| `ELASTICSEARCH_URL` | No | URL of Elasticsearch instance |  |  |
| `ELASTICSEARCH_USERNAME` | No | Elasticsearch user for authentication |  |  |
| `ELASTICSEARCH_PASSWORD` | No | Elasticsearch password for authentication |  |  |
| `ELASTICSEARCH_INDEX_PREFIX` | No | The prefix that will be added to all indexes created in Elasticsearch |  |  |

> [!NOTE]
> * you should either set `DB_PASSWORD` (to use password authentication) enable MSI by setting `DB_MSI_ENABLED` to `yes`;
>   if both `DB_PASSWORD` and `DB_MSI_ENABLED` are defined, the MSI authentication will be used,
>   and the value of `DB_PASSWORD` will be ignored.
> * if `ELASTICSEARCH_URL` is not set (you are not planning to use Elasticsearch to store indexes),
>   other variables with `ELASTICSEARCH_` prefix will be ignored.
> * if you use single instance of Elasticsearch for multiple generic-rag deployments, setting
>   `ELASTICSEARCH_INDEX_PREFIX` is mandatory, and it should be unique for every single deployment of generic-rag.
> * setting `DIAL_API_KEY` allows to enable limited support of background jobs execution (such as indexing);
>   if set, indexing will be performed via background jobs on behalf of this api-key for only applications
>   which are accessible using this api-key, other applications will not be affected

## DIAL Core Configuration

In order to enable Generic RAG applications support by the DIAL installation, DIAL configuration file
(`config.json`) should contain the application type schema in `applicationTypeSchemas` section as illustrated
by the following snippet:

```json
{ 
  "applicationTypeSchemas": [
    {
      "$schema": "https://dial.epam.com/application_type_schemas/schema#",
      "$id": "https://dial.epam.com/application_type_schemas/generic-rag",
      "dial:applicationTypeCompletionEndpoint": "<GENERIC_RAG_URL>/openai/deployments/generic-rag/chat/completions",
      "dial:applicationTypeConfigurationEndpoint": "<GENERIC_RAG_URL>/openai/deployments/generic-rag/configuration",
      "dial:applicationTypeSchemaEndpoint": "<GENERIC_RAG_URL>/application-type-schema",
      "dial:applicationTypeDisplayName": "Generic RAG",
      "dial:applicationTypeIconUrl": "RAG_files_search.svg",
      "dial:applicationTypeRoutes": {
        "channel": {
          "dial:paths": [
            "/channel(/[^/]+)*$"
          ],
          "dial:rewritePath": true,
          "dial:methods": [
            "GET",
            "POST",
            "PUT",
            "DELETE"
          ],
          "dial:upstreams": [
            {
              "dial:endpoint": "<GENERIC_RAG_URL>"
            }
          ]
        }
      },
      "dial:applicationTypeMcp": {
        "dial:endpoint": "<GENERIC_RAG_URL>/mcp/streamable-http",
        "dial:transport": "HTTP",
        "dial:mcpConfigDelivery": "HEADER",
        "dial:forwardPerRequestKey": true
      },
      "dial:appendApplicationPropertiesHeader": false
    }
  ]
}
```

> [!IMPORTANT]
> Replace `<GENERIC_RAG_URL>` with correct URL of Generic RAG service where DIAL core will send requests
> (this URL can be internal URL available only within cluster when deployed with kubernetes).

# Local Development

## Pre-requisites

**1. Install [Make](https://www.gnu.org/software/make/)**

* MacOS - should already be installed
* [Windows](https://gnuwin32.sourceforge.net/packages/make.htm)
* [Windows, using Chocolatey](https://community.chocolatey.org/packages/make)
* Make sure that `make` is in the PATH (run `which make`).

**2. Install Docker Engine and Docker Compose suitable for your OS.**

You can use one of the following alternatives:

* [Docker Desktop](https://docs.docker.com/desktop/)
* [Docker Engine and Docker Compose on Linux](https://docs.docker.com/engine/install/)
* [Rancher Desktop](https://rancherdesktop.io/) on Windows or MacOS

**3. Install Python 3.13**

Direct installation:

* [MacOS, using Homebrew](https://formulae.brew.sh/formula/python@3.13) - `brew install python@3.13`
* [Windows or MacOS, using official repository](https://www.python.org/downloads/)
* [Windows, using Chocolatey](https://community.chocolatey.org/packages/python313)
* Make sure that `python3` or `python3.13` is in the PATH and works properly (run `python3.13 --version`).

Alternative: use [pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation):

* `pyenv` allows to manage different python versions on the same machine
* execute following from the repository root folder:

  ```bash
  pyenv install 3.13
  pyenv local 3.13  # use Python 3.13 for the current project
  ```

**4. Install [Poetry](https://python-poetry.org/docs/#installation)**

Recommended way - system-wide, independent of any particular python venv:

- MacOS - recommended way to install poetry is to [use pipx](https://python-poetry.org/docs/#installing-with-pipx)
- Windows - recommended way to install poetry is to
  use [official installer](https://python-poetry.org/docs/#installing-with-the-official-installer)
- Make sure that `poetry` is in the PATH and works properly (run `poetry --version`).

**5. Create virtual environment for the project**

```bash
make init_venv
```

**6. Install python dependencies required by the app**

```bash
make install_all
```

**7. Create `.env` file using `.env.template`**

```bash
cp .env.template .env
```

Then put correct value for `REMOTE_DIAL_URL` and `REMOTE_DIAL_API_KEY` variables for upstream DIAL 
env so that you local setup will be able to proxy model requests here.

**8. Run the services the application depends on**

```bash
make up
```

> [!NOTE]
> ⚠️ **macOS users:** `elasticsearch:8.19.6` container might not work properly on macOS.
> You can use `elasticsearch:8.15.3` instead.

## Run the application

```bash
source .venv/bin/activate  # activate virtual environment
python ./src/main.py
```

Or alternatively, use `make` command:

```bash
make main  # handles environment automatically
```

Or use you favorite IDE.

Now you can open:
* <http://localhost:3000> for DIAL Chat UI
* <http://localhost:5000/docs> for generic-rag swagger documentation

> [!IMPORTANT]
> If your OS is not Linux, Docker Engine could already support `host.docker.internal` resolution,
> and defining it in the way as it's done in `docker-compose.yml` will break it. So if this is the case,
> comment out the `extra_hosts` section for the `core` service in `docker-compose.yml`.

* Use `make down` to stop the containers that continue running in background
* Use `make cleanup` to clean data volumes created during run
