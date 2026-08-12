#!/usr/bin/env python
import asyncio
import io
import json
import logging
import os
import sys
import tempfile
from collections.abc import AsyncGenerator
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import aiofiles
import click
from aiohttp import ClientResponseError, ClientSession, FormData
from deepdiff import DeepDiff
from deepdiff.helper import COLORED_COMPACT_VIEW
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()

DIAL_URL = os.getenv("DIAL_URL", "http://localhost:8080")
DIAL_API_KEY = os.getenv("DIAL_API_KEY", "dial_api_key")

APPLICATION_TYPE_SCHEMA_IDS = {
    "https://dial.epam.com/application_type_schemas/generic-rag",
    "https://dial.epam.com/application_type_schemas/statgpt-generic-rag",
}

MAX_CONCURRENCY = 10
MAX_RETRIES = 3
MAX_DELAY = 10

APPLICATION_HELP = f"""
The application can specified either as {click.style("{deployment_name}", fg="cyan")}
(for applications within the organization), or {click.style("applications/{bucket}/{application_name}", fg="cyan")}
(for applications in user bucket).
"""

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(process)s | %(threadName)s | %(name)s | %(message)s",
    level=os.getenv("LOG_LEVEL", logging.INFO),
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)

semaphore = asyncio.Semaphore(MAX_CONCURRENCY)


class OperationError(RuntimeError): ...


async def _validate_application(session: ClientSession, application_id: str):
    """Check the application exists and it's RAG application."""
    async with session.get(f"/openai/applications/{application_id}") as response:
        if response.status == 404:  # noqa: PLR2004
            raise OperationError(f"application not found, {application_id=}")
        response.raise_for_status()
        application_info: dict[str, Any] = await response.json()

    if application_info.get("application_type_schema_id") not in APPLICATION_TYPE_SCHEMA_IDS:
        raise OperationError("the application is not Generic RAG application")


async def _list_documents(session: ClientSession, application_id: str) -> AsyncGenerator[dict]:
    """List documents uploaded to RAG"""
    offset = 0
    limit = 100
    application_route = f"/v1/deployments/{application_id}/route"
    while True:
        params = {"offset": offset, "limit": limit}
        async with session.get(f"{application_route}/channel/documents", params=params) as response:
            if response.content_type == "application/json":
                body = await response.json()

                for doc in body.get("results", []):
                    yield doc
                if body["results"]:
                    offset += len(body["results"])
                else:
                    break
            else:
                text = await response.text()
                logger.debug(f"{response.status}\n{text}")

            response.raise_for_status()


async def _get_channel_config(session: ClientSession, application_id: str) -> dict[str, Any]:
    """Return channel config for given application."""
    application_route = f"/v1/deployments/{application_id}/route"
    async with session.get(f"{application_route}/channel/config") as response:
        response.raise_for_status()
        return await response.json()


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    retry=retry_if_exception_type((ClientResponseError, asyncio.TimeoutError)),
    wait=wait_exponential(),
)
async def _export_document(
    session: ClientSession, application_id, *, document_id: int, target_dir: str
) -> str | None:
    """Export document from given application as a new file created in `target_dir`."""
    application_route = f"/v1/deployments/{application_id}/route"

    async with session.get(f"{application_route}/channel/documents/{document_id}/export") as response:
        response.raise_for_status()
        assert response.content_disposition
        assert response.content_disposition.type == "attachment"
        assert response.content_disposition.filename
        target_path = os.path.join(
            target_dir,
            response.content_disposition.filename,
        )
        logger.info(f"exported document '{response.content_disposition.filename}'")
        try:
            async with aiofiles.open(target_path, "wb") as fp:
                async for chunk in response.content.iter_chunked(512 * 1024):
                    await fp.write(chunk)
        except Exception as e:
            logger.warning(f"{document_id=}: {str(e)}")
        else:
            return target_path

        return None


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    retry=retry_if_exception_type(ClientResponseError),
    wait=wait_exponential(),
)
async def _import_document(session: ClientSession, application_id, *, source_path: str):
    """Import document into given application."""
    application_route = f"/v1/deployments/{application_id}/route"
    attachment_filename = os.path.basename(source_path)

    data = FormData()
    params = {"overwrite": "true"}

    async with aiofiles.open(source_path, "rb") as fp:
        data.add_field(
            name="attachment",
            content_type="application/octet-stream",
            value=io.BytesIO(await fp.read()),
            filename=attachment_filename,
        )

    logger.info(f"importing '{source_path}'")

    async with session.post(
        f"{application_route}/channel/documents/import", data=data, params=params
    ) as response:
        response.raise_for_status()
        body = await response.json()
        return body["id"]


async def _export_channel(application_id: str, archive_path: str):
    logger.info(f"{DIAL_URL=}")
    logger.info(f"{application_id=}")

    async with ClientSession(base_url=DIAL_URL, headers={"api-key": DIAL_API_KEY}) as session:
        await _validate_application(session, application_id)

        channel_config = await _get_channel_config(session, application_id)

        with tempfile.TemporaryDirectory(prefix="export_") as workdir:

            async def _export_document_task(document_id: int):
                async with semaphore:
                    try:
                        return await _export_document(
                            session, application_id, document_id=document_id, target_dir=workdir
                        )
                    except Exception as e:
                        logger.warning(f"unable to export document '{document_id}': {str(e)}")
                        return None

            tasks = [
                _export_document_task(document["id"])
                async for document in _list_documents(session, application_id)
            ]
            results = await asyncio.gather(*tasks)
            exported_files = [item for item in results if item]

            if exported_files:
                with (
                    open(archive_path, "wb") as fp,
                    ZipFile(fp, mode="w", compression=ZIP_DEFLATED) as zip_file,
                ):
                    zip_file.writestr("_channel.json", json.dumps(channel_config, indent=2))
                    for source_path in exported_files:
                        filename = os.path.relpath(source_path, workdir)
                        logger.info(f"> appending '{filename}'")
                        zip_file.write(source_path, arcname=filename)

                logger.info(f"data of '{application_id}' saved as '{zip_file.filename}'")

        total_processed = len(results)
        total_exported = len(exported_files)
        total_errors = total_processed - total_exported

        logger.info(f"{total_processed=}, {total_exported=}, {total_errors=}")
        logger.info(
            f"export of '{application_id}' completed {'successfully' if not total_errors else 'with errors'}."
        )


async def _import_channel(application_id: str, archive_path: str):
    logger.info(f"{DIAL_URL=}")
    logger.info(f"{application_id=}")

    async with ClientSession(base_url=DIAL_URL, headers={"api-key": DIAL_API_KEY}) as session:
        await _validate_application(session, application_id)

        logger.info("verifying channel config")
        channel_config = await _get_channel_config(session, application_id)

        with open(archive_path, "rb") as fp, ZipFile(fp, mode="r") as zip_file:
            expected_channel_config = json.load(zip_file.open("_channel.json"))

            if diff := DeepDiff(
                expected_channel_config,
                channel_config,
                exclude_paths=["channel_key", "retriever", "generation"],
                view=COLORED_COMPACT_VIEW,
            ):
                raise OperationError(f"channel config mismatch: {diff}")

            with tempfile.TemporaryDirectory(prefix="export_") as workdir:
                logger.info(f"extracting '{zip_file.filename}' into {workdir}")
                exported_files = []

                for zip_info in zip_file.filelist:
                    if not zip_info.filename.endswith(".msgpack"):
                        continue

                    logger.info(f"> extracting '{zip_info.filename}'")
                    exported_files.append(zip_file.extract(zip_info, workdir))

                async def _import_document_task(source_path: str):
                    async with semaphore:
                        try:
                            await _import_document(session, application_id, source_path=source_path)
                            return True
                        except Exception as e:
                            logger.warning(f"unable to import '{source_path}': {str(e)}")
                    return False

                tasks = [_import_document_task(item) for item in exported_files]
                results = await asyncio.gather(*tasks)

                total_processed = len(results)
                total_imported = len([item for item in results if item])
                total_errors = total_processed - total_imported

            logger.info(f"{total_processed=}, {total_imported=}, {total_errors=}")
            logger.info(
                f"import of '{zip_file.filename}' into application '{application_id}' completed "
                f"{'successfully' if not total_errors else 'with errors'}."
            )


async def _reindex_channel(application_id: str, index_names: set[str] | None, force: bool):
    logger.info(f"{DIAL_URL=}")
    logger.info(f"{application_id=}")

    application_route = f"/v1/deployments/{application_id}/route"

    params = [("force", str(force).lower())] if force else []
    if index_names:
        params.extend(("index", name) for name in index_names)

    async with ClientSession(base_url=DIAL_URL, headers={"api-key": DIAL_API_KEY}) as session:
        await _validate_application(session, application_id)

        async for document in _list_documents(session, application_id):
            document_id = document.get("id")
            document_name = document.get("display_name")
            status = "unknown"
            delay = 0.5

            try:
                async with session.put(
                    f"{application_route}/channel/documents/{document_id}/reindex", params=params
                ) as response:
                    response.raise_for_status()
                    body = await response.json()
                    status: str = body.get("status", "unknown")

            except ClientResponseError as e:
                logger.warning(str(e))

            while status not in {"error", "ready"}:
                await asyncio.sleep(delay)
                delay = min(MAX_DELAY, delay * 2)

                async with session.get(f"{application_route}/channel/documents/{document_id}") as response:
                    response.raise_for_status()
                    body = await response.json()
                    status: str = body.get("status", "unknown")

            logger.info(f"[ {status.upper()} ] '{document_name}' (id: {document_id})")


@click.group()
def cli():
    """RAG cli tools."""


@cli.command(name="export", help=f"Export data of RAG channel as single archive.\n\n{APPLICATION_HELP}")
@click.argument("application", required=True)
@click.option("-o", "--output", "output_path", required=True, help="the output file path")
def export_channel(application: str, output_path: str):
    asyncio.run(_export_channel(application, output_path))
    logger.info("completed")


@cli.command(
    name="import", help=f"Import previously exported RAG channel data into application.\n\n{APPLICATION_HELP}"
)
@click.argument("application", required=True)
@click.option("-s", "--source", "source_path", required=True, help="the source file path")
def import_channel(application: str, source_path: str):
    """Import previously exported RAG channel data into application."""
    asyncio.run(_import_channel(application, source_path))
    logger.info("completed")


@cli.command(name="reindex", help=f"Reindex all documents in the channel.\n\n{APPLICATION_HELP}")
@click.argument("application", required=True)
@click.option(
    "-i",
    "--index",
    "index_names",
    multiple=True,
    help=(
        "names of the index to update (can be specified multiple times; if not defined - all indexes will be updated)"
    ),
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help=(
        "perform whole process, including document re-processing and rebuilding of all indexes; "
        "it not set, document processing will be performed only if the document wasn't processed yet"
    ),
)
def reindex_channel(application: str, index_names: set[str] | None, force: bool):
    """Reindex all documents in the channel."""
    asyncio.run(_reindex_channel(application, index_names, force))
    logger.info("completed")


def main():
    try:
        cli()
    except OperationError as e:
        logger.warning(f"operation could not be performed: {str(e)}")
        sys.exit(-1)
    except Exception as e:
        logger.error(f"unexpected error: {str(e)}", exc_info=e)
        sys.exit(-1)


if __name__ == "__main__":
    main()
