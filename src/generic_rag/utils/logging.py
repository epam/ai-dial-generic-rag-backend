import logging
import os
import sys

import click
from uvicorn.logging import AccessFormatter, ColourizedFormatter, DefaultFormatter

LOG_FORMAT_PREFIX = "%(asctime)s | %(levelname)s | [%(trace)s] | %(pid)s | %(threadName)s | %(name)s"


def _format_log_record(record: logging.LogRecord, formatter: ColourizedFormatter) -> logging.LogRecord:
    trace_id = record.__dict__.get("otelTraceID", "0")
    span_id = record.__dict__.get("otelSpanID", "0")

    record.trace = f"{trace_id},{span_id}"
    record.pid = str(record.process)

    if formatter.use_colors:
        record.pid = click.style(record.pid, fg="magenta")
        record.levelname = formatter.color_level_name(record.levelname, record.levelno)
        record.name = click.style(record.name, fg="cyan")

    return record


class LogFormatter(DefaultFormatter):
    def formatMessage(self, record: logging.LogRecord) -> str:  # noqa: N802
        return super().formatMessage(_format_log_record(record, self))


class AccessLogFormatter(AccessFormatter):
    def formatMessage(self, record: logging.LogRecord) -> str:  # noqa: N802
        return super().formatMessage(_format_log_record(record, self))


def configure_logging(level, use_color: bool):
    # set these vars to prevent ai-dial-sdk from overriding loggers configuration
    os.environ.setdefault("OTEL_PYTHON_LOG_CORRELATION", "true")
    os.environ.setdefault("OTEL_TRACES_EXPORTER", "oltp")

    # configure default logger
    default_handler = logging.StreamHandler(stream=sys.stdout)
    default_handler.setFormatter(
        LogFormatter(
            fmt=f"{LOG_FORMAT_PREFIX} | %(message)s",
            use_colors=use_color,
        )
    )

    logging.basicConfig(handlers=[default_handler], level=level, force=True)

    # configure uvicorn access logger
    access_handler = logging.StreamHandler(stream=sys.stdout)
    access_handler.setFormatter(
        AccessLogFormatter(
            fmt=f'{LOG_FORMAT_PREFIX} | %(client_addr)s - "%(request_line)s" %(status_code)s',
            use_colors=use_color,
        )
    )

    access_logger = logging.getLogger("uvicorn.access")
    access_logger.addHandler(access_handler)
    access_logger.propagate = False
