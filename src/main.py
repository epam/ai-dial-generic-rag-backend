#!/usr/bin/env python
import logging
import os
import sys
from pathlib import Path

import pydantic
import uvicorn
from dotenv import load_dotenv

src_path = Path(__file__).parent.parent.absolute()
sys.path.append(str(src_path))

load_dotenv()

LOG_LEVEL = os.environ.get("LOG_LEVEL", logging.INFO)
LOG_USE_COLORS = pydantic.TypeAdapter(bool).validate_python(
    os.getenv("LOG_USE_COLORS", "no")
)

from generic_rag.utils.logging import configure_logging  # noqa: E402

configure_logging(level=LOG_LEVEL, use_color=LOG_USE_COLORS)

if __name__ == "__main__":
    from generic_rag.app.factory import create_app

    uvicorn.run(create_app(), host="0.0.0.0", port=5000, log_config=None)
