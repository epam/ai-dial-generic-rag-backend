import logging
import os.path

from injection import inject
from sqlalchemy import URL
from yoyo import get_backend, read_migrations

from generic_rag.app.settings import DatabaseConfig
from generic_rag.db.auth import TokenProvider

logger = logging.getLogger(__name__)


@inject
def apply_migrations(config: DatabaseConfig, token_provider: TokenProvider | None = NotImplemented):
    migration_source = os.path.join(str(os.path.dirname(__file__)), "migrations")
    migrations = read_migrations(migration_source)
    logger.info(f"Loaded {len(migrations)} migration(s)")

    url = URL.create(
        drivername="postgresql+psycopg",
        host=config.host,
        port=config.port,
        database=config.dbname,
        username=config.username,
        password=(
            token_provider.token
            if token_provider is not None
            else (config.password.get_secret_value() if config.password else None)
        ),
    )

    backend = get_backend(url.render_as_string(hide_password=False))
    backend.apply_migrations(migrations=backend.to_apply(migrations))
    logger.info("All migrations applied")
