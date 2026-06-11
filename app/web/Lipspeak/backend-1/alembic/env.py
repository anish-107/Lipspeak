from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

from app.core.config import settings

from app.core.database import Base

from app.models.user import User # noqa: F401
from app.models.video import Video # noqa: F401

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in offline mode."""

    database_url = (
        f"mysql+pymysql://"
        f"{settings.MYSQL_USER}:"
        f"{settings.MYSQL_PASSWORD}@"
        f"{settings.MYSQL_HOST}:"
        f"{settings.MYSQL_PORT}/"
        f"{settings.MYSQL_DATABASE}"
    )

    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={
            "paramstyle": "named",
        },
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in online mode."""

    config.set_main_option(
        "sqlalchemy.url",
        (
            f"mysql+pymysql://"
            f"{settings.MYSQL_USER}:"
            f"{settings.MYSQL_PASSWORD}@"
            f"{settings.MYSQL_HOST}:"
            f"{settings.MYSQL_PORT}/"
            f"{settings.MYSQL_DATABASE}"
        ),
    )

    connectable = engine_from_config(
        config.get_section(
            config.config_ini_section,
            {},
        ),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()