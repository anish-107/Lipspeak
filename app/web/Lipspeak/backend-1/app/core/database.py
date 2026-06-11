"""database.py

@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Database connection and session management.
@date: 11 June 2026
@returns: SQLAlchemy engine and session factory.

"""


# Imports
from sqlalchemy import create_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    sessionmaker,
)

from app.core.config import (
    settings,
)


# Database URL
DATABASE_URL = (
    f"mysql+pymysql://"
    f"{settings.MYSQL_USER}:"
    f"{settings.MYSQL_PASSWORD}@"
    f"{settings.MYSQL_HOST}:"
    f"{settings.MYSQL_PORT}/"
    f"{settings.MYSQL_DATABASE}"
)


# Engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)


# Session Factory
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


# Base Model
class Base(DeclarativeBase):
    """Base ORM model."""

    pass


# Dependency
def get_db():
    """Database session dependency."""

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()