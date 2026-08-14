# -*- coding: utf-8 -*-
"""
Database connection / session setup for the Arabic writing-teaching app.

Set the connection string with the DATABASE_URL environment variable:

  PostgreSQL:  postgresql+psycopg2://user:password@localhost:5432/arabic_writing_app
  MySQL:       mysql+pymysql://user:password@localhost:3306/arabic_writing_app

Drivers:
  pip install sqlalchemy psycopg2-binary      # PostgreSQL
  pip install sqlalchemy pymysql              # MySQL

Create all tables:
  python database.py
"""
import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Base

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/arabic_writing_app",
)

engine = create_engine(DATABASE_URL, echo=False, future=True, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """Create every table defined in models.py (no-op for existing tables)."""
    Base.metadata.create_all(engine)


def drop_db() -> None:
    """Drop every table. Destructive -- use only in development."""
    Base.metadata.drop_all(engine)


@contextmanager
def session_scope():
    """Transactional session: commits on success, rolls back on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    print(f"Connecting to: {engine.url}")
    init_db()
    print("All tables created.")
