from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


def _normalize_database_url(database_url: str) -> str:
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return database_url


def _ensure_sqlite_parent_dir(database_url: str) -> None:
    if not database_url.startswith("sqlite"):
        return
    database = make_url(database_url).database
    if not database or database == ":memory:":
        return
    Path(database).expanduser().parent.mkdir(parents=True, exist_ok=True)


database_url = _normalize_database_url(settings.database_url)
_ensure_sqlite_parent_dir(database_url)
connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
engine = create_engine(database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    import app.db_models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _ensure_eval_question_columns()


def _ensure_eval_question_columns() -> None:
    inspector = inspect(engine)
    if "eval_questions" not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns("eval_questions")}
    additions = {
        "question_type": "VARCHAR(64) DEFAULT 'direct'",
        "expected_sources": "JSON",
        "reference_facts": "JSON",
        "evaluation_notes": "TEXT",
    }
    with engine.begin() as connection:
        for name, ddl in additions.items():
            if name not in existing:
                connection.execute(text(f"ALTER TABLE eval_questions ADD COLUMN {name} {ddl}"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
