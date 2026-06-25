from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class DocumentRecord(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_path: Mapped[str] = mapped_column(String(255), unique=True)
    title: Mapped[str] = mapped_column(String(255))
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


class ChunkRecord(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    document: Mapped[str] = mapped_column(String(255), index=True)
    section: Mapped[str] = mapped_column(String(255), index=True)
    text: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


class EvalQuestionRecord(Base):
    __tablename__ = "eval_questions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    question: Mapped[str] = mapped_column(Text)
    reference_answer: Mapped[str] = mapped_column(Text)
    expected_doc: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    expected_section: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    should_refuse: Mapped[bool] = mapped_column(default=False)
    question_type: Mapped[str] = mapped_column(String(64), default="direct")
    expected_sources: Mapped[list[dict]] = mapped_column(JSON, default=list)
    reference_facts: Mapped[list[str]] = mapped_column(JSON, default=list)
    evaluation_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class ExperimentRunRecord(Base):
    __tablename__ = "experiment_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_name: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), default="running")
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    config_grid_json: Mapped[dict] = mapped_column(JSON, default=dict)
    summary_json: Mapped[dict] = mapped_column(JSON, default=dict)


class DashboardStateRecord(Base):
    __tablename__ = "dashboard_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    active_run_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    dashboard_state: Mapped[str] = mapped_column(String(32), default="empty")
    last_ingested_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ExperimentResultRecord(Base):
    __tablename__ = "experiment_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("experiment_runs.id"), index=True)
    question_id: Mapped[str] = mapped_column(String(64), index=True)
    model: Mapped[str] = mapped_column(String(64), index=True)
    alpha: Mapped[float] = mapped_column(Float)
    retrieve_top_k: Mapped[int] = mapped_column(Integer)
    rerank_top_n: Mapped[int] = mapped_column(Integer)
    quality_score: Mapped[float] = mapped_column(Float)
    cost_usd: Mapped[float] = mapped_column(Float)
    latency_ms: Mapped[int] = mapped_column(Integer)
    failure_category: Mapped[str] = mapped_column(String(64), index=True)
    answer_json: Mapped[dict] = mapped_column(JSON, default=dict)
    retrieved_json: Mapped[list] = mapped_column(JSON, default=list)
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    trace_id: Mapped[str] = mapped_column(String(64), index=True)


class TraceRecord(Base):
    __tablename__ = "traces"

    trace_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    question_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    payload_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
