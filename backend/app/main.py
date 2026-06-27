from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
from itertools import product
from typing import Any

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.core.types import RAGConfig
from app.db import SessionLocal, get_db, init_db
from app.schemas import AskRequest, AskResponse, EvalRunRequest, EvalRunResponse, IngestResponse
from app.services.model_clients import model_catalog
from app.services.progress import create_job, get_job, increment_job, update_job
from app.services.rag_service import rag_service
from app.services.repository import ExperimentRepository
from app.services.serialization import serialize_answer, serialize_candidate

@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    if not settings.use_pinecone:
        rag_service.ingest()
    yield


app = FastAPI(title="SupportBench RAGOps", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def list_models() -> dict[str, list[dict[str, Any]]]:
    return {"models": model_catalog()}


@app.get("/retrieval/status")
def retrieval_status() -> dict[str, Any]:
    dense_retrieval = "local_hash"
    if settings.use_pinecone:
        dense_retrieval = f"pinecone_{settings.pinecone_embedding_mode.lower()}"
    return {
        "dense_retrieval": dense_retrieval,
        "use_pinecone": settings.use_pinecone,
        "pinecone_index": settings.pinecone_index,
        "pinecone_namespace": settings.pinecone_namespace,
        "pinecone_embedding_mode": settings.pinecone_embedding_mode,
        "pinecone_embed_model": settings.pinecone_embed_model,
        "pinecone_text_field": settings.pinecone_text_field,
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "bm25_retrieval": "local_rank_bm25",
        "reranker": "local_lexical_reranker",
        "dataset": rag_service.dataset_status(),
    }


@app.post("/documents/ingest", response_model=IngestResponse)
def ingest_documents(db: Session = Depends(get_db)) -> IngestResponse:
    repo = ExperimentRepository(db)
    repo.clear_active_dashboard()
    try:
        documents, chunks = rag_service.ingest()
    except httpx.HTTPStatusError as error:
        raise HTTPException(status_code=503, detail=format_upstream_error(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    repo.replace_chunks(
        [
            {
                "chunk_id": chunk.chunk_id,
                "document": chunk.document,
                "section": chunk.section,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            for chunk in rag_service.chunks
        ]
    )
    repo.replace_eval_questions(rag_service.load_eval_question_payload())
    return IngestResponse(documents=documents, chunks=chunks, dataset=rag_service.dataset_status())


@app.post("/documents/ingest/start")
def start_ingest_documents(background_tasks: BackgroundTasks) -> dict[str, Any]:
    job = create_job(
        "ingest",
        steps=[
            "Clear active dashboard",
            "Parse support docs and create chunks",
            "Build dense/BM25 retrieval stores",
            "Persist chunks and eval questions",
        ],
    )
    background_tasks.add_task(run_ingest_job, job["job_id"])
    return job


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/eval/questions")
def list_eval_questions(limit: int | None = None) -> list[dict[str, Any]]:
    questions = rag_service.load_eval_questions(limit=limit)
    return [
        {
            "id": question.id,
            "question": question.question,
            "reference_answer": question.reference_answer,
            "expected_doc": question.expected_doc,
            "expected_section": question.expected_section,
            "expected_sources": question.expected_sources,
            "question_type": question.question_type,
            "reference_facts": question.reference_facts,
            "evaluation_notes": question.evaluation_notes,
            "tags": question.tags,
            "should_refuse": question.should_refuse,
        }
        for question in questions
    ]


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, db: Session = Depends(get_db)) -> AskResponse:
    config = RAGConfig(
        model=request.model,
        alpha=request.alpha,
        retrieve_top_k=request.retrieve_top_k,
        rerank_top_n=request.rerank_top_n,
    )
    try:
        result = rag_service.ask(request.question, config)
    except httpx.HTTPStatusError as error:
        raise HTTPException(status_code=503, detail=format_upstream_error(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    answer = result["answer"]
    candidates = result["candidates"]
    trace_id = result["trace_id"]
    retrieved = [serialize_candidate(candidate) for candidate in candidates]
    answer_json = serialize_answer(answer)

    db.add(
        __import__("app.db_models", fromlist=["TraceRecord"]).TraceRecord(
            trace_id=trace_id,
            payload_json={
                "question": request.question,
                "config": asdict(config),
                "answer": answer_json,
                "retrieved_chunks": retrieved,
            },
        )
    )
    db.commit()

    return AskResponse(
        answer=answer.answer,
        citations=answer.citations,
        metrics={
            "latency_ms": answer.latency_ms,
            "input_tokens": answer.input_tokens,
            "output_tokens": answer.output_tokens,
            "estimated_cost_usd": answer.estimated_cost_usd,
        },
        retrieved_chunks=retrieved,
        trace_id=trace_id,
    )


@app.post("/eval/run", response_model=EvalRunResponse)
def run_eval(request: EvalRunRequest, db: Session = Depends(get_db)) -> EvalRunResponse:
    try:
        rag_service.ensure_ready()
    except httpx.HTTPStatusError as error:
        raise HTTPException(status_code=503, detail=format_upstream_error(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    repo = ExperimentRepository(db)
    config_grid = request.model_dump()
    run = repo.create_run(request.run_name, config_grid)
    questions = rag_service.load_eval_questions(
        limit=None if request.question_ids else request.question_limit,
        question_ids=request.question_ids,
    )
    combinations = list(
        product(request.models, request.alphas, request.retrieve_top_k, request.rerank_top_n)
    )

    try:
        for model, alpha, retrieve_top_k, rerank_top_n in combinations:
            config = RAGConfig(
                model=model,
                alpha=alpha,
                retrieve_top_k=retrieve_top_k,
                rerank_top_n=rerank_top_n,
            )
            for question in questions:
                evaluated = rag_service.evaluate_question(question, config)
                answer_json = serialize_answer(evaluated["answer"])
                retrieved_json = [serialize_candidate(candidate) for candidate in evaluated["candidates"]]
                metrics_json = evaluated["metrics"].as_dict()
                repo.add_result(
                    run_id=run.id,
                    question_id=question.id,
                    model=model,
                    alpha=alpha,
                    retrieve_top_k=retrieve_top_k,
                    rerank_top_n=rerank_top_n,
                    answer_json=answer_json,
                    retrieved_json=retrieved_json,
                    metrics_json=metrics_json,
                    trace_id=evaluated["trace_id"],
                )
    except httpx.HTTPStatusError as error:
        message = format_upstream_error(error)
        repo.fail_run(run.id, message)
        raise HTTPException(status_code=503, detail=message) from error
    except RuntimeError as error:
        message = str(error)
        repo.fail_run(run.id, message)
        raise HTTPException(status_code=503, detail=message) from error

    db.commit()
    summary = repo.summarize_run(run.id)
    repo.complete_run(run.id, summary)
    summary = repo.dashboard_summary()
    total_results = len(combinations) * len(questions)
    return EvalRunResponse(
        run_id=run.id,
        status="completed",
        combinations=len(combinations),
        questions=len(questions),
        total_results=total_results,
        summary=summary,
    )


@app.post("/eval/run/start")
def start_eval_run(request: EvalRunRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)) -> dict[str, Any]:
    repo = ExperimentRepository(db)
    run = repo.create_run(request.run_name, request.model_dump())
    selected_question_count = len(request.question_ids) if request.question_ids else request.question_limit
    total_results = (
        len(request.models)
        * len(request.alphas)
        * len(request.retrieve_top_k)
        * len(request.rerank_top_n)
        * selected_question_count
    )
    job = create_job(
        "benchmark",
        run_id=run.id,
        steps=[
            "Prepare selected questions",
            "Retrieve and rerank chunks",
            "Generate model answers",
            "Evaluate answers and citations",
            "Publish dashboard summary",
        ],
    )
    update_job(
        job["job_id"],
        status="running",
        run_id=run.id,
        total_items=total_results,
        detail=f"Queued run {run.id} with {total_results} expected test cases.",
    )
    background_tasks.add_task(run_eval_job, job["job_id"], run.id, request.model_dump())
    return get_job(job["job_id"]) or job


@app.get("/eval/runs")
def list_runs(db: Session = Depends(get_db)) -> list[dict]:
    return ExperimentRepository(db).list_runs()


@app.get("/eval/runs/{run_id}")
def get_run(run_id: int, db: Session = Depends(get_db)) -> dict:
    run = ExperimentRepository(db).get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/eval/runs/{run_id}/export.csv")
def export_run_csv(run_id: int, db: Session = Depends(get_db)) -> StreamingResponse:
    csv_text = ExperimentRepository(db).export_run_csv(run_id)
    if csv_text is None:
        raise HTTPException(status_code=404, detail="Run not found")
    filename = f"supportbench-run-{run_id}.csv"
    return StreamingResponse(
        iter([csv_text]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/dashboard/summary")
def dashboard_summary(db: Session = Depends(get_db)) -> dict:
    return ExperimentRepository(db).dashboard_summary()


@app.get("/dashboard/history")
def dashboard_history(db: Session = Depends(get_db)) -> dict:
    return ExperimentRepository(db).dashboard_history()


@app.get("/traces/{trace_id}")
def get_trace(trace_id: str, db: Session = Depends(get_db)) -> dict:
    trace = ExperimentRepository(db).get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace


def run_ingest_job(job_id: str) -> None:
    with SessionLocal() as db:
        repo = ExperimentRepository(db)
        try:
            update_job(job_id, status="running", step_index=0, detail="Clearing active dashboard.")
            repo.clear_active_dashboard()
            update_job(
                job_id,
                step_index=1,
                detail="Parsing markdown docs, creating chunks, and indexing retrieval stores.",
            )
            documents, chunks = rag_service.ingest()
            update_job(job_id, step_index=3, detail="Writing chunks and golden questions to the database.")
            repo.replace_chunks(
                [
                    {
                        "chunk_id": chunk.chunk_id,
                        "document": chunk.document,
                        "section": chunk.section,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                    }
                    for chunk in rag_service.chunks
                ]
            )
            repo.replace_eval_questions(rag_service.load_eval_question_payload())
            update_job(
                job_id,
                status="completed",
                step_index=3,
                completed_items=chunks,
                total_items=chunks,
                detail=f"Ingested {documents} documents into {chunks} chunks.",
                result={"documents": documents, "chunks": chunks},
            )
        except httpx.HTTPStatusError as error:
            update_job(job_id, status="failed", error=format_upstream_error(error), detail="Ingestion failed.")
        except RuntimeError as error:
            update_job(job_id, status="failed", error=str(error), detail="Ingestion failed.")


def run_eval_job(job_id: str, run_id: int, request_payload: dict[str, Any]) -> None:
    request = EvalRunRequest(**request_payload)
    with SessionLocal() as db:
        repo = ExperimentRepository(db)
        try:
            update_job(job_id, status="running", step_index=0, detail="Loading selected eval questions.")
            rag_service.ensure_ready()
            questions = rag_service.load_eval_questions(
                limit=None if request.question_ids else request.question_limit,
                question_ids=request.question_ids,
            )
            combinations = list(
                product(request.models, request.alphas, request.retrieve_top_k, request.rerank_top_n)
            )
            total_results = len(combinations) * len(questions)
            update_job(
                job_id,
                total_items=total_results,
                detail=f"Running {total_results} test cases across {len(combinations)} configs.",
            )
            for model, alpha, retrieve_top_k, rerank_top_n in combinations:
                config = RAGConfig(
                    model=model,
                    alpha=alpha,
                    retrieve_top_k=retrieve_top_k,
                    rerank_top_n=rerank_top_n,
                )
                for question in questions:
                    update_job(
                        job_id,
                        step_index=1,
                        detail=f"Retrieving context for {question.id} using {model}.",
                    )
                    evaluated = rag_service.evaluate_question(question, config)
                    answer_json = serialize_answer(evaluated["answer"])
                    retrieved_json = [
                        serialize_candidate(candidate) for candidate in evaluated["candidates"]
                    ]
                    metrics_json = evaluated["metrics"].as_dict()
                    update_job(
                        job_id,
                        step_index=3,
                        detail=f"Scored {question.id}: {metrics_json['failure_category']}.",
                    )
                    repo.add_result(
                        run_id=run_id,
                        question_id=question.id,
                        model=model,
                        alpha=alpha,
                        retrieve_top_k=retrieve_top_k,
                        rerank_top_n=rerank_top_n,
                        answer_json=answer_json,
                        retrieved_json=retrieved_json,
                        metrics_json=metrics_json,
                        trace_id=evaluated["trace_id"],
                    )
                    db.commit()
                    increment_job(
                        job_id,
                        detail=f"{question.id} on {model}: {metrics_json['failure_category']}",
                    )

            update_job(job_id, step_index=4, detail="Calculating leaderboard and dashboard metrics.")
            summary = repo.summarize_run(run_id)
            repo.complete_run(run_id, summary)
            summary = repo.dashboard_summary()
            update_job(
                job_id,
                status="completed",
                step_index=4,
                detail=f"Benchmark complete: {total_results} test cases stored.",
                result={
                    "run_id": run_id,
                    "status": "completed",
                    "combinations": len(combinations),
                    "questions": len(questions),
                    "total_results": total_results,
                    "summary": summary,
                },
            )
        except httpx.HTTPStatusError as error:
            message = format_upstream_error(error)
            repo.fail_run(run_id, message)
            update_job(job_id, status="failed", error=message, detail="Benchmark failed.")
        except RuntimeError as error:
            message = str(error)
            repo.fail_run(run_id, message)
            update_job(job_id, status="failed", error=message, detail="Benchmark failed.")


def format_upstream_error(error: httpx.HTTPStatusError) -> str:
    url = str(error.request.url)
    status = error.response.status_code
    if "api.openai.com/v1/embeddings" in url and status == 429:
        return (
            "OpenAI embeddings request was rate limited or quota-limited "
            "(HTTP 429). Pinecone mode needs OPENAI_API_KEY credits/rate limit "
            "when PINECONE_EMBEDDING_MODE=openai. Use integrated Pinecone embeddings "
            "to remove the OpenAI embedding dependency."
        )
    if "api.openai.com/v1/embeddings" in url:
        return f"OpenAI embeddings request failed with HTTP {status}."
    if "pinecone.io" in url:
        return f"Pinecone request failed with HTTP {status}."
    if "api.openai.com" in url:
        return f"OpenAI model request failed with HTTP {status}."
    if "generativelanguage.googleapis.com" in url:
        return f"Gemini model request failed with HTTP {status}."
    if "api.groq.com" in url:
        return f"Groq model request failed with HTTP {status}."
    return f"Upstream API request failed with HTTP {status}."
