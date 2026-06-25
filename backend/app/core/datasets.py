from __future__ import annotations

import json
from pathlib import Path
from typing import Any


GENERATED_DATASET_NAME = "souled_store"


def generated_dataset_dir(root: Path) -> Path:
    return root / "generated" / GENERATED_DATASET_NAME


def is_dataset_ready(path: Path) -> bool:
    return (path / "support_docs").is_dir() and (path / "eval_questions.json").is_file()


def active_data_dir(root: Path) -> Path:
    generated = generated_dataset_dir(root)
    if is_dataset_ready(generated):
        return generated
    return root


def eval_questions_path(root: Path) -> Path:
    return active_data_dir(root) / "eval_questions.json"


def dataset_status(root: Path) -> dict[str, Any]:
    active = active_data_dir(root)
    manifest_path = active / "source_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "active_dataset": manifest.get("dataset_name", active.name),
            "dataset_key": active.name,
            "data_dir": str(active),
            "is_generated": active == generated_dataset_dir(root),
            "source_page_count": manifest.get("source_page_count", 0),
            "faq_pair_count": manifest.get("faq_pair_count", 0),
            "support_doc_count": manifest.get("support_doc_count", 0),
            "eval_question_count": manifest.get("eval_question_count", 0),
            "generated_at": manifest.get("generated_at"),
            "source_pages": manifest.get("source_pages", []),
        }

    eval_count = 0
    eval_path = active / "eval_questions.json"
    if eval_path.is_file():
        eval_count = len(json.loads(eval_path.read_text(encoding="utf-8")))
    return {
        "active_dataset": "Sample SaaS support dataset",
        "dataset_key": "sample",
        "data_dir": str(active),
        "is_generated": False,
        "source_page_count": 0,
        "faq_pair_count": 0,
        "support_doc_count": len(list((active / "support_docs").glob("*.md"))),
        "eval_question_count": eval_count,
        "generated_at": None,
        "source_pages": [],
    }
