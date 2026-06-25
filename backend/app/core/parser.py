from __future__ import annotations

from pathlib import Path


def parse_markdown_sections(path: Path) -> list[dict[str, str]]:
    """Parse a Markdown support doc into section dictionaries."""
    document_title = path.stem
    current_section = "Overview"
    current_lines: list[str] = []
    sections: list[dict[str, str]] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("# "):
            document_title = line.removeprefix("# ").strip()
            continue
        if line.startswith("## "):
            if current_lines:
                sections.append(
                    {
                        "document": path.name,
                        "title": document_title,
                        "section": current_section,
                        "text": "\n".join(current_lines).strip(),
                    }
                )
                current_lines = []
            current_section = line.removeprefix("## ").strip()
            continue
        if line:
            current_lines.append(line)

    if current_lines:
        sections.append(
            {
                "document": path.name,
                "title": document_title,
                "section": current_section,
                "text": "\n".join(current_lines).strip(),
            }
        )

    return sections


def parse_support_docs(data_dir: Path) -> list[dict[str, str]]:
    docs_dir = data_dir / "support_docs"
    sections: list[dict[str, str]] = []
    for path in sorted(docs_dir.glob("*.md")):
        sections.extend(parse_markdown_sections(path))
    return sections
