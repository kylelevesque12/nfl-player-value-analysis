"""Export notebooks to Markdown for reliable GitHub viewing."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbconvert import MarkdownExporter


def find_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def export_notebook(path: Path, output_dir: Path) -> Path:
    notebook = nbformat.read(path, as_version=4)
    exporter = MarkdownExporter()
    body, _ = exporter.from_notebook_node(notebook)

    output_path = output_dir / path.with_suffix(".md").name
    output_path.write_text(body.rstrip() + "\n")
    return output_path


def main() -> int:
    project_root = find_project_root()
    notebook_dir = project_root / "notebooks"
    output_dir = project_root / "notebooks_markdown"
    output_dir.mkdir(parents=True, exist_ok=True)

    notebook_paths = sorted(notebook_dir.glob("*.ipynb"))
    for path in notebook_paths:
        output_path = export_notebook(path, output_dir)
        print("Exported", output_path.relative_to(project_root))

    print("Exported", len(notebook_paths), "notebooks to Markdown.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
