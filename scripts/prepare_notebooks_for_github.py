"""Strip notebook outputs so GitHub can render notebooks reliably.

The project keeps generated figures, tables, Excel files, and reports as the
reviewable outputs. Notebooks stay as the readable source narrative. Clearing
saved cell outputs avoids GitHub renderer failures caused by embedded HTML,
large display payloads, or stale execution metadata.
"""

from __future__ import annotations

from pathlib import Path

import nbformat


def find_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def clean_notebook(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        cell["outputs"] = []
        cell["execution_count"] = None
        cell.metadata.pop("ExecuteTime", None)
        cell.metadata.pop("execution", None)

    notebook.metadata.pop("widgets", None)
    nbformat.write(notebook, path)


def main() -> int:
    project_root = find_project_root()
    notebook_paths = sorted((project_root / "notebooks").glob("*.ipynb"))

    for path in notebook_paths:
        clean_notebook(path)
        print("Cleaned", path.relative_to(project_root))

    print("Prepared", len(notebook_paths), "notebooks for GitHub rendering.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
