"""Normalize notebooks so GitHub can render them reliably.

The project keeps generated figures, tables, Excel files, and reports as the
reviewable outputs. Notebooks stay as the readable source narrative. Clearing
saved cell outputs avoids GitHub renderer failures caused by embedded HTML,
large display payloads, or stale execution metadata.

This script also removes optional notebook v4.5 cell IDs and writes notebooks
as nbformat 4.4. That keeps the committed files in a simpler compatibility
format for GitHub's notebook renderer.
"""

from __future__ import annotations

from pathlib import Path

import nbformat


def find_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def clean_notebook(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    notebook.nbformat = 4
    notebook.nbformat_minor = 4
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }

    for cell in notebook.cells:
        cell.pop("id", None)
        cell.metadata = {}
        if cell.cell_type != "code":
            continue
        cell["outputs"] = []
        cell["execution_count"] = None

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
