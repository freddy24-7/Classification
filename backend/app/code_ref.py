"""Source-code references for a check: the function's source, file and line range."""
import inspect
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def code_reference(func) -> dict:
    lines, start = inspect.getsourcelines(func)
    path = Path(inspect.getsourcefile(func)).resolve()
    return {
        "function": func.__name__,
        "file": str(path.relative_to(PROJECT_ROOT)),
        "absolute_path": str(path),
        "start_line": start,
        "end_line": start + len(lines) - 1,
        "source": "".join(lines),
    }
