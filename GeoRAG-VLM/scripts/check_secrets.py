"""Fail on common credential patterns or notebook outputs in publication files."""
import json
import pathlib
import re
import sys


def main():
    root = pathlib.Path(__file__).resolve().parents[2]
    patterns = [re.compile(r"\bsk" + r"-[A-Za-z0-9_-]{16,}\b"),
                re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
                re.compile(r"-----BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----")]
    errors = []
    for path in root.rglob("*"):
        if not path.is_file() or any(p in {".git", ".venv", "__pycache__"} for p in path.parts):
            continue
        if path.suffix in {".pyc", ".db"} or "runs" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern.search(text) for pattern in patterns):
            errors.append(str(path.relative_to(root)) + ": credential pattern")
        if path.suffix == ".ipynb":
            notebook = json.loads(text)
            for cell in notebook["cells"]:
                if cell.get("outputs") or cell.get("execution_count") is not None:
                    errors.append(str(path.relative_to(root)) + ": uncleared output")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Credential-pattern and notebook-output checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
