"""Redact site-private identifiers from result files before they are committed.

    python scripts/scrub_private.py results/raw/<run>/*      # files (.gz handled by content)

Every match of a pattern in .private-patterns (git-ignored, the same list the
pre-commit check uses) becomes <redacted>. Server logs carry the node-local scratch
path, which includes the account name; nothing in the measurements depends on it.
Prints the number of redactions per changed file, so an unexpected match in data
would show up.
"""
from __future__ import annotations

import gzip
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    pats = [p for p in (ROOT / ".private-patterns").read_text().splitlines() if p.strip()]
    if not pats:
        raise SystemExit("scrub_private: .private-patterns missing or empty; refusing to guess")
    rx = re.compile("|".join(f"(?:{p})" for p in pats), re.IGNORECASE)
    for name in sys.argv[1:]:
        f = Path(name)
        if not f.is_file() or f.suffix in (".png", ".jpg", ".pdf"):
            continue
        gz = f.suffix == ".gz"
        try:
            text = gzip.open(f, "rt").read() if gz else f.read_text()
        except UnicodeDecodeError:
            print(f"skip (binary): {f}")
            continue
        new, n = rx.subn("<redacted>", text)
        if n:
            if gz:
                with gzip.open(f, "wt", compresslevel=9) as out:
                    out.write(new)
            else:
                f.write_text(new)
            print(f"{n:6d} redacted  {f}")


if __name__ == "__main__":
    main()
