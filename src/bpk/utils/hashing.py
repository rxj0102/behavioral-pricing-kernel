"""Content hashing for the data manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Hex SHA-256 digest of a file, read in chunks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()
