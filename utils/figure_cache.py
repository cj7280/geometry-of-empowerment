"""Store figure results in pickle caches and reject mismatched schemas or keys."""

from __future__ import annotations

import pickle
from pathlib import Path


FIGURE_CACHE_SCHEMA_VERSION = 2


def load_matching_cache(path, expected_key, *, recompute=False):
    """Load matching cached results; raise if the schema or key has changed.

    Return None when recomputing or when the file is absent. Use trusted files.
    """
    path = Path(path)
    if recompute or not path.exists():
        return None

    with path.open("rb") as handle:
        blob = pickle.load(handle)

    matches = (
        blob.get("schema_version") == FIGURE_CACHE_SCHEMA_VERSION
        and blob.get("key") == expected_key
    )
    if not matches:
        raise RuntimeError(
            f"stale figure cache: {path}; rerun with --recompute-fig4"
        )
    return blob["results"]


def write_matching_cache(path, key, results):
    """Write results, the cache key, and the schema version to a pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(
            {
                "schema_version": FIGURE_CACHE_SCHEMA_VERSION,
                "key": key,
                "results": results,
            },
            handle,
        )
