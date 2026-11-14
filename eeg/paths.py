"""Repository locations, resolved once so every module agrees."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
DOCS = REPO / "docs"


def _external():
    """data/external of this checkout, or of the main checkout when running from a worktree."""
    for base in (REPO, *REPO.parents):
        if (base / "data" / "external").exists():
            return base / "data" / "external"
    return REPO / "data" / "external"


EXTERNAL = _external()


def trust_system_certificates():
    """Use the OS certificate store for downloads (needed behind TLS-intercepting proxies); no-op if
    the optional `truststore` package is absent."""
    try:
        import truststore
    except ImportError:
        return
    truststore.inject_into_ssl()
