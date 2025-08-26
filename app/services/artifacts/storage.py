from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Optional


try:
    from google.cloud import storage  # type: ignore
    _HAS_GCS = True
except Exception:  # pragma: no cover
    storage = None  # type: ignore
    _HAS_GCS = False


class ArtifactStorage:
    """
    Storage helper for artifacts. Supports:
      - Local filesystem with optional base URL mapping
      - Optional upload to Google Cloud Storage when configured

    Env variables:
      - ARTIFACT_DIR (default: ./artifacts)
      - ARTIFACT_BASE_URL (optional; e.g., http://localhost:8000/static)
      - GCS_ARTIFACTS_BUCKET (optional; if set and google-cloud-storage available, uploads to GCS)
    """

    def __init__(self,
                 artifact_dir: Optional[str] = None,
                 base_url: Optional[str] = None,
                 gcs_bucket: Optional[str] = None) -> None:
        self.artifact_dir = Path(artifact_dir or os.getenv("ARTIFACT_DIR", "./artifacts")).resolve()
        self.base_url = base_url or os.getenv("ARTIFACT_BASE_URL")
        self.gcs_bucket_name = gcs_bucket or os.getenv("GCS_ARTIFACTS_BUCKET")
        try:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        self._gcs_client = None
        self._gcs_bucket = None
        if self.gcs_bucket_name and _HAS_GCS:
            try:
                self._gcs_client = storage.Client()  # type: ignore[attr-defined]
                self._gcs_bucket = self._gcs_client.bucket(self.gcs_bucket_name)
            except Exception:
                self._gcs_client = None
                self._gcs_bucket = None

    def _relpath(self, p: Path) -> Optional[str]:
        try:
            rel = p.resolve().relative_to(self.artifact_dir)
            # Normalize to POSIX separators for URLs
            return rel.as_posix()
        except Exception:
            return None

    def url_for_local(self, file_path: str) -> str:
        p = Path(file_path)
        rel = self._relpath(p)
        if self.base_url and rel is not None:
            return f"{self.base_url.rstrip('/')}/{rel}"
        # Fallback to file:// URL
        return p.resolve().as_uri()

    def upload_gcs(self, file_path: str, object_name: Optional[str] = None, content_type: Optional[str] = None) -> Optional[str]:
        if not (self._gcs_bucket and self._gcs_client):
            return None
        src = Path(file_path).resolve()
        if not src.exists():
            return None
        if object_name is None:
            rel = self._relpath(src)
            object_name = rel if rel is not None else src.name
        try:
            blob = self._gcs_bucket.blob(object_name)  # type: ignore[union-attr]
            if content_type is None:
                content_type = mimetypes.guess_type(src.name)[0] or "application/octet-stream"
            blob.upload_from_filename(str(src), content_type=content_type)
            # Return gs:// URI by default to avoid assuming public readability
            return f"gs://{self.gcs_bucket_name}/{object_name}"
        except Exception:
            return None

    def store(self, file_path: str) -> str:
        """Store artifact and return a URL. Prefers GCS when configured; otherwise local URL."""
        content_type = mimetypes.guess_type(file_path)[0]
        gcs_url = self.upload_gcs(file_path, content_type=content_type)
        if gcs_url:
            return gcs_url
        return self.url_for_local(file_path)


__all__ = ["ArtifactStorage"]

