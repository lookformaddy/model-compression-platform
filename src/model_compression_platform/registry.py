"""Versioned artifact registry with rollback.

Each artifact is a directory <registry>/<name>/v<semver>/ containing
artifact.json (metadata + weights for dense models, or a pointer for
quantized models, whose int8 codes are stored inline). The registry keeps
a `current` pointer per artifact name; rollback only moves the pointer,
so a bad rollback is itself reversible, and history is never deleted.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ArtifactMeta:
    name: str
    version: str
    technique: str
    accuracy: float
    size_bytes: int
    promoted: bool
    parent: str | None = None
    notes: str = ""


class ArtifactRegistry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _artifact_dir(self, name: str) -> Path:
        return self.root / name

    def _version_dir(self, name: str, version: str) -> Path:
        return self._artifact_dir(name) / f"v{version}"

    def save(self, meta: ArtifactMeta, payload: dict) -> Path:
        """Persist a new version and point `current` at it. Returns dir."""
        vdir = self._version_dir(meta.name, meta.version)
        if vdir.exists():
            raise ValueError(f"version {meta.version} of {meta.name} already exists")
        vdir.mkdir(parents=True)
        (vdir / "artifact.json").write_text(json.dumps(
            {"meta": meta.__dict__, "payload": payload}, indent=2))
        (self._artifact_dir(meta.name) / "current").write_text(meta.version)
        return vdir

    def load(self, name: str, version: str | None = None) -> dict:
        """Load a version, or the current version when version is None."""
        if version is None:
            current_file = self._artifact_dir(name) / "current"
            if not current_file.exists():
                raise KeyError(f"no current version for {name}")
            version = current_file.read_text().strip()
        path = self._version_dir(name, version) / "artifact.json"
        if not path.exists():
            raise KeyError(f"unknown artifact {name} v{version}")
        return json.loads(path.read_text())

    def current_version(self, name: str) -> str | None:
        current_file = self._artifact_dir(name) / "current"
        return current_file.read_text().strip() if current_file.exists() else None

    def rollback(self, name: str, version: str) -> str:
        """Point `current` at an older version. Returns the new current."""
        if not self._version_dir(name, version).exists():
            raise KeyError(f"cannot roll back {name} to unknown version {version}")
        (self._artifact_dir(name) / "current").write_text(version)
        return version

    def list_versions(self, name: str) -> list[str]:
        adir = self._artifact_dir(name)
        if not adir.exists():
            return []
        versions = sorted(
            (p.name[1:] for p in adir.iterdir() if p.is_dir() and p.name.startswith("v")),
            key=lambda v: [int(x) for x in v.split(".")],
        )
        return versions

    def list_artifacts(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())
