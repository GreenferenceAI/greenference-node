"""Lium-style persistent volume management with backup/restore."""

from __future__ import annotations

import shutil
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from greenference_protocol import VolumeRecord


def _utcnow() -> datetime:
    return datetime.now(UTC)


class VolumeError(RuntimeError):
    def __init__(self, message: str, *, failure_class: str = "volume_error") -> None:
        super().__init__(message)
        self.failure_class = failure_class


class LocalVolumeManager:
    """Manages local directory-backed persistent volumes with tar.gz backup support."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir = self.base_dir / ".backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_volume(
        self,
        *,
        deployment_id: str,
        hotkey: str,
        node_id: str,
        size_gb: int,
        volume_id: str | None = None,
    ) -> VolumeRecord:
        record = VolumeRecord(
            deployment_id=deployment_id,
            hotkey=hotkey,
            node_id=node_id,
            path=str(self._volume_path(deployment_id)),
            size_gb=size_gb,
            status="created",
        )
        if volume_id:
            record = record.model_copy(update={"volume_id": volume_id})
        vol_path = Path(record.path)
        vol_path.mkdir(parents=True, exist_ok=True)
        return record

    def delete_volume(self, volume: VolumeRecord) -> None:
        vol_path = Path(volume.path)
        if vol_path.exists():
            shutil.rmtree(str(vol_path), ignore_errors=True)

    def backup_volume(self, volume: VolumeRecord) -> VolumeRecord:
        vol_path = Path(volume.path)
        if not vol_path.exists():
            raise VolumeError(
                f"volume path missing: {volume.path}",
                failure_class="volume_backup_failure",
            )
        backup_path = self.backup_dir / f"{volume.volume_id}.tar.gz"
        try:
            with tarfile.open(str(backup_path), "w:gz") as tar:
                tar.add(str(vol_path), arcname="volume")
        except (OSError, tarfile.TarError) as exc:
            raise VolumeError(
                f"backup failed: {exc}",
                failure_class="volume_backup_failure",
            ) from exc
        return volume.model_copy(
            update={
                "backup_uri": str(backup_path),
                "last_backed_up_at": _utcnow(),
                "status": "backed_up",
            }
        )

    def restore_volume(self, volume: VolumeRecord, backup_uri: str) -> VolumeRecord:
        backup_path = Path(backup_uri)
        if not backup_path.exists():
            raise VolumeError(
                f"backup archive not found: {backup_uri}",
                failure_class="volume_restore_failure",
            )
        vol_path = Path(volume.path)
        if vol_path.exists():
            shutil.rmtree(str(vol_path), ignore_errors=True)
        vol_path.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(str(backup_path), "r:gz") as tar:
                # Validate all members stay within vol_path (prevent path traversal)
                for member in tar.getmembers():
                    member_path = (vol_path / member.name).resolve()
                    if not str(member_path).startswith(str(vol_path.resolve())):
                        raise VolumeError(
                            f"unsafe path in archive: {member.name}",
                            failure_class="volume_restore_failure",
                        )
                tar.extractall(str(vol_path))  # noqa: S202
        except (OSError, tarfile.TarError) as exc:
            raise VolumeError(
                f"restore failed: {exc}",
                failure_class="volume_restore_failure",
            ) from exc
        return volume.model_copy(update={"status": "attached"})

    def _volume_path(self, deployment_id: str) -> Path:
        return self.base_dir / deployment_id
