"""Хранилище прогонов.

Один прогон — один JSON-файл в ``$MBP_DATA_DIR/runs``. Артефакт модели (npz)
лежит в ``models`` рядом, под тем же идентификатором, поэтому его видит и
сервис magicbrain-api. Никакой базы: прогонов немного, а файл переживает
перезапуск контейнера и его можно скопировать целиком.
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
import time
import uuid
from typing import Any

from .config import settings


def new_run_id(kind: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return f"{kind}-{stamp}-{uuid.uuid4().hex[:6]}"


def _path(run_id: str) -> pathlib.Path:
    return settings.runs_dir / f"{run_id}.json"


def _newest_first() -> list[pathlib.Path]:
    """Свежие сверху — по времени файла, а не по имени.

    Имя прогона начинается с вида сценария (``act-``, ``train-``), поэтому
    сортировка по имени перемешала бы прогоны разных сценариев и ротация
    удаляла бы не самые старые, а те, чьё имя раньше по алфавиту.
    """
    files = list(settings.runs_dir.glob("*.json"))
    return sorted(files, key=lambda p: (p.stat().st_mtime, p.name), reverse=True)


def ensure_dirs() -> None:
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)


def prune(keep: int | None = None) -> list[str]:
    """Оставляет ``MBP_MAX_RUNS`` самых свежих прогонов, остальные удаляет.

    Каталог состояния лежит на общем разделе хоста, а каждый прогон — это ещё
    и модель на 0.25–0.7 MB. Без ротации настройка MAX_RUNS была бы мёртвой, а
    каталог рос бы неограниченно.
    """
    limit = settings.MAX_RUNS if keep is None else keep
    if limit <= 0:
        return []
    files = _newest_first()
    removed = []
    for path in files[limit:]:
        run_id = path.stem
        if delete(run_id):
            removed.append(run_id)
    return removed


def save(run: dict[str, Any]) -> None:
    ensure_dirs()
    path = _path(run["run_id"])
    # Пишем через временный файл: половина JSON на диске хуже, чем его отсутствие.
    fd, tmp = tempfile.mkstemp(dir=str(settings.runs_dir), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(run, fh, ensure_ascii=False, indent=2, sort_keys=False)
        os.replace(tmp, path)
    except Exception:
        pathlib.Path(tmp).unlink(missing_ok=True)
        raise
    prune()


def load(run_id: str) -> dict[str, Any] | None:
    path = _path(run_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def delete(run_id: str) -> bool:
    path = _path(run_id)
    existed = path.exists()
    path.unlink(missing_ok=True)
    (settings.models_dir / f"{run_id}.npz").unlink(missing_ok=True)
    return existed


def summary(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run.get("run_id"),
        "kind": run.get("kind"),
        "created_at": run.get("created_at"),
        "wall_time_sec": run.get("wall_time_sec"),
        "source": run.get("source", {}),
        "headline": run.get("headline", {}),
        "code": run.get("code", {}),
        "has_model": bool(run.get("model_id")),
    }


def list_runs(kind: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    ensure_dirs()
    items = []
    for path in _newest_first():
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if kind and run.get("kind") != kind:
            continue
        items.append(summary(run))
        if len(items) >= limit:
            break
    return items


def model_path(run_id: str) -> pathlib.Path:
    return settings.models_dir / f"{run_id}.npz"
