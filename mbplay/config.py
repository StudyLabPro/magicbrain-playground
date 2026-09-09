"""Настройки плейграунда. Всё берётся из окружения, значений по умолчанию хватает
для запуска без единой переменной."""

from __future__ import annotations

import os
import pathlib


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


DISCLOSURE_MODES = ("internal", "public")


class InvalidDisclosure(ValueError):
    """Режим раскрытия задан значением, которого не существует."""


def _disclosure() -> str:
    """Единственная задача переключателя — закрыть закрытую зону.

    Поэтому опечатка не должна означать «показывать всё»: раньше любое
    значение, кроме точного 'public', молча давало режим internal со всеми
    геномами. Теперь неизвестное значение — отказ на старте.
    """
    value = os.environ.get("MBP_DISCLOSURE", "internal").strip().lower()
    if value not in DISCLOSURE_MODES:
        raise InvalidDisclosure(
            f"MBP_DISCLOSURE={value!r}: допустимы только {', '.join(DISCLOSURE_MODES)}"
        )
    return value


class Settings:
    # Каталоги
    DATA_DIR = pathlib.Path(os.environ.get("MBP_DATA_DIR", "/data"))
    CORPORA_DIR = pathlib.Path(os.environ.get("MBP_CORPORA_DIR", "/corpora"))

    # Сколько одновременных прогонов держим. Контейнеру выделено 3 CPU,
    # каждый прогон однопоточный по numpy-ядру, но ACT-сравнение считает
    # два обучения последовательно внутри одной задачи.
    MAX_WORKERS = _int("MBP_MAX_WORKERS", 2)

    # Потолки, чтобы одна задача не съела весь контейнер.
    MAX_STEPS = _int("MBP_MAX_STEPS", 200_000)
    MAX_CHARS = _int("MBP_MAX_CHARS", 200_000)
    # Сколько прогонов держим на диске. Прогон — JSON плюс модель 0.25–0.7 MB;
    # раздел общий со всеми сервисами хоста, поэтому ротация обязательна.
    MAX_RUNS = _int("MBP_MAX_RUNS", 500)

    # Режим раскрытия. "internal" — экземпляр для своих: можно удалять прогоны
    # и видно, где корпуса лежат на диске. "public" — открытый показ: сценарии
    # и прогоны те же, но стенд только показывает, а не редактирует, и путей
    # хоста не отдаёт.
    DISCLOSURE = _disclosure()

    # Соседний сервис MagicBrain API (тот самый api/ из репозитория).
    MB_API_URL = os.environ.get("MBP_MB_API_URL", "http://magicbrain-api:8000")

    # Версия кода MagicBrain, из которой собран образ (пишет prepare-build.sh).
    MB_CODE_SHA = os.environ.get("MBP_MB_CODE_SHA", "unknown")
    MB_CODE_DIRTY = os.environ.get("MBP_MB_CODE_DIRTY", "unknown")
    BUILD_ID = os.environ.get("MBP_BUILD_ID", "dev")
    PLAYGROUND_SHA = os.environ.get("MBP_PLAYGROUND_SHA", "unknown")
    PLAYGROUND_DIRTY = os.environ.get("MBP_PLAYGROUND_DIRTY", "unknown")

    @property
    def runs_dir(self) -> pathlib.Path:
        return self.DATA_DIR / "runs"

    @property
    def models_dir(self) -> pathlib.Path:
        # Тот же каталог, что MODEL_STORAGE_PATH сервиса magicbrain-api,
        # поэтому обученная здесь модель сразу видна его /api/v1/models.
        return self.DATA_DIR / "models"

    @property
    def public_mode(self) -> bool:
        return self.DISCLOSURE == "public"


settings = Settings()
