"""HTTP-слой плейграунда.

Отдаёт статический интерфейс и тонкий API поверх очереди задач и хранилища
прогонов. Тяжёлого кода здесь нет — вся работа в mbplay/experiments.
"""

from __future__ import annotations

import copy
import json
import pathlib
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from . import __version__, corpora, store
from .config import settings
from .experiments import REGISTRY
from .jobs import runner


@asynccontextmanager
async def lifespan(_: FastAPI):
    store.ensure_dirs()
    yield


app = FastAPI(
    title="MagicBrain Playground",
    version=__version__,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

UI_DIR = pathlib.Path(__file__).parent / "ui"


class ExperimentRequest(BaseModel):
    # extra="forbid" намеренно: неизвестное поле должно быть ошибкой, а не тихо
    # исчезнуть. Один раз это уже стоило прогона — параметр damage_target не был
    # объявлен здесь, запрос принимался, и оба прогона повреждали одно и то же.
    model_config = ConfigDict(extra="forbid")

    corpus_id: str | None = None
    text: str | None = None
    max_chars: int | None = Field(default=None, ge=200)
    steps: int | None = Field(default=None, ge=200)
    genome: str | None = None
    seed: int | None = None
    use_act: bool | None = None
    holdout_frac: float | None = Field(default=None, gt=0.01, lt=0.5)
    sample_chars: int | None = Field(default=None, ge=0, le=4000)
    seed_text: str | None = None
    # нейрогенез
    strategy: str | None = None
    genome_length: int | None = Field(default=None, ge=24, le=64)
    use_cppn: bool | None = None
    reconstruct_length: int | None = Field(default=None, ge=50, le=4000)
    attractor_probes: int | None = Field(default=None, ge=0, le=500)
    # самовосстановление
    base_run_id: str | None = None
    damage_target: str | None = Field(default=None, pattern="^(recurrent|readout|both)$")
    pretrain_steps: int | None = Field(default=None, ge=200)
    damage_fracs: list[float] | None = None
    recovery_steps: int | None = Field(default=None, ge=200)
    report_every: int | None = None

    def clean(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "healthy", "service": "magicbrain-playground", "version": __version__}


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    import numpy

    return {
        "version": __version__,
        "disclosure": settings.DISCLOSURE,
        "public_mode": settings.public_mode,
        "enabled_experiments": _enabled_experiments(),
        "limits": {
            "max_steps": settings.MAX_STEPS,
            "max_chars": settings.MAX_CHARS,
            "max_workers": settings.MAX_WORKERS,
        },
        "code": {
            "magicbrain_sha": settings.MB_CODE_SHA,
            "magicbrain_dirty": settings.MB_CODE_DIRTY,
            "build_id": settings.BUILD_ID,
            "playground_sha": settings.PLAYGROUND_SHA,
            "playground_dirty": settings.PLAYGROUND_DIRTY,
            "numpy": numpy.__version__,
            "balansis": _balansis_version(),
        },
        "magicbrain_api": settings.MB_API_URL,
    }


def _balansis_version() -> str:
    try:
        import balansis
        return str(getattr(balansis, "__version__", "?"))
    except Exception:  # noqa: BLE001
        return "недоступен"


def _enabled_experiments() -> list[str]:
    # Все сценарии доступны в обоих режимах. Нейрогенез скрывался, пока
    # механизм был закрыт; библиотека опубликована под AGPL, и скрывать в
    # демонстрации то, что лежит в открытом репозитории, нечего.
    return list(REGISTRY.keys())


def _corpus_view(corpus: corpora.Corpus) -> dict[str, Any]:
    """Публичный режим не показывает абсолютные пути файлов на хосте."""
    out = corpus.as_dict()
    if settings.public_mode:
        out.pop("paths", None)
    return out


@app.get("/api/corpora")
def list_corpora() -> dict[str, Any]:
    return {"corpora": [_corpus_view(c) for c in corpora.list_corpora()]}


@app.get("/api/corpora/{corpus_id}/preview")
def preview_corpus(corpus_id: str, chars: int = 1200) -> dict[str, Any]:
    corpus = corpora.get_corpus(corpus_id)
    if corpus is None:
        raise HTTPException(404, f"неизвестный корпус: {corpus_id}")
    return {
        "corpus": _corpus_view(corpus),
        "preview": corpora.read_text(corpus_id, max(200, min(chars, 8000))),
    }


@app.post("/api/experiments/{kind}", status_code=202)
def start_experiment(kind: str, request: ExperimentRequest) -> dict[str, Any]:
    if kind not in REGISTRY:
        raise HTTPException(404, f"неизвестный сценарий: {kind}")
    if kind not in _enabled_experiments():
        raise HTTPException(
            403,
            "сценарий выключен в текущем режиме раскрытия "
            f"({settings.DISCLOSURE}); см. README, раздел про ip-publication-gate",
        )
    params = request.clean()
    fn = REGISTRY[kind]
    job = runner.submit(kind, lambda progress: fn(params, progress), params)
    return job


@app.get("/api/jobs")
def list_jobs(limit: int = 50) -> dict[str, Any]:
    return {"jobs": runner.list(limit)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = runner.get(job_id)
    if job is None:
        raise HTTPException(404, f"задача {job_id} не найдена")
    return job


@app.get("/api/runs")
def list_runs(kind: str | None = None, limit: int = 200) -> dict[str, Any]:
    return {"runs": store.list_runs(kind, limit)}


def _load_visible(run_id: str) -> dict[str, Any]:
    """Прогон по идентификатору. Содержимое прогонов одинаково в обоих режимах."""
    run = store.load(run_id)
    if run is None:
        raise HTTPException(404, f"прогон {run_id} не найден")
    return run


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    return _load_visible(run_id)


@app.get("/api/runs/{run_id}/export")
def export_run(run_id: str) -> Response:
    run = _load_visible(run_id)
    body = json.dumps(run, ensure_ascii=False, indent=2)
    return Response(
        content=body,
        media_type="application/json; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{run_id}.json"'},
    )


@app.delete("/api/runs/{run_id}", status_code=204)
def delete_run(run_id: str) -> Response:
    if settings.public_mode:
        # Публичный экземпляр показывает, а не редактирует: удалять чужие
        # прогоны через открытый эндпоинт не должен никто.
        raise HTTPException(403, "удаление прогонов выключено в публичном режиме")
    if not store.delete(run_id):
        raise HTTPException(404, f"прогон {run_id} не найден")
    return Response(status_code=204)


@app.post("/api/runs/{run_id}/publish")
def publish_model(run_id: str) -> dict[str, Any]:
    """Регистрирует модель прогона в соседнем сервисе magicbrain-api.

    Каталог моделей у сервисов общий, поэтому файл никуда не копируется —
    достаточно попросить runtime загрузить его.
    """
    run = _load_visible(run_id)
    if not run.get("model_id"):
        raise HTTPException(404, f"прогон {run_id} без модели")
    model_id = run["model_id"]
    url = f"{settings.MB_API_URL}/api/v1/runtime/models/{model_id}/load"
    try:
        resp = httpx.post(url, timeout=60.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"magicbrain-api недоступен: {exc}") from exc
    return {"model_id": model_id, "magicbrain_api": resp.json()}


class SampleRequest(BaseModel):
    model_id: str
    seed_text: str = "The "
    n_tokens: int = Field(default=200, ge=1, le=2000)
    # Нижняя граница та же, что в схеме mb-api: иначе значение из зазора
    # проходило валидацию здесь и возвращалось пользователю как 422 чужого
    # сервиса.
    temperature: float = Field(default=0.75, ge=0.1, le=2.0)


@app.post("/api/sample")
def sample_from_api(request: SampleRequest) -> dict[str, Any]:
    """Генерация через соседний сервис magicbrain-api — доказательство, что
    сохранённая здесь модель действительно читается штатным рантаймом."""
    url = f"{settings.MB_API_URL}/api/v1/inference/sample"
    payload = {
        "model_id": request.model_id,
        "seed_text": request.seed_text,
        "n_tokens": request.n_tokens,
        "temperature": request.temperature,
    }
    try:
        resp = httpx.post(url, json=payload, timeout=120.0)
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"magicbrain-api недоступен: {exc}") from exc
    return resp.json()


@app.get("/api/magicbrain/models")
def magicbrain_models() -> dict[str, Any]:
    try:
        resp = httpx.get(f"{settings.MB_API_URL}/api/v1/models/", timeout=30.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"magicbrain-api недоступен: {exc}") from exc
    return resp.json()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")
