"""Тесты плейграунда: то, что должно быть верно без поднятого стека.

Проверяются вещи, ошибка в которых портит именно достоверность результата:
опорные модели, честность отложенной оценки, редактирование прогона в
публичном режиме и хранилище.
"""
from __future__ import annotations

import math
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MBP_CORPORA_DIR", str(ROOT / "data" / "corpora"))


def test_baselines_order_makes_sense():
    from mbplay.experiments import common

    text = "abcabcabcabc" * 200
    stoi, _ = common.vocab_of(text)
    train, holdout = common.split_text(text, 0.1)
    base = common.baselines(train, holdout, stoi)
    # На идеально предсказуемой строке биграмма обязана быть много лучше
    # униграммы, а униграмма — не хуже равномерной.
    assert base["bigram"] < base["unigram"] <= base["uniform"] + 1e-9
    assert base["bigram"] < 0.1


def test_split_keeps_holdout_out_of_training():
    from mbplay.experiments import common

    text = "".join(chr(97 + i % 20) for i in range(5000))
    train, holdout = common.split_text(text, 0.2)
    assert len(train) + len(holdout) == len(text)
    assert text.startswith(train)
    assert text.endswith(holdout)
    assert len(holdout) > 200


def test_eval_loss_does_not_change_weights():
    import numpy as np
    from magicbrain import TextBrain
    from mbplay.experiments import common

    text = "the quick brown fox jumps over the lazy dog. " * 40
    stoi, _ = common.vocab_of(text)
    brain = TextBrain("30121033102301230112332100123", len(stoi))
    common.train_chunked(brain, text, stoi, 600, chunk=600)
    w_before = brain.w_slow.copy()
    r_before = brain.R.copy()
    common.eval_loss(brain, text, stoi)
    assert np.array_equal(brain.w_slow, w_before)
    assert np.array_equal(brain.R, r_before)


def test_eval_loss_is_finite_and_bounded():
    from magicbrain import TextBrain
    from mbplay.experiments import common

    text = "abcdefghij" * 200
    stoi, _ = common.vocab_of(text)
    brain = TextBrain("30121033102301230112332100123", len(stoi))
    loss = common.eval_loss(brain, text, stoi)
    # Необученная сеть не может быть заметно лучше равномерного распределения.
    assert 0.0 < loss < math.log(len(stoi)) + 1.0


def test_corpora_manifest_declares_provenance():
    from mbplay import corpora

    items = corpora.list_corpora()
    assert items, "манифест корпусов пуст"
    for c in items:
        assert c.data_class in {"D0", "D1", "D2"}, c.id
        assert c.license and c.license != "не указано", c.id
        assert len(c.sha256) == 64, c.id


def test_corpus_content_matches_declared_hash():
    import hashlib

    from mbplay import corpora

    for c in corpora.list_corpora():
        text = corpora.read_text(c.id)
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == c.sha256, c.id


def test_inline_text_is_marked_unknown_provenance():
    from mbplay import corpora

    _, source = corpora.resolve_text(None, "x" * 500, 1000)
    assert source["data_class"] == "unknown"
    assert source["corpus_id"] is None


def test_short_text_is_rejected():
    from mbplay import corpora

    with pytest.raises(ValueError):
        corpora.resolve_text(None, "коротко", 1000)


def test_store_roundtrip(tmp_path):
    os.environ["MBP_DATA_DIR"] = str(tmp_path)
    import importlib

    from mbplay import config as config_module

    importlib.reload(config_module)
    store = importlib.reload(importlib.import_module("mbplay.store"))

    run = {"run_id": "train-test-1", "kind": "train", "created_at": 1.0, "source": {}}
    store.save(run)
    assert store.load("train-test-1") == run
    assert any(r["run_id"] == "train-test-1" for r in store.list_runs())
    assert store.delete("train-test-1") is True
    assert store.load("train-test-1") is None


def test_experiment_request_rejects_unknown_field():
    """Неизвестный параметр должен быть ошибкой, а не молча исчезнуть."""
    import pydantic

    from mbplay.app import ExperimentRequest

    with pytest.raises(pydantic.ValidationError):
        ExperimentRequest(corpus_id="x", damage_targt="readout")


def test_experiment_request_carries_damage_target():
    from mbplay.app import ExperimentRequest

    req = ExperimentRequest(corpus_id="x", damage_target="readout")
    assert req.clean()["damage_target"] == "readout"
    with pytest.raises(Exception):
        ExperimentRequest(corpus_id="x", damage_target="everything")


def test_unknown_disclosure_value_is_refused():
    """Опечатка в режиме раскрытия не должна означать «показывать всё»."""
    import importlib

    from mbplay import config as config_module

    os.environ["MBP_DISCLOSURE"] = "puplic"
    try:
        # Перезагрузка модуля создаёт новый класс исключения, поэтому ловим
        # базовый ValueError, от которого он наследуется.
        with pytest.raises(ValueError, match="MBP_DISCLOSURE"):
            importlib.reload(config_module)
    finally:
        os.environ["MBP_DISCLOSURE"] = "internal"
        importlib.reload(config_module)


def test_public_mode_hides_corpus_paths(monkeypatch):
    """Пути корпуса D1 — абсолютные пути на хосте."""
    from mbplay import app as app_module

    monkeypatch.setattr(app_module.settings.__class__, "public_mode", property(lambda self: True))
    try:
        for c in app_module.list_corpora()["corpora"]:
            assert "paths" not in c, c["id"]
    finally:
        monkeypatch.undo()
    assert any("paths" in c for c in app_module.list_corpora()["corpora"])


def test_public_mode_refuses_deletion(monkeypatch):
    from fastapi import HTTPException

    from mbplay import app as app_module

    monkeypatch.setattr(app_module.settings.__class__, "public_mode", property(lambda self: True))
    try:
        with pytest.raises(HTTPException) as exc:
            app_module.delete_run("whatever")
        assert exc.value.status_code == 403
    finally:
        monkeypatch.undo()


def test_runs_are_rotated_by_max_runs(tmp_path):
    """MBP_MAX_RUNS должен что-то делать: раздел общий со всеми сервисами хоста."""
    import importlib
    import time

    os.environ["MBP_DATA_DIR"] = str(tmp_path)
    os.environ["MBP_MAX_RUNS"] = "3"
    try:
        importlib.reload(importlib.import_module("mbplay.config"))
        store = importlib.reload(importlib.import_module("mbplay.store"))
        store.ensure_dirs()
        for i in range(5):
            run_id = f"train-{i}"
            store.save({"run_id": run_id, "kind": "train"})
            store.model_path(run_id).write_bytes(b"x")
            time.sleep(0.01)
        left = sorted(p.stem for p in (tmp_path / "runs").glob("*.json"))
        assert left == ["train-2", "train-3", "train-4"]
        # Модель удаляется вместе с прогоном, иначе ротация не освобождает место.
        assert not store.model_path("train-0").exists()
        assert store.model_path("train-4").exists()
    finally:
        os.environ.pop("MBP_MAX_RUNS", None)
        os.environ["MBP_DATA_DIR"] = "/data"
        importlib.reload(importlib.import_module("mbplay.config"))
        importlib.reload(importlib.import_module("mbplay.store"))


def test_synthetic_source_entropy_matches_generator():
    """Заявленная энтропия источника должна совпадать с фактической.

    Раньше в манифест попадала только энтропия марковской цепи, а вставка
    разделителей — процесс, тоже несущий энтропию, — не учитывалась, и число
    было занижено примерно вдвое.
    """
    import importlib.util
    import json as _json

    spec = importlib.util.spec_from_file_location(
        "fetch_corpora", ROOT / "scripts" / "fetch_corpora.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    manifest = _json.loads((ROOT / "data" / "corpora" / "manifest.json").read_text(encoding="utf-8"))
    entry = next(i for i in manifest if i["id"] == "synthetic_markov")
    source = next(s for s in module.SOURCES if s["id"] == "synthetic_markov")

    _, details = module.synthetic_markov(source["limit"], source["seed"])
    analytic = module.entropy_per_char(details)
    assert abs(entry["source_entropy_nats_per_char"] - analytic) < 1e-3
    # Аналитика против эмпирики по фактически сделанным выборам.
    assert abs(details["empirical_nats_per_char"] - analytic) < 0.01


def test_publish_overlay_pins_basic_auth():
    """Basic-auth не должен сниматься переменной окружения."""
    overlay = (ROOT / "docker-compose.publish.yml").read_text(encoding="utf-8")
    middlewares = [ln for ln in overlay.splitlines() if ".middlewares=" in ln]
    assert middlewares, "лейбл middlewares пропал"
    # Имя middleware обязательно и не имеет значения по умолчанию: пустая или
    # забытая переменная должна ронять compose, а не поднимать стенд без пароля.
    assert all("MBP_AUTH_MIDDLEWARE:?" in ln for ln in middlewares)

    base = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "MBP_EDGE_NETWORK" not in base, (
        "в базовом файле не должно быть сети прокси: без решения о публикации "
        "стенд к ней не подключается"
    )

    # Открытый показ — отдельный файл, и в нём basic-auth нет намеренно.
    public = (ROOT / "docker-compose.public.yml").read_text(encoding="utf-8")
    assert not [ln for ln in public.splitlines() if ".middlewares=" in ln and "AUTH" in ln]


def test_public_mode_keeps_every_scenario(monkeypatch):
    """Нейрогенез скрывался, пока механизм был закрыт. Библиотека опубликована."""
    from mbplay import app as app_module

    monkeypatch.setattr(app_module.settings.__class__, "public_mode", property(lambda self: True))
    try:
        assert "neurogenesis" in app_module._enabled_experiments()
    finally:
        monkeypatch.undo()


def test_public_mode_serves_neurogenesis_runs(monkeypatch):
    """Прогон нейрогенеза виден в списке и по прямой ссылке, геном не вырезается."""
    from mbplay import app as app_module

    summaries = [
        {"run_id": "a", "kind": "train", "headline": {"genome": "0123"}},
        {"run_id": "b", "kind": "neurogenesis", "headline": {"genome": "0123"}},
    ]
    run = {"run_id": "b", "kind": "neurogenesis", "network": {"N": 1216, "genome": "0123"},
           "stages": [{"name": "compile"}]}
    monkeypatch.setattr(app_module.store, "list_runs", lambda *a, **k: summaries)
    monkeypatch.setattr(app_module.store, "load", lambda rid: run if rid == "b" else None)
    monkeypatch.setattr(app_module.settings.__class__, "public_mode", property(lambda self: True))
    try:
        out = app_module.list_runs()
        assert [r["run_id"] for r in out["runs"]] == ["a", "b"]
        served = app_module.get_run("b")
        assert served["network"]["genome"] == "0123"
        assert served["stages"] == [{"name": "compile"}]
    finally:
        monkeypatch.undo()


def test_public_mode_serves_jobs_in_full(monkeypatch):
    """Задача отдаёт и ход выполнения, и заголовок: скрывать в нём нечего."""
    from mbplay import app as app_module

    job = {"job_id": "job-1", "kind": "neurogenesis", "status": "done", "progress": 1.0,
           "message": "готово", "run_id": "r1", "error": None,
           "headline": {"genome": "0123"}, "params": {"strategy": "statistical"}}
    monkeypatch.setattr(app_module.runner, "get", lambda jid: dict(job))
    monkeypatch.setattr(app_module.runner, "list", lambda limit=50: [dict(job)])
    monkeypatch.setattr(app_module.settings.__class__, "public_mode", property(lambda self: True))
    try:
        out = app_module.get_job("job-1")
        assert out["headline"]["genome"] == "0123"
        assert out["status"] == "done"
        assert app_module.list_jobs()["jobs"][0]["params"]["strategy"] == "statistical"
    finally:
        monkeypatch.undo()
