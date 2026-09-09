"""Сценарий (г): сравнение с ACT-бэкендом Balansis и без него.

Меряются три разные вещи, и их важно не путать.

1. Обучение: одинаковый геном, одинаковый seed, одинаковый текст — один прогон
   с ``use_act=False``, второй с ``use_act=True``. Сравниваются качество на
   отложенном хвосте и время.
2. Дрейф компенсации во время обучения: ACTMetricsTracker показывает, насколько
   компенсированная сумма весов расходится с наивной.
3. Микробенчмарк арифметики на специально плохо обусловленных данных, где
   наивная сумма float ошибается заметно. Это и есть то, за что платят
   временем; на весах сети масштаба N~1000 разница обычно ниже уровня шума,
   и прогон это показывает, а не скрывает.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from magicbrain import TextBrain
from magicbrain.diagnostics.act_metrics import ACTMetricsTracker
from magicbrain.integration.act_backend import ACTBackend

from .. import store
from ..config import settings
from ..corpora import resolve_text
from . import common
from .train import DEFAULT_GENOME


def _microbenchmark() -> dict[str, Any]:
    """Плохо обусловленные суммы и скалярные произведения.

    Эталон — ``math.fsum`` (точное суммирование с округлением один раз).
    """
    act = ACTBackend()
    rng = np.random.default_rng(20260908)
    out: dict[str, Any] = {"act_available": bool(act.available)}

    n = 20_000
    # Классический плохой случай: большие величины разных знаков плюс мелочь.
    big = rng.normal(0, 1e8, size=n // 2)
    small = rng.normal(0, 1e-8, size=n // 2)
    arr = np.empty(n, dtype=np.float64)
    arr[0::2] = big
    arr[1::2] = small
    arr = np.concatenate([arr, -big])  # сумма больших членов взаимно уничтожается

    exact = math.fsum(arr.tolist())
    t0 = time.perf_counter()
    naive = float(np.sum(arr))
    t_naive = time.perf_counter() - t0
    t0 = time.perf_counter()
    compensated = float(act.kahan_sum(arr)) if act.available else naive
    t_comp = time.perf_counter() - t0

    scale = max(abs(exact), 1e-30)
    out["sum"] = {
        "n": int(arr.size),
        "exact_fsum": exact,
        "naive_numpy": naive,
        "act_kahan": compensated,
        "naive_abs_error": abs(naive - exact),
        "act_abs_error": abs(compensated - exact),
        "naive_rel_error": abs(naive - exact) / scale,
        "act_rel_error": abs(compensated - exact) / scale,
        "naive_seconds": t_naive,
        "act_seconds": t_comp,
    }

    m = 5_000
    a = rng.normal(0, 1e6, size=m)
    b = rng.normal(0, 1e-6, size=m)
    a = np.concatenate([a, a])
    b = np.concatenate([b, -b])  # точное значение скалярного произведения — 0
    exact_dot = math.fsum((a * b).tolist())
    t0 = time.perf_counter()
    naive_dot = float(np.dot(a, b))
    t_naive_dot = time.perf_counter() - t0
    t0 = time.perf_counter()
    act_dot = float(act.dot(a, b)) if act.available else naive_dot
    t_act_dot = time.perf_counter() - t0
    out["dot"] = {
        "n": int(a.size),
        "exact_fsum": exact_dot,
        "naive_numpy": naive_dot,
        "act_dot2": act_dot,
        "naive_abs_error": abs(naive_dot - exact_dot),
        "act_abs_error": abs(act_dot - exact_dot),
        "naive_seconds": t_naive_dot,
        "act_seconds": t_act_dot,
    }
    return out


def _one_side(
    use_act: bool, genome: str, text: str, train_text: str, holdout_text: str,
    stoi: dict, steps: int, seed: int | None, progress, offset: float,
) -> dict[str, Any]:
    brain = TextBrain(genome, len(stoi), seed_override=seed, use_act=use_act)
    tracker = ACTMetricsTracker(brain._act)
    before = common.eval_loss(brain, holdout_text, stoi)
    snapshots: list[dict[str, Any]] = []
    chunk = max(500, steps // 20)

    def on_chunk(done: int, loss: float, point: dict[str, Any]) -> None:
        snap = tracker.record(brain, done)
        snapshots.append({
            "step": done,
            "w_slow_l2": snap.w_slow_l2,
            "w_slow_compensation_ratio": snap.w_slow_compensation_ratio,
            "w_fast_compensation_ratio": snap.w_fast_compensation_ratio,
            "combined_weight_std": snap.combined_weight_std,
        })
        label = "ACT" if use_act else "numpy"
        progress(offset + 0.35 * (done / steps), f"{label}: {done}/{steps}, loss {loss:.3f}")

    t0 = time.time()
    curve = common.train_chunked(brain, train_text, stoi, steps, chunk=chunk, on_chunk=on_chunk)
    seconds = time.time() - t0
    after = common.eval_loss(brain, holdout_text, stoi)

    return {
        "use_act": use_act,
        "act_active": bool(brain._act is not None and brain._act.available),
        "seconds": seconds,
        "steps_per_second": steps / seconds if seconds else None,
        "holdout_nats_before": before,
        "holdout_nats_after": after,
        "train_loss_last": curve[-1]["loss"] if curve else None,
        "curve": curve,
        "act_snapshots": snapshots,
        "act_summary": tracker.summary(),
        "final_w_slow_l2": float(np.linalg.norm(brain.w_slow)),
    }


def run(params: dict[str, Any], progress) -> dict[str, Any]:
    text, source = resolve_text(
        params.get("corpus_id"), params.get("text"), params.get("max_chars", 40_000)
    )
    steps = min(int(params.get("steps", 10_000)), settings.MAX_STEPS)
    genome = (params.get("genome") or DEFAULT_GENOME).strip()
    seed = params.get("seed", 20260908)
    seed = int(seed) if seed not in (None, "") else None

    run_id = store.new_run_id("act")
    t_start = time.time()

    stoi, itos = common.vocab_of(text)
    train_text, holdout_text = common.split_text(text, float(params.get("holdout_frac", 0.1)))

    progress(0.03, "микробенчмарк арифметики")
    micro = _microbenchmark()

    progress(0.08, "прогон без ACT")
    plain = _one_side(False, genome, text, train_text, holdout_text, stoi, steps, seed, progress, 0.08)
    progress(0.5, "прогон с ACT")
    with_act = _one_side(True, genome, text, train_text, holdout_text, stoi, steps, seed, progress, 0.5)

    base = common.baselines(train_text, holdout_text, stoi)
    d_quality = with_act["holdout_nats_after"] - plain["holdout_nats_after"]
    slowdown = (
        with_act["seconds"] / plain["seconds"] if plain["seconds"] else None
    )

    run = {
        "run_id": run_id,
        "kind": "act",
        "created_at": time.time(),
        "wall_time_sec": time.time() - t_start,
        "source": source,
        "params": {"genome": genome, "steps": steps, "seed": seed},
        "sides": {"numpy": plain, "act": with_act},
        "microbenchmark": micro,
        "metrics": {
            "baselines_nats": base,
            "holdout_delta_nats": d_quality,
            "holdout_delta_relative": (
                d_quality / plain["holdout_nats_after"] if plain["holdout_nats_after"] else None
            ),
            "act_slowdown_x": slowdown,
        },
        "caveats": [
            "TextBrain держит веса в float32, а ACT считает в float64 и возвращает "
            "результат обратно в float32 — часть выигранной точности теряется на "
            "обратном приведении.",
            "matvec_add в ACTBackend только повышает разрядность и НЕ компенсирует "
            "суммирование (так написано в самом коде), поэтому основной путь "
            "forward компенсирован лишь частично.",
            "Разница качества на этом масштабе сопоставима с шумом инициализации; "
            "именно поэтому в прогоне есть микробенчмарк — он показывает, где "
            "компенсация действительно меняет ответ.",
        ],
        "code": common.code_block(balansis_version=_balansis_version()),
    }
    run["headline"] = {
        "act_available": with_act["act_active"],
        "slowdown_x": slowdown,
        "holdout_delta_nats": d_quality,
        "sum_error_ratio": (
            micro["sum"]["naive_abs_error"] / micro["sum"]["act_abs_error"]
            if micro.get("sum", {}).get("act_abs_error") else None
        ),
    }
    store.save(run)
    return run


def _balansis_version() -> str:
    try:
        import balansis
        return str(getattr(balansis, "__version__", "?"))
    except Exception:  # noqa: BLE001
        return "недоступен"
