"""Общая механика экспериментов: разбиение корпуса, честная оценка и базовые
модели, с которыми сравнивается сеть.

Ключевое решение: словарь строится по ВСЕМУ тексту, а loss меряется на
отложенном хвосте, который в обучении не участвует. Иначе «результат обучения»
сводится к запоминанию тренировочной строки и ничего не говорит.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from typing import Any, Callable

import numpy as np

from magicbrain import TextBrain
from magicbrain.tasks.text_task import build_vocab

from ..config import settings

NATS_TO_BITS = 1.0 / math.log(2.0)


def split_text(text: str, holdout_frac: float = 0.1) -> tuple[str, str]:
    """Отрезает хвост под отложенную оценку."""
    holdout_frac = float(min(max(holdout_frac, 0.02), 0.5))
    cut = int(len(text) * (1.0 - holdout_frac))
    cut = max(200, min(cut, len(text) - 200))
    return text[:cut], text[cut:]


def vocab_of(text: str) -> tuple[dict, dict]:
    return build_vocab(text)


def train_chunked(
    brain: TextBrain,
    text: str,
    stoi: dict,
    steps: int,
    chunk: int = 1000,
    on_chunk: Callable[[int, float, dict], None] | None = None,
) -> list[dict[str, Any]]:
    """Обучение с телеметрией.

    Возвращает точки кривой: шаг, средний loss на участке и состояние субстрата
    (частота разрядов, средний порог, средняя |W|, дофамин).
    """
    ids = np.asarray([stoi[c] for c in text], dtype=np.int32)
    if ids.size < 2:
        raise ValueError("текст слишком короткий")
    n = ids.size - 1
    curve: list[dict[str, Any]] = []
    done = 0
    idx = brain.step % n

    while done < steps:
        take = min(chunk, steps - done)
        losses = np.empty(take, dtype=np.float64)
        for i in range(take):
            x = int(ids[idx])
            y = int(ids[(idx + 1) % n])
            probs = brain.forward(x)
            losses[i] = brain.learn(y, probs)
            idx = (idx + 1) % n
        done += take
        point = {
            "step": done,
            "loss": float(np.mean(losses)),
            "firing_rate": float(brain.firing_rate()),
            "avg_theta": float(brain.avg_theta()),
            "mean_abs_w": float(brain.mean_abs_w()),
            "dopamine": float(brain.dopamine),
        }
        curve.append(point)
        if on_chunk:
            on_chunk(done, point["loss"], point)
    return curve


def eval_loss(brain: TextBrain, text: str, stoi: dict, warmup: int = 50) -> float:
    """Средний -log p(следующий символ) без обучения, в натах.

    Веса не трогаются: ``forward`` только считает, ``learn`` не вызывается.
    Первые ``warmup`` шагов прогоняются, чтобы задержечная линия и трассы
    успели наполниться, и в счёт не идут.
    """
    ids = [stoi[c] for c in text if c in stoi]
    if len(ids) < warmup + 2:
        raise ValueError("текст для оценки слишком короткий")
    brain.reset_state()
    for i in range(warmup):
        brain.forward(ids[i])
    total = 0.0
    count = 0
    for i in range(warmup, len(ids) - 1):
        probs = brain.forward(ids[i])
        total += -math.log(float(probs[ids[i + 1]]) + 1e-12)
        count += 1
    return total / max(1, count)


def baselines(train_text: str, holdout_text: str, stoi: dict) -> dict[str, float]:
    """Опорные модели на том же отложенном тексте, в натах на символ.

    - uniform: равномерное распределение по словарю, ln(V);
    - unigram: частоты символов обучающей части;
    - bigram: марковская модель первого порядка с аддитивным сглаживанием.

    Сеть имеет смысл только если она бьёт unigram; bigram — честная планка,
    которую посимвольная модель обязана перерасти, чтобы называться моделью
    последовательности.
    """
    V = len(stoi)
    uniform = math.log(V) if V > 1 else 0.0

    unigram_counts = Counter(c for c in train_text if c in stoi)
    total = sum(unigram_counts.values())
    alpha = 1.0
    denom = total + alpha * V

    bigram_counts: dict[str, Counter] = {}
    prev = None
    for ch in train_text:
        if ch not in stoi:
            prev = None
            continue
        if prev is not None:
            bigram_counts.setdefault(prev, Counter())[ch] += 1
        prev = ch

    uni_sum = 0.0
    bi_sum = 0.0
    count = 0
    prev = None
    for ch in holdout_text:
        if ch not in stoi:
            prev = None
            continue
        if prev is not None:
            p_uni = (unigram_counts.get(ch, 0) + alpha) / denom
            row = bigram_counts.get(prev)
            if row is None:
                p_bi = 1.0 / V
            else:
                p_bi = (row.get(ch, 0) + alpha) / (sum(row.values()) + alpha * V)
            uni_sum += -math.log(p_uni)
            bi_sum += -math.log(p_bi)
            count += 1
        prev = ch

    if count == 0:
        return {"uniform": uniform, "unigram": uniform, "bigram": uniform}
    return {
        "uniform": uniform,
        "unigram": uni_sum / count,
        "bigram": bi_sum / count,
    }


def to_bits(nats: float | None) -> float | None:
    return None if nats is None else float(nats) * NATS_TO_BITS


def brain_facts(brain: TextBrain) -> dict[str, Any]:
    """Что за сеть на самом деле выросла из генома."""
    return {
        "genome": brain.genome_str,
        "N": int(brain.N),
        "K": int(brain.K),
        "edges": int(brain.src.shape[0]),
        "vocab_size": int(brain.vocab_size),
        "k_active": int(brain.p["k_active"]),
        "target_rate": float(brain.target_rate),
        "p_inhib": float(brain.p["p_inhib"]),
        "prune_every": int(brain.p["prune_every"]),
        "sens_fanout": int(brain.sens_fanout),
        "delay_histogram": {
            str(d): int(brain.idx_by_delay[d].size) for d in range(1, 6)
        },
    }


class Stopwatch:
    def __init__(self) -> None:
        self._marks: dict[str, float] = {}
        self._t0 = time.time()

    def mark(self, name: str) -> float:
        now = time.time()
        elapsed = now - self._t0
        self._marks[name] = elapsed
        self._t0 = now
        return elapsed

    @property
    def marks(self) -> dict[str, float]:
        return dict(self._marks)


def code_block(**extra) -> dict:
    """Происхождение кода, из которого получены числа прогона.

    Пишется в каждый прогон целиком, чтобы шапку docs/RUNS.md можно было
    собрать из самих прогонов: раньше она бралась из /api/meta, снятого один
    раз в начале сценария, и после переигровки одного прогона другой сборкой
    описывала не тот код.
    """
    import numpy

    block = {
        "magicbrain_sha": settings.MB_CODE_SHA,
        "magicbrain_dirty": settings.MB_CODE_DIRTY,
        "build_id": settings.BUILD_ID,
        "playground_sha": settings.PLAYGROUND_SHA,
        "playground_dirty": settings.PLAYGROUND_DIRTY,
        "numpy_version": numpy.__version__,
    }
    block.update(extra)
    return block
