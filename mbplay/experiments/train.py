"""Сценарий (а): обучить TextBrain на выбранном тексте и увидеть результат.

Результат — не «loss упал», а ответ на проверяемый вопрос: обгоняет ли сеть
униграмму и биграмму на отложенном хвосте того же текста.
"""

from __future__ import annotations

import time
from typing import Any

from magicbrain import TextBrain, sample, save_model

from .. import store
from ..config import settings
from ..corpora import resolve_text
from . import common

DEFAULT_GENOME = "30121033102301230112332100123"


def run(params: dict[str, Any], progress) -> dict[str, Any]:
    text, source = resolve_text(
        params.get("corpus_id"), params.get("text"), params.get("max_chars", 60_000)
    )
    steps = min(int(params.get("steps", 20_000)), settings.MAX_STEPS)
    genome = (params.get("genome") or DEFAULT_GENOME).strip()
    use_act = bool(params.get("use_act", False))
    seed = params.get("seed")
    seed = int(seed) if seed not in (None, "") else None
    holdout_frac = float(params.get("holdout_frac", 0.1))
    seed_text = params.get("seed_text") or text[:40]
    sample_chars = int(params.get("sample_chars", 400))

    run_id = store.new_run_id("train")
    t_start = time.time()

    progress(0.02, "готовим словарь и отложенный хвост")
    stoi, itos = common.vocab_of(text)
    train_text, holdout_text = common.split_text(text, holdout_frac)

    progress(0.05, "выращиваем сеть из генома")
    brain = TextBrain(genome, len(stoi), seed_override=seed, use_act=use_act)
    facts = common.brain_facts(brain)

    progress(0.08, "оценка до обучения")
    loss_before = common.eval_loss(brain, holdout_text, stoi)

    chunk = max(500, steps // 40)
    holdout_curve: list[dict[str, Any]] = []

    def on_chunk(done: int, loss: float, point: dict[str, Any]) -> None:
        frac = 0.10 + 0.75 * (done / steps)
        progress(frac, f"обучение {done}/{steps}, loss {loss:.3f}")
        # Отложенную оценку считаем редко: она сама по себе стоит времени.
        if done % max(chunk, steps // 8) == 0 or done == steps:
            holdout_curve.append({
                "step": done,
                "holdout_nats": common.eval_loss(brain, holdout_text, stoi),
            })

    t_train = time.time()
    curve = common.train_chunked(brain, train_text, stoi, steps, chunk=chunk, on_chunk=on_chunk)
    train_seconds = time.time() - t_train

    progress(0.88, "оценка после обучения")
    loss_after = common.eval_loss(brain, holdout_text, stoi)
    base = common.baselines(train_text, holdout_text, stoi)

    progress(0.93, "генерируем образец текста")
    generated = sample(brain, stoi, itos, seed=seed_text, n=sample_chars, temperature=0.75)

    progress(0.97, "сохраняем модель")
    store.ensure_dirs()
    model_path = store.model_path(run_id)
    save_model(brain, stoi, itos, str(model_path))

    run = {
        "run_id": run_id,
        "kind": "train",
        "created_at": time.time(),
        "wall_time_sec": time.time() - t_start,
        "source": source,
        "params": {
            "genome": genome, "steps": steps, "seed": seed, "use_act": use_act,
            "holdout_frac": holdout_frac, "sample_chars": sample_chars,
        },
        "network": facts,
        "model_id": run_id,
        "model_bytes": model_path.stat().st_size,
        "curve": curve,
        "holdout_curve": holdout_curve,
        "metrics": {
            "holdout_nats_before": loss_before,
            "holdout_nats_after": loss_after,
            "holdout_bits_before": common.to_bits(loss_before),
            "holdout_bits_after": common.to_bits(loss_after),
            "train_loss_first": curve[0]["loss"] if curve else None,
            "train_loss_last": curve[-1]["loss"] if curve else None,
            "baselines_nats": base,
            "baselines_bits": {k: common.to_bits(v) for k, v in base.items()},
            "beats_unigram": loss_after < base["unigram"],
            "beats_bigram": loss_after < base["bigram"],
            "train_seconds": train_seconds,
            "steps_per_second": steps / train_seconds if train_seconds else None,
        },
        "sample": generated,
        "code": common.code_block(),
    }
    run["headline"] = {
        "holdout_bits_after": run["metrics"]["holdout_bits_after"],
        "bigram_bits": common.to_bits(base["bigram"]),
        "beats_bigram": run["metrics"]["beats_bigram"],
        "steps": steps,
        "N": facts["N"],
    }
    store.save(run)
    return run
