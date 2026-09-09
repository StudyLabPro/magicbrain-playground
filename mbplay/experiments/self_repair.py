"""Сценарий (в): бенчмарк самовосстановления.

Убираем долю весов и смотрим, насколько сеть возвращается к прежнему качеству,
дообучаясь на том же тексте. Каждая доля повреждения стартует с одной и той же
обученной сети: модель сохраняется на диск и перечитывается перед каждым
повреждением, поэтому доли сравнимы между собой.

Точность перечитывания обеспечивает исправление в magicbrain/io.py: топология
после prune/rewire теперь сохраняется вместе с весами. До него перечитанная
модель получала веса от одной сети на рёбра другой.

**Цель повреждения выбирается.** Штатный `TextBrain.damage_edges` обнуляет
рекуррентные синапсы — и первый же прогон показал, что предсказание от этого
почти не страдает: работу несёт линейный readout `R`, которого повреждение не
касается. Поэтому здесь есть три режима: `recurrent` (как в библиотеке),
`readout` (обнуляем долю весов `R`) и `both`. Без этого выбора «бенчмарк
самовосстановления» измеряет устойчивость к удалению того, что и так почти не
влияет на ответ.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from magicbrain import TextBrain, load_model, save_model

from .. import store
from ..config import settings
from ..corpora import resolve_text
from . import common
from .train import DEFAULT_GENOME


def run(params: dict[str, Any], progress) -> dict[str, Any]:
    base_run_id = params.get("base_run_id")
    damage_target = str(params.get("damage_target", "recurrent")).lower()
    if damage_target not in ("recurrent", "readout", "both"):
        raise ValueError("damage_target: recurrent | readout | both")
    damage_fracs = params.get("damage_fracs") or [0.1, 0.2, 0.4]
    damage_fracs = [float(f) for f in damage_fracs][:5]
    recovery_steps = min(int(params.get("recovery_steps", 8_000)), settings.MAX_STEPS)
    report_every = max(500, int(params.get("report_every", recovery_steps // 8)))

    run_id = store.new_run_id("self-repair")
    t_start = time.time()
    store.ensure_dirs()

    if base_run_id:
        base = store.load(base_run_id)
        if base is None or not base.get("model_id"):
            raise ValueError(f"прогон {base_run_id} не найден или без модели")
        model_file = store.model_path(base["model_id"])
        if not model_file.exists():
            raise ValueError(f"файл модели для {base_run_id} отсутствует")
        source = base["source"]
        if not source.get("corpus_id") and not params.get("text"):
            raise ValueError(
                f"прогон {base_run_id} обучался на присланном тексте, который "
                "не сохраняется; передайте тот же текст полем text"
            )
        text, _ = resolve_text(
            source.get("corpus_id"), None if source.get("corpus_id") else params.get("text"),
            source.get("used_chars", 60_000),
        )
        progress(0.05, f"берём обученную сеть из прогона {base_run_id}")
        pretrain_steps = base.get("params", {}).get("steps")
        genome = base.get("network", {}).get("genome")
    else:
        text, source = resolve_text(
            params.get("corpus_id"), params.get("text"), params.get("max_chars", 60_000)
        )
        genome = (params.get("genome") or DEFAULT_GENOME).strip()
        pretrain_steps = min(int(params.get("pretrain_steps", 15_000)), settings.MAX_STEPS)
        progress(0.05, "предварительное обучение сети")
        stoi_tmp, itos_tmp = common.vocab_of(text)
        train_text_tmp, _ = common.split_text(text, float(params.get("holdout_frac", 0.1)))
        brain_tmp = TextBrain(genome, len(stoi_tmp))
        common.train_chunked(
            brain_tmp, train_text_tmp, stoi_tmp, pretrain_steps,
            on_chunk=lambda d, l, p: progress(
                0.05 + 0.25 * (d / pretrain_steps), f"предобучение {d}/{pretrain_steps}"
            ),
        )
        model_file = store.model_path(run_id)
        save_model(brain_tmp, stoi_tmp, itos_tmp, str(model_file))

    stoi, itos = common.vocab_of(text)
    train_text, holdout_text = common.split_text(text, float(params.get("holdout_frac", 0.1)))

    # Опорная точка: неповреждённая сеть на отложенном хвосте.
    reference, _, _, meta = load_model(str(model_file))
    if not meta.get("topology_restored", False):
        # Старый файл без топологии — считаем результат недостоверным и говорим об этом.
        topology_note = (
            "файл модели записан в старом формате без топологии: "
            "рёбра восстановлены из генома, результат ниже сравним лишь приблизительно"
        )
    else:
        topology_note = None
    pre_damage = common.eval_loss(reference, holdout_text, stoi)
    facts = common.brain_facts(reference)

    results = []
    total = len(damage_fracs)
    for i, frac in enumerate(damage_fracs):
        progress(0.35 + 0.6 * (i / total), f"повреждение {frac:.0%}")
        brain, _, _, _ = load_model(str(model_file))
        edges_total = int(brain.src.shape[0])
        readout_total = int(brain.R.size)
        nonzero_before = int((brain.w_slow != 0).sum()) + int((brain.R != 0).sum())
        if damage_target in ("recurrent", "both"):
            brain.damage_edges(frac)
        if damage_target in ("readout", "both"):
            _damage_readout(brain, frac)
        nonzero_after = int((brain.w_slow != 0).sum()) + int((brain.R != 0).sum())
        post_damage = common.eval_loss(brain, holdout_text, stoi)

        curve = []
        done = 0
        while done < recovery_steps:
            take = min(report_every, recovery_steps - done)
            common.train_chunked(brain, train_text, stoi, take, chunk=take)
            done += take
            curve.append({"step": done, "holdout_nats": common.eval_loss(brain, holdout_text, stoi)})
            progress(
                0.35 + 0.6 * ((i + done / recovery_steps) / total),
                f"повреждение {frac:.0%}: восстановление {done}/{recovery_steps}",
            )

        final = curve[-1]["holdout_nats"] if curve else post_damage
        gap = post_damage - pre_damage
        results.append({
            "damage_frac": frac,
            "damage_target": damage_target,
            "edges_total": edges_total,
            "readout_weights_total": readout_total,
            "weights_zeroed": nonzero_before - nonzero_after,
            "pre_damage_nats": pre_damage,
            "post_damage_nats": post_damage,
            "final_nats": final,
            "recovery_curve": curve,
            "recovery_ratio": float((post_damage - final) / gap) if abs(gap) > 1e-9 else None,
            "damage_cost_nats": post_damage - pre_damage,
            "residual_nats": final - pre_damage,
        })

    base_stats = common.baselines(train_text, holdout_text, stoi)

    run = {
        "run_id": run_id,
        "kind": "self-repair",
        "created_at": time.time(),
        "wall_time_sec": time.time() - t_start,
        "source": source,
        "params": {
            "base_run_id": base_run_id,
            "genome": genome,
            "pretrain_steps": pretrain_steps,
            "damage_fracs": damage_fracs,
            "damage_target": damage_target,
            "recovery_steps": recovery_steps,
            "report_every": report_every,
        },
        "network": facts,
        "model_id": run_id if not base_run_id else base_run_id,
        "metrics": {
            "pre_damage_nats": pre_damage,
            "baselines_nats": base_stats,
            "by_damage": results,
        },
        "caveats": [
            "Повреждение обнуляет ВЕСА, а не удаляет рёбра: структура графа не "
            "меняется, поэтому восстановление идёт по уже существующим связям.",
            "Цель повреждения — " + damage_target + ". Обнуление только "
            "рекуррентных синапсов почти не влияет на предсказание: его несёт "
            "линейный readout, который в этом режиме не трогается.",
            "Восстановление меряется на отложенном хвосте, который в дообучении "
            "не участвует.",
        ] + ([topology_note] if topology_note else []),
        "code": common.code_block(),
    }
    run["headline"] = {
        "pre_damage_nats": pre_damage,
        "damage_target": damage_target,
        "worst_damage": max(damage_fracs),
        "recovery_ratios": {str(r["damage_frac"]): r["recovery_ratio"] for r in results},
    }
    store.save(run)
    return run


def _damage_readout(brain, frac: float) -> None:
    """Обнуляет долю весов линейного readout.

    Именно readout превращает состояние сети в распределение по символам,
    поэтому повреждение здесь видно в качестве сразу — в отличие от
    рекуррентных синапсов.
    """
    frac = float(np.clip(frac, 0.0, 1.0))
    total = int(brain.R.size)
    count = int(total * frac)
    if count <= 0:
        return
    flat = brain.R.reshape(-1)
    idx = brain.rng.choice(total, size=count, replace=False)
    flat[idx] = 0.0
