"""Сценарий (б): пайплайн нейрогенеза со стадиями и метриками.

Пять стадий вызываются по отдельности, а не через
``run_neurogenesis_pipeline``, чтобы каждая стадия отдала своё время и свои
метрики: иначе «нейрогенез» выглядит как чёрный ящик с одним числом на выходе.

Честная граница: это компиляция генома из статистики текста и морфогенез
ДО обучения. Роста числа нейронов во время обучения в коде нет — N задаётся
геномом и дальше не меняется, меняются только рёбра (prune/rewire).
"""

from __future__ import annotations

import bz2
import gzip
import lzma
import time
from typing import Any

from magicbrain import TextBrain, save_model
from magicbrain.neurogenesis.attractor_dynamics import AttractorDynamics
from magicbrain.neurogenesis.compiler import GenomeCompiler
from magicbrain.neurogenesis.development import DevelopmentOperator
from magicbrain.neurogenesis.energy import EnergyFunction
from magicbrain.neurogenesis.reconstruction import ReconstructionOperator

from .. import store
from ..config import settings
from ..corpora import resolve_text
from . import common

STAGES = ["compile", "develop", "train", "reconstruct", "evaluate"]


def run(params: dict[str, Any], progress) -> dict[str, Any]:
    text, source = resolve_text(
        params.get("corpus_id"), params.get("text"), params.get("max_chars", 40_000)
    )
    strategy = params.get("strategy", "statistical")
    genome_length = int(params.get("genome_length", 28))
    steps = min(int(params.get("steps", 20_000)), settings.MAX_STEPS)
    use_cppn = bool(params.get("use_cppn", True))
    reconstruct_length = int(params.get("reconstruct_length", 500))
    n_probes = int(params.get("attractor_probes", 100))
    holdout_frac = float(params.get("holdout_frac", 0.1))

    run_id = store.new_run_id("neurogenesis")
    t_start = time.time()
    stages: list[dict[str, Any]] = []

    # --- 1. Компиляция генома -------------------------------------------------
    progress.stage("компиляция генома", 1, 5)
    t0 = time.time()
    compiler = GenomeCompiler()
    meta = compiler.compile_with_metadata(text, strategy, genome_length)
    genome, metrics = compiler.compile_with_metrics(text, strategy, genome_length)
    stages.append({
        "name": "compile",
        "title": "Компиляция генома из статистики текста",
        "seconds": time.time() - t0,
        "metrics": {
            "genome": meta["genome"],
            "genome_length": meta["genome_length"],
            "strategy_requested": strategy,
            "strategy_used": metrics.strategy_used,
            "genome_quality_score": metrics.genome_quality_score,
            "data_hash_prefix": meta["data_hash"][:16],
            "data_size_bytes": meta["data_size"],
            "dataset_stats": meta["stats"],
        },
    })
    genome = meta["genome"]

    # --- 2. Морфогенез --------------------------------------------------------
    progress.stage("морфогенез и синаптогенез", 2, 5)
    stoi, itos = common.vocab_of(text)
    train_text, holdout_text = common.split_text(text, holdout_frac)
    t0 = time.time()
    dev = DevelopmentOperator()
    tissue, dev_metrics = dev.develop_with_metrics(genome, len(stoi), use_cppn=use_cppn)
    # Ткань разворачивается один раз: develop_and_build_brain внутри вызвал бы
    # develop повторно, а это самая дорогая стадия после обучения.
    brain = TextBrain(genome, len(stoi))
    brain.w_slow = tissue.w_slow.copy()
    brain.w_fast = tissue.w_fast.copy()
    brain.theta = tissue.theta.copy()
    facts = common.brain_facts(brain)
    stages.append({
        "name": "develop",
        "title": "Развитие ткани: позиции, связи, веса из CPPN, созревание порогов",
        "seconds": time.time() - t0,
        "metrics": {
            "n_neurons": dev_metrics.n_neurons,
            "n_edges": dev_metrics.n_edges,
            "weight_mean": dev_metrics.weight_mean,
            "weight_std": dev_metrics.weight_std,
            "cppn_used": dev_metrics.cppn_used,
            "delay_histogram": facts["delay_histogram"],
            "inhibitory_share": float(tissue.is_inhib.mean()),
            "theta_mean": float(tissue.theta.mean()),
            "develop_seconds": dev_metrics.time_seconds,
        },
    })

    # --- 3. Обучение ----------------------------------------------------------
    progress.stage("обучение", 3, 5)
    loss_before = common.eval_loss(brain, holdout_text, stoi)
    chunk = max(500, steps // 30)

    def on_chunk(done: int, loss: float, point: dict[str, Any]) -> None:
        progress(0.4 + 0.35 * (done / steps), f"обучение {done}/{steps}, loss {loss:.3f}")

    t0 = time.time()
    curve = common.train_chunked(brain, train_text, stoi, steps, chunk=chunk, on_chunk=on_chunk)
    train_seconds = time.time() - t0
    loss_after = common.eval_loss(brain, holdout_text, stoi)
    base = common.baselines(train_text, holdout_text, stoi)
    stages.append({
        "name": "train",
        "title": "Обучение развитой сети на том же тексте",
        "seconds": train_seconds,
        "metrics": {
            "steps": steps,
            "holdout_nats_before": loss_before,
            "holdout_nats_after": loss_after,
            "baselines_nats": base,
            "beats_bigram": loss_after < base["bigram"],
            "steps_per_second": steps / train_seconds if train_seconds else None,
        },
    })

    # --- 4. Реконструкция -----------------------------------------------------
    progress.stage("реконструкция", 4, 5)
    t0 = time.time()
    recon = ReconstructionOperator()
    result = recon.reconstruct_autoregressive(
        brain, stoi, itos, seed=text[: min(10, len(text))], length=reconstruct_length
    )
    stages.append({
        "name": "reconstruct",
        "title": "Авторегрессивное воспроизведение текста из обученной сети",
        "seconds": time.time() - t0,
        "metrics": {"length": len(result.text)},
        "text": result.text,
    })

    # --- 5. Оценка ------------------------------------------------------------
    progress.stage("оценка верности и аттракторов", 5, 5)
    t0 = time.time()
    fidelity = recon.measure_fidelity(text[:reconstruct_length], result.text)
    compression = recon.measure_compression(genome, text, model_size_bytes=None)
    raw = text.encode("utf-8")
    codecs = {
        "gzip": len(gzip.compress(raw, 9)),
        "bz2": len(bz2.compress(raw, 9)),
        "lzma": len(lzma.compress(raw)),
    }
    attractors: dict[str, Any] = {"count": None, "note": "не считалось"}
    if n_probes > 0:
        try:
            dynamics = AttractorDynamics(max_iterations=50)
            W_dense = recon._build_dense_weights(brain)
            found = dynamics.find_attractors(brain.N, W_dense, brain.theta, n_probes=n_probes)
            energy_fn = EnergyFunction()
            attractors = {
                "count": len(found),
                "probes": n_probes,
                "basin_sizes": [int(a.basin_size) for a in found[:20]],
                "energies": [float(a.energy) for a in found[:20]],
                "energy_of_best": (
                    float(energy_fn.energy(found[0].state, W_dense, brain.theta))
                    if found else None
                ),
                "mean_activity": (
                    float(sum(float(a.state.mean()) for a in found) / len(found))
                    if found else None
                ),
            }
        except Exception as exc:  # noqa: BLE001
            attractors = {"count": None, "note": f"поиск не удался: {type(exc).__name__}: {exc}"}

    stages.append({
        "name": "evaluate",
        "title": "Верность воспроизведения, отношение размеров, аттракторы",
        "seconds": time.time() - t0,
        "metrics": {
            "fidelity": fidelity,
            "compression": compression,
            "codec_baselines_bytes": codecs,
            "codec_baseline_ratios": {
                k: v / max(1, len(raw)) for k, v in codecs.items()
            },
            "attractors": attractors,
        },
    })

    store.ensure_dirs()
    model_path = store.model_path(run_id)
    save_model(brain, stoi, itos, str(model_path))

    run = {
        "run_id": run_id,
        "kind": "neurogenesis",
        "created_at": time.time(),
        "wall_time_sec": time.time() - t_start,
        "source": source,
        "params": {
            "strategy": strategy, "genome_length": genome_length, "steps": steps,
            "use_cppn": use_cppn, "reconstruct_length": reconstruct_length,
            "attractor_probes": n_probes,
        },
        "network": facts,
        "model_id": run_id,
        "stages": stages,
        "curve": curve,
        "caveats": [
            "Роста числа нейронов во время обучения нет: N фиксирован геномом, "
            "меняются только рёбра (prune/rewire).",
            "genome_ratio — отношение длины генома к размеру данных, а не коэффициент "
            "сжатия: обратного преобразования с гарантией нет, верность измеряется "
            "отдельно и она далека от единицы. Строки gzip/bz2/lzma приведены как "
            "напоминание, что сравнение с кодеками некорректно, пока верность не 1.0.",
        ],
        "code": common.code_block(),
    }
    run["headline"] = {
        "genome": genome,
        "N": facts["N"],
        "char_accuracy": fidelity.get("char_accuracy"),
        "genome_ratio": compression.get("genome_ratio"),
        "attractors": attractors.get("count"),
    }
    store.save(run)
    return run
