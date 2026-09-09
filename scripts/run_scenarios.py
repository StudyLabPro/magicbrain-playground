#!/usr/bin/env python3
"""Прогоняет все сценарии плейграунда через HTTP и печатает выжимку.

Скрипт ходит в поднятый стек, а не импортирует библиотеку напрямую: так
проверяется именно то, что увидит человек в интерфейсе, вместе с очередью
задач, сохранением прогонов и соседним сервисом magicbrain-api.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("MBP_BASE", "http://127.0.0.1:8007")


def call(path: str, payload=None, method=None, timeout=60):
    url = BASE + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"))
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        body = resp.read().decode()
    return json.loads(body) if body else None


def wait(job_id: str, label: str, timeout: float = 3600) -> dict:
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout:
        job = call(f"/api/jobs/{job_id}")
        msg = f"  [{label}] {job['status']} {job['progress'] * 100:5.1f}% {job.get('message', '')}"
        if msg != last:
            print(msg, flush=True)
            last = msg
        if job["status"] == "done":
            return call(f"/api/runs/{job['run_id']}")
        if job["status"] == "failed":
            raise RuntimeError(f"{label}: {job.get('error')}\n{job.get('traceback', '')}")
        time.sleep(2)
    raise TimeoutError(label)


def start(kind: str, params: dict, label: str) -> dict:
    print(f"== {label}", flush=True)
    job = call(f"/api/experiments/{kind}", params)
    return wait(job["job_id"], label)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="shakespeare_en")
    ap.add_argument("--ru-corpus", default="rachinsky_ru")
    ap.add_argument("--syn-corpus", default="synthetic_markov")
    # По умолчанию тот же масштаб, на котором получены числа в docs/RUNS.md.
    # Значение меньше давало документ с числами другого масштаба и расходилось
    # с README.
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--out", default="var/scenarios-summary.json")
    ap.add_argument("--note", default="", help="произвольная пометка, попадает в сводку")
    args = ap.parse_args()

    meta = call("/api/meta")
    print(json.dumps(meta, ensure_ascii=False, indent=2))

    results = {"meta": meta, "runs": {}}
    objects: dict[str, dict] = {}

    train = start("train", {
        "corpus_id": args.corpus, "steps": args.steps, "max_chars": 100000,
    }, "а) обучение на публичном корпусе")
    results["runs"]["train_en"] = train["run_id"]
    objects["train_en"] = train
    m = train["metrics"]
    print(f"   отложенный хвост: {m['holdout_bits_after']:.3f} бит/симв "
          f"(биграмма {m['baselines_bits']['bigram']:.3f}, униграмма {m['baselines_bits']['unigram']:.3f}), "
          f"бьёт биграмму: {m['beats_bigram']}")

    train_syn = start("train", {
        "corpus_id": args.syn_corpus, "steps": args.steps, "max_chars": 80000,
    }, "а') обучение на синтетике с известной энтропией источника")
    results["runs"]["train_syn"] = train_syn["run_id"]
    objects["train_syn"] = train_syn

    ng = start("neurogenesis", {
        "corpus_id": args.ru_corpus, "steps": args.steps, "max_chars": 60000,
        "strategy": "statistical", "use_cppn": True, "attractor_probes": 100,
    }, "б) пайплайн нейрогенеза")
    results["runs"]["neurogenesis"] = ng["run_id"]
    objects["neurogenesis"] = ng

    repair = start("self-repair", {
        "base_run_id": train["run_id"],
        "damage_target": "recurrent",
        "damage_fracs": [0.1, 0.2, 0.4],
        "recovery_steps": 8000,
    }, "в) самовосстановление: повреждение рекуррентных синапсов")
    results["runs"]["self_repair"] = repair["run_id"]
    objects["self_repair"] = repair

    repair_readout = start("self-repair", {
        "base_run_id": train["run_id"],
        "damage_target": "readout",
        "damage_fracs": [0.1, 0.2, 0.4],
        "recovery_steps": 8000,
    }, "в') самовосстановление: повреждение линейного readout")
    results["runs"]["self_repair_readout"] = repair_readout["run_id"]
    objects["self_repair_readout"] = repair_readout

    act = start("act", {
        "corpus_id": args.corpus, "steps": 10000, "max_chars": 60000, "seed": 20260908,
    }, "г) сравнение с ACT-бэкендом Balansis")
    results["runs"]["act"] = act["run_id"]
    objects["act"] = act

    print("== д) сохранение и повторное открытие")
    listed = call("/api/runs")["runs"]
    ids = {r["run_id"] for r in listed}
    # Проверяется КАЖДЫЙ прогон и целиком: раньше перечитывался один и
    # сравнивалось одно число, а документ утверждал «все прогоны … побайтово».
    identical = []
    differing = []
    for key, run_id in results["runs"].items():
        assert run_id in ids, f"{run_id} не попал в список прогонов"
        reopened = call(f"/api/runs/{run_id}")
        a = json.dumps(objects[key], ensure_ascii=False, sort_keys=True)
        b = json.dumps(reopened, ensure_ascii=False, sort_keys=True)
        (identical if a == b else differing).append(run_id)
    results["reread_check"] = {
        "method": "полное сравнение сериализованного JSON прогона с перечитанным по HTTP",
        "checked": len(results["runs"]),
        "identical": len(identical),
        "differing": differing,
    }
    assert not differing, f"перечитанные прогоны разошлись: {differing}"
    print(f"   прогонов в хранилище: {len(listed)}; перечитано и сравнено целиком: "
          f"{len(identical)} из {len(results['runs'])}")

    print("== публикация модели в magicbrain-api")
    pub = call(f"/api/runs/{train['run_id']}/publish", {})
    sample = call("/api/sample", {"model_id": pub["model_id"], "seed_text": "The ", "n_tokens": 120})
    results["publish"] = {"model_id": pub["model_id"], "sample_head": sample["generated_text"][:120]}
    print("   сгенерировано рантаймом:", sample["generated_text"][:80].replace("\n", " "))

    results["steps"] = args.steps
    if args.note:
        results["note"] = args.note
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print("\nсводка:", args.out)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.HTTPError as exc:
        print("HTTP", exc.code, exc.read().decode()[:800], file=sys.stderr)
        sys.exit(1)
