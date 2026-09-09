#!/usr/bin/env python3
"""Сборка корпусов плейграунда из законных источников.

Скрипт идемпотентен: он скачивает исходники, срезает служебные врезки
Project Gutenberg, нормализует пробелы и кладёт результат в
``data/corpora/<id>.txt``. Рядом пишется ``manifest.json`` с URL, размером
и SHA-256 каждого корпуса, чтобы происхождение текста можно было проверить
без повторного скачивания.

Запускается вручную; результат коммитится в репозиторий, поэтому во время
работы плейграунда сеть не нужна.
"""

from __future__ import annotations



import argparse
import hashlib
import json
import os
import math
import pathlib
import re
import sys
import urllib.request

_BALANSIS_SRC = os.environ.get("BALANSIS_SRC", "../Balansis")

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "corpora"

PG_START = re.compile(r"\*\*\* ?START OF TH[EIS][^\n]*\*\*\*")
PG_END = re.compile(r"\*\*\* ?END OF TH[EIS][^\n]*\*\*\*")


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "magicbrain-playground/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 - фиксированный список URL
        return resp.read().decode("utf-8", errors="replace")


def strip_gutenberg(raw: str) -> str:
    """Убирает заголовок и подвал Project Gutenberg.

    Оставляем только само произведение, находящееся в общественном достоянии.
    Товарный знак и лицензионный текст PG вырезаются целиком, поэтому
    лицензия PG на результат не распространяется.
    """
    m = PG_START.search(raw)
    if m:
        raw = raw[m.end():]
    m = PG_END.search(raw)
    if m:
        raw = raw[: m.start()]
    raw = re.sub(r"Produced by[^\n]*\n", "", raw)
    return raw


def normalize(text: str, limit: int) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()
    if limit and len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        text = text[: cut if cut > limit // 2 else limit]
    return text.strip() + "\n"


def from_repo_files(paths: list[pathlib.Path]) -> str:
    parts = []
    for p in paths:
        if p.exists():
            parts.append(p.read_text(encoding="utf-8", errors="replace"))
    return "\n\n".join(parts)


def synthetic_markov(n_chars: int, seed: int) -> tuple[str, dict]:
    """Синтетический корпус с известной скоростью энтропии источника.

    Источник — марковская цепь первого порядка над слоговым алфавитом, после
    каждого слога с вероятностью 0.28 вставляется разделитель. Энтропию несут
    **оба** процесса: раньше считалась только цепь, а вставка разделителя (её
    факт и выбор одного из четырёх различимых вариантов) в расчёт не входила,
    из-за чего заявленный «предел» был занижен примерно вдвое.

    Слоги двухсимвольные, разделители — не буквы, поэтому строка разбирается на
    шаги однозначно: скорость энтропии символьного процесса равна энтропии
    одного шага, делённой на среднюю длину шага в символах.

    Возвращает текст и разбор энтропии: аналитический расчёт и эмпирическую
    оценку, накопленную по фактически сделанным на этом seed выборам.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    syllables = ["ба", "ве", "ги", "до", "жу", "зя", "ле", "ми", "но", "ра", "со", "ту"]
    seps = [" ", " ", " ", ", ", ". ", "\n"]
    p_sep = 0.28
    n_states = len(syllables)
    # Разреженная переходная матрица: у каждого состояния ровно 3 преемника.
    trans = np.zeros((n_states, n_states))
    for i in range(n_states):
        succ = rng.choice(n_states, size=3, replace=False)
        probs = rng.dirichlet([2.0, 2.0, 2.0])
        trans[i, succ] = probs
    # Стационарное распределение через степенной метод.
    pi = np.ones(n_states) / n_states
    for _ in range(2000):
        pi = pi @ trans
    pi = pi / pi.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(trans > 0, np.log(trans), 0.0)
    chain_nats = float(-(pi[:, None] * trans * logs).sum())

    # Разделитель наблюдается как строка, а не как индекс: три позиции списка
    # дают один и тот же пробел и на письме неразличимы.
    sep_probs: dict[str, float] = {}
    for sep in seps:
        sep_probs[sep] = sep_probs.get(sep, 0.0) + 1.0 / len(seps)
    sep_choice_nats = float(-sum(p * math.log(p) for p in sep_probs.values()))
    bernoulli_nats = float(-(p_sep * math.log(p_sep) + (1 - p_sep) * math.log(1 - p_sep)))
    sep_nats = bernoulli_nats + p_sep * sep_choice_nats

    mean_syllable = float(np.mean([len(s) for s in syllables]))
    mean_sep = float(sum(p * len(s) for s, p in sep_probs.items()))
    mean_step_chars = mean_syllable + p_sep * mean_sep

    out: list[str] = []
    state = 0
    total = 0
    # Эмпирическая оценка: копим -log p фактических выборов, не трогая rng.
    nats_used = 0.0
    chars_used = 0
    prev_p: float | None = None
    while total < n_chars:
        piece = syllables[state]
        if prev_p is not None:
            # Переход, приведший сюда: его стоимость платится, когда слог
            # действительно выписан, иначе последний неиспользованный выбор
            # попал бы в счёт.
            nats_used += -math.log(prev_p)
        out.append(piece)
        total += len(piece)
        chars_used += len(piece)
        if rng.random() < p_sep:
            sep = seps[int(rng.integers(0, len(seps)))]
            out.append(sep)
            total += len(sep)
            chars_used += len(sep)
            nats_used += -math.log(p_sep) - math.log(sep_probs[sep])
        else:
            nats_used += -math.log(1 - p_sep)
        nxt = int(rng.choice(n_states, p=trans[state]))
        prev_p = float(trans[state, nxt])
        state = nxt
    text = "".join(out)
    details = {
        "chain_nats_per_step": round(chain_nats, 4),
        "separator_nats_per_step": round(sep_nats, 4),
        "mean_chars_per_step": round(mean_step_chars, 4),
        "empirical_nats_per_char": round(nats_used / max(1, chars_used), 4),
    }
    return text[:n_chars] + "\n", details


def entropy_per_char(details: dict) -> float:
    return (details["chain_nats_per_step"] + details["separator_nats_per_step"]) / details["mean_chars_per_step"]


SOURCES = [
    {
        "id": "rachinsky_ru",
        "title": "С. А. Рачинский. 1001 задача для умственного счёта",
        "lang": "ru",
        "limit": 120_000,
        "url": "https://www.gutenberg.org/cache/epub/16527/pg16527.txt",
        "kind": "gutenberg",
        "data_class": "D0",
        "license": "public domain (автор ум. 1902; врезки Project Gutenberg вырезаны)",
    },
    {
        "id": "moskoviya_ru",
        "title": "П. Н. Апостол. Московия в представлении иностранцев XVI-XVII в.",
        "lang": "ru",
        "limit": 100_000,
        "url": "https://www.gutenberg.org/cache/epub/30774/pg30774.txt",
        "kind": "gutenberg",
        "data_class": "D0",
        "license": "public domain (издание 1918 г.; врезки Project Gutenberg вырезаны)",
    },
    {
        "id": "shakespeare_en",
        "title": "William Shakespeare, selected plays (tiny-shakespeare)",
        "lang": "en",
        "limit": 120_000,
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "kind": "plain",
        "data_class": "D0",
        "license": "public domain (Шекспир, ум. 1616); файл-агрегат из char-rnn, MIT",
    },
    {
        "id": "balansis_docs_en",
        "title": "Balansis: собственная документация экосистемы (AGPL-3.0, публичный репозиторий)",
        "lang": "en",
        "limit": 100_000,
        "kind": "repo",
        "data_class": "D1",
        "license": "собственный код экосистемы; репозиторий StudyLabPro/Balansis публичный",
        "paths": [
            _BALANSIS_SRC + "/README.md",
            _BALANSIS_SRC + "/ROADMAP.md",
            _BALANSIS_SRC + "/CHANGELOG.md",
            _BALANSIS_SRC + "/docs/index.md",
            _BALANSIS_SRC + "/docs/glossary.md",
        ],
    },
    {
        "id": "synthetic_markov",
        "title": "Синтетическая марковская цепь (сгенерирована этим скриптом)",
        "lang": "synthetic",
        "limit": 80_000,
        "kind": "synthetic",
        "data_class": "D2",
        "license": "сгенерировано этим скриптом, реальных данных не содержит",
        "seed": 20260908,
    },
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", action="append", default=None,
                    help="пересобрать только эти id (остальные записи манифеста сохраняются)")
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    previous = []
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in previous}

    selected = [s for s in SOURCES if not args.only or s["id"] in args.only]
    if args.only:
        unknown = set(args.only) - {s["id"] for s in SOURCES}
        if unknown:
            raise SystemExit(f"неизвестные id: {sorted(unknown)}")

    for src in selected:
        print(f"-> {src['id']}")
        extra: dict = {}
        if src["kind"] == "gutenberg":
            text = normalize(strip_gutenberg(fetch(src["url"])), src["limit"])
        elif src["kind"] == "plain":
            text = normalize(fetch(src["url"]), src["limit"])
        elif src["kind"] == "repo":
            text = normalize(from_repo_files([pathlib.Path(p) for p in src["paths"]]), src["limit"])
        elif src["kind"] == "synthetic":
            raw, details = synthetic_markov(src["limit"], src["seed"])
            text = normalize(raw, src["limit"])
            extra["source_entropy_nats_per_char"] = round(entropy_per_char(details), 4)
            extra["source_entropy_details"] = details
        else:
            raise ValueError(src["kind"])

        path = OUT / f"{src['id']}.txt"
        path.write_text(text, encoding="utf-8")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        entry = {
            "id": src["id"],
            "title": src["title"],
            "lang": src["lang"],
            "kind": src["kind"],
            "chars": len(text),
            "bytes": len(text.encode("utf-8")),
            "vocab_size": len(set(text)),
            "sha256": digest,
            "data_class": src["data_class"],
            "license": src["license"],
        }
        if "url" in src:
            entry["url"] = src["url"]
        if "paths" in src:
            entry["paths"] = src["paths"]
        entry.update(extra)
        by_id[src["id"]] = entry
        print(f"   {entry['chars']} символов, словарь {entry['vocab_size']}, sha256 {digest[:12]}")

    manifest = [by_id[s["id"]] for s in SOURCES if s["id"] in by_id]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
