#!/usr/bin/env python3
"""Собирает docs/RUNS.md из фактических файлов прогонов.

Документ генерируется, а не пишется руками: числа в нём — те же, что лежат в
var/runs/*.json, и разойтись они не могут. Прозаические части — здесь же,
рядом с кодом, который их подставляет.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "var" / "runs"
SUMMARY = ROOT / "var" / "scenarios-summary.json"
AB = ROOT / "var" / "delay-fix-ab.json"
OUT = ROOT / "docs" / "RUNS.md"

B = 1.0 / math.log(2.0)


def load(run_id: str) -> dict:
    return json.loads((RUNS / f"{run_id}.json").read_text(encoding="utf-8"))


def f(x, n=3):
    return "—" if x is None else f"{x:.{n}f}"


def src_line(run: dict) -> str:
    s = run["source"]
    return (f"`{s.get('corpus_id') or 'текст из запроса'}` ({s['data_class']}), "
            f"{s['used_chars']} символов, sha256 `{s['used_sha256'][:12]}`")


def section_train(run: dict, title: str, extra: str = "") -> str:
    m = run["metrics"]
    n = run["network"]
    return f"""### {title}

Прогон `{run['run_id']}` · данные: {src_line(run)} · сеть: N={n['N']}, K={n['K']},
рёбер {n['edges']}, геном `{n['genome']}` · {run['params']['steps']} шагов ·
{f(run['wall_time_sec'], 1)} с ({f(m['steps_per_second'], 0)} шагов/с).

| модель | бит/символ на отложенном хвосте |
|---|---|
| равномерная | {f(m['baselines_bits']['uniform'])} |
| униграмма | {f(m['baselines_bits']['unigram'])} |
| биграмма | {f(m['baselines_bits']['bigram'])} |
| сеть до обучения | {f(m['holdout_bits_before'])} |
| **сеть после обучения** | **{f(m['holdout_bits_after'])}** |

Бьёт униграмму: **{'да' if m['beats_unigram'] else 'нет'}**.
Бьёт биграмму: **{'да' if m['beats_bigram'] else 'нет'}**.
{extra}
Что сеть генерирует (первые 200 символов, затравка — начало текста):

```
{run.get('sample', '')[:200]}
```
"""


def attractor_note(att: dict) -> str:
    """Один аттрактор, собравший все пробы, — это не «память», а коллапс."""
    count = att.get("count")
    activity = att.get("mean_activity")
    if count == 1 and activity is not None and activity > 0.5:
        return ("\n\nОдин аттрактор, в который сошлись все пробы, и почти все нейроны в нём "
                "активны — это не набор запомненных образов, а коллапс динамики в одно "
                "состояние «всё включено». Аттракторная память на этом прогоне не "
                "демонстрируется, и выдавать число «найден 1 аттрактор» за неё нельзя.")
    return ""


def section_neurogenesis(run: dict) -> str:
    stages = {s["name"]: s for s in run["stages"]}
    c = stages["compile"]["metrics"]
    d = stages["develop"]["metrics"]
    t = stages["train"]["metrics"]
    e = stages["evaluate"]["metrics"]
    fid = e["fidelity"]
    comp = e["compression"]
    codecs = e["codec_baseline_ratios"]
    att = e["attractors"]
    rows = "\n".join(
        f"| {s['title']} | {f(s['seconds'], 2)} |" for s in run["stages"]
    )
    return f"""### б) Пайплайн нейрогенеза

Прогон `{run['run_id']}` · данные: {src_line(run)} · стратегия
`{c['strategy_used']}` · {run['params']['steps']} шагов · {f(run['wall_time_sec'], 1)} с.

| стадия | секунд |
|---|---|
{rows}

**Компиляция.** Геном `{c['genome']}` длиной {c['genome_length']}; оценка качества
генома {f(c['genome_quality_score'])}. Статистика текста: энтропия
{f(c['dataset_stats']['entropy'], 2)} нат/символ, словарь
{c['dataset_stats']['vocab_size']}, повторяемость
{f(c['dataset_stats']['repetitiveness'], 3)}.

**Морфогенез.** Нейронов {d['n_neurons']}, рёбер {d['n_edges']}, CPPN
{'использован' if d['cppn_used'] else 'не использован'}; веса ткани
mean={f(d['weight_mean'], 4)}, std={f(d['weight_std'], 4)}; тормозных
{f(d['inhibitory_share'] * 100, 1)}%; средний порог после созревания
{f(d['theta_mean'], 4)}. Развитие заняло {f(d['develop_seconds'], 2)} с.

**Обучение.** Отложенный хвост {f(t['holdout_nats_before'] * B)} →
{f(t['holdout_nats_after'] * B)} бит/символ при биграмме
{f(t['baselines_nats']['bigram'] * B)}; биграмму
{'обошёл' if t['beats_bigram'] else '**не обошёл**'}.

**Воспроизведение и оценка.** Точность посимвольного совпадения
{f(fid['char_accuracy'], 3)}, совпадение биграмм {f(fid['bigram_overlap'], 3)},
триграмм {f(fid['trigram_overlap'], 3)}, пересечение словарей
{f(fid['vocab_overlap'], 3)}.

Отношение длины генома к размеру данных: **{f(comp['genome_ratio'], 5)}**
({comp['genome_size_bytes']} байт генома на {comp['data_size_bytes']} байт текста).
Для сравнения, доля от исходного размера у настоящих кодеков:
gzip {f(codecs['gzip'], 3)}, bz2 {f(codecs['bz2'], 3)}, lzma {f(codecs['lzma'], 3)} —
но кодек восстанавливает текст точно, а здесь совпадение символов
{f(fid['char_accuracy'], 3)}. Это разные величины, и складывать их в одну
таблицу «во сколько раз сжали» нельзя.

Аттракторов найдено {att.get('count')} на {att.get('probes')} проб;
размеры бассейнов {att.get('basin_sizes', [])[:8]}. Средняя доля активных
нейронов в найденном состоянии — {f(att.get('mean_activity'), 3)}.{attractor_note(att)}
"""


RECURRENT_NOTE = """
Это штатный `TextBrain.damage_edges`. Результат неожиданный и важный: обнуление
до 40% рекуррентных синапсов почти не меняет качество предсказания. Значит,
рекуррентная часть в этой задаче несёт мало сигнала, а работу делает линейный
readout, которого такое повреждение не касается. «Восстанавливать» после такого
удара нечего, поэтому доля отыгранного здесь и не считается.
"""

READOUT_NOTE = """
Здесь обнуляется доля весов матрицы readout `R` — той самой, что превращает
состояние сети в распределение по символам. Просадка видна сразу, и вот на ней
уже видно, восстанавливается сеть дообучением или нет.
"""


def section_repair(run: dict, title: str, note: str, meta_build: str = "") -> str:
    m = run["metrics"]
    build = run.get("code", {}).get("build_id", "")
    build_note = ""
    if meta_build and build and build != meta_build:
        build_note = (f"\n> Этот прогон сделан сборкой `{build}` — она новее остальных: "
                      "параметр `damage_target` не был объявлен в схеме запроса, "
                      "запрос принимался, и повреждение молча уходило в значение по "
                      "умолчанию. После исправления схема отвергает неизвестные поля.\n")
    def recovered(d):
        # Доля отыгранного имеет смысл только если удар вообще что-то сломал.
        # Порог — 0.01 нат/символ: ниже него «просадка» неотличима от шума
        # оценки, и деление на неё даёт числа вроде −80%.
        gap = d["post_damage_nats"] - d["pre_damage_nats"]
        if gap < 0.01 or d["recovery_ratio"] is None:
            return "не измеримо"
        return f(d["recovery_ratio"] * 100, 1) + "%"

    rows = "\n".join(
        f"| {int(d['damage_frac'] * 100)}% | {d['weights_zeroed']} | "
        f"{f(d['pre_damage_nats'] * B)} | {f(d['post_damage_nats'] * B)} | "
        f"{f(d['final_nats'] * B)} | {recovered(d)} | "
        f"{f(d['residual_nats'] * B)} |"
        for d in m["by_damage"]
    )
    return f"""### {title}
{build_note}
Прогон `{run['run_id']}` · цель повреждения
`{run['params'].get('damage_target', 'recurrent')}` · та же обученная сеть
перечитывается с диска перед каждым повреждением · восстановление
{run['params']['recovery_steps']} шагов · {f(run['wall_time_sec'], 1)} с.

| доля | обнулено весов | до, бит/симв | сразу после удара | после восстановления | отыграно | остаточный ущерб |
|---|---|---|---|---|---|---|
{rows}

Все значения — на отложенном хвосте, который в дообучении не участвует.
«Отыграно» — доля просадки, которую вернуло дообучение; «не измеримо» значит,
что просадка меньше 0.01 нат/символ, то есть удар ничего заметного не сломал и
восстанавливать нечего.
{note}"""


def section_act(run: dict) -> str:
    a, n = run["sides"]["act"], run["sides"]["numpy"]
    mb = run["microbenchmark"]
    m = run["metrics"]
    return f"""### г) С ACT-бэкендом Balansis и без него

Прогон `{run['run_id']}` · один геном, один seed
({run['params']['seed']}), один текст · по {run['params']['steps']} шагов на сторону ·
balansis {run['code'].get('balansis_version')}.

| | numpy | ACT |
|---|---|---|
| ACT реально активен | {n['act_active']} | {a['act_active']} |
| отложенный хвост, бит/символ | {f(n['holdout_nats_after'] * B)} | {f(a['holdout_nats_after'] * B)} |
| время, с | {f(n['seconds'], 1)} | {f(a['seconds'], 1)} |
| шагов/с | {f(n['steps_per_second'], 0)} | {f(a['steps_per_second'], 0)} |

ACT медленнее в **{f(m['act_slowdown_x'], 2)}×**; разница качества
{f(m['holdout_delta_nats'], 5)} нат/символ — то есть на этой задаче
компенсированная арифметика ничего не меняет в предсказании.

Где она всё же меняет ответ — микробенчмарк на специально плохо обусловленных
данных, эталон `math.fsum`:

| операция | ошибка numpy | ошибка ACT | точнее в |
|---|---|---|---|
| сумма {mb['sum']['n']} чисел | {mb['sum']['naive_abs_error']:.3e} | {mb['sum']['act_abs_error']:.3e} | {'∞' if mb['sum']['act_abs_error'] == 0 else f(mb['sum']['naive_abs_error'] / mb['sum']['act_abs_error'], 1)}× |
| скалярное произведение ({mb['dot']['n']}) | {mb['dot']['naive_abs_error']:.3e} | {mb['dot']['act_abs_error']:.3e} | {'∞' if mb['dot']['act_abs_error'] == 0 else f(mb['dot']['naive_abs_error'] / mb['dot']['act_abs_error'], 1)}× |

Вывод без прикрас: ACT честно делает то, ради чего написан — в микробенчмарке
его ошибка ровно ноль против эталона. Но веса `TextBrain` — float32 в диапазоне
порядка 0.01–0.1, где наивная сумма и так точна. На этой задаче мы платим
замедлением в {f(m['act_slowdown_x'], 2)}× за точность, которая не нужна.
"""


def section_ab(ab: dict) -> str:
    rows = ""
    for corpus, data in ab["corpora"].items():
        rows += (f"| `{corpus}` | {f(data['before_mean'])} ± {f(data['before_sd'])} | "
                 f"{f(data['after_mean'])} ± {f(data['after_sd'])} | "
                 f"{data['delta']:+.4f} |\n")
    return f"""## Что дало исправление аксональных задержек

Контролируемое сравнение: тот же геном, {ab['seeds']} разных seed, по
{ab['steps']} шагов, «до» — коммит `{ab['before_sha']}`, «после» — `{ab['after_sha']}`.
Метрика — бит/символ на отложенном хвосте, меньше лучше.

| корпус | до исправления | после исправления | разница средних |
|---|---|---|---|
{rows}
Разница лежит внутри разброса по seed. Это и есть результат: механизм
аксональных задержек 1–5 шагов, вынесенный в заголовок модуля как одна из
ключевых особенностей архитектуры, до исправления фактически не работал —
и его починка **не улучшила предсказание следующего символа**. Задержки на
этой задаче не несут полезного сигнала.

Исправление всё равно нужно: без него код делал не то, что описывает, а любые
дальнейшие выводы о роли задержек были бы выводами о несуществующем механизме.
"""


def _code_of(runs: list[dict], fallback: dict) -> dict:
    """Метка кода для шапки — из самих прогонов, а не из сводки.

    Сводка снимает /api/meta один раз, в начале прогона. Если часть сценариев
    переигрывалась другой сборкой, шапка начинала описывать не тот код, из
    которого получены числа. Здесь собирается объединение по включённым
    прогонам, и расхождение видно прямо в документе.
    """
    out = dict(fallback)
    for field in ("magicbrain_sha", "magicbrain_dirty", "build_id", "playground_sha",
                  "balansis_version", "numpy_version"):
        values = sorted({str(r.get("code", {}).get(field)) for r in runs
                         if r.get("code", {}).get(field) is not None})
        if values:
            out[field] = values[0] if len(values) == 1 else " / ".join(values)
    out.setdefault("balansis", out.get("balansis_version", fallback.get("balansis")))
    out.setdefault("numpy", out.get("numpy_version", fallback.get("numpy")))
    if "balansis_version" in out:
        out["balansis"] = out["balansis_version"]
    if "numpy_version" in out:
        out["numpy"] = out["numpy_version"]
    return out


def main() -> int:
    if not SUMMARY.exists():
        print(f"нет {SUMMARY}: сначала `make smoke`", file=sys.stderr)
        return 1
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    ids = summary["runs"]
    code = summary["meta"]["code"]

    train_en = load(ids["train_en"])
    train_syn = load(ids["train_syn"])
    ng = load(ids["neurogenesis"])
    repair = load(ids["self_repair"])
    repair_readout = load(ids["self_repair_readout"]) if "self_repair_readout" in ids else None
    act = load(ids["act"])

    syn_extra = ""
    manifest = json.loads((ROOT / "data" / "corpora" / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest:
        if item["id"] == train_syn["source"].get("corpus_id") and "source_entropy_nats_per_char" in item:
            ent = item["source_entropy_nats_per_char"] * B
            det = item.get("source_entropy_details", {})
            emp = det.get("empirical_nats_per_char")
            check = (f" Аналитический расчёт сходится с эмпирической оценкой по фактически "
                     f"сделанным выборам: {f(det['chain_nats_per_step'], 4)} нат/шаг у цепи плюс "
                     f"{f(det['separator_nats_per_step'], 4)} у разделителей на "
                     f"{f(det['mean_chars_per_step'], 4)} символа шага против {emp} нат/символ "
                     "накопленных при генерации.") if emp is not None else ""
            syn_extra = (f"\nСкорость энтропии генератора — {f(ent)} бит/символ. Это оценка "
                         "самого источника, а не результат модели: строка разбирается на шаги "
                         "однозначно, поэтому энтропия одного шага, делённая на его среднюю "
                         "длину, и есть предел, к которому может стремиться модель, знающая "
                         "параметры цепи; обучающейся на выборке модели опуститься ниже неё на "
                         f"отложенном хвосте в среднем не даёт ничего.{check} "
                         f"Сеть остановилась на {f(train_syn['metrics']['holdout_bits_after'])}.\n")

    included = [train_en, train_syn, ng, repair, act] + ([repair_readout] if repair_readout else [])
    code = _code_of(included, code)
    n_runs = len(ids)
    check = summary.get("reread_check")
    if check:
        reread_line = (
            f"Все {check['checked']} прогонов выше сохранены файлами в `var/runs/` и после "
            f"завершения перечитаны через HTTP; сериализованный JSON сравнивался целиком, "
            f"совпали {check['identical']} из {check['checked']}"
            + (f", разошлись: {', '.join(check['differing'])}" if check.get("differing") else "")
            + "."
        )
    else:
        reread_line = (
            f"Все {n_runs} прогонов выше сохранены файлами в `var/runs/`; сводка этого "
            "прогона сделана версией скрипта без сплошной проверки перечитывания, "
            "поэтому здесь утверждается только факт сохранения."
        )
    note = summary.get("note")
    note_block = f"\n> {note}\n" if note else ""
    body = f"""# Фактические прогоны

Все числа получены в поднятом стеке через тот же
HTTP-интерфейс, которым пользуется человек. Документ **генерируется** скриптом
`scripts/render_runs.py` из файлов `var/runs/*.json` — переписать число в нём
и не переписать в прогоне нельзя.

Версия кода: MagicBrain `{code['magicbrain_sha']}`
(рабочее дерево на момент сборки — `{code['magicbrain_dirty']}`; в образ попал
`git archive HEAD`, чужие незакоммиченные правки исключены),
плейграунд `{code.get('playground_sha', '?')}`,
balansis {code['balansis']}, numpy {code['numpy']}, сборка `{code['build_id']}`.

> Оговорка о времени: хост в момент прогонов был занят параллельной работой
> других агентов (load average 10–16 на 8 ядрах). Абсолютные секунды поэтому
> завышены и бенчмарком скорости считаться не могут;
> сравнения внутри одного прогона (ACT против numpy, стадия против стадии)
> остаются осмысленными, потому что стороны считались последовательно в
> одинаковых условиях.

{note_block}
## Сценарии

{section_train(train_en, "а) Обучение на публичном корпусе")}
{section_train(train_syn, "а') Обучение на синтетике с известной энтропией источника", syn_extra)}
{section_neurogenesis(ng)}
{section_repair(repair, "в) Самовосстановление: повреждение рекуррентных синапсов", RECURRENT_NOTE, code['build_id'])}
{section_repair(repair_readout, "в') Самовосстановление: повреждение линейного readout", READOUT_NOTE, code['build_id']) if repair_readout else ""}
{section_act(act)}
### д) Сохранение и повторное открытие

{reread_line} Модель прогона
`{ids['train_en']}` передана в соседний контейнер со штатным MagicBrain API
(`POST /api/v1/runtime/models/{{id}}/load`), после чего его собственный рантайм
сгенерировал текст:

```
{summary.get('publish', {}).get('sample_head', '').strip()}
```

Это и есть проверка, что артефакт настоящий: файл, записанный плейграундом,
читается штатным рантаймом библиотеки без единой правки в нём.
"""

    if AB.exists():
        body += "\n" + section_ab(json.loads(AB.read_text(encoding="utf-8")))

    OUT.write_text(body, encoding="utf-8")
    print(f"записано {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
