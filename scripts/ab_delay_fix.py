#!/usr/bin/env python3
"""Контролируемое сравнение «до/после» исправления линии аксональных задержек.

Запускает обучение на нескольких seed двумя ревизиями MagicBrain: текущей и
предыдущим коммитом, распакованным в отдельный каталог. Обе ревизии считаются
одним и тем же кодом плейграунда, поэтому различаться может только библиотека.

Подготовка предыдущей ревизии:

    rm -rf /tmp/mb_prev && mkdir -p /tmp/mb_prev
    git -C "$MB_SRC" archive HEAD~1 | tar -x -C /tmp/mb_prev

Результат печатается и складывается в var/delay-fix-ab.json, откуда его
забирает scripts/render_runs.py.
"""
import json
import os
import pathlib
import statistics
import subprocess
import sys

MB_SRC = os.environ.get("MB_SRC", "../MagicBrain")
os.environ.setdefault("PYTHONUNBUFFERED","1")
CODE = '''
import sys, math, os
sys.path.insert(0, {prefix!r}); sys.path.insert(0, ".")
os.environ["MBP_CORPORA_DIR"]="data/corpora"; os.environ["MBP_DATA_DIR"]="var"
from mbplay.experiments import common
from mbplay.corpora import read_text
from magicbrain import TextBrain
text = read_text({corpus!r}, 100000)
stoi, itos = common.vocab_of(text)
tr, ho = common.split_text(text, 0.1)
b = TextBrain("30121033102301230112332100123", len(stoi), seed_override={seed})
common.train_chunked(b, tr, stoi, {steps}, chunk=5000)
print("%.6f" % (common.eval_loss(b, ho, stoi)/math.log(2)))
'''
def run(prefix, corpus, seed, steps):
    out = subprocess.run([sys.executable,"-c",CODE.format(prefix=prefix,corpus=corpus,seed=seed,steps=steps)],
                         capture_output=True, text=True, cwd=".")
    if out.returncode: print(out.stderr[-1500:]); sys.exit(1)
    return float(out.stdout.strip().split("\n")[-1])


SEEDS = (11, 23, 37, 51, 73)
STEPS = 50000

res={}
for corpus in ("shakespeare_en","synthetic_markov"):
    for label, prefix in (("before","/tmp/mb_prev"),("after", MB_SRC)):
        vals=[run(prefix,corpus,s,STEPS) for s in SEEDS]
        res[(corpus,label)]=vals
        print(f"{corpus:18s} {label:7s} mean={statistics.mean(vals):.4f} sd={statistics.pstdev(vals):.4f} vals={[round(v,4) for v in vals]}", flush=True)
def sha(rev):
    return subprocess.run(
        ["git", "-C", MB_SRC, "rev-parse", "--short=12", rev],
        capture_output=True, text=True).stdout.strip()


out = {
    "seeds": len(SEEDS), "seed_values": list(SEEDS), "steps": STEPS,
    "metric": "бит/символ на отложенном хвосте, меньше лучше",
    "before_sha": sha("HEAD~1"), "after_sha": sha("HEAD"),
    "corpora": {},
}
for corpus in ("shakespeare_en", "synthetic_markov"):
    b = res[(corpus, "before")]
    a = res[(corpus, "after")]
    out["corpora"][corpus] = {
        "before": b, "after": a,
        "before_mean": statistics.mean(b), "before_sd": statistics.pstdev(b),
        "after_mean": statistics.mean(a), "after_sd": statistics.pstdev(a),
        "delta": statistics.mean(a) - statistics.mean(b),
    }
    print(f"{corpus}: разница средних {out['corpora'][corpus]['delta']:+.4f} бит/симв, "
          f"разброс по seed ±{max(statistics.pstdev(a), statistics.pstdev(b)):.4f}")

pathlib.Path("var").mkdir(exist_ok=True)
pathlib.Path("var/delay-fix-ab.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("записано var/delay-fix-ab.json")
