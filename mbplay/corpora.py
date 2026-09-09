"""Реестр корпусов.

Тексты и manifest.json готовит ``scripts/fetch_corpora.py`` и коммитит в
репозиторий, поэтому сервису сеть не нужна. Каждый корпус несёт класс данных
(D0 публичный / D1 собственный / D2 синтетика) и основание законности —
это единственный способ попасть в обучение: произвольный текст пользователь
может прислать телом запроса, но он не сохраняется в реестр.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from .config import settings


@dataclass
class Corpus:
    id: str
    title: str
    lang: str
    kind: str
    chars: int
    bytes: int
    vocab_size: int
    sha256: str
    data_class: str
    license: str
    url: str | None = None
    paths: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out = {
            "id": self.id,
            "title": self.title,
            "lang": self.lang,
            "kind": self.kind,
            "chars": self.chars,
            "bytes": self.bytes,
            "vocab_size": self.vocab_size,
            "sha256": self.sha256,
            "data_class": self.data_class,
            "license": self.license,
        }
        if self.url:
            out["url"] = self.url
        if self.paths:
            out["paths"] = self.paths
        out.update(self.extra)
        return out


_KNOWN = {
    "id", "title", "lang", "kind", "chars", "bytes", "vocab_size",
    "sha256", "data_class", "license", "url", "paths",
}


@lru_cache(maxsize=1)
def _manifest() -> list[Corpus]:
    path = settings.CORPORA_DIR / "manifest.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for item in raw:
        out.append(Corpus(
            id=item["id"],
            title=item.get("title", item["id"]),
            lang=item.get("lang", "?"),
            kind=item.get("kind", "?"),
            chars=int(item.get("chars", 0)),
            bytes=int(item.get("bytes", 0)),
            vocab_size=int(item.get("vocab_size", 0)),
            sha256=item.get("sha256", ""),
            data_class=item.get("data_class", "?"),
            license=item.get("license", "не указано"),
            url=item.get("url"),
            paths=item.get("paths", []),
            extra={k: v for k, v in item.items() if k not in _KNOWN},
        ))
    return out


def list_corpora() -> list[Corpus]:
    return _manifest()


def get_corpus(corpus_id: str) -> Corpus | None:
    for c in _manifest():
        if c.id == corpus_id:
            return c
    return None


def read_text(corpus_id: str, max_chars: int | None = None) -> str:
    corpus = get_corpus(corpus_id)
    if corpus is None:
        raise KeyError(f"неизвестный корпус: {corpus_id}")
    text = (settings.CORPORA_DIR / f"{corpus_id}.txt").read_text(encoding="utf-8")
    if max_chars:
        text = text[:max_chars]
    return text


def resolve_text(corpus_id: str | None, raw_text: str | None, max_chars: int) -> tuple[str, dict[str, Any]]:
    """Возвращает текст и его происхождение.

    Либо корпус из реестра, либо присланный текст — тогда класс данных
    неизвестен, и это честно записывается в прогон.
    """
    limit = min(int(max_chars or settings.MAX_CHARS), settings.MAX_CHARS)
    if corpus_id:
        text = read_text(corpus_id, limit)
        corpus = get_corpus(corpus_id)
        source = {
            "kind": "corpus",
            "corpus_id": corpus_id,
            "title": corpus.title if corpus else corpus_id,
            "data_class": corpus.data_class if corpus else "?",
            "license": corpus.license if corpus else "?",
            "sha256_full": corpus.sha256 if corpus else "",
        }
    elif raw_text:
        text = raw_text[:limit]
        source = {
            "kind": "inline",
            "corpus_id": None,
            "title": "текст из запроса",
            "data_class": "unknown",
            "license": "предоставлен вызывающей стороной; не сохраняется в реестр",
            "sha256_full": "",
        }
    else:
        raise ValueError("нужен либо corpus_id, либо text")

    if len(text) < 200:
        raise ValueError("слишком короткий текст: нужно хотя бы 200 символов")

    source["used_chars"] = len(text)
    source["used_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return text, source
