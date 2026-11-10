"""Dos tuberías de preprocesamiento, deliberadamente distintas (§4).

Léxica (§4.1, solo BM25): minúsculas → plegado de diacríticos preservando ñ →
tokenización [a-z0-9ñ]+ → stopwords español embebidas → SnowballStemmer("spanish").

Semántica (§4.2): normalización de espacios y nada más.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from nltk.stem.snowball import SnowballStemmer

from src.common import ROOT, load_config

_TOKEN_RE = re.compile(r"[a-z0-9ñ]+")
_WS_RE = re.compile(r"\s+")
_ENIE_SENTINEL = "\x00"


def fold_accents(text: str) -> str:
    """Plegado NFD eliminando marcas combinantes, preservando la ñ (§4.1 paso 2-3)."""
    text = text.replace("ñ", _ENIE_SENTINEL)
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.replace(_ENIE_SENTINEL, "ñ")


@lru_cache(maxsize=1)
def _stemmer() -> SnowballStemmer:
    return SnowballStemmer("spanish")


@lru_cache(maxsize=1)
def load_stopwords() -> frozenset[str]:
    """Lista embebida en el repositorio (§4.1 paso 4), plegada igual que los tokens."""
    path = ROOT / load_config()["paths"]["stopwords"]
    if not path.exists():
        raise FileNotFoundError(
            f"Stopwords embebidas no encontradas en {path}; ejecutar scripts/setup_assets.py"
        )
    words = [w.strip() for w in path.read_text(encoding="utf-8").splitlines() if w.strip()]
    return frozenset(fold_accents(w.lower()) for w in words)


@lru_cache(maxsize=1)
def _stem_cache():
    # caché de stems por token (el corpus repite mucho vocabulario)
    cache: dict[str, str] = {}
    stem = _stemmer().stem

    def cached(tok: str) -> str:
        s = cache.get(tok)
        if s is None:
            s = stem(tok)
            cache[tok] = s
        return s

    return cached


def lexical_tokens(text: str) -> list[str]:
    """Tubería léxica completa §4.1 (consulta y documentos de BM25)."""
    if not isinstance(text, str):
        return []
    folded = fold_accents(text.lower())
    toks = _TOKEN_RE.findall(folded)
    sw = load_stopwords()
    stem = _stem_cache()
    return [stem(t) for t in toks if t not in sw]


def semantic_text(text: str) -> str:
    """Tubería semántica §4.2: colapsar espacios y strip. Nada más."""
    if not isinstance(text, str):
        return ""
    return _WS_RE.sub(" ", text).strip()
