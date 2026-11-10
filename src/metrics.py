"""Métricas de ranking §5 (relevancia binaria): Precision@K, NDCG@K, RR/MRR,
desempate determinista (−score, AVISOID ascendente) y conteo de empates.

Regla §5.4: un perfil sin relevantes lanza excepción (no devuelve 0 silencioso).
"""
from __future__ import annotations

import math

import numpy as np


class NoRelevantError(ValueError):
    """Perfil sin ítem relevante: no define tarea de ranking (§2.4, §5.4)."""


def rank_by_score(avisoids, scores) -> dict:
    """Orden determinista §5.2: (−score, AVISOID ascendente).

    Devuelve índices de ranking y estadística de empates (scores exactamente iguales).
    """
    avisoids = np.asarray(avisoids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if avisoids.shape != scores.shape:
        raise ValueError("avisoids y scores deben tener la misma longitud")
    # np.lexsort ordena por la última clave primero: score desc, luego AVISOID asc
    order = np.lexsort((avisoids, -scores))
    uniq, counts = np.unique(scores, return_counts=True)
    tie_groups = int((counts > 1).sum())
    tied_items = int(counts[counts > 1].sum())
    return {
        "order": order,
        "avisoids_ranked": avisoids[order],
        "scores_ranked": scores[order],
        "n_grupos_empate": tie_groups,
        "n_items_empatados": tied_items,
    }


def precision_at_k(rel: list[int], k: int) -> float:
    if k > len(rel):
        raise ValueError(f"K={k} mayor que la lista ({len(rel)})")
    return float(sum(rel[:k])) / k


def dcg_at_k(rel: list[int], k: int) -> float:
    return float(sum(r / math.log2(i + 2) for i, r in enumerate(rel[:k])))


def ndcg_at_k(rel: list[int], k: int) -> float:
    if k > len(rel):
        raise ValueError(f"K={k} mayor que la lista ({len(rel)})")
    ideal = sorted(rel, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        raise NoRelevantError("NDCG indefinido: perfil sin relevantes")
    return dcg_at_k(rel, k) / idcg


def reciprocal_rank(rel: list[int]) -> float:
    for i, r in enumerate(rel):
        if r == 1:
            return 1.0 / (i + 1)
    raise NoRelevantError("RR indefinido: perfil sin relevantes")


def profile_metrics(rel: list[int]) -> dict:
    """Las 5 métricas congeladas (§5.1) para un perfil, con rel en orden de ranking."""
    rel = [int(r) for r in rel]
    if sum(rel) == 0:
        raise NoRelevantError("Perfil sin relevantes: excluido del cálculo (§2.4)")
    return {
        "P@1": precision_at_k(rel, 1),
        "P@3": precision_at_k(rel, 3),
        "NDCG@3": ndcg_at_k(rel, 3),
        "NDCG@5": ndcg_at_k(rel, 5),
        "RR": reciprocal_rank(rel),
    }


AGG_KEYS = ["P@1", "P@3", "NDCG@3", "NDCG@5", "MRR"]


def aggregate(per_profile: list[dict]) -> dict:
    """Macro-promedio §5.2: cada perfil pesa igual. RR agregado se reporta como MRR."""
    if not per_profile:
        raise ValueError("Sin perfiles que agregar")
    out = {}
    for k in ["P@1", "P@3", "NDCG@3", "NDCG@5"]:
        out[k] = float(np.mean([p[k] for p in per_profile]))
    out["MRR"] = float(np.mean([p["RR"] for p in per_profile]))
    return out
