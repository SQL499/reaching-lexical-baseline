"""Vectores de prueba del harness de métricas (§5.4, tolerancia ±0.0005) + caso de error
(perfil sin relevantes lanza excepción) + prueba del desempate determinista (§5.2).

PROHIBIDO editar los valores esperados: provienen del protocolo congelado.
"""
import numpy as np
import pytest

from src import metrics as M

TOL = 0.0005

CASES = {
    "A": ([0, 1, 0, 0, 1, 0], {"P@1": 0.0, "P@3": 0.3333, "NDCG@3": 0.3869, "NDCG@5": 0.6240, "RR": 0.5}),
    "B": ([1, 0, 0, 0, 0, 0], {"P@1": 1.0, "P@3": 0.3333, "NDCG@3": 1.0000, "NDCG@5": 1.0000, "RR": 1.0}),
    "C": ([0, 0, 0, 1, 0, 0], {"P@1": 0.0, "P@3": 0.0000, "NDCG@3": 0.0000, "NDCG@5": 0.4307, "RR": 0.25}),
}


@pytest.mark.parametrize("name", list(CASES))
def test_vectores_54(name):
    rel, expected = CASES[name]
    got = M.profile_metrics(rel)
    for k, v in expected.items():
        assert abs(got[k] - v) <= TOL, f"caso {name}, {k}: esperado {v}, obtenido {got[k]}"


def test_agregacion_macro_54():
    per_profile = [M.profile_metrics(rel) for rel, _ in CASES.values()]
    agg = M.aggregate(per_profile)
    expected = {"P@1": 0.3333, "P@3": 0.2222, "NDCG@3": 0.4623, "NDCG@5": 0.6849, "MRR": 0.5833}
    for k, v in expected.items():
        assert abs(agg[k] - v) <= TOL, f"agregación {k}: esperado {v}, obtenido {agg[k]}"


def test_perfil_sin_relevantes_lanza_excepcion():
    with pytest.raises(M.NoRelevantError):
        M.profile_metrics([0, 0, 0, 0, 0, 0])
    with pytest.raises(M.NoRelevantError):
        M.reciprocal_rank([0, 0, 0])
    with pytest.raises(M.NoRelevantError):
        M.ndcg_at_k([0, 0, 0, 0], 3)


def test_desempate_determinista():
    # scores empatados -> orden por AVISOID ascendente (§5.2)
    avisoids = [50, 10, 30, 20, 40, 60]
    scores = [1.0, 0.5, 0.5, 0.5, 2.0, 0.1]
    rk = M.rank_by_score(avisoids, scores)
    assert rk["avisoids_ranked"].tolist() == [40, 50, 10, 20, 30, 60]
    assert rk["n_grupos_empate"] == 1
    assert rk["n_items_empatados"] == 3
    # empate total (p. ej., BM25 con score 0): orden = AVISOID ascendente
    rk0 = M.rank_by_score([3, 1, 2], [0.0, 0.0, 0.0])
    assert rk0["avisoids_ranked"].tolist() == [1, 2, 3]
    assert rk0["n_grupos_empate"] == 1
    assert rk0["n_items_empatados"] == 3


def test_k_mayor_que_lista_falla():
    with pytest.raises(ValueError):
        M.precision_at_k([1, 0], 3)
    with pytest.raises(ValueError):
        M.ndcg_at_k([1, 0], 5)


def test_orden_estable_scores_distintos():
    avisoids = np.array([7, 3, 9], dtype=np.int64)
    scores = np.array([0.2, 0.9, 0.5])
    rk = M.rank_by_score(avisoids, scores)
    assert rk["avisoids_ranked"].tolist() == [3, 9, 7]
    assert rk["n_grupos_empate"] == 0
