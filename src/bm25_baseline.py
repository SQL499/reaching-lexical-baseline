"""Baseline léxico BM25 (§6, congelado): BM25Okapi k1=1.5, b=0.75, epsilon=0.25.
Índice/IDF sobre el corpus completo de ofertas; el ranking se restringe a las
candidatas del perfil con get_batch_scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from src.preprocess import lexical_tokens


class BM25Index:
    def __init__(self, ofertas: pd.DataFrame, cfg: dict, log=print):
        p = cfg["bm25"]
        avisoids = ofertas["AVISOID"].to_numpy()
        log(f"[bm25] tokenizando corpus completo ({len(ofertas)} ofertas, tubería léxica §4.1)…")
        corpus = [lexical_tokens(t) for t in ofertas["TEXTO_OFERTA"].tolist()]
        n_empty = sum(1 for c in corpus if not c)
        log(f"[bm25] documentos con 0 tokens tras la tubería: {n_empty}")
        log(f"[bm25] construyendo BM25Okapi(k1={p['k1']}, b={p['b']}, epsilon={p['epsilon']})…")
        self.bm25 = BM25Okapi(corpus, k1=p["k1"], b=p["b"], epsilon=p["epsilon"])
        self.index_of = {int(a): i for i, a in enumerate(avisoids)}
        self.n_docs = len(corpus)

    def score_candidates(self, query_text: str, candidate_avisoids) -> np.ndarray:
        """Scores BM25 de las candidatas del perfil (índices sobre el corpus completo)."""
        q = lexical_tokens(query_text)
        idx = [self.index_of[int(a)] for a in candidate_avisoids]
        if not q:
            return np.zeros(len(idx), dtype=np.float64)
        return np.asarray(self.bm25.get_batch_scores(q, idx), dtype=np.float64)
