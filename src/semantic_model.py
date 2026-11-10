"""Modelo semántico (§7, congelado): paraphrase-multilingual-MiniLM-L12-v2 desde caché
local, max_seq_length=256 (§4.2), embeddings normalizados, afinidad = producto punto.
"""
from __future__ import annotations

import numpy as np

from src.common import ROOT


def load_model(cfg: dict, model_dir: str | None = None, device: str = "cuda"):
    from sentence_transformers import SentenceTransformer

    path = ROOT / (model_dir or cfg["paths"]["base_model_dir"])
    if not path.exists():
        raise FileNotFoundError(f"Caché del modelo no encontrada en {path}; ejecutar scripts/setup_assets.py")
    model = SentenceTransformer(str(path), device=device)
    model.max_seq_length = int(cfg["semantic"]["max_seq_length"])
    return model


def encode(model, texts: list[str], cfg: dict, batch_size: int | None = None) -> np.ndarray:
    """Embeddings fp32 normalizados (batch fijo en config para reproducibilidad, I16)."""
    bs = batch_size or int(cfg["semantic"]["eval_encode_batch_size"])
    emb = model.encode(
        texts,
        batch_size=bs,
        normalize_embeddings=bool(cfg["semantic"]["normalize_embeddings"]),
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(emb, dtype=np.float32)


def dot_scores(query_emb: np.ndarray, cand_embs: np.ndarray) -> np.ndarray:
    """Afinidad §7: producto punto (equivale a coseno con embeddings normalizados)."""
    return (cand_embs @ query_emb).astype(np.float64)
