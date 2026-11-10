"""Medición de eficiencia (E5, §10, RNF-02): embeddings del corpus precalculados,
50 repeticiones tras 5 de calentamiento, mediana y p95 de codificación / similitud+top-10 /
extremo a extremo, en GPU y CPU (DECISIONES.md I8). Referencia opcional BM25 incluida.
"""
from __future__ import annotations

import os
import platform
import subprocess
import time

import numpy as np


def _pcts(samples: list[float]) -> dict:
    a = np.asarray(samples, dtype=np.float64) * 1000.0  # ms
    return {
        "n": len(samples),
        "mediana_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


def timed_reps(fn, warmup: int, reps: int, sync=None) -> dict:
    for _ in range(warmup):
        fn()
        if sync:
            sync()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        out.append(time.perf_counter() - t0)
    return _pcts(out)


def precompute_corpus(model, texts: list[str], cfg: dict, log=print):
    """Precálculo §10: embeddings de todo el corpus, tiempo total y por lote."""
    bs = int(cfg["semantic"]["corpus_encode_batch_size"])
    chunks = [texts[i:i + bs] for i in range(0, len(texts), bs)]
    embs, batch_times = [], []
    t0 = time.perf_counter()
    for ch in chunks:
        tb = time.perf_counter()
        e = model.encode(ch, batch_size=bs, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False)
        batch_times.append(time.perf_counter() - tb)
        embs.append(e)
    total = time.perf_counter() - t0
    mat = np.vstack(embs).astype(np.float32)
    log(f"[E5] precálculo: {len(texts)} ofertas en {total:.1f} s "
        f"({len(chunks)} lotes de hasta {bs})")
    stats = {
        "n_ofertas": len(texts),
        "batch_size": bs,
        "n_lotes": len(chunks),
        "tiempo_total_s": round(total, 2),
        "por_lote": _pcts(batch_times),
        "dim": int(mat.shape[1]),
    }
    return mat, stats


def bench_device(model, query_text: str, corpus_mat: np.ndarray, cfg: dict,
                 device: str, log=print) -> dict:
    """Pasos §10 en un dispositivo: (1) codificar 1 perfil, (2) similitud + top-10,
    (3) extremo a extremo."""
    import torch

    warm, reps = int(cfg["efficiency"]["warmup"]), int(cfg["efficiency"]["reps"])
    k = int(cfg["efficiency"]["top_k"])
    sync = (lambda: torch.cuda.synchronize()) if device == "cuda" else None

    def enc():
        return model.encode([query_text], batch_size=1, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False)[0]

    r_enc = timed_reps(enc, warm, reps, sync)
    log(f"[E5] ({device}) codificación 1 perfil: mediana {r_enc['mediana_ms']:.1f} ms, "
        f"p95 {r_enc['p95_ms']:.1f} ms")

    q = enc().astype(np.float32)
    if device == "cuda":
        mat_t = torch.from_numpy(corpus_mat).cuda()
        q_t = torch.from_numpy(q).cuda()

        def sim():
            scores = mat_t @ q_t
            torch.topk(scores, k)
    else:
        def sim():
            scores = corpus_mat @ q
            idx = np.argpartition(-scores, k)[:k]
            idx[np.argsort(-scores[idx])]

    r_sim = timed_reps(sim, warm, reps, sync)
    log(f"[E5] ({device}) similitud {corpus_mat.shape[0]}x{corpus_mat.shape[1]} + top-{k}: "
        f"mediana {r_sim['mediana_ms']:.2f} ms, p95 {r_sim['p95_ms']:.2f} ms")

    if device == "cuda":
        def e2e():
            e = model.encode([query_text], batch_size=1, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False)[0]
            qt = torch.from_numpy(e.astype(np.float32)).cuda()
            scores = mat_t @ qt
            torch.topk(scores, k)
    else:
        def e2e():
            e = model.encode([query_text], batch_size=1, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False)[0].astype(np.float32)
            scores = corpus_mat @ e
            idx = np.argpartition(-scores, k)[:k]
            idx[np.argsort(-scores[idx])]

    r_e2e = timed_reps(e2e, warm, reps, sync)
    log(f"[E5] ({device}) extremo a extremo: mediana {r_e2e['mediana_ms']:.1f} ms, "
        f"p95 {r_e2e['p95_ms']:.1f} ms")
    return {"codificacion_1_perfil": r_enc, "similitud_top10": r_sim, "extremo_a_extremo": r_e2e}


def bench_bm25(bm25_index, query_text: str, cfg: dict, log=print) -> dict:
    """Referencia opcional §10: consulta BM25 sobre el corpus completo + top-10."""
    from src.preprocess import lexical_tokens

    warm, reps = int(cfg["efficiency"]["warmup"]), int(cfg["efficiency"]["reps"])
    k = int(cfg["efficiency"]["top_k"])

    def q():
        toks = lexical_tokens(query_text)
        scores = np.asarray(bm25_index.bm25.get_scores(toks))
        idx = np.argpartition(-scores, k)[:k]
        idx[np.argsort(-scores[idx])]

    r = timed_reps(q, warm, reps)
    log(f"[E5] BM25 consulta corpus completo + top-{k}: mediana {r['mediana_ms']:.0f} ms, "
        f"p95 {r['p95_ms']:.0f} ms")
    return r


def hardware_info() -> dict:
    import psutil
    import torch

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    return {
        "os": platform.platform(),
        "cpu": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
        "nucleos_fisicos": psutil.cpu_count(logical=False),
        "nucleos_logicos": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "gpu_nvidia_smi": gpu,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(),
    }
