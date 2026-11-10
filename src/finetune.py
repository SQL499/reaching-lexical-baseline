"""Fine-tuning condicional (§9): GroupKFold(5) por ID_POSTULANTE sobre TRAIN,
rejilla congelada A1–A3 / B1–B3 (§9.2), selección por NDCG@3 macro CV (desempate MRR)
y criterio GO/NO-GO mecánico (§9.3). El split de evaluación NO interviene.
"""
from __future__ import annotations

import gc
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.common import ROOT
from src.evaluate import aggregate_from_df, semantic_per_profile
from src.preprocess import semantic_text

TMP_DIR = ROOT / "tmp_finetune"


def make_folds(train_canon: pd.DataFrame, n_folds: int) -> list[dict]:
    """GroupKFold determinista (shuffle=False) agrupando por ID_POSTULANTE (§9.1)."""
    gkf = GroupKFold(n_splits=n_folds)
    folds = []
    groups = train_canon["ID_POSTULANTE"].to_numpy()
    for k, (tr_idx, te_idx) in enumerate(gkf.split(train_canon, groups=groups)):
        tr = train_canon.iloc[tr_idx]
        te = train_canon.iloc[te_idx]
        # perfiles utilizables del pliegue retenido: >=1 positivo entre sus 5 candidatas (I13)
        pos = te.groupby("ID_POSTULANTE", sort=False)["ETIQUETA"].sum()
        usable = [pid for pid in pos.index if pos[pid] >= 1]
        folds.append({
            "k": k,
            "train_pairs": tr,
            "eval_pairs": te,
            "eval_usable_profiles": usable,
            "n_perfiles_train": tr["ID_POSTULANTE"].nunique(),
            "n_perfiles_eval": te["ID_POSTULANTE"].nunique(),
        })
    return folds


def build_train_dataset(pairs: pd.DataFrame, loss_kind: str):
    """Datos de entrenamiento §9.2: A=solo positivos (anchor, positive);
    B=todos los pares con etiqueta 1.0/0.0. Textos con tubería semántica §4.2."""
    from datasets import Dataset

    if loss_kind == "mnrl":
        pos = pairs[pairs["ETIQUETA"] == 1]
        return Dataset.from_dict({
            "anchor": [semantic_text(t) for t in pos["TEXTO_PERFIL"]],
            "positive": [semantic_text(t) for t in pos["TEXTO_OFERTA_CANON"]],
        })
    if loss_kind == "cosine":
        return Dataset.from_dict({
            "sentence1": [semantic_text(t) for t in pairs["TEXTO_PERFIL"]],
            "sentence2": [semantic_text(t) for t in pairs["TEXTO_OFERTA_CANON"]],
            "score": [float(e) for e in pairs["ETIQUETA"]],
        })
    raise ValueError(f"Pérdida desconocida: {loss_kind}")


def train_one(cfg: dict, train_pairs: pd.DataFrame, loss_kind: str, epochs: int,
              batch_size: int, run_name: str, log=print, save_dir: Path | None = None):
    """Entrena una corrida de la rejilla desde el checkpoint base (semilla 42, fp32,
    AdamW default, scheduler lineal con warmup_ratio 0.10; ver DECISIONES.md I14)."""
    import torch
    from sentence_transformers import (SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    from sentence_transformers.losses import (CosineSimilarityLoss,
                                              MultipleNegativesRankingLoss)

    from src.semantic_model import load_model

    t0 = time.perf_counter()
    model = load_model(cfg, device="cuda")
    ds = build_train_dataset(train_pairs, loss_kind)
    loss = MultipleNegativesRankingLoss(model) if loss_kind == "mnrl" else CosineSimilarityLoss(model)
    out_dir = TMP_DIR / run_name
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=float(cfg["finetune"]["learning_rate"]),
        warmup_ratio=float(cfg["finetune"]["warmup_ratio"]),
        lr_scheduler_type="linear",
        seed=int(cfg["seed"]),
        data_seed=int(cfg["seed"]),
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
        logging_strategy="no",
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss)
    trainer.train()
    secs = time.perf_counter() - t0
    log(f"[E3] {run_name}: {len(ds)} ejemplos, {epochs} época(s), batch {batch_size}, "
        f"{secs:.1f} s")
    if save_dir is not None:
        model.save(str(save_dir))
        log(f"[E3] modelo final guardado en {save_dir}")
    shutil.rmtree(out_dir, ignore_errors=True)

    def cleanup():
        nonlocal model, trainer
        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return model, secs, len(ds), cleanup


def eval_fold(model, fold: dict, cfg: dict) -> dict:
    """NDCG@3 macro (+ resto de métricas) sobre los perfiles utilizables del pliegue,
    rankeando sus 5 candidatas (§9.1)."""
    df = semantic_per_profile(model, fold["eval_pairs"], fold["eval_usable_profiles"], cfg)
    agg = aggregate_from_df(df)
    return {"per_profile": df, "agg": agg}


def gonogo_decision(cv_table: pd.DataFrame, zs_by_fold: dict[int, float]) -> dict:
    """Criterio §9.3, aplicado mecánicamente.

    cv_table: filas (config, fold, NDCG@3, MRR). zs_by_fold: NDCG@3 zero-shot por pliegue.
    GO si la mejor config (media CV NDCG@3, desempate MRR) supera estrictamente la media
    zero-shot Y mejora (estrictamente) en >=3 de 5 pliegues (I5).
    """
    means = (cv_table.groupby("config")
             .agg(ndcg3_medio=("NDCG@3", "mean"), mrr_medio=("MRR", "mean"))
             .reset_index()
             .sort_values(["ndcg3_medio", "mrr_medio"], ascending=False, kind="mergesort"))
    best = means.iloc[0]
    zs_mean = float(np.mean(list(zs_by_fold.values())))
    best_rows = cv_table[cv_table["config"] == best["config"]].sort_values("fold")
    wins = int(sum(float(r["NDCG@3"]) > zs_by_fold[int(r["fold"])] for _, r in best_rows.iterrows()))
    cond_a = bool(best["ndcg3_medio"] > zs_mean)
    cond_b = bool(wins >= 3)
    return {
        "mejor_config": str(best["config"]),
        "ndcg3_cv_mejor": float(best["ndcg3_medio"]),
        "mrr_cv_mejor": float(best["mrr_medio"]),
        "ndcg3_cv_zeroshot": zs_mean,
        "mejoras_por_pliegue": wins,
        "condicion_a_media_estrictamente_mayor": cond_a,
        "condicion_b_mejora_en_3_de_5": cond_b,
        "decision": "GO" if (cond_a and cond_b) else "NO-GO",
        "ranking_configs": means.to_dict("records"),
    }
