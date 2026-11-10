"""Evaluación de rankings sobre el split de evaluación (E1/E2/E4) y análisis
estadístico §9.4: bootstrap (IC 95%) y test de permutación pareado.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import metrics as M

METRIC_KEYS = ["P@1", "P@3", "NDCG@3", "NDCG@5"]  # + RR/MRR


def evaluate_ranking(pairs_canon: pd.DataFrame, profile_ids: list[str],
                     score_fn, query_text_col: str = "TEXTO_PERFIL") -> pd.DataFrame:
    """Evalúa un método sobre los perfiles dados.

    score_fn(profile_id, query_text, cand_df) -> np.ndarray de scores alineados a cand_df.
    Devuelve una fila por perfil con las 5 métricas, empates y el ranking completo con scores.
    """
    rows = []
    by_profile = dict(tuple(pairs_canon.groupby("ID_POSTULANTE", sort=False)))
    for pid in profile_ids:
        cand = by_profile[pid]
        qtext = cand[query_text_col].iloc[0]
        scores = np.asarray(score_fn(pid, qtext, cand), dtype=np.float64)
        rk = M.rank_by_score(cand["AVISOID"].to_numpy(), scores)
        rel_ranked = cand["ETIQUETA"].to_numpy()[rk["order"]].tolist()
        met = M.profile_metrics(rel_ranked)
        ranking_str = ";".join(
            f"{pos + 1}:{int(a)}:{s:.6f}:{int(r)}"
            for pos, (a, s, r) in enumerate(zip(rk["avisoids_ranked"], rk["scores_ranked"], rel_ranked))
        )
        rows.append({
            "ID_POSTULANTE": pid,
            "n_candidatas": len(cand),
            "n_relevantes": int(cand["ETIQUETA"].sum()),
            **met,
            "n_grupos_empate": rk["n_grupos_empate"],
            "n_items_empatados": rk["n_items_empatados"],
            "ranking": ranking_str,
        })
    return pd.DataFrame(rows)


def aggregate_from_df(per_profile: pd.DataFrame) -> dict:
    recs = per_profile[["P@1", "P@3", "NDCG@3", "NDCG@5", "RR"]].to_dict("records")
    return M.aggregate(recs)


def tie_summary(per_profile: pd.DataFrame) -> dict:
    """§5.2: los empates existen y deben ser visibles."""
    return {
        "perfiles_con_empates": int((per_profile["n_grupos_empate"] > 0).sum()),
        "total_grupos_empate": int(per_profile["n_grupos_empate"].sum()),
        "total_items_empatados": int(per_profile["n_items_empatados"].sum()),
    }


def _metric_matrix(per_profile: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Matriz (n_perfiles × 5) en el orden canónico de métricas agregadas."""
    cols = ["P@1", "P@3", "NDCG@3", "NDCG@5", "RR"]
    return per_profile.sort_values("ID_POSTULANTE")[cols].to_numpy(dtype=np.float64), cols


def bootstrap_cis(per_profile_by_method: dict[str, pd.DataFrame], cfg: dict) -> dict:
    """IC 95% bootstrap §9.4: 10 000 remuestreos con reemplazo sobre los perfiles
    utilizables (semilla 42), percentiles 2.5/97.5, por métrica y método, y para la
    diferencia pareada entre cada par de métodos declarado.

    El mismo conjunto de índices de remuestreo se aplica a todos los métodos
    (consistencia pareada).
    """
    n_boot = int(cfg["stats"]["bootstrap_n"])
    pcts = cfg["stats"]["ci_percentiles"]
    methods = list(per_profile_by_method.keys())
    mats = {}
    ids_ref = None
    for m in methods:
        df = per_profile_by_method[m].sort_values("ID_POSTULANTE")
        ids = df["ID_POSTULANTE"].tolist()
        if ids_ref is None:
            ids_ref = ids
        elif ids != ids_ref:
            raise ValueError(f"Perfiles no alineados entre métodos ({m})")
        mats[m], cols = _metric_matrix(per_profile_by_method[m])
    n = len(ids_ref)
    rng = np.random.default_rng(int(cfg["seed"]))
    idx = rng.integers(0, n, size=(n_boot, n))

    out = {"n_perfiles": n, "n_bootstrap": n_boot, "percentiles": pcts, "metodos": {}}
    agg_names = {"RR": "MRR"}
    for m in methods:
        boot_means = mats[m][idx].mean(axis=1)  # (n_boot, 5)
        entry = {}
        for j, c in enumerate(cols):
            name = agg_names.get(c, c)
            lo, hi = np.percentile(boot_means[:, j], pcts)
            entry[name] = {"punto": float(mats[m][:, j].mean()), "ci95": [float(lo), float(hi)]}
        out["metodos"][m] = entry
    return out, idx, mats, cols, ids_ref


def paired_diff_ci(mats: dict, idx: np.ndarray, cols: list[str], a: str, b: str, pcts) -> dict:
    """IC bootstrap de la diferencia pareada (a − b) por métrica."""
    diff = mats[a] - mats[b]
    boot = diff[idx].mean(axis=1)
    agg_names = {"RR": "MRR"}
    out = {}
    for j, c in enumerate(cols):
        name = agg_names.get(c, c)
        lo, hi = np.percentile(boot[:, j], pcts)
        out[name] = {"punto": float(diff[:, j].mean()), "ci95": [float(lo), float(hi)]}
    return out


def semantic_per_profile(model, pairs_canon: pd.DataFrame, profile_ids: list[str],
                         cfg: dict, profile_text_override: dict[str, str] | None = None) -> pd.DataFrame:
    """Evaluación de un modelo de embeddings sobre pares canónicos (§7):
    codifica textos únicos (batch fijo, fp32, normalizados) y rankea por producto punto.
    profile_text_override permite sustituir el texto de consulta de perfiles concretos (R1).
    """
    from src.preprocess import semantic_text
    from src.semantic_model import dot_scores, encode

    sub = pairs_canon[pairs_canon["ID_POSTULANTE"].isin(set(profile_ids))]
    prof_text = {}
    for pid, g in sub.groupby("ID_POSTULANTE", sort=False):
        raw = g["TEXTO_PERFIL"].iloc[0]
        if profile_text_override and pid in profile_text_override:
            raw = profile_text_override[pid]
        prof_text[pid] = semantic_text(raw)
    offers = sub[["AVISOID", "TEXTO_OFERTA_CANON"]].drop_duplicates("AVISOID").sort_values("AVISOID")
    offer_texts = [semantic_text(t) for t in offers["TEXTO_OFERTA_CANON"].tolist()]

    prof_ids_order = list(profile_ids)
    prof_embs = encode(model, [prof_text[p] for p in prof_ids_order], cfg)
    offer_embs = encode(model, offer_texts, cfg)
    emb_of_profile = dict(zip(prof_ids_order, prof_embs))
    emb_of_offer = dict(zip(offers["AVISOID"].tolist(), offer_embs))

    def score_fn(pid, qtext, cand):
        cand_mat = np.stack([emb_of_offer[int(a)] for a in cand["AVISOID"]])
        return dot_scores(emb_of_profile[pid], cand_mat)

    return evaluate_ranking(sub, prof_ids_order, score_fn)


def permutation_tests(per_profile_a: pd.DataFrame, per_profile_b: pd.DataFrame, cfg: dict) -> dict:
    """Test de permutación pareado §9.4 (a vs b): 10 000 permutaciones de signo sobre
    las diferencias por perfil (semilla 42), bilateral, para NDCG@3, MRR y P@1.
    p = (1 + #{|T_perm| >= |T_obs|}) / (1 + N)  (DECISIONES.md I3).
    """
    n_perm = int(cfg["stats"]["permutation_n"])
    target = cfg["stats"]["permutation_metrics"]  # nombres agregados
    col_of = {"NDCG@3": "NDCG@3", "MRR": "RR", "P@1": "P@1"}
    a = per_profile_a.sort_values("ID_POSTULANTE")
    b = per_profile_b.sort_values("ID_POSTULANTE")
    if a["ID_POSTULANTE"].tolist() != b["ID_POSTULANTE"].tolist():
        raise ValueError("Perfiles no alineados para el test de permutación")
    n = len(a)
    rng = np.random.default_rng(int(cfg["seed"]))
    signs = rng.integers(0, 2, size=(n_perm, n)) * 2 - 1
    out = {"n_permutaciones": n_perm, "n_perfiles": n, "alpha": cfg["stats"]["alpha"], "tests": {}}
    for name in target:
        d = a[col_of[name]].to_numpy(np.float64) - b[col_of[name]].to_numpy(np.float64)
        t_obs = d.mean()
        t_perm = (signs * d).mean(axis=1)
        p = (1 + int((np.abs(t_perm) >= abs(t_obs)).sum())) / (1 + n_perm)
        out["tests"][name] = {
            "diferencia_media": float(t_obs),
            "p_valor": float(p),
            "significativo_alpha_0.05": bool(p < cfg["stats"]["alpha"]),
        }
    return out
