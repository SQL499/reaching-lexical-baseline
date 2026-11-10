"""Carga de datos (utf-8-sig), aserciones de integridad (§2.2, bloqueantes),
textos canónicos (§2.3) y verificaciones documentales (§2.5, §3)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common import ROOT, GateError


def _read_csv(path: Path, encoding: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding=encoding, dtype={"ID_POSTULANTE": str})


def load_all(cfg: dict) -> dict:
    enc = cfg["encoding"]
    p = cfg["paths"]
    d = {
        "train": _read_csv(ROOT / p["train_pairs"], enc),
        "val": _read_csv(ROOT / p["val_pairs"], enc),
        "ofertas": _read_csv(ROOT / p["ofertas"], enc),
        "perfiles": _read_csv(ROOT / p["perfiles"], enc),
        "just_train": _read_csv(ROOT / p["just_train"], enc),
        "just_val": _read_csv(ROOT / p["just_val"], enc),
    }
    for key in ("train", "val"):
        d[key]["AVISOID"] = d[key]["AVISOID"].astype("int64")
        d[key]["ETIQUETA"] = d[key]["ETIQUETA"].astype("int64")
        d[key]["PERFIL_SIN_RELEVANTE"] = d[key]["PERFIL_SIN_RELEVANTE"].astype("int64")
    d["ofertas"]["AVISOID"] = d["ofertas"]["AVISOID"].astype("int64")
    d["just_train"]["AVISOID"] = d["just_train"]["AVISOID"].astype("int64")
    d["just_val"]["AVISOID"] = d["just_val"]["AVISOID"].astype("int64")
    return d


def run_assertions(d: dict, cfg: dict, log=print) -> None:
    """Aserciones §2.2 (+ distribución §3, ver DECISIONES.md I10). Aborta si alguna falla."""
    train, val, ofertas, perfiles = d["train"], d["val"], d["ofertas"], d["perfiles"]
    checks: list[tuple[str, bool, str]] = []

    def add(desc: str, ok: bool, detail: str = ""):
        checks.append((desc, bool(ok), detail))

    # --- TRAIN ---
    add("TRAIN: 800 filas", len(train) == 800, f"obtenido {len(train)}")
    n_prof_tr = train["ID_POSTULANTE"].nunique()
    add("TRAIN: 160 ID_POSTULANTE únicos", n_prof_tr == 160, f"obtenido {n_prof_tr}")
    cand_tr = train.groupby("ID_POSTULANTE").size()
    add("TRAIN: exactamente 5 candidatas por perfil", (cand_tr == 5).all(),
        f"min={cand_tr.min()}, max={cand_tr.max()}")
    n_pos_tr = int(train["ETIQUETA"].sum())
    add("TRAIN: 272 pares con ETIQUETA=1", n_pos_tr == 272, f"obtenido {n_pos_tr}")
    n_av_tr = train["AVISOID"].nunique()
    add("TRAIN: 643 AVISOID únicos", n_av_tr == 643, f"obtenido {n_av_tr}")
    pos_por_perfil_tr = train.groupby("ID_POSTULANTE")["ETIQUETA"].sum()
    n_ut_tr = int((pos_por_perfil_tr >= 1).sum())
    add("TRAIN: 109 perfiles con >=1 positivo", n_ut_tr == 109, f"obtenido {n_ut_tr}")

    # --- VALIDATION ---
    add("VALIDATION: 288 filas", len(val) == 288, f"obtenido {len(val)}")
    n_prof_v = val["ID_POSTULANTE"].nunique()
    add("VALIDATION: 48 perfiles", n_prof_v == 48, f"obtenido {n_prof_v}")
    cand_v = val.groupby("ID_POSTULANTE").size()
    add("VALIDATION: exactamente 6 candidatas por perfil", (cand_v == 6).all(),
        f"min={cand_v.min()}, max={cand_v.max()}")
    n_pos_v = int(val["ETIQUETA"].sum())
    add("VALIDATION: 48 positivos", n_pos_v == 48, f"obtenido {n_pos_v}")
    n_av_v = val["AVISOID"].nunique()
    add("VALIDATION: 285 AVISOID únicos", n_av_v == 285, f"obtenido {n_av_v}")
    sin_rel = val.groupby("ID_POSTULANTE")["PERFIL_SIN_RELEVANTE"].first()
    n_usable = int((sin_rel == 0).sum())
    n_sinrel = int((sin_rel == 1).sum())
    add("VALIDATION: 32 perfiles utilizables", n_usable == 32, f"obtenido {n_usable}")
    add("VALIDATION: 16 perfiles con PERFIL_SIN_RELEVANTE=1", n_sinrel == 16, f"obtenido {n_sinrel}")

    # --- Cruzadas ---
    inter = set(train["ID_POSTULANTE"]) & set(val["ID_POSTULANTE"])
    add("Intersección ID_POSTULANTE train∩validation = 0", len(inter) == 0, f"obtenido {len(inter)}")

    flag_const = val.groupby("ID_POSTULANTE")["PERFIL_SIN_RELEVANTE"].nunique()
    add("PERFIL_SIN_RELEVANTE constante dentro de cada perfil", (flag_const == 1).all(), "")
    pos_v = val.groupby("ID_POSTULANTE")["ETIQUETA"].sum()
    incons = int(((pos_v == 0) != (sin_rel == 1)).sum())
    add("PERFIL_SIN_RELEVANTE consistente con etiquetas (0 inconsistencias)", incons == 0,
        f"inconsistencias {incons}")

    dup_tr = int(train.duplicated(subset=["ID_POSTULANTE", "AVISOID"]).sum())
    dup_v = int(val.duplicated(subset=["ID_POSTULANTE", "AVISOID"]).sum())
    add("Sin pares (ID_POSTULANTE, AVISOID) duplicados", dup_tr == 0 and dup_v == 0,
        f"train {dup_tr}, validation {dup_v}")

    of_ids = set(ofertas["AVISOID"])
    miss_of = set(train["AVISOID"]) | set(val["AVISOID"])
    miss_of -= of_ids
    add("Todo AVISOID de los pares existe en ofertas_consolidadas", len(miss_of) == 0,
        f"faltantes {sorted(miss_of)[:10]}")
    pf_ids = set(perfiles["ID_POSTULANTE"])
    miss_pf = (set(train["ID_POSTULANTE"]) | set(val["ID_POSTULANTE"])) - pf_ids
    add("Todo ID_POSTULANTE existe en perfiles_consolidados", len(miss_pf) == 0,
        f"faltantes {sorted(miss_pf)[:10]}")

    # --- §3: distribución de positivos por perfil utilizable (sustenta anclajes §5.3) ---
    dist = pos_v[sin_rel == 0].value_counts().to_dict()
    expected_dist = {int(k): v for k, v in cfg["eval_structure"]["positives_distribution"].items()}
    add("§3: distribución de positivos por perfil utilizable = {1:20, 2:8, 3:4}",
        {int(k): int(v) for k, v in dist.items()} == expected_dist, f"obtenido {dist}")

    failures = [(desc, det) for desc, ok, det in checks if not ok]
    for desc, ok, det in checks:
        log(f"[aserción §2.2] {'OK ' if ok else 'FALLO'} — {desc}" + (f" ({det})" if det and not ok else ""))
    if failures:
        detail = "\n".join(f"  - {desc}: {det}" for desc, det in failures)
        raise GateError(f"COMPUERTA §2.2 FALLIDA — {len(failures)} aserción(es) de integridad:\n{detail}")
    log(f"[aserción §2.2] Las {len(checks)} aserciones pasan.")


def build_canonical(pairs: pd.DataFrame, ofertas: pd.DataFrame) -> pd.DataFrame:
    """Textos canónicos §2.3: TEXTO_PERFIL desde el archivo de pares; TEXTO_OFERTA por
    join con ofertas_consolidadas.csv vía AVISOID. TIPO_CANDIDATA no se usa como entrada."""
    of = ofertas[["AVISOID", "TEXTO_OFERTA"]].rename(columns={"TEXTO_OFERTA": "TEXTO_OFERTA_CANON"})
    out = pairs.merge(of, on="AVISOID", how="left", validate="many_to_one")
    if out["TEXTO_OFERTA_CANON"].isna().any():
        faltan = out.loc[out["TEXTO_OFERTA_CANON"].isna(), "AVISOID"].tolist()
        raise GateError(f"Join canónico §2.3 falló: AVISOID sin texto de oferta: {faltan[:10]}")
    return out


def usable_val_profiles(val: pd.DataFrame) -> list[str]:
    """§2.4: perfiles utilizables (PERFIL_SIN_RELEVANTE=0), en orden de aparición del CSV."""
    flags = val.groupby("ID_POSTULANTE", sort=False)["PERFIL_SIN_RELEVANTE"].first()
    return [pid for pid in flags.index if flags[pid] == 0]


def shared_vacancies(train: pd.DataFrame, val: pd.DataFrame) -> dict:
    """§2.5: vacantes compartidas entre particiones y su huella en evaluación."""
    shared = set(train["AVISOID"]) & set(val["AVISOID"])
    val_rows = val[val["AVISOID"].isin(shared)]
    return {
        "n_vacantes_compartidas": len(shared),
        "avisoids_compartidos": sorted(int(a) for a in shared),
        "n_pares_val_afectados": int(len(val_rows)),
        "n_positivos_val_afectados": int(val_rows["ETIQUETA"].sum()),
        "perfiles_val_afectados": sorted(val_rows["ID_POSTULANTE"].unique().tolist()),
    }


def truncated_profiles(pairs: pd.DataFrame, perfiles: pd.DataFrame) -> pd.DataFrame:
    """§2.3/R1: perfiles cuyo TEXTO_PERFIL de pares difiere del consolidado (prefijo de este).
    Devuelve DataFrame con ID, ambos textos y verificación de prefijo."""
    txt_pairs = pairs.groupby("ID_POSTULANTE", sort=False)["TEXTO_PERFIL"].first()
    cons = perfiles.set_index("ID_POSTULANTE")["TEXTO_PERFIL"]
    rows = []
    for pid, t in txt_pairs.items():
        tc = cons.get(pid)
        if tc is not None and isinstance(tc, str) and tc != t:
            rows.append({
                "ID_POSTULANTE": pid,
                "len_pares": len(t),
                "len_consolidado": len(tc),
                "es_prefijo": tc.startswith(t),
                "delta_chars": len(tc) - len(t),
            })
    return pd.DataFrame(rows)
