"""E0 — Preparación (§8): compuertas de hashes (§2.1) y aserciones (§2.2, +§3),
textos canónicos (§2.3), verificación por muestreo de TEXTO_OFERTA, harness de métricas
con pruebas §5.4 (pytest, bloqueante), QA de anclajes §5.3 y prechequeos R1/S1.
Artefactos C1 y C2 para la revisión final.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src import data as D  # noqa: E402
from src.common import (GateError, RunRecorder, harness_selfcheck,  # noqa: E402
                        load_config, set_determinism)


def expected_oferta_text(row):
    """Patrón del consolidado (I18, corregido en B1): concatenación con '. ' de las
    secciones título / 'Sector: …' / 'Ocupacion: …' / 'Competencias: …', omitiendo
    las secciones cuyo valor está vacío o son solo espacios (p. ej., ESCO=' ' en
    7 916 ofertas)."""
    import pandas as pd
    parts = []
    nombre = "" if pd.isna(row["NOMBREAVISO"]) else str(row["NOMBREAVISO"])
    if nombre.strip():
        parts.append(nombre)
    for label, col in (("Sector", "SECTOR"), ("Ocupacion", "ESCO"),
                       ("Competencias", "COMPETENCIAS")):
        v = "" if pd.isna(row[col]) else str(row[col])
        if v.strip():
            parts.append(f"{label}: {v}")
    return ". ".join(parts)


def verify_oferta_sampling(ofertas, rec, n=20, seed=42):
    """§2.3: TEXTO_OFERTA corresponde a la concatenación título + sector + ocupación +
    competencias (secciones vacías omitidas; I18/B1)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ofertas), size=n, replace=False)
    mismatches = []
    for i in sorted(int(j) for j in idx):
        row = ofertas.iloc[i]
        expected = expected_oferta_text(row)
        actual = row["TEXTO_OFERTA"]
        ok = expected == actual
        rec.log(f"[muestreo §2.3] fila {i} AVISOID={row['AVISOID']}: {'OK' if ok else 'MISMATCH'}")
        if not ok:
            mismatches.append({"fila": i, "AVISOID": int(row["AVISOID"]),
                               "esperado": expected, "obtenido": actual})
    if mismatches:
        for m in mismatches:
            rec.log(f"  esperado: {m['esperado'][:200]!r}")
            rec.log(f"  obtenido: {m['obtenido'][:200]!r}")
        raise GateError(f"E0: verificación por muestreo de TEXTO_OFERTA falló en "
                        f"{len(mismatches)}/{n} muestras (ver run.log)")
    rec.log(f"[muestreo §2.3] {n}/{n} muestras coinciden con la concatenación "
            f"título + sector + ocupación + competencias.")
    return {"n": n, "seed": seed, "mismatches": 0}


def simulate_anchors(cfg, rec, n_sims=20000, seed=42):
    """QA de anclajes §5.3 (I10): re-simulación desde la estructura de §3.
    No sustituye los valores congelados; una discrepancia material se reporta."""
    dist = {int(k): int(v) for k, v in cfg["eval_structure"]["positives_distribution"].items()}
    n_cand = int(cfg["eval_structure"]["n_candidates"])
    rng = np.random.default_rng(seed)
    w = 1.0 / np.log2(np.arange(2, n_cand + 2))
    cw = np.cumsum(w)
    rels = []
    for k, n_prof in sorted(dist.items()):
        u = rng.random((n_sims, n_prof, n_cand))
        perm = np.argsort(u, axis=2)
        rels.append((perm < k).astype(np.float64))
    rel = np.concatenate(rels, axis=1)
    kk = rel.sum(axis=2).astype(int)
    p1 = rel[:, :, 0]
    p3 = rel[:, :, :3].mean(axis=2)
    n3 = (rel[:, :, :3] @ w[:3]) / cw[np.minimum(kk, 3) - 1]
    n5 = (rel[:, :, :5] @ w[:5]) / cw[np.minimum(kk, 5) - 1]
    rr = 1.0 / (np.argmax(rel, axis=2) + 1)
    sim = {"P@1": float(p1.mean(axis=1).mean()), "P@3": float(p3.mean(axis=1).mean()),
           "NDCG@3": float(n3.mean(axis=1).mean()), "NDCG@5": float(n5.mean(axis=1).mean()),
           "MRR": float(rr.mean(axis=1).mean())}
    frozen = cfg["random_anchors"]
    report = {}
    for k, v in frozen.items():
        s = round(sim[k], 3)
        report[k] = {"congelado_5.3": v, "simulado": round(sim[k], 4), "coincide_3dec": s == round(v, 3)}
        rec.log(f"[anclajes §5.3] {k}: congelado {v} | simulado {sim[k]:.4f} "
                f"({'OK' if s == round(v, 3) else 'DISCREPANCIA'})")
    return report


def main():
    set_determinism(42)
    cfg = load_config()
    rec = RunRecorder("E0_preparacion", cfg)
    try:
        from src.common import verify_hashes
        verify_hashes(cfg, rec.log)

        d = D.load_all(cfg)
        D.run_assertions(d, cfg, rec.log)

        # Textos canónicos §2.3 (ambas particiones; falla si el join deja huecos)
        train_canon = D.build_canonical(d["train"], d["ofertas"])
        val_canon = D.build_canonical(d["val"], d["ofertas"])
        rec.log(f"[canónico §2.3] TRAIN {len(train_canon)} pares, "
                f"VALIDATION {len(val_canon)} pares construidos (join AVISOID sin huecos)")

        # Verificación por muestreo de TEXTO_OFERTA (bloqueante en E0)
        muestreo = verify_oferta_sampling(d["ofertas"], rec, n=20, seed=int(cfg["seed"]))

        # C2: 5 pares con textos canónicos construidos (muestra determinista de TRAIN)
        rng = np.random.default_rng(int(cfg["seed"]))
        sample_idx = sorted(int(i) for i in rng.choice(len(train_canon), size=5, replace=False))
        lines = ["# C2 — 5 pares con textos canónicos construidos (muestra TRAIN, semilla 42)\n"]
        for i in sample_idx:
            r = train_canon.iloc[i]
            lines += [f"\n## Par fila {i} — ID_POSTULANTE={r['ID_POSTULANTE']} | "
                      f"AVISOID={r['AVISOID']} | ETIQUETA={r['ETIQUETA']}",
                      f"\n**TEXTO_PERFIL (canónico, de pares):**\n{r['TEXTO_PERFIL']}",
                      f"\n**TEXTO_OFERTA (canónico, join ofertas_consolidadas):**\n{r['TEXTO_OFERTA_CANON']}"]
        (rec.dir / "C2_pares_canonicos.md").write_text("\n".join(lines), encoding="utf-8")
        rec.log(f"[C2] 5 pares canónicos en {rec.dir / 'C2_pares_canonicos.md'}")

        # Prechequeo §2.5 (S1): vacantes compartidas — verificación documental (I7)
        sv = D.shared_vacancies(d["train"], d["val"])
        rec.log(f"[§2.5] vacantes compartidas={sv['n_vacantes_compartidas']} (doc: 20), "
                f"pares val afectados={sv['n_pares_val_afectados']} (doc: 22), "
                f"positivos afectados={sv['n_positivos_val_afectados']} (doc: 9)")
        doc_match = (sv["n_vacantes_compartidas"] == 20 and sv["n_pares_val_afectados"] == 22
                     and sv["n_positivos_val_afectados"] == 9)
        if not doc_match:
            rec.log("[§2.5] AVISO: cifras difieren de las documentadas; se reporta como incidencia (I7)")

        # Prechequeo §2.3 (R1): perfiles truncados en origen
        tr_trunc = D.truncated_profiles(d["train"], d["perfiles"])
        val_trunc = D.truncated_profiles(d["val"], d["perfiles"])
        usable = set(D.usable_val_profiles(d["val"]))
        val_trunc["utilizable"] = val_trunc["ID_POSTULANTE"].isin(usable) if len(val_trunc) else []
        rec.log(f"[§2.3/R1] truncados: train={len(tr_trunc)} (doc: 13), "
                f"validation={len(val_trunc)} (doc: 4), "
                f"validation utilizables={int(val_trunc['utilizable'].sum()) if len(val_trunc) else 0}")
        prefix_ok = bool(tr_trunc["es_prefijo"].all()) if len(tr_trunc) else True
        prefix_ok &= bool(val_trunc["es_prefijo"].all()) if len(val_trunc) else True
        rec.log(f"[§2.3/R1] propiedad de prefijo en todos los truncados: {prefix_ok}")

        # QA de anclajes §5.3 (20 000 permutaciones, semilla 42)
        anchors = simulate_anchors(cfg, rec, n_sims=20000, seed=int(cfg["seed"]))

        # C1 + compuerta §5.4: selfcheck en proceso y pytest completo (bloqueante)
        harness_selfcheck()
        rec.log("[C1/§5.4] selfcheck del harness en proceso: OK (casos A/B/C + macro)")
        py = ROOT / ".venv" / "Scripts" / "python.exe"
        r = subprocess.run([str(py), "-m", "pytest", "-v", "tests/"], cwd=ROOT,
                           capture_output=True, text=True)
        (rec.dir / "pytest_report.txt").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
        rec.log(f"[pytest] returncode={r.returncode}; reporte en pytest_report.txt")
        if r.returncode != 0:
            print(r.stdout[-3000:])
            raise GateError("COMPUERTA §5.4 FALLIDA — pytest no está en verde; ver pytest_report.txt")
        rec.log("[pytest] 100% pruebas en verde — compuerta E0 superada")

        rec.save_json("e0_report.json", {
            "compuerta_hashes_2.1": "OK",
            "compuerta_aserciones_2.2": "OK (18 aserciones)",
            "muestreo_texto_oferta_2.3": muestreo,
            "vacantes_compartidas_2.5": sv,
            "coincide_documentacion_2.5": doc_match,
            "truncados_R1": {
                "train": int(len(tr_trunc)),
                "validation": int(len(val_trunc)),
                "validation_utilizables": int(val_trunc["utilizable"].sum()) if len(val_trunc) else 0,
                "prefijo_ok": prefix_ok,
                "detalle_validation": val_trunc.to_dict("records") if len(val_trunc) else [],
            },
            "anclajes_5.3_QA": anchors,
            "pytest": "verde",
        })
        rec.close({"estado": "OK"})
    except Exception:
        rec.close({"estado": "ABORTADO"})
        raise


if __name__ == "__main__":
    main()
