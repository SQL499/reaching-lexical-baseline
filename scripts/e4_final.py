"""E4 — Evaluación final (§8): UNA sola pasada del modelo final sobre evaluación.
Tabla completa (Aleatorio / BM25 / Zero-shot / Fine-tuned si GO) con IC bootstrap y
test de permutación (§9.4), robustez R1 (§2.3), sensibilidad S1 (§2.5) y volcado de
datos para el análisis cualitativo de errores (§9.4).

Disciplina de pasada única: si existe el marker E4_EJECUTADO.marker se aborta; una
re-ejecución solo procede por bug documentado en DECISIONES.md (variable de entorno
E4_RERUN_POR_BUG=1), regla 7 del kickoff.
"""
import datetime
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import data as D  # noqa: E402
from src.common import (GateError, RunRecorder, git_head,  # noqa: E402
                        harness_selfcheck, load_config, set_determinism,
                        verify_hashes)
from src.evaluate import (aggregate_from_df, bootstrap_cis, paired_diff_ci,  # noqa: E402
                          permutation_tests, semantic_per_profile, tie_summary)
from src.metrics import aggregate  # noqa: E402
from src.semantic_model import load_model  # noqa: E402


def parse_ranking(s: str):
    out = []
    for item in s.split(";"):
        pos, avisoid, score, rel = item.split(":")
        out.append({"pos": int(pos), "AVISOID": int(avisoid),
                    "score": float(score), "rel": int(rel)})
    return out


def main():
    set_determinism(42)
    cfg = load_config()
    marker = ROOT / cfg["paths"]["results"] / "E4_final" / "E4_EJECUTADO.marker"
    if marker.exists() and os.environ.get("E4_RERUN_POR_BUG") != "1":
        raise GateError(
            "E4 ya fue ejecutado (disciplina de pasada única, §8). Re-ejecutar SOLO por "
            "bug documentado en DECISIONES.md, con E4_RERUN_POR_BUG=1."
        )
    rec = RunRecorder("E4_final", cfg)
    try:
        verify_hashes(cfg, rec.log)
        d = D.load_all(cfg)
        D.run_assertions(d, cfg, rec.log)
        harness_selfcheck()

        gonogo_path = ROOT / cfg["paths"]["results"] / "E3_finetune" / "gonogo.json"
        if not gonogo_path.exists():
            raise GateError("E4 requiere E3 previo (orden §8); falta gonogo.json")
        decision = json.loads(gonogo_path.read_text(encoding="utf-8"))
        is_go = decision["decision"] == "GO"
        rec.log(f"[E4] decisión E3: {decision['decision']} (mejor config {decision['mejor_config']})")

        e1 = ROOT / cfg["paths"]["results"] / "E1_bm25" / "per_profile.csv"
        e2 = ROOT / cfg["paths"]["results"] / "E2_zeroshot" / "per_profile.csv"
        if not (e1.exists() and e2.exists()):
            raise GateError("E4 requiere E1 y E2 previos (orden §8)")
        per_bm25 = pd.read_csv(e1, encoding="utf-8", dtype={"ID_POSTULANTE": str})
        per_zs = pd.read_csv(e2, encoding="utf-8", dtype={"ID_POSTULANTE": str})

        val_canon = D.build_canonical(d["val"], d["ofertas"])
        usable = D.usable_val_profiles(d["val"])

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if is_go:
            model_dir = cfg["paths"]["finetuned_final_dir"]
            nombre_final = f"Fine-tuned ({decision['mejor_config']})"
        else:
            model_dir = cfg["paths"]["base_model_dir"]
            nombre_final = "Zero-shot (modelo final por NO-GO)"
        rec.log(f"[E4] modelo final: {nombre_final} desde {model_dir} | dispositivo {device}")
        model = load_model(cfg, model_dir=model_dir, device=device)

        # ÚNICA pasada del modelo final sobre evaluación
        per_final = semantic_per_profile(model, val_canon, usable, cfg)
        agg_final = aggregate_from_df(per_final)
        ties = tie_summary(per_final)
        per_final.to_csv(rec.dir / "per_profile.csv", index=False, encoding="utf-8")
        rec.log(f"[artefacto] {rec.dir / 'per_profile.csv'}")
        for k, v in agg_final.items():
            rec.log(f"[E4] modelo final {k} = {v:.4f} (azar {cfg['random_anchors'][k]})")
        rec.log(f"[E4] empates: {ties}")

        if not is_go:
            same = np.allclose(
                per_final.sort_values("ID_POSTULANTE")[["P@1", "P@3", "NDCG@3", "NDCG@5", "RR"]].to_numpy(),
                per_zs.sort_values("ID_POSTULANTE")[["P@1", "P@3", "NDCG@3", "NDCG@5", "RR"]].to_numpy(),
                atol=1e-12)
            rec.log(f"[E4] verificación NO-GO: métricas del modelo final idénticas a E2: {same}")
            if not same:
                raise GateError("E4: con NO-GO el modelo final es el zero-shot y debería "
                                "reproducir E2 bit-a-bit; discrepancia detectada (posible bug)")

        # §9.4 — tabla completa con IC bootstrap; diferencia pareada y permutación final vs BM25
        methods = {"BM25": per_bm25, "Zero-shot": per_zs}
        if is_go:
            methods["Fine-tuned"] = per_final
        final_key = "Fine-tuned" if is_go else "Zero-shot"
        cis, idx, mats, cols, ids = bootstrap_cis(methods, cfg)
        diff_ci = paired_diff_ci(mats, idx, cols, final_key, "BM25", cfg["stats"]["ci_percentiles"])
        perm = permutation_tests(per_final, per_bm25, cfg)
        for m, t in perm["tests"].items():
            rec.log(f"[E4] final vs BM25 — {m}: dif={t['diferencia_media']:+.4f}, p={t['p_valor']:.4f}")

        # R1 (§2.3): texto consolidado completo para perfiles utilizables truncados en origen
        val_trunc = D.truncated_profiles(d["val"], d["perfiles"])
        usable_set = set(usable)
        afectados = [r["ID_POSTULANTE"] for _, r in val_trunc.iterrows()
                     if r["ID_POSTULANTE"] in usable_set]
        rec.log(f"[R1] perfiles utilizables afectados por truncamiento: {len(afectados)} "
                f"(protocolo: 4 identificados sobre v1)")
        cons_text = d["perfiles"].set_index("ID_POSTULANTE")["TEXTO_PERFIL"].to_dict()
        override = {pid: cons_text[pid] for pid in afectados}
        per_r1 = semantic_per_profile(model, val_canon, usable, cfg, profile_text_override=override)
        agg_r1 = aggregate_from_df(per_r1)
        per_r1.to_csv(rec.dir / "r1_per_profile.csv", index=False, encoding="utf-8")
        deltas = []
        pf = per_final.set_index("ID_POSTULANTE")
        pr = per_r1.set_index("ID_POSTULANTE")
        for pid in afectados:
            deltas.append({"ID_POSTULANTE": pid,
                           **{f"delta_{k}": float(pr.loc[pid, k] - pf.loc[pid, k])
                              for k in ["P@1", "P@3", "NDCG@3", "NDCG@5", "RR"]}})
        r1 = {"perfiles_afectados": afectados, "n_afectados": len(afectados),
              "agregados_texto_canonico": agg_final, "agregados_texto_consolidado": agg_r1,
              "deltas_por_perfil_afectado": deltas}
        rec.save_json("r1_robustez.json", r1)
        rec.log(f"[R1] canónico NDCG@3={agg_final['NDCG@3']:.4f} vs consolidado "
                f"NDCG@3={agg_r1['NDCG@3']:.4f}")

        # S1 (§2.5): subconjunto complementario sin vacantes compartidas
        sv = D.shared_vacancies(d["train"], d["val"])
        shared = set(sv["avisoids_compartidos"])
        cand_by_prof = val_canon.groupby("ID_POSTULANTE")["AVISOID"].apply(set)
        con_compartida = [p for p in usable if cand_by_prof[p] & shared]
        complemento = [p for p in usable if p not in set(con_compartida)]
        recs_comp = per_final[per_final["ID_POSTULANTE"].isin(complemento)]
        agg_comp = aggregate(recs_comp[["P@1", "P@3", "NDCG@3", "NDCG@5", "RR"]].to_dict("records"))
        s1 = {"n_vacantes_compartidas": sv["n_vacantes_compartidas"],
              "perfiles_utilizables_con_vacante_compartida": sorted(con_compartida),
              "n_con_compartida": len(con_compartida),
              "n_complemento": len(complemento),
              "agregados_32_perfiles": agg_final,
              "agregados_subconjunto_complementario": agg_comp}
        rec.save_json("s1_sensibilidad.json", s1)
        rec.log(f"[S1] 32 perfiles NDCG@3={agg_final['NDCG@3']:.4f} | complemento "
                f"(n={len(complemento)}) NDCG@3={agg_comp['NDCG@3']:.4f}")

        # §9.4 — volcado de datos para análisis cualitativo (>=5 perfiles por |ΔNDCG@3|)
        pb = per_bm25.set_index("ID_POSTULANTE")
        diffs = [(pid, float(pf.loc[pid, "NDCG@3"] - pb.loc[pid, "NDCG@3"])) for pid in usable]
        diffs.sort(key=lambda x: -abs(x[1]))
        umbral = abs(diffs[4][1])
        seleccion = [x for x in diffs if abs(x[1]) >= umbral and abs(x[1]) > 0]
        if len(seleccion) < 5:
            seleccion = diffs[:5]
        just = d["just_val"].set_index(["ID_POSTULANTE", "AVISOID"])["JUSTIFICACION"].to_dict()
        nombres = d["ofertas"].set_index("AVISOID")["NOMBREAVISO"].to_dict()
        lines = ["# E4 — Datos para análisis cualitativo (§9.4)",
                 f"\nSelección: los {len(seleccion)} perfiles con mayor |ΔNDCG@3| "
                 f"({final_key} − BM25).\n"]
        for pid, dlt in seleccion:
            g = val_canon[val_canon["ID_POSTULANTE"] == pid]
            lines.append(f"\n---\n## Perfil {pid} | ΔNDCG@3 = {dlt:+.4f} "
                         f"(final {pf.loc[pid, 'NDCG@3']:.4f} vs BM25 {pb.loc[pid, 'NDCG@3']:.4f})")
            lines.append(f"\n**TEXTO_PERFIL:** {g['TEXTO_PERFIL'].iloc[0]}\n")
            for met, per in ((final_key, pf), ("BM25", pb)):
                lines.append(f"\n**Ranking {met}:**\n")
                for it in parse_ranking(per.loc[pid, "ranking"]):
                    j = just.get((pid, it["AVISOID"]), "")
                    lines.append(f"{it['pos']}. [{'REL' if it['rel'] else '—'}] "
                                 f"AVISOID {it['AVISOID']} — {nombres.get(it['AVISOID'], '?')} "
                                 f"(score {it['score']:.4f})"
                                 + (f" | Justificación: {j}" if j else ""))
        (rec.dir / "analisis_cualitativo_datos.md").write_text("\n".join(lines), encoding="utf-8")
        rec.log(f"[artefacto] {rec.dir / 'analisis_cualitativo_datos.md'} "
                f"({len(seleccion)} perfiles)")

        rec.save_json("metrics.json", {
            "experimento": "E4_final",
            "modelo_final": nombre_final,
            "decision_e3": decision["decision"],
            "n_perfiles": len(usable),
            "agregados_modelo_final": agg_final,
            "anclaje_aleatorio_5.3": cfg["random_anchors"],
            "empates": ties,
            "bootstrap_ic95": cis,
            "diferencia_pareada_final_menos_bm25_ic95": diff_ci,
            "test_permutacion_pareado_final_vs_bm25": perm,
            "r1_resumen": {"n_afectados": r1["n_afectados"],
                           "NDCG@3_canonico": agg_final["NDCG@3"],
                           "NDCG@3_consolidado": agg_r1["NDCG@3"]},
            "s1_resumen": {"n_complemento": s1["n_complemento"],
                           "NDCG@3_32": agg_final["NDCG@3"],
                           "NDCG@3_complemento": agg_comp["NDCG@3"]},
        })

        head = git_head()
        marker.write_text(
            f"E4 ejecutado (pasada única, §8)\nfecha: {datetime.datetime.now().isoformat()}\n"
            f"commit: {head['commit']}\nmodelo_final: {nombre_final}\n", encoding="utf-8")
        rec.log("[E4] marker de pasada única escrito")
        rec.close({"estado": "OK", "modelo_final": nombre_final, "agregados": agg_final})
    except Exception:
        rec.close({"estado": "ABORTADO"})
        raise


if __name__ == "__main__":
    main()
