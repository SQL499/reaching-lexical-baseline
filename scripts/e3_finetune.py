"""E3 — Fine-tuning condicional con validación cruzada sobre TRAIN (§9).
El split de evaluación NO interviene en ninguna decisión (§0, §9.1): solo se lee para
las compuertas de integridad (§2.1/§2.2), sin usar sus contenidos.
30 corridas (6 configs × 5 pliegues), GO/NO-GO mecánico (§9.3) y, si GO, reentrenamiento
final sobre los 160 perfiles de TRAIN.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import data as D  # noqa: E402
from src import finetune as F  # noqa: E402
from src.common import (ROOT as SROOT, RunRecorder, harness_selfcheck,  # noqa: E402
                        load_config, set_determinism, verify_hashes)
from src.evaluate import aggregate_from_df  # noqa: E402
from src.semantic_model import load_model  # noqa: E402


def main():
    set_determinism(42)
    cfg = load_config()
    rec = RunRecorder("E3_finetune", cfg)
    try:
        verify_hashes(cfg, rec.log)
        d = D.load_all(cfg)
        D.run_assertions(d, cfg, rec.log)
        harness_selfcheck()
        rec.log("[E3] gates OK. El conjunto de evaluación NO se usa en E3 (§0/§9.1).")

        train_canon = D.build_canonical(d["train"], d["ofertas"])
        folds = F.make_folds(train_canon, int(cfg["finetune"]["n_folds"]))
        rec.save_json("folds.json", [{
            "fold": f["k"],
            "n_perfiles_train": f["n_perfiles_train"],
            "n_perfiles_eval": f["n_perfiles_eval"],
            "n_perfiles_eval_utilizables": len(f["eval_usable_profiles"]),
            "perfiles_eval": sorted(f["eval_pairs"]["ID_POSTULANTE"].unique().tolist()),
        } for f in folds])
        for f in folds:
            rec.log(f"[E3] pliegue {f['k']}: train {f['n_perfiles_train']} perfiles | "
                    f"eval {f['n_perfiles_eval']} (utilizables {len(f['eval_usable_profiles'])})")
        total_usable = sum(len(f["eval_usable_profiles"]) for f in folds)
        rec.log(f"[E3] perfiles utilizables totales en CV: {total_usable} (esperado 109, §2.2)")

        # Comparador zero-shot sobre los mismos pliegues (§9.1)
        rec.log("[E3] evaluando zero-shot por pliegue…")
        zs_model = load_model(cfg, device="cuda")
        zs_rows, zs_by_fold = [], {}
        per_prof_dir = rec.dir / "cv_per_profile"
        per_prof_dir.mkdir(exist_ok=True)
        for f in folds:
            r = F.eval_fold(zs_model, f, cfg)
            zs_by_fold[f["k"]] = r["agg"]["NDCG@3"]
            zs_rows.append({"fold": f["k"], **r["agg"],
                            "n_perfiles_utilizables": len(f["eval_usable_profiles"])})
            r["per_profile"].to_csv(per_prof_dir / f"zeroshot_f{f['k']}.csv",
                                    index=False, encoding="utf-8")
            rec.log(f"[E3] zero-shot pliegue {f['k']}: NDCG@3={r['agg']['NDCG@3']:.4f}, "
                    f"MRR={r['agg']['MRR']:.4f}")
        del zs_model
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
        pd.DataFrame(zs_rows).to_csv(rec.dir / "zeroshot_folds.csv", index=False, encoding="utf-8")

        # Rejilla congelada §9.2: 6 configs × 5 pliegues
        grid = cfg["finetune"]["grid"]
        cv_rows = []
        t_grid = time.perf_counter()
        for cname, gc_ in grid.items():
            for f in folds:
                run_name = f"{cname}_f{f['k']}"
                model, secs, n_ex, cleanup = F.train_one(
                    cfg, f["train_pairs"], gc_["loss"], int(gc_["epochs"]),
                    int(gc_["batch_size"]), run_name, rec.log)
                r = F.eval_fold(model, f, cfg)
                cleanup()
                r["per_profile"].to_csv(per_prof_dir / f"{run_name}.csv",
                                        index=False, encoding="utf-8")
                cv_rows.append({"config": cname, "fold": f["k"], **r["agg"],
                                "n_ejemplos_train": n_ex, "segundos_train": round(secs, 1)})
                rec.log(f"[E3] {run_name}: NDCG@3={r['agg']['NDCG@3']:.4f}, "
                        f"MRR={r['agg']['MRR']:.4f} "
                        f"(zs pliegue: {zs_by_fold[f['k']]:.4f})")
        cv_table = pd.DataFrame(cv_rows)
        cv_table.to_csv(rec.dir / "cv_results.csv", index=False, encoding="utf-8")
        rec.log(f"[E3] rejilla completa: {len(cv_rows)} corridas en "
                f"{(time.perf_counter() - t_grid) / 60:.1f} min")

        # GO/NO-GO mecánico §9.3
        decision = F.gonogo_decision(cv_table, zs_by_fold)
        rec.save_json("gonogo.json", decision)
        rec.log(f"[E3][GO/NO-GO §9.3] mejor config: {decision['mejor_config']} "
                f"(NDCG@3 CV {decision['ndcg3_cv_mejor']:.4f} vs zero-shot "
                f"{decision['ndcg3_cv_zeroshot']:.4f}; mejoras en "
                f"{decision['mejoras_por_pliegue']}/5 pliegues) → {decision['decision']}")

        if decision["decision"] == "GO":
            g = grid[decision["mejor_config"]]
            rec.log(f"[E3] GO: reentrenando {decision['mejor_config']} sobre los 160 perfiles "
                    f"de TRAIN (misma receta, semilla 42)…")
            final_dir = SROOT / cfg["paths"]["finetuned_final_dir"]
            model, secs, n_ex, cleanup = F.train_one(
                cfg, train_canon, g["loss"], int(g["epochs"]), int(g["batch_size"]),
                "final_160", rec.log, save_dir=final_dir)
            cleanup()
            rec.save_json("final_training.json", {
                "config": decision["mejor_config"], **g,
                "n_ejemplos": n_ex, "segundos": round(secs, 1),
                "modelo_final_dir": str(final_dir.relative_to(SROOT)),
            })
        else:
            rec.log("[E3] NO-GO: el modelo final para E4 es el zero-shot (§9.3); "
                    "el fine-tuning se reporta como resultado negativo (hallazgo legítimo).")

        rec.close({"estado": "OK", "decision": decision["decision"],
                   "mejor_config": decision["mejor_config"]})
    except Exception:
        rec.close({"estado": "ABORTADO"})
        raise


if __name__ == "__main__":
    main()
