"""E2 — Zero-shot sobre evaluación (§7, §8): paraphrase-multilingual-MiniLM-L12-v2 sin
ajuste; comparación E2 vs E1 con IC bootstrap 95% y test de permutación pareado (§9.4).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import data as D  # noqa: E402
from src.common import (GateError, RunRecorder, harness_selfcheck,  # noqa: E402
                        load_config, set_determinism, verify_hashes)
from src.evaluate import (aggregate_from_df, bootstrap_cis, paired_diff_ci,  # noqa: E402
                          permutation_tests, semantic_per_profile, tie_summary)
from src.semantic_model import load_model  # noqa: E402


def main():
    set_determinism(42)
    cfg = load_config()
    rec = RunRecorder("E2_zeroshot", cfg)
    try:
        verify_hashes(cfg, rec.log)
        d = D.load_all(cfg)
        D.run_assertions(d, cfg, rec.log)
        harness_selfcheck()
        rec.log("[gate §5.4] harness OK")

        e1_path = ROOT / cfg["paths"]["results"] / "E1_bm25" / "per_profile.csv"
        if not e1_path.exists():
            raise GateError("E2 requiere E1 previo (orden §8); falta results/E1_bm25/per_profile.csv")
        per_bm25 = pd.read_csv(e1_path, encoding="utf-8", dtype={"ID_POSTULANTE": str})

        val_canon = D.build_canonical(d["val"], d["ofertas"])
        usable = D.usable_val_profiles(d["val"])

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rec.log(f"[E2] dispositivo: {device}; modelo desde caché local "
                f"(revisión en models/MODEL_INFO.txt), max_seq_length={cfg['semantic']['max_seq_length']}")
        model = load_model(cfg, device=device)
        per_zs = semantic_per_profile(model, val_canon, usable, cfg)
        agg = aggregate_from_df(per_zs)
        ties = tie_summary(per_zs)

        per_zs.to_csv(rec.dir / "per_profile.csv", index=False, encoding="utf-8")
        rec.log(f"[artefacto] {rec.dir / 'per_profile.csv'}")
        for k, v in agg.items():
            rec.log(f"[E2] {k} = {v:.4f} (azar esperado {cfg['random_anchors'][k]})")
        rec.log(f"[E2] empates (§5.2): {ties}")

        # §9.4 — IC bootstrap por método y para la diferencia pareada; permutación pareada
        cis, idx, mats, cols, ids = bootstrap_cis({"BM25": per_bm25, "Zero-shot": per_zs}, cfg)
        diff_ci = paired_diff_ci(mats, idx, cols, "Zero-shot", "BM25", cfg["stats"]["ci_percentiles"])
        perm = permutation_tests(per_zs, per_bm25, cfg)
        for m, t in perm["tests"].items():
            rec.log(f"[E2 vs E1] {m}: dif={t['diferencia_media']:+.4f}, p={t['p_valor']:.4f}")

        rec.save_json("metrics.json", {
            "experimento": "E2_zeroshot",
            "metodo": "Zero-shot paraphrase-multilingual-MiniLM-L12-v2 (§7)",
            "n_perfiles": len(usable),
            "agregados": agg,
            "anclaje_aleatorio_5.3": cfg["random_anchors"],
            "empates": ties,
            "bootstrap_ic95": cis,
            "diferencia_pareada_zeroshot_menos_bm25_ic95": diff_ci,
            "test_permutacion_pareado": perm,
        })
        rec.close({"estado": "OK", "agregados": agg})
    except Exception:
        rec.close({"estado": "ABORTADO"})
        raise


if __name__ == "__main__":
    main()
