"""E1 — Baseline BM25 sobre evaluación (§6, §8): índice sobre el corpus completo,
ranking restringido a las 6 candidatas por perfil, métricas macro sobre los 32
perfiles utilizables, conteo de empates (C3).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data as D  # noqa: E402
from src.bm25_baseline import BM25Index  # noqa: E402
from src.common import (RunRecorder, harness_selfcheck, load_config,  # noqa: E402
                        set_determinism, verify_hashes)
from src.evaluate import aggregate_from_df, evaluate_ranking, tie_summary  # noqa: E402


def main():
    set_determinism(42)
    cfg = load_config()
    rec = RunRecorder("E1_bm25", cfg)
    try:
        verify_hashes(cfg, rec.log)
        d = D.load_all(cfg)
        D.run_assertions(d, cfg, rec.log)
        harness_selfcheck()
        rec.log("[gate §5.4] harness OK antes de E1")

        val_canon = D.build_canonical(d["val"], d["ofertas"])
        usable = D.usable_val_profiles(d["val"])
        rec.log(f"[E1] perfiles utilizables: {len(usable)} (§2.4)")

        index = BM25Index(d["ofertas"], cfg, rec.log)

        def score_fn(pid, qtext, cand):
            return index.score_candidates(qtext, cand["AVISOID"].tolist())

        per_profile = evaluate_ranking(val_canon, usable, score_fn)
        agg = aggregate_from_df(per_profile)
        ties = tie_summary(per_profile)

        per_profile.to_csv(rec.dir / "per_profile.csv", index=False, encoding="utf-8")
        rec.log(f"[artefacto] {rec.dir / 'per_profile.csv'}")
        for k, v in agg.items():
            rec.log(f"[E1] {k} = {v:.4f} (azar esperado {cfg['random_anchors'][k]})")
        rec.log(f"[E1] empates (§5.2): {ties}")
        rec.save_json("metrics.json", {
            "experimento": "E1_bm25",
            "metodo": "BM25 (k1=1.5, b=0.75, epsilon=0.25; índice sobre corpus completo)",
            "n_perfiles": len(usable),
            "agregados": agg,
            "anclaje_aleatorio_5.3": cfg["random_anchors"],
            "empates": ties,
        })
        rec.close({"estado": "OK", "agregados": agg})
    except Exception:
        rec.close({"estado": "ABORTADO"})
        raise


if __name__ == "__main__":
    main()
