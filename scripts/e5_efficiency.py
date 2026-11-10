"""E5 — Eficiencia (§10, RNF-02): precálculo de embeddings del corpus completo,
latencias (mediana/p95, 50 reps tras 5 de calentamiento) de codificación, similitud+top-10
y extremo a extremo, en GPU y CPU (I8); referencia opcional BM25; hardware exacto.

La consulta de prueba es el primer perfil utilizable de TRAIN (evita tocar evaluación
fuera de E1/E2/E4; la latencia no depende del split).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data as D  # noqa: E402
from src import efficiency as EF  # noqa: E402
from src.bm25_baseline import BM25Index  # noqa: E402
from src.common import (GateError, RunRecorder, load_config,  # noqa: E402
                        set_determinism, verify_hashes)
from src.preprocess import semantic_text  # noqa: E402
from src.semantic_model import load_model  # noqa: E402


def main():
    set_determinism(42)
    cfg = load_config()
    rec = RunRecorder("E5_eficiencia", cfg)
    try:
        verify_hashes(cfg, rec.log)
        d = D.load_all(cfg)
        D.run_assertions(d, cfg, rec.log)

        gonogo_path = ROOT / cfg["paths"]["results"] / "E3_finetune" / "gonogo.json"
        if not gonogo_path.exists():
            raise GateError("E5 requiere E3 (identidad del modelo final)")
        decision = json.loads(gonogo_path.read_text(encoding="utf-8"))
        is_go = decision["decision"] == "GO"
        model_dir = cfg["paths"]["finetuned_final_dir"] if is_go else cfg["paths"]["base_model_dir"]
        rec.log(f"[E5] modelo final ({decision['decision']}): {model_dir}")

        # consulta: primer perfil de TRAIN con >=1 positivo (determinista)
        pos = d["train"].groupby("ID_POSTULANTE", sort=False)["ETIQUETA"].sum()
        qpid = next(p for p in pos.index if pos[p] >= 1)
        qtext = semantic_text(
            d["train"].loc[d["train"]["ID_POSTULANTE"] == qpid, "TEXTO_PERFIL"].iloc[0])
        rec.log(f"[E5] perfil de consulta: {qpid} ({len(qtext)} caracteres)")

        corpus_texts = [semantic_text(t) for t in d["ofertas"]["TEXTO_OFERTA"].tolist()]

        model_gpu = load_model(cfg, model_dir=model_dir, device="cuda")
        mat, pre_stats = EF.precompute_corpus(model_gpu, corpus_texts, cfg, rec.log)

        res_gpu = EF.bench_device(model_gpu, qtext, mat, cfg, "cuda", rec.log)
        import gc, torch
        del model_gpu
        gc.collect()
        torch.cuda.empty_cache()

        model_cpu = load_model(cfg, model_dir=model_dir, device="cpu")
        res_cpu = EF.bench_device(model_cpu, qtext, mat, cfg, "cpu", rec.log)
        del model_cpu
        gc.collect()

        rec.log("[E5] referencia opcional BM25 sobre corpus completo…")
        index = BM25Index(d["ofertas"], cfg, rec.log)
        raw_q = d["train"].loc[d["train"]["ID_POSTULANTE"] == qpid, "TEXTO_PERFIL"].iloc[0]
        res_bm25 = EF.bench_bm25(index, raw_q, cfg, rec.log)

        hw = EF.hardware_info()
        rec.log(f"[E5] hardware: {hw}")
        rec.save_json("efficiency.json", {
            "experimento": "E5_eficiencia",
            "modelo_final": model_dir,
            "esquema": f"{cfg['efficiency']['warmup']} calentamiento + "
                       f"{cfg['efficiency']['reps']} repeticiones (§10)",
            "precalculo_corpus": pre_stats,
            "gpu": res_gpu,
            "cpu": res_cpu,
            "referencia_bm25_corpus_completo": res_bm25,
            "hardware": hw,
        })
        rec.close({"estado": "OK"})
    except Exception:
        rec.close({"estado": "ABORTADO"})
        raise


if __name__ == "__main__":
    main()
