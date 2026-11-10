"""Utilidades transversales: determinismo (§12.1), compuerta de hashes (§2.1),
configuración y registro de corridas (§12.3)."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


class GateError(RuntimeError):
    """Fallo de compuerta bloqueante: abortar y reportar (regla 3 del kickoff)."""


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_determinism(seed: int = 42, torch_too: bool = True) -> None:
    """§12.1: semillas y flags. PYTHONHASHSEED se fija también en el lanzador
    (debe existir antes de iniciar el intérprete; ver DECISIONES.md I15)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch_too:
        try:
            import torch

            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_hashes(cfg: dict, log=print) -> None:
    """Compuerta §2.1: verifica las 6 huellas SHA-256. Aborta con detalle si alguna difiere."""
    data_dir = ROOT / cfg["paths"]["data_raw"]
    failures = []
    for fname, expected in cfg["sha256"].items():
        p = data_dir / fname
        if not p.exists():
            failures.append(f"  {fname}: ARCHIVO NO ENCONTRADO en {p}")
            continue
        actual = sha256_file(p)
        if actual != expected:
            failures.append(f"  {fname}:\n    esperado {expected}\n    obtenido {actual}")
        else:
            log(f"[hash-gate] OK  {fname}  sha256={actual}")
    if failures:
        raise GateError(
            "COMPUERTA §2.1 FALLIDA — hashes SHA-256 no coinciden con el manifiesto:\n"
            + "\n".join(failures)
            + "\nProhibido continuar (regla 3). Verificar procedencia de los datos."
        )
    log("[hash-gate] Las 6 huellas SHA-256 coinciden con §2.1.")


def git_head() -> dict:
    def run(args):
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()

    commit = run(["rev-parse", "HEAD"]) or "(sin commits)"
    dirty = bool(run(["status", "--porcelain"]))
    return {"commit": commit, "dirty": dirty}


class RunRecorder:
    """Registro por corrida (§12.3): carpeta en results/, run.log, run_meta.json,
    copia de config.yaml, hash de commit y marca temporal."""

    def __init__(self, experiment_dirname: str, cfg: dict):
        self.t0 = datetime.datetime.now()
        self.dir = ROOT / cfg["paths"]["results"] / experiment_dirname
        self.dir.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(self.dir / "run.log", "a", encoding="utf-8")
        head = git_head()
        self.meta = {
            "experimento": experiment_dirname,
            "inicio": self.t0.isoformat(timespec="seconds"),
            "git_commit": head["commit"],
            "git_dirty": head["dirty"],
            "python": sys.version,
            "argv": sys.argv,
        }
        shutil.copyfile(ROOT / "config.yaml", self.dir / "config.yaml")
        self.log(f"===== {experiment_dirname} | inicio {self.meta['inicio']} "
                 f"| commit {head['commit'][:12]}{'+dirty' if head['dirty'] else ''} =====")

    def log(self, msg: str) -> None:
        line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self._log_fh.write(line + "\n")
        self._log_fh.flush()

    def save_json(self, name: str, obj) -> Path:
        p = self.dir / name
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)
        self.log(f"[artefacto] {p.relative_to(ROOT)}")
        return p

    def close(self, extra_meta: dict | None = None) -> None:
        t1 = datetime.datetime.now()
        self.meta["fin"] = t1.isoformat(timespec="seconds")
        self.meta["duracion_seg"] = round((t1 - self.t0).total_seconds(), 1)
        if extra_meta:
            self.meta.update(extra_meta)
        with open(self.dir / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2, default=_json_default)
        self.log(f"===== fin | duración {self.meta['duracion_seg']} s =====")
        self._log_fh.close()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"No serializable: {type(o)}")


def harness_selfcheck() -> None:
    """Compuerta §5.4 en cada entry-point de evaluación: los vectores de prueba deben
    pasar antes de E1 (y se re-verifican en E2/E3/E4). Aborta si fallan."""
    from src import metrics as M

    tol = load_config()["metrics"]["test_tolerance"]
    cases = {
        "A": ([0, 1, 0, 0, 1, 0], {"P@1": 0.0, "P@3": 0.3333, "NDCG@3": 0.3869, "NDCG@5": 0.6240, "RR": 0.5}),
        "B": ([1, 0, 0, 0, 0, 0], {"P@1": 1.0, "P@3": 0.3333, "NDCG@3": 1.0000, "NDCG@5": 1.0000, "RR": 1.0}),
        "C": ([0, 0, 0, 1, 0, 0], {"P@1": 0.0, "P@3": 0.0000, "NDCG@3": 0.0000, "NDCG@5": 0.4307, "RR": 0.25}),
    }
    agg_expected = {"P@1": 0.3333, "P@3": 0.2222, "NDCG@3": 0.4623, "NDCG@5": 0.6849, "MRR": 0.5833}
    per_profile = []
    for name, (rel, expected) in cases.items():
        got = M.profile_metrics(rel)
        per_profile.append(got)
        for k, v in expected.items():
            if abs(got[k] - v) > tol:
                raise GateError(
                    f"COMPUERTA §5.4 FALLIDA — caso {name}, métrica {k}: esperado {v}, obtenido {got[k]:.6f}"
                )
    agg = M.aggregate(per_profile)
    for k, v in agg_expected.items():
        if abs(agg[k] - v) > tol:
            raise GateError(f"COMPUERTA §5.4 FALLIDA — agregación macro, {k}: esperado {v}, obtenido {agg[k]:.6f}")
