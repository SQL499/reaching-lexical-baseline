"""Setup único de assets (previo a E0):
1. Copia la lista de stopwords español de NLTK a src/resources/stopwords_es.txt (versionada;
   el runtime nunca descarga nada, §4.1).
2. Descarga el modelo HF a models/ fijando la revisión exacta (§7) y la registra en
   models/MODEL_INFO.txt.
"""
import datetime
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_DIR = ROOT / "models" / "paraphrase-multilingual-MiniLM-L12-v2"
STOPWORDS_DST = ROOT / "src" / "resources" / "stopwords_es.txt"


def fetch_stopwords():
    import nltk

    tmp = Path(tempfile.mkdtemp(prefix="nltk_"))
    nltk.download("stopwords", download_dir=str(tmp), quiet=True)
    src = tmp / "corpora" / "stopwords" / "spanish"
    if not src.exists():  # zip no extraído (según versión de nltk)
        import zipfile

        with zipfile.ZipFile(tmp / "corpora" / "stopwords.zip") as z:
            z.extract("stopwords/spanish", tmp / "corpora")
        src = tmp / "corpora" / "stopwords" / "spanish"
    STOPWORDS_DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, STOPWORDS_DST)
    n = len([l for l in STOPWORDS_DST.read_text(encoding="utf-8").splitlines() if l.strip()])
    print(f"[stopwords] copiadas {n} stopwords de NLTK spanish -> {STOPWORDS_DST}")
    shutil.rmtree(tmp, ignore_errors=True)
    return n


def fetch_model():
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    info = api.model_info(MODEL_NAME)
    sha = info.sha
    print(f"[modelo] {MODEL_NAME} revision={sha}")
    path = snapshot_download(MODEL_NAME, revision=sha, local_dir=str(MODEL_DIR))
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    (ROOT / "models" / "MODEL_INFO.txt").write_text(
        f"model_name: {MODEL_NAME}\n"
        f"hf_revision_sha: {sha}\n"
        f"local_dir: {MODEL_DIR.relative_to(ROOT)}\n"
        f"descargado: {stamp}\n",
        encoding="utf-8",
    )
    print(f"[modelo] snapshot en {path}")
    return sha


if __name__ == "__main__":
    n = fetch_stopwords()
    sha = fetch_model()
    print(f"SETUP_ASSETS_OK stopwords={n} revision={sha}")
    sys.exit(0)
