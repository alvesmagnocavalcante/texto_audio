import os
import time
import tempfile
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import whisper
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Transcription API", version="1.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SUPPORTED_FORMATS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg", ".flac"}
MAX_FILE_SIZE_MB = 100
AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large"]
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "base")

_model_cache: dict = {}


def get_model(model_name: str) -> whisper.Whisper:
    if model_name not in _model_cache:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Carregando modelo {model_name} em {device}")
        _model_cache[model_name] = whisper.load_model(model_name, device=device)
    return _model_cache[model_name]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "models_loaded": list(_model_cache.keys()),
        "default_model": DEFAULT_MODEL,
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    language: Optional[str] = Form(None),
    word_timestamps: bool = Form(False),
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(400, f"Formato não suportado: {suffix}")
    if model not in AVAILABLE_MODELS:
        raise HTTPException(400, f"Modelo inválido: {model}. Disponíveis: {AVAILABLE_MODELS}")

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(413, f"Arquivo muito grande: {size_mb:.1f}MB. Limite: {MAX_FILE_SIZE_MB}MB")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        t0 = time.time()
        result = get_model(model).transcribe(
            tmp_path,
            task="transcribe",
            language=language,
            word_timestamps=word_timestamps,
        )

        segments = []
        for seg in result.get("segments", []):
            s = {
                "id": seg["id"],
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "text": seg["text"].strip(),
            }
            if word_timestamps and "words" in seg:
                s["words"] = [
                    {"word": w["word"], "start": round(w["start"], 3), "end": round(w["end"], 3)}
                    for w in seg["words"]
                ]
            segments.append(s)

        return {
            "text": result["text"].strip(),
            "language": result.get("language"),
            "model": model,
            "duration_s": round(result["segments"][-1]["end"], 2) if result["segments"] else None,
            "processing_time_s": round(time.time() - t0, 2),
            "size_mb": round(size_mb, 2),
            "segments": segments,
        }

    except Exception as e:
        logger.error(f"Erro na transcrição: {e}")
        raise HTTPException(500, str(e))
    finally:
        os.unlink(tmp_path)
