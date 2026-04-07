import os
import time
import tempfile
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import whisper
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Whisper Transcription API",
    description="API de transcrição de áudio usando OpenAI Whisper",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurações
SUPPORTED_FORMATS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg", ".flac"}
MAX_FILE_SIZE_MB = 100
AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large"]
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "tiny")

# Cache de modelos carregados
_model_cache: dict = {}


def get_model(model_name: str) -> whisper.Whisper:
    """Carrega e faz cache do modelo Whisper."""
    if model_name not in _model_cache:
        logger.info(f"Carregando modelo Whisper: {model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model_cache[model_name] = whisper.load_model(model_name, device=device)
        logger.info(f"Modelo {model_name} carregado em {device}")
    return _model_cache[model_name]


@app.on_event("startup")
async def startup_event():
    """Pré-carrega o modelo padrão na inicialização."""
    logger.info(f"Iniciando API — pré-carregando modelo: {DEFAULT_MODEL}")
    get_model(DEFAULT_MODEL)


@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Whisper Transcription API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["Info"])
def health():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "status": "ok",
        "device": device,
        "models_loaded": list(_model_cache.keys()),
        "default_model": DEFAULT_MODEL,
    }


@app.get("/models", tags=["Info"])
def list_models():
    """Lista os modelos disponíveis com informações de tamanho/velocidade."""
    return {
        "available": AVAILABLE_MODELS,
        "default": DEFAULT_MODEL,
        "info": {
            "tiny":   {"params": "39M",  "speed": "~32x", "vram": "~1GB"},
            "base":   {"params": "74M",  "speed": "~16x", "vram": "~1GB"},
            "small":  {"params": "244M", "speed": "~6x",  "vram": "~2GB"},
            "medium": {"params": "769M", "speed": "~2x",  "vram": "~5GB"},
            "large":  {"params": "1550M","speed": "1x",   "vram": "~10GB"},
        },
    }


@app.post("/transcribe", tags=["Transcrição"])
async def transcribe(
    file: UploadFile = File(..., description="Arquivo de áudio"),
    model: str = Form(DEFAULT_MODEL, description="Modelo Whisper a usar"),
    language: Optional[str] = Form(None, description="Código do idioma (ex: pt, en, es). Auto-detecta se omitido."),
    task: str = Form("transcribe", description="'transcribe' ou 'translate' (traduz para inglês)"),
    word_timestamps: bool = Form(False, description="Incluir timestamps por palavra"),
):
    """
    Transcreve um arquivo de áudio para texto.

    - **file**: Arquivo de áudio (mp3, wav, m4a, ogg, flac, webm, mp4...)
    - **model**: Modelo Whisper (tiny, base, small, medium, large)
    - **language**: Idioma do áudio (auto-detectado se não informado)
    - **task**: `transcribe` mantém o idioma original; `translate` converte para inglês
    - **word_timestamps**: Retorna timestamps para cada palavra
    """
    # Valida extensão
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Formato não suportado: {suffix}. Use: {', '.join(SUPPORTED_FORMATS)}",
        )

    # Valida modelo
    if model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Modelo inválido: {model}. Disponíveis: {AVAILABLE_MODELS}",
        )

    # Valida task
    if task not in ("transcribe", "translate"):
        raise HTTPException(status_code=400, detail="task deve ser 'transcribe' ou 'translate'")

    # Lê e valida tamanho
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"Arquivo muito grande: {size_mb:.1f}MB. Limite: {MAX_FILE_SIZE_MB}MB",
        )

    # Transcreve
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        t0 = time.time()
        whisper_model = get_model(model)

        options = {
            "task": task,
            "word_timestamps": word_timestamps,
        }
        if language:
            options["language"] = language

        result = whisper_model.transcribe(tmp_path, **options)
        elapsed = round(time.time() - t0, 2)

        # Monta segmentos
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
                    {
                        "word": w["word"],
                        "start": round(w["start"], 3),
                        "end": round(w["end"], 3),
                        "probability": round(w.get("probability", 0), 4),
                    }
                    for w in seg["words"]
                ]
            segments.append(s)

        return {
            "text": result["text"].strip(),
            "language": result.get("language"),
            "model": model,
            "task": task,
            "duration_audio_s": round(result["segments"][-1]["end"], 2) if result["segments"] else None,
            "processing_time_s": elapsed,
            "file": file.filename,
            "size_mb": round(size_mb, 2),
            "segments": segments,
        }

    except Exception as e:
        logger.error(f"Erro na transcrição: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/detect-language", tags=["Utilitários"])
async def detect_language(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
):
    """Detecta o idioma de um arquivo de áudio sem transcrever."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail=f"Formato não suportado: {suffix}")

    content = await file.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        whisper_model = get_model(model)
        audio = whisper.load_audio(tmp_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
        _, probs = whisper_model.detect_language(mel)

        top5 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
        return {
            "detected_language": top5[0][0],
            "confidence": round(top5[0][1], 4),
            "top_5": [{"language": l, "probability": round(p, 4)} for l, p in top5],
        }
    finally:
        os.unlink(tmp_path)
