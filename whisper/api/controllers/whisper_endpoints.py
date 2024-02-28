from fastapi import APIRouter, File, UploadFile, HTTPException
from ..services.whisper_services import transcribe_audio
from typing import Dict, Optional
from ..dto.request_model import WhisperRequestBody
import json
import os

whisper_router = APIRouter()

@whisper_router.get("/")
async def root():
    return {"message": "Whisper API"}


@whisper_router.post("/transcribe")
async def transcribe(options: Optional[str], audio_file: UploadFile = File(...),):
    try:
        options = json.loads(options)
        audio_content = await audio_file.read()
        filename = f"audio.wav"  # Or any desired filename
        with open(filename, "wb") as file:
            file.write(audio_content)
        transcribed_audio = await transcribe_audio(filename, options)
        os.remove(filename)
        return transcribed_audio
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# @whisper_router.post("/translate")
