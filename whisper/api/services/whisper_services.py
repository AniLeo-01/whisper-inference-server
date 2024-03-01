import os
import json
from fastapi import HTTPException
# from ..dto.request_model import WhisperRequestBody
from typing import Dict, Optional
from whisper.config.config import get_settings
from transformers import pipeline
import torch
from transformers.utils import is_flash_attn_2_available
    

def transcribe_audio(filepath: str, options: Optional[Dict]):
    print(options)
    try:
        #load model
        pipe = pipeline(
            "automatic-speech-recognition",
            model=options.get("model_name","openai/whisper-tiny"),
            torch_dtype=torch.float32 if get_settings().DEVICE_ID == "cpu" else torch.float16,
            device= "cpu" if get_settings().DEVICE_ID == "cpu" else f"cuda:{get_settings().DEVICE_ID}",
            model_kwargs={"attn_implementation": "flash_attention_2"} if get_settings().FLASH_ATTN else {"attn_implementation": "sdpa"},
        )

        # if get_settings().DEVICE_ID == "mps":
        #     torch.mps.empty_cache()

        generate_kwargs = {"task": options.get('task', 'transcribe'), "language": options.get('language', None)}

        outputs = pipe(
                filepath,
                chunk_length_s=options.get('chunk_size', 30),
                batch_size=options.get('batch_size', 8),
                generate_kwargs=generate_kwargs,
                return_timestamps=True,
            )
        return outputs
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


