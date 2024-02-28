from fastapi import FastAPI
from whisper.api.controllers.whisper_endpoints import whisper_router
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Whisper API started.....")
    yield
    print("Whisper API shutdown....")

app = FastAPI(title = "Whisper API", version = "1.0.0")

app.include_router(whisper_router, tags=['whisper'], prefix="/whisper")
