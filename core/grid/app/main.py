from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.transcribe import router as transcribe_router


app = FastAPI(title="Lipspeak API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(transcribe_router)