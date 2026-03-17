import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.config import settings
from app.schemas.response import TranscriptionResponse
from app.services.inference import inference_service
import tensorflow as tf
import string


router = APIRouter()


vocab = list(string.ascii_lowercase + "'?! ")
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    invert=True,
    oov_token=""
)


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(video: UploadFile = File(...)):
    if not video.filename:
        raise HTTPException(status_code=400, detail="No video file provided.")

    file_ext = video.filename.split(".")[-1]
    file_name = f"{uuid.uuid4()}.{file_ext}"
    file_path = settings.upload_dir / file_name

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        text = inference_service.predict(str(file_path), num_to_char)

        return TranscriptionResponse(
            success=True,
            transcription=text,
            filename=file_name
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if file_path.exists():
            file_path.unlink()