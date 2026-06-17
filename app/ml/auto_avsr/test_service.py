from app.services.inference_service import (
    InferenceService,
)

model = InferenceService()

result = model.predict(
    "/home/anish/Downloads/4.mp4"
)

print(result)