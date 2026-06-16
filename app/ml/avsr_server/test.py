''' test.py
@authors: Anish Kumar, Bidipta Barua,
Dibyasmita Hati, Arpan Haldar
@description: Test script for
validating AVSR inference.
@date: 11 June 2026
@returns: Console transcript output.

'''


# Imports
from app.services.inference_service import (
    InferenceService,
)



# Main
if __name__ == "__main__":

    service = (
        InferenceService()
    )

    transcript = (
        service.predict(
            "test-videos/bbbs4n.mpg"
        )
    )

    print(
        "\nTranscript:"
    )

    print(
        transcript
    )