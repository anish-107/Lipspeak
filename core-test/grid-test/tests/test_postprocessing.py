''' test_postprocessing.py
@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Unit tests for sentence
postprocessing utilities.
@date: 10 May 2026
@returns: Postprocessing Test Suite

'''


# Imports
from app.services.postprocessing_service import (
    postprocess_prediction
)



# Test Sentence Postprocessing
def test_postprocess_prediction() -> None:

    '''
    @description: Tests GRID sentence
    correction.

    '''

    sentence: str = (
        "bin gren at a won please"
    )


    corrected: str = (
        postprocess_prediction(
            sentence
        )
    )


    assert corrected is not None

    assert isinstance(
        corrected,
        str
    )