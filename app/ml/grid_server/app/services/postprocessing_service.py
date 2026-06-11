''' postprocessing_service.py
@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Service functions for GRID
sentence post-processing and correction.
@date: 10 May 2026
@returns: Corrected GRID Sentences

'''


# Imports
from app.utils.text_utils import (
    correct_word
)



# GRID Sentence Structure
GRID_STRUCTURE: list[set[str]] = [
    {
        "bin",
        "lay",
        "place",
        "set"
    },

    {
        "blue",
        "green",
        "red",
        "white"
    },

    {
        "at",
        "by",
        "in",
        "with"
    },

    set(
        "abcdefghijklmnopqrstuvwxyz"
    ),

    {
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine"
    },

    {
        "again",
        "now",
        "please",
        "soon"
    }
]



# Postprocess Prediction
def postprocess_prediction(
    sentence: str
) -> str:

    '''
    @description: Corrects predicted
    GRID sentence structure.

    @args:
        sentence:
            Predicted sentence.

    @returns:
        Corrected sentence.

    '''

    # Sentence Words
    words: list[str] = (
        sentence.split()
    )


    # Corrected Words
    corrected: list[str] = []


    # Correct Sentence
    for i, word in enumerate(words):

        if i >= len(
            GRID_STRUCTURE
        ):

            corrected.append(
                word
            )

            continue


        corrected.append(
            correct_word(
                word,
                GRID_STRUCTURE[i]
            )
        )


    return " ".join(
        corrected
    )