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
        "abcdefghijklmnopqrstuvxyz"
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
    GRID sentence structure and aligns
    missing single letter words.

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


    # Word Index Tracker
    w_idx: int = 0


    # Correct Sentence
    for i in range(
        len(GRID_STRUCTURE)
    ):

        # Handle Missing End Words
        if w_idx >= len(words):

            corrected.append(
                correct_word(
                    "",
                    GRID_STRUCTURE[i]
                )
            )

            continue


        # Current Word
        current_word: str = (
            words[w_idx]
        )


        # Check For Dropped Letter
        if (
            i == 3
            and
            current_word in GRID_STRUCTURE[4]
        ):

            corrected.append(
                correct_word(
                    "",
                    GRID_STRUCTURE[i]
                )
            )

            continue


        # Correct Current Word
        corrected.append(
            correct_word(
                current_word,
                GRID_STRUCTURE[i]
            )
        )


        w_idx += 1


    # Append Remaining Words
    while w_idx < len(words):

        corrected.append(
            words[w_idx]
        )

        w_idx += 1


    return " ".join(
        corrected
    ).strip()