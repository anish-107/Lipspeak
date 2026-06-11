''' text_utils.py
@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Utility functions for text correction
and sentence post-processing.
@date: 10 May 2026
@returns: Text Processing Utilities

'''


# Imports
from rapidfuzz.distance import Levenshtein



# Correct Word
def correct_word(
    word: str,
    dictionary: set[str]
) -> str:

    '''
    @description: Corrects a word using
    Levenshtein distance matching.

    @args:
        word:
            Input word.

        dictionary:
            Valid dictionary words.

    @returns:
        Closest matching dictionary word.

    '''

    # Best Matching Word
    best_match: str = word


    # Best Distance Score
    best_score: float = float(
        "inf"
    )


    # Compare Against Dictionary
    for candidate in dictionary:

        score: int = (
            Levenshtein.distance(
                word,
                candidate
            )
        )


        # Update Best Match
        if score < best_score:

            best_score = score

            best_match = candidate


    return best_match



# Correct Sentence
def correct_sentence(
    sentence: str,
    dictionary: set[str]
) -> str:

    '''
    @description: Corrects sentence words
    using dictionary-based matching.

    @args:
        sentence:
            Input sentence.

        dictionary:
            Valid dictionary words.

    @returns:
        Corrected sentence.

    '''

    # Split Sentence Words
    words: list[str] = sentence.split()


    # Corrected Words
    corrected: list[str] = [
        correct_word(
            word,
            dictionary
        )
        for word in words
    ]


    return " ".join(
        corrected
    )