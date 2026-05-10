''' decoding_service.py
@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Service functions for decoding
TensorFlow lip reading predictions.
@date: 10 May 2026
@returns: Decoded Text Predictions

'''


# Imports
import string
import tensorflow as tf



# Vocabulary
vocab: list[str] = list(
    string.ascii_lowercase
    + "'?! "
)



# Character To Number Mapping
char_to_num = (
    tf.keras.layers.StringLookup(
        vocabulary=vocab,
        oov_token=""
    )
)



# Number To Character Mapping
num_to_char = (
    tf.keras.layers.StringLookup(
        vocabulary=(
            char_to_num.get_vocabulary()
        ),
        invert=True,
        oov_token=""
    )
)



# Decode Prediction
def decode_prediction(
    yhat: tf.Tensor
) -> str:

    '''
    @description: Decodes model prediction
    using CTC decoding.

    @args:
        yhat:
            Model output tensor.

    @returns:
        Decoded prediction text.

    '''

    # Output Sequence Length
    output_len_shape: int | None = (
        yhat.shape[1]
    )
    
    output_len: int = int(
        output_len_shape or 0
    )
    
    
    # Input Sequence Length
    input_len: list[int] = [
        output_len
    ]


    # Decode Prediction
    decoded = (
        tf.keras.backend.ctc_decode(
            yhat,
            input_length=input_len,
            greedy=False
        )[0][0]
    )


    # Convert Tokens To Text
    text: str = (
        tf.strings.reduce_join(
            num_to_char(decoded)
        ).numpy().decode(
            "utf-8"
        )
    )


    return text.strip()