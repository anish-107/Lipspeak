import tensorflow as tf
from app.config import settings
from app.services.preprocessing import load_video


def CTCLoss(y_true, y_pred):
    batch_size = tf.cast(tf.shape(y_true)[0], tf.int64)
    input_len = tf.cast(tf.shape(y_pred)[1], tf.int64)
    label_len = tf.cast(tf.shape(y_true)[1], tf.int64)

    input_len = input_len * tf.ones((batch_size, 1), tf.int64)
    label_len = label_len * tf.ones((batch_size, 1), tf.int64)

    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_len, label_len)


class InferenceService:
    def __init__(self):
        self.model = tf.keras.models.load_model(
            str(settings.model_path),
            custom_objects={"CTCLoss": CTCLoss},
            compile=False
        )

    def decode(self, yhat):
        decoded = tf.keras.backend.ctc_decode(
            yhat,
            input_length=[settings.MAX_FRAMES],
            greedy=True
        )[0][0]

        return decoded

    def predict(self, video_path, num_to_char):
        video_tensor = load_video(video_path)
        video_tensor = tf.expand_dims(video_tensor, axis=0)

        yhat = self.model.predict(video_tensor, verbose=0)
        decoded = self.decode(yhat)

        text = tf.strings.reduce_join(num_to_char(decoded)).numpy().decode("utf-8")
        return text.strip()


inference_service = InferenceService()