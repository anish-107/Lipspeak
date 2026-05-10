''' exceptions.py
@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Custom application exception classes
for GRID lip reading backend.
@date: 10 May 2026
@returns: Application Exceptions

'''



# Base Application Exception
class ApplicationError(
    Exception
):

    '''
    @description: Base application exception.

    '''

    pass



# Invalid Video Exception
class InvalidVideoError(
    ApplicationError
):

    '''
    @description: Raised when uploaded
    video file is invalid.

    '''

    pass



# Video Processing Exception
class VideoProcessingError(
    ApplicationError
):

    '''
    @description: Raised when video
    preprocessing fails.

    '''

    pass



# Face Detection Exception
class FaceDetectionError(
    ApplicationError
):

    '''
    @description: Raised when no face
    or mouth is detected.

    '''

    pass



# Model Loading Exception
class ModelLoadingError(
    ApplicationError
):

    '''
    @description: Raised when TensorFlow
    model loading fails.

    '''

    pass



# Inference Exception
class InferenceError(
    ApplicationError
):

    '''
    @description: Raised when model
    inference fails.

    '''

    pass