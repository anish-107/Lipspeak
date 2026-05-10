''' config.py
@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Application configuration settings for the GRID lip reading model.
@date: 10 May 2026
@returns: Application Settings

'''


# Imports
from pathlib import Path
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict
)



# Application Settings
class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


    # Base Directory
    BASE_DIR: Path = Path(
        __file__
    ).resolve().parent.parent


    # Model Paths
    MODEL_PATH: str

    DLIB_LANDMARK_PATH: str

    FACE_LANDMARKER_PATH: str


    # Video Configuration
    MAX_FRAMES: int = 75

    FRAME_WIDTH: int = 140

    FRAME_HEIGHT: int = 46


    # Upload Configuration
    UPLOAD_DIR: str


    # Runtime Configuration
    USE_CPU_ONLY: bool = False


    # Upload Directory Path
    @property
    def upload_dir(self) -> Path:

        '''
        @description: Returns the upload directory path.
        @returns: Upload directory path

        '''

        return (
            self.BASE_DIR /
            self.UPLOAD_DIR
        )


    # Model Path
    @property
    def model_path(self) -> Path:

        '''
        @description: Returns the lip reading model path.
        @returns: Lip reading model path

        '''

        return (
            self.BASE_DIR /
            self.MODEL_PATH
        )


    # Dlib Landmark Path
    @property
    def dlib_landmark_path(self) -> Path:

        '''
        @description: Returns the dlib landmark model path.
        @returns: Dlib landmark model path

        '''

        return (
            self.BASE_DIR /
            self.DLIB_LANDMARK_PATH
        )


    # MediaPipe Landmark Path
    @property
    def face_landmarker_path(
        self
    ) -> Path:

        '''
        @description: Returns the MediaPipe face landmarker path.
        @returns: MediaPipe face landmarker path

        '''

        return (
            self.BASE_DIR /
            self.FACE_LANDMARKER_PATH
        )



# Settings Instance
settings = Settings() #type: ignore