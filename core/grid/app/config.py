from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    UPLOAD_DIR: str
    MODEL_PATH: str
    DLIB_LANDMARK_PATH: str
    FACE_LANDMARKER_PATH: str

    MAX_FRAMES: int
    FRAME_WIDTH: int
    FRAME_HEIGHT: int

    USE_CPU_ONLY: bool

    @property
    def upload_dir(self) -> Path:
        return self.BASE_DIR / self.UPLOAD_DIR

    @property
    def model_path(self) -> Path:
        return self.BASE_DIR / self.MODEL_PATH

    @property
    def dlib_landmark_path(self) -> Path:
        return self.BASE_DIR / self.DLIB_LANDMARK_PATH
        
    @property
    def face_landmarker_path(self) -> Path:
        return self.BASE_DIR / self.FACE_LANDMARKER_PATH


settings = Settings() # type: ignore