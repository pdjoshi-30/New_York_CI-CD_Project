from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_dir: str = Field("models", env="MODEL_DIR")
    model_name: str = Field("lstm_ae.pth", env="MODEL_NAME")
    scaler_name: str = Field("scaler.npz", env="SCALER_NAME")
    threshold_name: str = Field("threshold.txt", env="THRESHOLD_NAME")
    mongo_uri: str = Field("mongodb://mongo:27017", env="MONGO_URI")
    mongo_db: str = Field("anomalydb", env="MONGO_DB")
    mongo_col: str = Field("predictions", env="MONGO_COL")
    listen_host: str = Field("0.0.0.0", env="HOST")
    listen_port: int = Field(8000, env="PORT")
    class Config:
        env_file = ".env"
