from pydantic_settings import BaseSettings
from enum import Enum
from functools import lru_cache

class Profile(str, Enum):
    STAGING = "staging"

class ProfileSetting(BaseSettings):
    profile: Profile

    def get_settings(self):
        return Settings(_env_file="whisper/environments/.env" + "." + self.profile.lower()) # type: ignore

    class Config:
        # env_file = "../.env"
        env_file = "whisper/environments/.env"
        env_file_encoding = "utf-8"


class Settings(BaseSettings):
    DEVICE_ID: str
    FLASH_ATTN: bool

    class Config:
        case_sensitive = True
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings():
    profile = ProfileSetting() # type: ignore
    return profile.get_settings()