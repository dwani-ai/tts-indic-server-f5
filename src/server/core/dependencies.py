# core/dependencies.py
from fastapi import HTTPException
from core.managers import registry, settings

def get_tts_manager():
    if registry.tts_manager is None:
        raise HTTPException(status_code=500, detail="TTS manager not initialized")
    return registry.tts_manager


def get_settings():
    return settings