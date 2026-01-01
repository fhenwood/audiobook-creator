
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AnyHttpUrl
from typing import Optional

class Settings(BaseSettings):
    """
    Application configuration settings.
    Loads values from environment variables (and .env file).
    """
    # TTS Settings
    tts_base_url: str = Field(default="http://localhost:8880/v1")
    tts_api_key: str = Field(default="not-needed")
    tts_max_parallel_requests_batch_size: int = Field(default=4)
    default_tts_engine: str = Field(default="orpheus")
    default_tts_voice: str = Field(default="zac")
    default_output_format: str = Field(default="m4b")

    # LLM & GPU Settings
    llm_server_host: str = Field(default="llm_server")
    llm_server_port: str = Field(default="8000")
    llm_base_url: str = Field(default="http://localhost:8000/v1")
    llm_api_key: str = Field(default="not-needed")
    llm_model_name: str = Field(default="gpt-oss-20b")
    llm_request_timeout: int = Field(default=120)
    llm_max_retries: int = Field(default=3)
    llm_retry_delay: float = Field(default=2.0)
    llm_max_parallel_requests_batch_size: int = Field(default=1)
    
    # Emotion Processing
    use_emotion_tags: bool = Field(default=True)
    max_input_tokens: int = Field(default=500)
    emotion_context_window_size: int = Field(default=2)
    
    # Orpheus (Docker/GPU)
    orpheus_llama_host: str = Field(default="orpheus_llama")
    orpheus_llama_port: str = Field(default="5006")
    gpu_idle_timeout: int = Field(default=60)
    
    # Feature Flags
    no_think_mode: str = Field(default="true") # "true" or "false" string for env var compat
    enable_postprocessing: bool = Field(default=False)
    
    # Concurrency & VRAM Management
    max_concurrent_jobs: int = Field(default=1, description="Max parallel audiobook jobs")
    # VRAM requirements per engine (GB):
    # - orpheus: ~6GB
    # - vibevoice: ~14GB
    # - maya: ~6GB
    available_vram_gb: float = Field(default=24.0, description="Available GPU VRAM in GB")
    orpheus_vram_gb: float = Field(default=6.0)
    vibevoice_vram_gb: float = Field(default=14.0)
    maya_vram_gb: float = Field(default=6.0)
    
    # Model Management
    models_dir: str = Field(default="models")

    # Character Extraction
    max_context_window: int = Field(default=10240)
    max_batch_tokens: int = Field(default=2000)
    max_context_tokens_per_direction: int = Field(default=1000)
    presence_penalty: float = Field(default=0.6)
    frequency_penalty: float = Field(default=0.3)
    repeat_penalty: float = Field(default=1.1)

    # Orpheus Engine
    orpheus_temperature: float = Field(default=0.6)
    orpheus_top_p: float = Field(default=0.9)
    orpheus_max_tokens: int = Field(default=8192)

    # Config for pydantic-settings
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore" # Ignore extra env vars
    )

# Global settings instance
settings = Settings()
