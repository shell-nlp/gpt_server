from pydantic_settings import BaseSettings


class ModelConfig(BaseSettings):
    backend: str = "vllm"
    enable_prefix_caching: bool = False
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.8
    kv_cache_quant_policy: int = 0
    dtype: str = "auto"
    num_gpus: int = 1
    lora: str | None = None


def get_model_config() -> ModelConfig:
    """获取模型配置"""
    return ModelConfig()
