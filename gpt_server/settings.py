from pydantic_settings import BaseSettings


class ModelConfig(BaseSettings):
    model_name_or_path: str | None = None
    """模型名称或者路径"""
    backend: str = "vllm"
    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.8
    kv_cache_quant_policy: int = 0
    dtype: str = "auto"
    num_gpus: int = 1
    lora: str | None = None
    hf_overrides: dict | None = None
    """HuggingFace 配置覆盖参数"""


def get_model_config() -> ModelConfig:
    """获取模型配置"""
    return ModelConfig()
