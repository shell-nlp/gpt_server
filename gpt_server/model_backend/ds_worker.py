import deepspeed
from deepspeed.inference.config import DeepSpeedTPConfig
import torch
import torch.distributed as dist
from transformers import AutoModel, AutoModelForCausalLM


def get_ds_model(model_path):
    deepspeed.init_distributed(
        # dist_backend="nccl"
        # distributed_port=master_port
    )

    # model = get_acc_model(model_path=model_path)
    try:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    except ValueError as e:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    world_size = dist.get_world_size()
    print(f"world size {world_size}")
    tensor_parallel_config = DeepSpeedTPConfig(
        enabled=True, tp_size=world_size, mpu=None, tp_group=None
    )
    ds_model = deepspeed.init_inference(
        model=model,  # # Transformers models
        tensor_parallel=tensor_parallel_config,  # Number of GPU    ==   world_size
        dtype=torch.half,  # dtype of the weights (fp16)
        replace_method="auto",  # Lets DS autmatically identify the layer to replace
        replace_with_kernel_inject=True,  # replace the model with the kernel injector
    )
    print(f"model is loaded on device {ds_model.module.device}")

    return ds_model.module
