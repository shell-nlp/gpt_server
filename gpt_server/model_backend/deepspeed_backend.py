import os
import deepspeed
from deepspeed.inference.config import DeepSpeedTPConfig
import torch
import torch.distributed as dist
from transformers import AutoModel, AutoModelForCausalLM


def get_ds_model(model_path, model_class):
    deepspeed.init_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    replace_with_kernel_inject = (
        False if "yi" in os.getenv("WORKER_NAME").lower() else True
    )

    model = model_class.from_pretrained(
        model_path, torch_dtype="auto", trust_remote_code=True
    )
    print(model)
    world_size = dist.get_world_size()
    print(f"world size {world_size}")

    old_current_device_function = deepspeed.get_accelerator().current_device_name

    def tmp_current_device_fn():
        deepspeed.get_accelerator().set_device(local_rank)
        deepspeed.get_accelerator().current_device_name = old_current_device_function
        return deepspeed.get_accelerator().current_device_name()

    deepspeed.get_accelerator().current_device_name = tmp_current_device_fn

    tensor_parallel_config = DeepSpeedTPConfig(
        enabled=True, tp_size=world_size, mpu=None, tp_group=None
    )
    ds_model = deepspeed.init_inference(
        model=model,  # # Transformers models
        tensor_parallel=tensor_parallel_config,  # Number of GPU    ==   world_size
        dtype=torch.half,  # dtype of the weights (fp16)
        # replace_method="auto",  # Lets DS autmatically identify the layer to replace
        replace_with_kernel_inject=replace_with_kernel_inject,  # TODO replace the model with the kernel injector
    )
    # torch.cuda.empty_cache()
    print(f"model is loaded on device {ds_model.module.device}")

    return ds_model.module
