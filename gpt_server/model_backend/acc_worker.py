from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import get_balanced_memory, infer_auto_device_map
from transformers import AutoModel, AutoConfig
import torch


def get_acc_model(model_path: str):
    with init_empty_weights():
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        model = AutoModel.from_config(
            config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
    no_split_module_classes = getattr(model, "_no_split_modules", None)
    max_memory = get_balanced_memory(
        model,
        dtype=torch.bfloat16,
        low_zero=False,
        no_split_module_classes=no_split_module_classes,
    )

    device_map = infer_auto_device_map(
        model,
        dtype=torch.bfloat16,
        max_memory=max_memory,
        no_split_module_classes=no_split_module_classes,
    )
    print(device_map)
    model = load_checkpoint_and_dispatch(
        model=model, checkpoint=model_path, device_map=device_map, offload_buffers=False
    )
    return model
