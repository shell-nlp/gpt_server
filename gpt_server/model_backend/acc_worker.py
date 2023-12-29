import time
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import get_balanced_memory, infer_auto_device_map
from transformers import AutoModel, AutoConfig, AutoTokenizer
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


if __name__ == "__main__":
    model_path = "/home/dev/model/chatglm3-6b/"
    acc_model = get_acc_model(model_path=model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model = acc_model
    for response, new_history in model.stream_chat(query="你是谁", tokenizer=tokenizer):
        pass
    t1 = time.time()
    all_text = 0
    for i in range(10):
        text = ""
        for response, new_history in model.stream_chat(
            query="你是谁", tokenizer=tokenizer
        ):
            text = response
        all_text += len(text)
    t2 = time.time() - t1
    print(t2)
    print("吞吐量", all_text / t2)
