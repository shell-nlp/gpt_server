import time
import deepspeed
from deepspeed.inference.config import DeepSpeedTPConfig, DeepSpeedZeroConfig
import torch
import torch.distributed as dist
from transformers import AutoModel, AutoConfig, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import get_balanced_memory, infer_auto_device_map
from acc_worker import get_acc_model
from fastchat_adapter.utils import get_free_tcp_port

# mp.set_start_method("spawn")
# 创建进程对象


# MASTER_ADDR和MASTER_PORT是通信模块初始化需要的两个环境变量。
# 由于是在单机上，所以用localhost的ip就可以了。
# os.environ["MASTER_ADDR"] = "localhost"
# os.environ["MASTER_PORT"] = "29500"
# 用于初始化通信模块的函数。只有当world size和实际启动的进程数匹配，init_process_group才可以初始化成功。
# 通信模块初始化
# 进程会阻塞在该函数，直到确定所有进程都可以通信。
# dist.init_process_group(
#     "nccl", rank=rank, world_size=size, group_name=f"default_{rank}"
# )


def get_ds_model(model_path):
    deepspeed.init_distributed(
        # dist_backend="nccl"
        # distributed_port=master_port
    )

    # model = get_acc_model(model_path=model_path)

    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

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


if __name__ == "__main__":
    model_path = "/home/dev/model/chatglm3-6b/"
    ds_model = get_ds_model(model_path=model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model = ds_model

    t1 = time.time()
    all_text = 0
    for i in range(1):
        text = ""
        for response, new_history in model.stream_chat(
            query="你是谁", tokenizer=tokenizer
        ):
            text = response
            print(text)
        all_text += len(text)
    t2 = time.time() - t1
    print(t2)
    print("吞吐量", all_text / t2)
