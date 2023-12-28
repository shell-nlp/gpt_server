import yaml
from pprint import pprint
import os
import sys

# 配置根目录
root_dir = os.path.join(os.path.dirname(__file__), "..")
root_dir = os.path.abspath(root_dir)
sys.path.append(root_dir)
print(sys.path)
with open("./config.yaml", "r") as f:
    config = yaml.safe_load(f)

for model_name, model_config in config["models"].items():
    pprint(model_config)
    print()
    if model_config["enable"]:
        model_type = model_config["model_type"]
        if model_type == "chatglm3":
            from fastchat_adapter.model_worker.chatglm3 import main

        # 获取 worker 数目 并获取每个 worker 的资源
        workers = model_config["workers"]
        if model_config["work_mode"] == "deepspeed":
            for worker in workers:
                gpus = worker["gpus"]
                # 将gpus int ---> str
                gpus = [str(i) for i in gpus]
                num_gpus = len(gpus)
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)

                print("命令如下：")
                print()
