import yaml
from pprint import pprint
import os
import sys
from multiprocessing import Process
import signal
import ray
import torch

ray.shutdown()

# 配置根目录
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
root_dir = os.path.abspath(root_dir)

original_pythonpath = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = original_pythonpath + ":" + root_dir
sys.path.append(root_dir)
os.environ["LOGDIR"] = os.path.join(root_dir, "./logs")
from gpt_server.utils import (
    start_server,
    start_api_server,
    run_cmd,
    stop_server,
    delete_log,
)

# 删除日志
delete_log(root_dir)


def signal_handler(signum, frame):
    stop_server()
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, signal_handler)
config_path = os.path.join(root_dir, "gpt_server/script/config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
# print(config)
# ----------------------------启动 Controller 和 Openai API 服务----------------------------------------------------
host = config["serve_args"]["host"]
port = config["serve_args"]["port"]
start_api_server(host, port)
# ----------------------------启动 Controller 和 Openai API 服务----------------------------------------------------
process = []
for model_config_ in config["models"]:
    for model_name, model_config in model_config_.items():
        # 启用的模型
        if model_config["enable"]:
            # pprint(model_config)
            print()
            # 模型地址
            model_name_or_path = model_config["model_name_or_path"]
            # 模型类型
            model_type = model_config["model_type"]

            # model type 校验
            # py_path = f"{root_dir}/gpt_server/model_worker/{model_type}.py"
            py_path = f"-m gpt_server.model_worker.{model_type}"

            model_names = model_name
            if model_config["alias"]:
                model_names = model_name + "," + model_config["alias"]

            # 获取 worker 数目 并获取每个 worker 的资源
            workers = model_config["workers"]
            # if model_config["work_mode"] == "deepspeed":
            # 设置使用 deepspeed

            # process = []
            for worker in workers:
                gpus = worker["gpus"]
                # 将gpus int ---> str
                gpus = [str(i) for i in gpus]
                gpus_str = ",".join(gpus)
                num_gpus = len(gpus)
                run_mode = "python "
                CUDA_VISIBLE_DEVICES = ""
                if (
                    torch.cuda.is_available()
                    and model_config["device"].lower() == "gpu"
                ):
                    CUDA_VISIBLE_DEVICES = f"CUDA_VISIBLE_DEVICES={gpus_str} "
                elif model_config["device"].lower() == "cpu":
                    CUDA_VISIBLE_DEVICES = ""
                else:
                    raise Exception("目前仅支持 CPU/GPU设备!")
                backend = model_config["work_mode"]

                cmd = (
                    CUDA_VISIBLE_DEVICES
                    + run_mode
                    + py_path
                    + f" --num_gpus {num_gpus}"
                    + f" --model_name_or_path {model_name_or_path}"
                    + f" --model_names {model_names}"
                    + f" --backend {backend}"
                )

                p = Process(target=run_cmd, args=(cmd,))
                p.start()
                process.append(p)
for p in process:
    p.join()
