import yaml
from pprint import pprint
import os
import sys
from multiprocessing import Process
import signal

# 配置根目录
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
root_dir = os.path.abspath(root_dir)

original_pythonpath = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = original_pythonpath + ":" + root_dir
sys.path.append(root_dir)
from gpt_server.utils import (
    start_server,
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
config_path = os.path.join(root_dir, "gpt_server/serving/config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
# print(config)
# ----------------------------启动 Controller 和 Openai API 服务----------------------------------------------------
host = config["serve_args"]["host"]
port = config["serve_args"]["port"]
start_server(host, port)
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
                # os.environ["CUDA_VISIBLE_DEVICES"] = gpus_str
                CUDA_VISIBLE_DEVICES = f"CUDA_VISIBLE_DEVICES={gpus_str} "

                if model_config["work_mode"] == "vllm":
                    use_vllm = 1
                else:
                    use_vllm = 0

                cmd = (
                    CUDA_VISIBLE_DEVICES
                    + run_mode
                    + py_path
                    # + f" --gpus {gpus_str}"
                    + f" --model_name_or_path {model_name_or_path}"
                    + f" --model_names {model_names}"
                )

                p = Process(target=run_cmd, args=(cmd,), kwargs={"use_vllm": use_vllm})
                p.start()
                process.append(p)
for p in process:
    p.join()
