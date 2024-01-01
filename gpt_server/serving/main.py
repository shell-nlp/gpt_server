import yaml
from pprint import pprint
import os
import sys
from multiprocessing import Process
import subprocess
import signal

# 配置根目录
root_dir = os.path.join(os.path.dirname(__file__), "..")
root_dir = os.path.abspath(root_dir)
sys.path.append(root_dir)
from gpt_server.utils import get_free_tcp_port, start_server, run_cmd, stop_server,delete_log
# 删除日志
delete_log(root_dir)
def signal_handler(signum, frame):
    stop_server()
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, signal_handler)

with open("./config.yaml", "r") as f:
    config = yaml.safe_load(f)
print(config)
# ----------------------------启动 Controller 和 Openai API 服务----------------------------------------------------
host = config["serve_args"]["host"]
port = config["serve_args"]["port"]
start_server(host, port)
# ----------------------------启动 Controller 和 Openai API 服务----------------------------------------------------
for model_name, model_config in config["models"].items():
    # 启用的模型
    if model_config["enable"]:
        pprint(model_config)
        print()
        # 模型地址
        model_name_or_path = model_config["model_name_or_path"]
        # 模型类型
        model_type = model_config["model_type"]
        # model type 校验
        py_path = f"{root_dir}/model_worker/{model_type}.py"

        model_names = model_name
        if model_config["alias"]:
            model_names = model_name + "," + model_config["alias"]

        # 获取 worker 数目 并获取每个 worker 的资源
        workers = model_config["workers"]
        # if model_config["work_mode"] == "deepspeed":
        # 设置使用 deepspeed

        process = []
        for worker in workers:
            gpus = worker["gpus"]
            # 将gpus int ---> str
            gpus = [str(i) for i in gpus]
            gpus_str = ",".join(gpus)
            num_gpus = len(gpus)
            if model_config["work_mode"] == "deepspeed":
                os.environ["USE_DS"] = "1"
                run_mode = f"deepspeed --num_gpus {num_gpus} "
                pass
            elif model_config["work_mode"] == "accelerate":
                os.environ["USE_ACC"] = "1"
                os.environ["CUDA_VISIBLE_DEVICES"] = gpus_str
                run_mode = "python "
                pass
            elif model_config["work_mode"] == "hf":
                os.environ["CUDA_VISIBLE_DEVICES"] = gpus_str
                run_mode = "python "
                pass

            # DS 只能在代码内部起效
            # os.environ["CUDA_VISIBLE_DEVICES"] = gpus_str
            cmd = (
                run_mode
                + py_path
                + f" --gpus {gpus_str}"
                + f" --master_port {get_free_tcp_port()}"
                + f" --model_name_or_path {model_name_or_path}"
                + f" --model_names {model_names}"
            )

            p = Process(target=run_cmd, args=(cmd,))
            p.start()
            process.append(p)
        for p in process:
            p.join()
