import yaml
from pprint import pprint
import os
import sys
from multiprocessing import Process
import subprocess

# 配置根目录
root_dir = os.path.join(os.path.dirname(__file__), "..")
root_dir = os.path.abspath(root_dir)
sys.path.append(root_dir)
from gpt_server.utils import get_free_tcp_port

with open("./config.yaml", "r") as f:
    config = yaml.safe_load(f)

for model_name, model_config in config["models"].items():
    pprint(model_config)
    print()
    # 启用的模型
    if model_config["enable"]:
        # 模型地址
        model_name_or_path = model_config["model_name_or_path"]
        # 模型类型
        model_type = model_config["model_type"]
        # model type 校验
        py_path = f"{root_dir}/model_worker/{model_type}.py"
        model_names = model_name + "," + model_config["alias"]
        # 获取 worker 数目 并获取每个 worker 的资源
        workers = model_config["workers"]
        if model_config["work_mode"] == "deepspeed":
            process = []
            for worker in workers:
                gpus = worker["gpus"]
                # 将gpus int ---> str
                gpus = [str(i) for i in gpus]
                gpus_str = ",".join(gpus)
                num_gpus = len(gpus)
                # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
                cmd = (
                    f"deepspeed --num_gpus {num_gpus} "
                    + py_path
                    + f" --gpus {gpus_str}"
                    + f" --master_port {get_free_tcp_port()}"
                    + f" --model_name_or_path {model_name_or_path}"
                    + f" --model_names {model_names}"
                )

                def run_cmd(cmd):
                    print("执行命令命令如下：")
                    print(cmd)  # 执行
                    print()
                    subprocess.run(cmd, shell=True)

                p = Process(target=run_cmd, args=(cmd,))
                p.start()
                process.append(p)
            for p in process:
                p.join()
