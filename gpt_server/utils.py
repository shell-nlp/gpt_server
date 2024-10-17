import socket
from typing import List, Optional
import os
import json
from multiprocessing import Process
import subprocess
from loguru import logger
import torch

logger.add("logs/gpt_server.log", rotation="100 MB", level="INFO")


def run_cmd(cmd: str, *args, **kwargs):
    logger.info(f"执行命令如下：\n{cmd}\n")
    subprocess.run(cmd, shell=True)


def start_controller(controller_host, controller_port, dispatch_method):
    """启动fastchat控制器"""
    cmd = f"python -m fastchat.serve.controller --host {controller_host} --port {controller_port} --dispatch-method {dispatch_method} "
    controller_process = Process(target=run_cmd, args=(cmd,))
    controller_process.start()


def start_openai_server(host, port, controller_address, api_keys=None):
    """启动openai api 服务"""
    os.environ["FASTCHAT_WORKER_API_EMBEDDING_BATCH_SIZE"] = "100000"

    cmd = f"python -m gpt_server.serving.openai_api_server --host {host} --port {port} --controller-address {controller_address}"
    if api_keys:
        cmd += f" --api-keys {api_keys}"
    openai_server_process = Process(target=run_cmd, args=(cmd,))
    openai_server_process.start()


def start_api_server(config: dict):
    host = config["serve_args"]["host"]
    port = config["serve_args"]["port"]
    controller_address = config["serve_args"]["controller_address"]
    api_keys = config["serve_args"].get("api_keys", None)

    controller_host = config["controller_args"]["host"]
    controller_port = config["controller_args"]["port"]
    dispatch_method = config["controller_args"].get("dispatch_method", "shortest_queue")

    start_server(
        host=host,
        port=port,
        controller_address=controller_address,
        api_keys=api_keys,
        controller_host=controller_host,
        controller_port=controller_port,
        dispatch_method=dispatch_method,
    )


def start_model_worker(config: dict):
    process = []
    try:
        host = config["model_worker_args"]["host"]
        controller_address = config["model_worker_args"]["controller_address"]
    except KeyError as e:
        error_msg = f"请参照 https://github.com/shell-nlp/gpt_server/blob/main/gpt_server/script/config.yaml 设置正确的 model_worker_args"
        logger.error(error_msg)
        raise KeyError(error_msg)

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
                lora = model_config.get("lora", None)
                enable_prefix_caching = model_config.get("enable_prefix_caching", False)
                # model type 校验
                # py_path = f"{root_dir}/gpt_server/model_worker/{model_type}.py"
                py_path = f"-m gpt_server.model_worker.{model_type}"

                model_names = model_name
                if model_config["alias"]:
                    model_names = model_name + "," + model_config["alias"]
                    if lora:  # 如果使用lora,将lora的name添加到 model_names 中
                        lora_names = list(lora.keys())
                        model_names += "," + ",".join(lora_names)

                # 获取 worker 数目 并获取每个 worker 的资源
                workers = model_config["workers"]

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
                        + f" --host {host}"
                        + f" --controller_address {controller_address}"
                    )
                    if lora:
                        cmd += f" --lora '{json.dumps(lora)}'"
                    if enable_prefix_caching:  # 是否开启 prefix cache
                        cmd += f" --enable_prefix_caching {enable_prefix_caching}"

                    p = Process(target=run_cmd, args=(cmd,))
                    p.start()
                    process.append(p)
    for p in process:
        p.join()


def start_server(
    host: str = "0.0.0.0",
    port: int = 8081,
    controller_address: str = "http://localhost:21001",
    api_keys: Optional[List[str]] = None,
    controller_host: str = "localhost",
    controller_port: int = 21001,
    dispatch_method: str = "shortest_queue",
):
    """启动服务"""
    # 判断端口是否被占用
    used_ports = []
    if is_port_in_use(controller_port):
        used_ports.append(controller_port)
    if is_port_in_use(port):
        used_ports.append(port)
    if len(used_ports) > 0:
        logger.warning(
            f"端口：{used_ports} 已被占用!为了系统的正常运行,请确保是被已启动的gpt_server服务占用。"
        )
    if controller_port not in used_ports:
        # 启动控制器
        start_controller(controller_host, controller_port, dispatch_method)
    if port not in used_ports:
        # 启动openai_api服务
        start_openai_server(host, port, controller_address, api_keys)


def stop_controller():
    cmd = "ps -ef | grep fastchat.serve | awk '{print $2}' |xargs -I{} kill -9 {}"
    run_cmd(cmd=cmd)


def stop_openai_server():
    cmd = "ps -ef | grep gpt_server |grep serving | awk '{print $2}' |xargs -I{} kill -9 {}"
    run_cmd(cmd=cmd)


def stop_all_model_worker():
    cmd = "ps -ef | grep gpt_server |grep model_worker | awk '{print $2}' |xargs -I{} kill -9 {}"
    run_cmd(cmd=cmd)


def stop_server():
    """停止服务"""
    stop_all_model_worker()
    stop_controller()
    stop_openai_server()

    logger.info("停止服务成功！")


def delete_log():
    logs_path = os.environ.get("LOGDIR")
    logger.debug(f"logs_path: {logs_path}")
    # 如果目录不存在则创建
    if not os.path.exists(logs_path):
        os.makedirs(logs_path, exist_ok=True)

    logs_path_datanames = os.listdir(logs_path)  # 查找本目录下所有文件
    datanames = logs_path_datanames
    for dataname in datanames:
        if dataname.endswith(".log"):
            os.remove(os.path.join(logs_path, f"{dataname}"))


def get_free_tcp_port():
    """获取可用的端口"""
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(("", 0))
    _, port = tcp.getsockname()
    tcp.close()
    return port


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("localhost", port))
            return False
        except:
            return True


model_type_mapping = {
    "yi": "yi",
    "qwen": "qwen",
    "glm4": "chatglm",
    "chatglm3": "chatglm",
    "internvl2-internlm2": "internvl2",
    "internlm2": "internlm",
    "internlm": "internlm",
    "baichuan2": "baichuan",
    "llama3": "llama",
    "mistral": "mistral",
    "deepseek": "deepseek",
}


def infer_model_type(model_path: str) -> str:
    """自动推测模型类型，未使用此代码"""
    from lmdeploy.model import best_match_model
    from transformers import AutoConfig

    match_model_type = best_match_model(model_path)
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config_model_type = model_config.get("model_type", False)
    return model_type_mapping[match_model_type]


if __name__ == "__main__":
    # ckpt = "deepseek-ai/deepseek-moe-16b-base"  # internlm2
    # model_type = best_match_model(ckpt)
    # print(model_type)
    pass
