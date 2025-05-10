import socket
from typing import List, Optional
import os
import sys
import json
from multiprocessing import Process
import subprocess
from loguru import logger
import torch
import psutil
from rich import print
import signal

logger.add("logs/gpt_server.log", rotation="100 MB", level="INFO")


def kill_child_processes(parent_pid, including_parent=False):
    "杀死子进程/僵尸进程"
    try:
        parent = psutil.Process(parent_pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                print(f"终止子进程 {child.pid}...")
                os.kill(child.pid, signal.SIGTERM)  # 优雅终止
                child.wait(5)  # 等待子进程最多 5 秒
            except psutil.NoSuchProcess:
                pass
            except psutil.TimeoutExpired():
                print(f"终止子进程 {child.pid} 超时！强制终止...")
                os.kill(child.pid, signal.SIGKILL)  # 强制终止
        if including_parent:
            print(f"终止父进程 {parent_pid}...")
            os.kill(parent_pid, signal.SIGTERM)
    except psutil.NoSuchProcess:
        print(f"父进程 {parent_pid} 不存在！")


# 记录父进程 PID
parent_pid = os.getpid()


def signal_handler(signum, frame):
    print("\nCtrl-C detected! Cleaning up...")
    # kill_child_processes(parent_pid, including_parent=False)
    stop_server()
    exit(0)  # 正常退出程序


signal.signal(signal.SIGINT, signal_handler)


def run_cmd(cmd: str, *args, **kwargs):
    logger.info(f"执行命令如下：\n{cmd}\n")
    # subprocess.run(cmd, shell=True)
    process = subprocess.Popen(cmd, shell=True)
    # 等待命令执行完成
    process.wait()
    return process.pid


def start_controller(controller_host, controller_port, dispatch_method):
    """启动fastchat控制器"""
    cmd = f"python -m fastchat.serve.controller --host {controller_host} --port {controller_port} --dispatch-method {dispatch_method} "
    cmd += "> /dev/null 2>&1"  # 完全静默（Linux/macOS）
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
    server_enable = config["serve_args"].get("enable", True)
    host = config["serve_args"]["host"]
    port = config["serve_args"]["port"]
    controller_address = config["serve_args"]["controller_address"]
    api_keys = config["serve_args"].get("api_keys", None)

    controller_enable = config["controller_args"].get("enable", True)
    controller_host = config["controller_args"]["host"]
    controller_port = config["controller_args"]["port"]
    dispatch_method = config["controller_args"].get("dispatch_method", "shortest_queue")
    # -----------------------------------------------------------------------
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
    if controller_port not in used_ports and controller_enable:
        # 启动控制器
        start_controller(controller_host, controller_port, dispatch_method)
    if port not in used_ports and server_enable:
        # 启动openai_api服务
        start_openai_server(host, port, controller_address, api_keys)
    # -----------------------------------------------------------------------


def get_model_types():
    model_types = []
    root_dir = os.path.dirname(__file__)
    model_worker_path = os.path.join(root_dir, "model_worker")
    # 遍历目录及其子目录
    for root, dirs, files in os.walk(model_worker_path):
        for file in files:
            # 检查文件是否以 .py 结尾
            if file.endswith(".py") and file != "__init__.py":
                # 输出文件的完整路径
                model_type = file[:-3]
                model_types.append(model_type)
    return model_types


model_types = get_model_types()


def start_model_worker(config: dict):
    process = []
    try:
        host = config["model_worker_args"]["host"]
        controller_address = config["model_worker_args"]["controller_address"]
        log_level = config["model_worker_args"].get("log_level", "WARNING")
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
                engine_config = model_config.get("model_config", None)
                # TODO -------------- 向前兼容 --------------
                if engine_config:
                    # 新版本
                    # 模型地址
                    model_name_or_path = engine_config["model_name_or_path"]
                    enable_prefix_caching = engine_config.get(
                        "enable_prefix_caching", "False"
                    )
                    dtype = engine_config.get("dtype", "auto")
                    lora = engine_config.get("lora", None)
                    max_model_len = engine_config.get("max_model_len", None)
                    gpu_memory_utilization = engine_config.get(
                        "gpu_memory_utilization", 0.8
                    )
                    kv_cache_quant_policy = engine_config.get(
                        "kv_cache_quant_policy", 0
                    )
                    vad_model = engine_config.get("vad_model", "")

                else:
                    logger.error(
                        f"""模型： {model_name}的 model_name_or_path,model_name_or_path 参数的配置必须修改到 model_config 下面！形如：
- minicpmv:
    alias: null
    enable: false
    model_type: minicpmv
    model_config:
      model_name_or_path: /home/dev/model/OpenBMB/MiniCPM-V-2_6/
      enable_prefix_caching: false
      dtype: auto
    work_mode: lmdeploy-turbomind
    device: gpu
    workers:
    - gpus:
      - 3
 """
                    )
                    sys.exit()

                # -------------- 向前兼容 --------------
                # 模型类型
                model_type = model_config["model_type"]
                # 对model type 进行校验
                if model_type not in model_types:
                    logger.error(
                        f"不支持model_type: {model_type},仅支持{model_types}模型之一！"
                    )
                    sys.exit()
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
                        + f" --dtype {dtype}"
                        + f" --enable_prefix_caching {enable_prefix_caching}"  # 是否开启 prefix cache
                        + f" --gpu_memory_utilization {gpu_memory_utilization}"  # 占用GPU比例
                        + f" --kv_cache_quant_policy {kv_cache_quant_policy}"  # kv cache 量化策略
                        + f" --log_level {log_level}"  # 日志水平
                    )
                    # 处理为 None的情况
                    if lora:
                        cmd += f" --lora '{json.dumps(lora)}'"
                    if max_model_len:
                        cmd += f" --max_model_len '{max_model_len}'"
                    if vad_model:
                        cmd += f" --vad_model '{vad_model}'"
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


if __name__ == "__main__":
    # /home/dev/model/KirillR/QwQ-32B-Preview-AWQ
    # get_model_types()
    from lmdeploy.serve.async_engine import get_names_from_model
    from lmdeploy.archs import get_model_arch
    from lmdeploy.cli.utils import get_chat_template

    ckpt = "/home/dev/model/Qwen/Qwen3-32B/"  # internlm2
    chat_template = get_chat_template(ckpt)
    model_type = get_names_from_model(ckpt)
    arch = get_model_arch(ckpt)
    print(chat_template)
    # print(arch)
    print(model_type)
    print(model_type[1] == "base")
    print()
