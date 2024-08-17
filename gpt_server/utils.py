import socket
from typing import List, Optional
import os
from multiprocessing import Process
import subprocess
from loguru import logger

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
    if 21001 not in used_ports:
        # 启动控制器
        start_controller(controller_host, controller_port, dispatch_method)
    if port not in used_ports:
        # 启动openai_api服务
        start_openai_server(host, port, controller_address, api_keys)


def stop_server():
    """停止服务"""
    stop_fastchat = (
        "ps -ef | grep fastchat.serve | awk '{print $2}' |xargs -I{} kill -9 {}"
    )
    stop_gpt_server = (
        "ps -ef | grep gpt_server | awk '{print $2}' |xargs -I{} kill -9 {}"
    )
    run_cmd(stop_fastchat)
    run_cmd(stop_gpt_server)
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


if __name__ == "__main__":
    print(is_port_in_use(21001))
