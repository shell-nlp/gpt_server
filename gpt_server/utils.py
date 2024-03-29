import socket
import os
from multiprocessing import Process
import subprocess
from loguru import logger


def run_cmd(cmd: str, *args, **kwargs):
    use_vllm = kwargs.get("use_vllm", 0)
    if use_vllm:
        os.environ["USE_VLLM"] = "1"
    logger.info(f"执行命令如下：\n{cmd}\n")
    subprocess.run(cmd, shell=True)


def start_controller():
    """启动fastchat控制器"""
    cmd = "python -m fastchat.serve.controller"
    controller_process = Process(target=run_cmd, args=(cmd,))
    controller_process.start()


def start_openai_server(host, port):
    """启动openai api 服务"""
    os.environ["FASTCHAT_WORKER_API_EMBEDDING_BATCH_SIZE"] = "100000"
    # cmd = f"python -m fastchat.serve.openai_api_server --host {host} --port {port}"
    # my
    cmd = f"python -m gpt_server.serving.openai_api_server --host {host} --port {port}"
    openai_server_process = Process(target=run_cmd, args=(cmd,))
    openai_server_process.start()


def start_server(host, port):
    """启动服务"""
    start_controller()
    start_openai_server(host, port)


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


def delete_log(root_path):
    serving_path = os.path.join(root_path, "gpt_server/serving")
    datanames = os.listdir(serving_path)  # 查找本目录下所有文件
    for dataname in datanames:
        if dataname.endswith(".log"):
            os.remove(os.path.join(serving_path, f"{dataname}"))


def get_free_tcp_port():
    """获取可用的端口"""
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(("", 0))
    _, port = tcp.getsockname()
    tcp.close()
    return port
