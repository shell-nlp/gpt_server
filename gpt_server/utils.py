import socket
import uvicorn
import os
from multiprocessing import Process
import subprocess


def run_cmd(cmd):
    print("执行命令命令如下：")
    print(cmd)  # 执行
    print()
    subprocess.run(cmd, shell=True)


def start_controller():
    """启动fastchat控制器"""
    cmd = "python -m fastchat.serve.controller"
    controller_process = Process(target=run_cmd, args=(cmd,))
    controller_process.start()


def start_openai_server(host, port):
    """启动openai api 服务"""
    os.environ["FASTCHAT_WORKER_API_EMBEDDING_BATCH_SIZE"] = "100000"
    cmd = f"python -m fastchat.serve.openai_api_server --host {host} --port {port}"
    openai_server_process = Process(target=run_cmd, args=(cmd,))
    openai_server_process.start()


def start_server(host, port):
    """启动服务"""
    start_controller()
    start_openai_server(host, port)


def stop_server():
    """停止服务"""
    pass


def get_free_tcp_port():
    """获取可用的端口"""
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(("", 0))
    _, port = tcp.getsockname()
    tcp.close()
    return port
