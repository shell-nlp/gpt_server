import socket


def start_controller():
    """启动fastchat控制器"""
    pass


def start_openai_server():
    """启动openai api 服务"""


def start_server():
    """启动服务"""
    pass


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
