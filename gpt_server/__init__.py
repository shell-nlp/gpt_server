import os
import sys
import importlib.util
from loguru import logger


def get_module_path(module_name):
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return f"Module '{module_name}' not found."
    return spec.origin


def check_lmdeploy_lib():
    if os.path.exists(os.path.join(lmdeploy_path, "lib")):
        return True
    return False


# 示例
module_name = "lmdeploy"
lmdeploy_path = os.path.dirname(get_module_path(module_name))
if not check_lmdeploy_lib():
    logger.warning("不存在lmdeploy的lib文件目录,系统将会自动下载！")
    cmd = "pip install --force-reinstall lmdeploy==0.6.0 --no-deps"
    logger.info(f"正在执行命令：{cmd}")
    os.system(cmd)
    logger.info("安装成功，请重新启动服务！")
    sys.exit()
else:
    pass
