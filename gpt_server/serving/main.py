import yaml
import os
import sys
import ray
from dotenv import load_dotenv
from loguru import logger

load_dotenv()
os.environ["OPENBLAS_NUM_THREADS"] = (
    "1"  # 解决线程不足时，OpenBLAS blas_thread_init报错
)
ray.shutdown()

# 配置根目录
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
root_dir = os.path.abspath(root_dir)

original_pythonpath = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = original_pythonpath + ":" + root_dir
sys.path.append(root_dir)
os.environ["LOGDIR"] = os.path.join(root_dir, "logs")
from gpt_server.utils import (
    start_api_server,
    start_model_worker,
    delete_log,
)


def delete_flash_attn():
    "删除 flash_attn，避免报错"
    import shutil
    import os
    from pathlib import Path
    from loguru import logger

    root_path = Path(__file__).parent.parent.parent
    flash_attn_path = root_path.joinpath(
        ".venv/lib/python3.11/site-packages/flash_attn"
    )

    try:
        # 检查路径是否存在
        if os.path.exists(flash_attn_path):
            # 删除整个目录树
            shutil.rmtree(flash_attn_path)
            logger.info(f"成功删除: {flash_attn_path}")

    except PermissionError:
        logger.error("权限不足，无法删除 flash_attn")
    except Exception as e:
        logger.error(f"删除 flash_attn 失败: {e}")


# 删除日志
delete_log()

delete_flash_attn()

config_path = os.path.join(root_dir, "gpt_server/script/config.yaml")
env = os.getenv("ENV")
if env == "test":
    logger.warning("当前使用测试环境！开发测试专用")
    config_path = os.path.join(root_dir, "gpt_server/script/config_test.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)


# print(config)
def main():
    # ----------------------------启动 Controller 和 Openai API 服务----------------------------------------
    start_api_server(config=config)
    # ----------------------------启动 Model Worker 服务----------------------------------------------------
    start_model_worker(config=config)


if __name__ == "__main__":
    main()
