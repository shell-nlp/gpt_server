# 安装 lmdeploy
import os

# LMDEPLOY_VERSION = "0.2.1"
# PYTHON_VERSION = "310"
# proxy_ip = "http://10.20.20.32:7890"
# os.environ["HTTP_PROXY"] = proxy_ip
# os.environ["HTTPS_PROXY"] = proxy_ip
# cmd = f"pip install https://github.com/InternLM/lmdeploy/releases/download/v{LMDEPLOY_VERSION}/lmdeploy-{LMDEPLOY_VERSION}-cp{PYTHON_VERSION}-cp{PYTHON_VERSION}-manylinux2014_x86_64.whl"
# print(cmd)
# os.system(cmd)
from lmdeploy import (
    pipeline,
    GenerationConfig,
    TurbomindEngineConfig,
    PytorchEngineConfig,
)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # Turbomind
    # backend_config = TurbomindEngineConfig(tp=1)
    # Pytorch
    backend_config = PytorchEngineConfig()

    gen_config = GenerationConfig(top_p=0.8, temperature=0.8, max_new_tokens=1024)

    pipe = pipeline(
        "/home/dev/model/qwen/Qwen-14B-Chat/",
        backend_config=backend_config,
    )
    response = pipe(["写一个快排"], gen_config=gen_config)
    print(response)
