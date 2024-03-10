import os

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
