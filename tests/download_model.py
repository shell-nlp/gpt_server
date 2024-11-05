"""
如果使用   hf 下载 则：
pip install -U huggingface_hub hf_transfer

如果使用 modelscope 下载 则：
pip install modelscope
"""


def model_download(model_id, local_dir="/data", hub_name="hf", repo_type="model"):
    import os

    # 配置 hf镜像
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    if hub_name == "hf":
        cmd = f"huggingface-cli download --repo-type {repo_type} --resume-download {model_id} --local-dir {local_dir}/{model_id} --local-dir-use-symlinks False --token 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"
        # 启动下载
        os.system(cmd)
        print("下载完成！")
    elif hub_name == "modelscope":
        from modelscope.hub.snapshot_download import snapshot_download

        snapshot_download(model_id=model_id, cache_dir=local_dir)  # revision="v1.0.2"
        print("下载完成！")
    else:
        print("hub_name 只支持  hf 和 modelscope ! 请重新设置")


if __name__ == "__main__":
    import os

    # 设置保存的路径
    local_dir = "/home/dev/model"
    # 仓库类型 dataset / model
    repo_type = "model"

    data_model_id_list = [
        "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
    ]

    for model_id in data_model_id_list:
        # 设置仓库id
        model_download(model_id, local_dir, hub_name="hf", repo_type=repo_type)
    print("所有下载完毕！")
