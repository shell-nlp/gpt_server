pip install uv # 安装 uv
uv venv --seed # 创建 uv 虚拟环境，并设置seed
uv sync # 同步环境依赖
source .venv/bin/activate # 激活 uv 环境
pip install --force-reinstall lmdeploy==0.6.2 --no-deps # 强制安装lmdeploy==0.6.2