uv venv --seed # 创建 uv 虚拟环境，并设置seed
uv sync --index https://pypi.tuna.tsinghua.edu.cn/simple # 同步环境依赖
source .venv/bin/activate # 激活 uv 环境