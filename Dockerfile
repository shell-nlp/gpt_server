FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

COPY ./ /gpt_server

WORKDIR /gpt_server

RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    echo "开始安装python依赖环境" && apt-get update -y && apt install software-properties-common python3-dev build-essential git -y && add-apt-repository ppa:deadsnakes/ppa -y && \
    echo "开始安装python3.10" && apt-get install -y python3.10 curl && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.10 get-pip.py && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    ln -sf $(which python3.10) /usr/local/bin/python 

RUN sh install.sh && pip cache purge

CMD ["/bin/bash"]