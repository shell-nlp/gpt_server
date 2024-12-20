# FROM docker.rainbond.cc/506610466/cuda:12.2.0-runtime-ubuntu20.04-uv
FROM 506610466/cuda:12.2.0-runtime-ubuntu20.04-uv
COPY ./ /gpt_server
WORKDIR /gpt_server

RUN uv venv --seed && uv sync && uv cache clean && \
    echo '[[ -f .venv/bin/activate ]] && source .venv/bin/activate' >> ~/.bashrc

CMD ["/bin/bash"]