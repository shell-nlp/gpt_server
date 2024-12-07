<div align="center">

<a href="https://github.com/shell-nlp/gpt_server"><img src="assets/logo.png" width="252" height="116" alt="gpt_server logo"></a>

# GPT Server
[![License][license-shield]][license-url]
[![Stars][stars-shield]][stars-url]
[![Forks][forks-shield]][forks-url]
[![Docker pulls][docker-pulls]][docker-pulls]
[![CI Status][ci-shield]][ci-url]
[![issue resolution][closed-issues-shield]][closed-issues-url]

</div>

本项目依托fastchat的基础能力来提供**openai server**的能力.

1. **在此基础上完美适配了更多的模型**，**优化了fastchat兼容较差的模型**
2. 支持了**Function Calling (Tools)** 能力（现阶段支持Qwen/ChatGLM，对Qwen支持更好）
3. 重新适配了vllm对模型适配较差，导致解码内容和hf不对齐的问题。
4. 支持了**vllm**、**LMDeploy**和**hf**的加载方式
5. 支持所有兼容sentence_transformers的语义向量模型（Embedding和Reranker）
6. 支持了Infinity后端，推理速度大于onnx/tensorrt，支持动态组批
7. 支持guided_decoding,强制模型按照Schema的要求进行JSON格式输出。
8. Chat模板无角色限制，使其完美支持了**LangGraph Agent**框架
9. 支持多模态大模型
10. **降低了模型适配的难度和项目使用的难度**(新模型的适配仅需修改低于5行代码)，从而更容易的部署自己最新的模型。

（仓库初步构建中，构建过程中没有经过完善的回归测试，可能会发生已适配的模型不可用的Bug,欢迎提出改进或者适配模型的建议意见。）

## 最新消息
本项目将在下一个版本将Python版本环境管理工具由pip切换到 uv(https://github.com/astral-sh/uv)

## 特色

1. 支持多种推理后端引擎，vLLM和LMDeploy，**LMDeploy**后端引擎，每秒处理的请求数是 vLLM 的 1.36 ~ 1.85 倍
2. 支持了Infinity后端，推理速度大于onnx/tensorrt，支持动态组批
3. 全球唯一完美支持**Tools（Function Calling）**功能的开源框架。兼容**LangChain**的 **bind_tools**、**AgentExecutor**、**with_structured_output**写法（目前支持Qwen系列、GLM系列）
4. 全球唯一扩展了**openai**库,实现Reranker模型。(代码样例见gpt_server/tests/test_openai_rerank.py)
5. 支持多模态大模型
6. 与FastChat相同的分布式架构

## 更新信息

```plaintext
2024-12-7  支持了 /v1/rerank 接口
2024-12-1  支持了 QWQ-32B-Preview
2024-10-15 支持了 Qwen2-VL
2024-9-19  支持了 minicpmv 模型
2024-8-17  支持了 vllm/hf 后端的 lora 部署
2024-8-14  支持了 InternVL2 系列多模态模型
2024-7-28  支持embedding/reranker 的动态组批加速（infinity后端, 比onnx/tensorrt更快）
2024-7-19  支持了多模态模型 glm-4v-gb 的LMDeploy PyTorch后端
2024-6-22  支持了 Qwen系列、ChatGLM系列 function call (tools) 能力
2024-6-12  支持了 qwen-2
2024-6-5   支持了 Yinka、zpoint_large_embedding_zh 嵌入模型
2024-6-5   支持了 glm4-9b系列（hf和vllm）
2024-4-27  支持了 LMDeploy 加速推理后端
2024-4-20  支持了 llama-3
2024-4-13  支持了 deepseek
2024-4-4   支持了 embedding模型 acge_text_embedding
2024-3-9   支持了 reranker 模型 （ bge-reranker，bce-reranker-base_v1）
2024-3-3   支持了 internlm-1.0 ,internlm-2.0
2024-3-2   支持了 qwen-1.5 0.5B, 1.8B, 4B, 7B, 14B, and 72B
2024-2-4   支持了 vllm 实现
2024-1-6   支持了 Yi-34B
2023-12-31 支持了 qwen-7b, qwen-14b
2023-12-30 支持了 all-embedding(理论上支持所有的词嵌入模型)
2023-12-24 支持了 chatglm3-6b 
```

## 路线

* [X] 支持HF后端
* [X] 支持vLLM后端
* [X] 支持LMDeploy后端
* [X] 支持 function call 功能 (tools)（Qwen系列、ChatGLM系列已经支持,后面有需求再继续扩展）
* [X] 支持多模态模型（初步支持glm-4v,其它模型后续慢慢支持）
* [X] 支持Embedding模型动态组批(实现方式：infinity后端)
* [X] 支持Reranker模型动态组批(实现方式：infinity后端)
* [X] 可视化启动界面(不稳定)
* [ ] 支持 pip install 方式进行安装
* [ ] 内置部分 tools (image_gen,code_interpreter,weather等)
* [ ] 并行的function call功能（tools）

## 启用方式
### Python启动

#### 1. 配置python环境

##### 1.1 uv 启动 (推荐)

```bash
# 安装 uv 
pip install uv # 或查看教程 https://docs.astral.sh/uv/getting-started/installation/#standalone-installer
sh install_uv.sh
```

##### 1.2 conda 启动(后期将启用)

```bash
# 1. 创建conda 环境
conda create -n gpt_server python=3.10

# 2. 激活conda 环境
conda activate gpt_server

# 3. 安装仓库（一定要使用 install.sh 安装,否则无法解决依赖冲突）
sh install.sh
```


#### 2. 修改启动配置文件

修改模型后端方式（vllm,lmdeploy等）

config.yaml中：

```bash
work_mode: vllm  # vllm hf lmdeploy-turbomind  lmdeploy-pytorch
```

修改embedding/reranker后端方式（embedding或embedding_infinity）

config.yaml中：

```bash
model_type: embedding_infinity # embedding 或 embedding_infinity  embedding_infinity后端速度远远大于 embedding
```

[config.yaml](https://github.com/shell-nlp/gpt_server/blob/main/gpt_server/script/config.yaml "配置文件")

```bash
cd gpt_server/script
vim config.yaml
```

```yaml
serve_args:  # openai 服务的 host 和 pot
  host: 0.0.0.0
  port: 8082
  controller_address: http://localhost:21001 # 控制器的ip地址
  # api_keys: 111,222  # 用来设置 openai 密钥

# controller
controller_args: # 控制器的配置参数
  host: 0.0.0.0
  port: 21001
  dispatch_method: shortest_queue # lottery、shortest_queue # 现有两种请求分发策略，随机（lottery） 和 最短队列（shortest_queue），最短队列方法更推荐。

# model worker
model_worker_args: # 模型的配置参数，这里port 不能设置，程序自动分配，并注册到 控制器中。
  host: 0.0.0.0
  controller_address: http://localhost:21001 # 将模型注册到 控制器的 地址

models:
  - chatglm4:  #自定义的模型名称
      alias: null # 别名     例如  gpt4,gpt3
      enable: true  # false true 控制是否启动模型worker
      model_config:
        model_name_or_path: /home/dev/model/THUDM/glm-4-9b-chat/
      model_type: chatglm  # qwen  yi internlm
      work_mode: vllm  # vllm hf lmdeploy-turbomind  lmdeploy-pytorch
      # lora:  # lora 配置
      #   test_lora: /home/dev/project/LLaMA-Factory/saves/Qwen1.5-14B-Chat/lora/train_2024-03-22-09-01-32/checkpoint-100
      device: gpu  # gpu / cpu
      workers:
      - gpus:
        # - 1
        - 0

# - gpus:  表示 模型使用 gpu[0,1]，默认使用的 TP(张量并行)
#   - 0
#   - 1

# - gpus:  表示启动两个模型，模型副本1加载到 0卡， 模型副本2 加载到 1卡
#   - 0
# - gpus:
#   - 1


  - qwen:  #自定义的模型名称
      alias: gpt-4,gpt-3.5-turbo,gpt-3.5-turbo-16k # 别名     例如  gpt4,gpt3
      enable: true  # false true 控制是否启动模型worker
      model_config:
        model_name_or_path: /home/dev/model/qwen/Qwen1___5-14B-Chat/ 
        enable_prefix_caching: false
        dtype: auto
        max_model_len: 65536
      model_type: qwen  # qwen  yi internlm
      work_mode: vllm  # vllm hf lmdeploy-turbomind  lmdeploy-pytorch
      device: gpu  # gpu / cpu
      workers:
      - gpus:
        - 1
      # - gpus:
      #   - 3

  # Embedding 模型
  - bge-base-zh:
      alias: null # 别名   
      enable: true  # false true
      model_config:
        model_name_or_path: /home/dev/model/Xorbits/bge-base-zh-v1___5/
      model_type: embedding_infinity # embedding_infinity 
      work_mode: hf
      device: gpu  # gpu / cpu
      workers:
      - gpus:
        - 2
 # reranker 模型
  - bge-reranker-base:
      alias: null # 别名   
      enable: true  # false true  控制是否启动模型worker
      model_config:
        model_name_or_path: /home/dev/model/Xorbits/bge-reranker-base/
      model_type: embedding_infinity # embedding_infinity
      work_mode: hf
      device: gpu  # gpu / cpu
      workers:
      - gpus:
        - 2
```

#### 3. 运行命令

[start.sh](https://github.com/shell-nlp/gpt_server/blob/main/gpt_server/script/start.sh "服务主文件")

```bash
cd gpt_server/script
sh start.sh
```

#### 4. 可视化UI方式启动服务（可选）

```bash
cd gpt_server/gpt_server/serving
streamlit run server_ui.py
```

##### 4.1 Server UI界面:

![server_ui_demo.png](assets/server_ui_demo.png)


## 支持的模型以及推理后端

**推理速度：** LMDeploy TurboMind > vllm > LMDeploy PyTorch > HF

### **LLM**

|    Models / BackEnd   |model_type | HF | vllm | LMDeploy TurboMind | LMDeploy PyTorch |
| :--------------------: |:-: | :-: | :--: | :----------------: | :--------------: |
|      chatglm4-9b    |chatglm  | √ |  √  |         √         |        √        |
|      chatglm3-6b     |chatglm | √ |  √  |         ×         |        √        |
| Qwen (7B, 14B, etc.)) |qwen | √ |  √  |         √         |        √        |
|  Qwen-1.5 (0.5B--72B)  |qwen| √ |  √  |         √         |        √        |
|         Qwen-2         |qwen| √ |  √  |         √         |        √        |
|         Qwen-2.5       |qwen| √ |  √  |         √         |        √        |
|         Yi-34B         |yi| √ |  √  |         √         |        √        |
|      Internlm-1.0      |internlm| √ |  √  |         √         |        √        |
|      Internlm-2.0      |internlm| √ |  √  |         √         |        √        |
|        Deepseek        |deepseek| √ |  √  |         √         |        √        |
|        Llama-3        |llama| √ |  √  |         √         |        √        |
|        Baichuan-2        |baichuan| √ |  √  |         √         |        √        |
|        QWQ-32B-Preview |qwen| √ |  √  |         √         |        √        |

### **VLM** (视觉大模型榜单 https://rank.opencompass.org.cn/leaderboard-multimodal)

| Models / BackEnd |model_type| HF | vllm | LMDeploy TurboMind | LMDeploy PyTorch |
| :--------------: | :-: | :-: | :--: | :----------------: | :--------------: |
|    glm-4v-9b    |chatglm| × |  ×  |         ×         |        √        |
|    InternVL2    |internvl2| × |  ×  |         √         |        √        |
|    MiniCPM-V-2_6   |minicpmv | × |  √  |         √         |        ×        |
|    Qwen2-VL   |qwen | × |  √  |         ×         |        √        |
<br>

### Embedding模型

**原则上支持所有的Embedding/Rerank 模型**

**推理速度：** Infinity >> HF

以下模型经过测试可放心使用：

| Embedding/Rerank          | HF | Infinity |
| ------------------------- | -- | -------- |
| bge-reranker              | √ | √       |
| bce-reranker              | √ | √       |
| bge-embedding             | √ | √       |
| bce-embedding             | √ | √       |
|puff                       | √ | √       |
| piccolo-base-zh-embedding | √ | √       |
| acge_text_embedding       | √ | √       |
| Yinka                     | √ | √       |
| zpoint_large_embedding_zh | √ | √       |
| xiaobu-embedding          | √ | √       |
|Conan-embedding-v1         | √ | √       |

目前 TencentBAC的 **Conan-embedding-v1** C-MTEB榜单排行第一(MTEB: https://huggingface.co/spaces/mteb/leaderboard)

#### 5. 使用 openai 库 进行调用

**见 gpt_server/tests 目录 样例测试代码:
https://github.com/shell-nlp/gpt_server/tree/main/tests**

#### 6. 使用Chat UI

```bash
cd gpt_server/gpt_server/serving
streamlit run chat_ui.py
```

Chat UI界面:

![chat_ui_demo.png](assets/chat_ui_demo.png)

## Docker安装

### 0. 使用Docker Hub镜像
```bash
docker pull 506610466/gpt_server:latest # 如果拉取失败可尝试下面的方式


# 如果国内无法拉取docker镜像，可以尝试下面的国内镜像拉取的方式（不保证国内镜像源一直可用）
docker pull docker.rainbond.cc/506610466/gpt_server:latest 

```

### 1. 手动构建镜像（可选）
#### 1.1 构建镜像

```bash
docker build --rm -f "Dockerfile" -t gpt_server:latest "." 
```
#### 1.2 Docker Compose启动 (建议在项目里使用docker-compose启动)

```bash
docker-compose  -f "docker-compose.yml" up -d --build gpt_server
```
## 架构

![gpt_server_archs.png](assets/gpt_server_archs.png)

## 致谢

 [FastChat](https://github.com/lm-sys/FastChat) : https://github.com/lm-sys/FastChat

 [vLLM](https://github.com/vllm-project/vllm)   : https://github.com/vllm-project/vllm

[LMDeploy ](https://github.com/InternLM/lmdeploy)： https://github.com/InternLM/lmdeploy

[infinity](https://github.com/michaelfeil/infinity) ： https://github.com/michaelfeil/infinity

## 与我联系(会邀请进入交流群)

![wechat.png](assets/wechat.png)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=shell-nlp/gpt_server&type=Date)](https://star-history.com/#shell-nlp/gpt_server&Date)

[open-issues-url]: https://github.com/shell-nlp/gpt_server/issues
[open-issues-shield]: https://img.shields.io/github/issues-raw/shell-nlp/gpt_server
[closed-issues-shield]: https://img.shields.io/github/issues-closed-raw/shell-nlp/gpt_server
[closed-issues-url]: https://github.com/shell-nlp/gpt_server/issues

[forks-url]: https://github.com/shell-nlp/gpt_server/network/members
[forks-shield]: https://img.shields.io/github/forks/shell-nlp/gpt_server?color=9cf
[stars-url]: https://github.com/shell-nlp/gpt_server/stargazers
[stars-shield]: https://img.shields.io/github/stars/shell-nlp/gpt_server?color=yellow
[license-url]: https://github.com/shell-nlp/gpt_server/blob/main/LICENSE
[license-shield]: https://img.shields.io/github/license/shell-nlp/gpt_server
[docker-pulls]: https://img.shields.io/docker/pulls/506610466/gpt_server
[ci-shield]: https://github.com/shell-nlp/gpt_server/actions/workflows/docker-image.yml/badge.svg
[ci-url]: https://github.com/shell-nlp/gpt_server/actions/workflows/docker-image.yml