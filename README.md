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

1. 重新适配了vLLM对模型适配较差，导致解码内容和hf不对齐的问题。
2.  **降低了模型适配的难度和项目使用的难度**(新模型的适配仅需修改低于5行代码)，从而更容易的部署自己最新的模型。

（仓库初步构建中，构建过程中没有经过完善的回归测试，可能会发生已适配的模型不可用的Bug,欢迎提出改进或者适配模型的建议意见。）

## 最新消息
本项目将在下一个版本将Python版本环境管理工具由pip切换到 uv(https://github.com/astral-sh/uv)

## 特色

1. 支持多种推理后端引擎，vLLM和LMDeploy，**LMDeploy**后端引擎，每秒处理的请求数是 vLLM 的 1.36 ~ 1.85 倍
2. 支持了Infinity后端，推理速度大于onnx/tensorrt，支持动态组批
3. 支持所有兼容sentence_transformers的语义向量模型（Embedding和Reranker）
4. 支持guided_decoding,强制模型按照Schema的要求进行JSON格式输出。
5. 支持了**Tools（Function Calling）**功能,并优化Tools解析方式，大大提高tools的调用成功率。兼容**LangChain**的 **bind_tools**、**with_structured_output**写法（目前支持Qwen系列、GLM系列）
6. 支持了**cohere**库接口规范的 /v1/rerank 接口
7. 全球唯一扩展了**openai**库,实现Reranker模型（rerank, /v1/rerank）。(代码样例见gpt_server/tests/test_openai_rerank.py)
8. 全球唯一支持了**openai**库的文本审核模型接口（text-moderation, /v1/moderations）。(代码样例见gpt_server/tests/test_openai_moderation.py)
9. 全球唯一支持了**openai**库的TTS模型接口（tts, /v1/audio/speech）,自带edge-tts(免费的TTS)(代码样例见gpt_server/tests/test_openai_tts.py)
10. 全球唯一支持了**openai**库的ASR模型接口（asr, /v1/audio/transcriptions）,基于fanasr后端(代码样例见gpt_server/tests/test_openai_transcriptions.py)
11. 支持多模态大模型
12. 与FastChat相同的分布式架构
## 配置文档
通过这个样例文件，可以很快的掌握项目的配置方式。
<br>
**配置文件的详细说明信息位于：[config_example.yaml](https://github.com/shell-nlp/gpt_server/blob/main/gpt_server/script/config_example.yaml "配置文件")**

## 更新信息

```plaintext
2025-4-2   支持了 OpenAI的ASR接口 /v1/audio/transcriptions
2025-4-1   支持了 internvl2.5模型
2025-2-9   支持了 QVQ
2024-12-22 支持了 tts, /v1/audio/speech TTS模型
2024-12-21 支持了 text-moderation, /v1/moderations 文本审核模型 
2024-12-14 支持了 phi-4
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
* [X] 可视化启动界面(不稳定,对开发人员来说比较鸡肋，后期将弃用！)
* [X] 并行的function call功能（tools）
* [ ] 支持 pip install 方式进行安装


## 快速开始

### 1. 配置python环境

#### 1.1 uv 方式 安装 (推荐,迄今最优秀的 库 管理工具, 性能和易用性远高于 pip、conda、poetry等,各大优秀开源项目都在使用。)

```bash
# 安装 uv 
pip install uv -U # 或查看教程 https://docs.astral.sh/uv/getting-started/installation/#standalone-installer
# uv venv --seed # （可选）创建 uv 虚拟环境，并设置seed
uv sync
source .venv/bin/activate # 激活 uv 环境
```

#### 1.2 conda  方式 安装(后期将弃用，可选)

```bash
# 1. 创建conda 环境
conda create -n gpt_server python=3.10

# 2. 激活conda 环境
conda activate gpt_server

# 3. 安装仓库（一定要使用 install.sh 安装,否则无法解决依赖冲突）
bash install.sh
```

### 2. 修改启动配置文件

#### 2.1 复制样例配置文件:
**配置文件的详细说明信息位于：[config_example.yaml](https://github.com/shell-nlp/gpt_server/blob/main/gpt_server/script/config_example.yaml "配置文件")**

```bash
# 进入script目录
cd gpt_server/script
# 复制样例配置文件
cp config_example.yaml config.yaml
```

### 3. 启动服务
#### 3.1 命令启动

```bash
uv run gpt_server/serving/main.py
```
或者
```bash
sh gpt_server/script/start.sh
```
或者
```bash
python gpt_server/serving/main.py
```

#### 3.2 Docker启动

##### 3.2.0 使用Docker Hub镜像
```bash
docker pull 506610466/gpt_server:latest # 如果拉取失败可尝试下面的方式
# 如果国内无法拉取docker镜像，可以尝试下面的国内镜像拉取的方式（不保证国内镜像源一直可用）
docker pull hub.littlediary.cn/506610466/gpt_server:latest
```

##### 3.2.1 手动构建镜像（可选）
- 构建镜像

```bash
docker build --rm -f "Dockerfile" -t gpt_server:latest "." 
```
##### 3.2.2 Docker Compose 启动 (建议在项目里使用docker-compose启动)

```bash
docker-compose  -f "docker-compose.yml" up -d --build gpt_server
```


#### 3.3 可视化UI方式启动服务（可选,有Bug，不建议使用，欢迎大佬优化代码）

```bash
cd gpt_server/serving
streamlit run server_ui.py
```

##### 3.3.1 Server UI界面:

![server_ui_demo.png](assets/server_ui_demo.png)


### 4. 使用 openai 库 进行调用

**见 gpt_server/tests 目录 样例测试代码:
https://github.com/shell-nlp/gpt_server/tree/main/tests**

### 5. 使用Chat UI

```bash
cd gpt_server/gpt_server/serving
streamlit run chat_ui.py
```

Chat UI界面:

![chat_ui_demo.png](assets/chat_ui_demo.png)



## 支持的模型以及推理后端

**推理速度：** LMDeploy TurboMind > vllm > LMDeploy PyTorch > HF

### **LLM**

|   Models / BackEnd    | model_type |  HF   | vllm  | LMDeploy TurboMind | LMDeploy PyTorch |
| :-------------------: | :--------: | :---: | :---: | :----------------: | :--------------: |
|      chatglm4-9b      |  chatglm   |   √   |   √   |         √          |        √         |
|      chatglm3-6b      |  chatglm   |   √   |   √   |         ×          |        √         |
| Qwen (7B, 14B, etc.)) |    qwen    |   √   |   √   |         √          |        √         |
| Qwen-1.5 (0.5B--72B)  |    qwen    |   √   |   √   |         √          |        √         |
|        Qwen-2         |    qwen    |   √   |   √   |         √          |        √         |
|       Qwen-2.5        |    qwen    |   √   |   √   |         √          |        √         |
|        Yi-34B         |     yi     |   √   |   √   |         √          |        √         |
|     Internlm-1.0      |  internlm  |   √   |   √   |         √          |        √         |
|     Internlm-2.0      |  internlm  |   √   |   √   |         √          |        √         |
|       Deepseek        |  deepseek  |   √   |   √   |         √          |        √         |
|        Llama-3        |   llama    |   √   |   √   |         √          |        √         |
|      Baichuan-2       |  baichuan  |   √   |   √   |         √          |        √         |
|        QWQ-32B        |    qwen    |   √   |   √   |         √          |        √         |
|         Phi-4         |    phi     |   √   |   √   |         ×          |        ×         |
### **VLM** (视觉大模型榜单 https://rank.opencompass.org.cn/leaderboard-multimodal)

| Models / BackEnd | model_type |  HF   | vllm  | LMDeploy TurboMind | LMDeploy PyTorch |
| :--------------: | :--------: | :---: | :---: | :----------------: | :--------------: |
|    glm-4v-9b     |  chatglm   |   ×   |   ×   |         ×          |        √         |
|    InternVL2     |  internvl  |   ×   |   ×   |         √          |        √         |
|   InternVL2.5    |  internvl  |   ×   |   ×   |         √          |        √         |
|  MiniCPM-V-2_6   |  minicpmv  |   ×   |   √   |         √          |        ×         |
|     Qwen2-VL     |    qwen    |   ×   |   √   |         ×          |        √         |
|    Qwen2.5-VL    |    qwen    |   ×   |   ×   |         ×          |        √         |
|       QVQ        |    qwen    |   ×   |   √   |         ×          |        ×         |
<br>

### Embedding/Rerank/Classify模型

**原则上支持所有的Embedding/Rerank/Classify模型**

**推理速度：** Infinity >> HF

以下模型经过测试可放心使用：

| Embedding/Rerank/Classify                                                           | HF  | Infinity |
| ----------------------------------------------------------------------------------- | --- | -------- |
| bge-reranker                                                                        | √   | √        |
| bce-reranker                                                                        | √   | √        |
| bge-embedding                                                                       | √   | √        |
| bce-embedding                                                                       | √   | √        |
| puff                                                                                | √   | √        |
| piccolo-base-zh-embedding                                                           | √   | √        |
| acge_text_embedding                                                                 | √   | √        |
| Yinka                                                                               | √   | √        |
| zpoint_large_embedding_zh                                                           | √   | √        |
| xiaobu-embedding                                                                    | √   | √        |
| Conan-embedding-v1                                                                  | √   | √        |
| KoalaAI/Text-Moderation（文本审核/多分类，审核文本是否存在暴力、色情等）            | ×   | √        |
| protectai/deberta-v3-base-prompt-injection-v2（提示注入/2分类，审核文本为提示注入） | ×   | √        |

目前 TencentBAC的 **Conan-embedding-v1** C-MTEB榜单排行第一(MTEB: https://huggingface.co/spaces/mteb/leaderboard)

<br>

### **ASR** (支持FunASR非实时模型 https://github.com/modelscope/FunASR/blob/main/README_zh.md)
目前只测试了SenseVoiceSmall模型（性能最优的），其它模型的支持情况只是从官方文档中拷贝过来，不一定可以正常使用，欢迎测试/提issue。

|    Models / BackEnd    | model_type |
| :--------------------: | :--------: |
|    SenseVoiceSmall     |   funasr   |
|     paraformer-zh      |   funasr   |
|     paraformer-en      |   funasr   |
|      conformer-en      |   funasr   |
|    Whisper-large-v3    |   funasr   |
| Whisper-large-v3-turbo |   funasr   |
|       Qwen-Audio       |   funasr   |
|    Qwen-Audio-Chat     |   funasr   |

<br>

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