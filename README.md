# gpt_server

本项目借助fastchat的基础能力来提供**openai server**的能力，**在此基础上适配了更多的模型**，**优化了fastchat兼容较差的模型**、支持了**vllm**和**hf**的加载方式、**降低了模型适配的难度和项目使用的难度**，从而更容易的部署自己最新的模型。

（仓库初步构建中，欢迎提出改进或者适配模型的建议。）

## 更新信息

```plaintext
2-1   支持了 vllm 实现
1-1   支持了 Yi-34B
12-31 支持了 qwen-14b
12-29 支持了 all-embedding(所有的词嵌入模型)
12-28 支持了 chatglm3-6b 
```

## 支持的模型

| 模型          | HF | vllm |
| ------------- | -- | ---- |
| All-embedding | √ | ×   |
| chatglm-6b    | √ | √   |
| Qwen-14B      | √ | √   |
| Yi-34B        | √ | √   |

## 使用方式

### 1. 修改配置文件

[config.yaml](https://github.com/shell-nlp/gpt_server/blob/main/gpt_server/serving/config.yaml "配置文件")

```bash
cd gpt_erver/serving
vim config.yaml
```

```yaml
serve_args:
  host: 0.0.0.0
  port: 8082

models:
  chatglm3-6b:  # 自定义的模型名称
    alias: gpt4,gpt3 # 模型别名     例如  gpt4,gpt3
    enable: false  # 是否启动这个模型   false / true
    model_name_or_path: /home/dev/model/chatglm3-6b/  # 模型的路径
    model_type: chatglm3  # 模型的类型 现在暂时 只有 chatglm3  embedding
    work_mode: hf # 启动方式  vllm  hf

    workers: 
    - gpus: # 第一个 worker 每一个 -gpus 表示一个 worker
      - 1  # 每个worker 使用的gpu
      - 2
      # - 3
    # - gpus:   # 第二个 worker
    #   - 1

  # Embedding 模型 同上
  embedding:
    alias: piccolo-base-zh# 别名   
    enable: true  # false true
    model_name_or_path: /home/dev/model/assets/embeddings/sensenova/piccolo-base-zh/
    model_type: embedding
    work_mode: hf

    workers:
    - gpus:
      - 0

```

### 2. 运行命令

[main.py](https://github.com/shell-nlp/gpt_server/blob/main/gpt_server/serving/main.py "服务主文件")

```bash
sh start.sh
```

## 致谢

    FastChat :  https://github.com/lm-sys/FastChat

    vllm :  https://github.com/vllm-project/vllm
