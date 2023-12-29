# gpt_server

本项目借助fastchat的基础能力来提供**openai server**的能力，**在此基础上适配了更多的模型**，**优化了fastchat兼容较差的模型**、支持了**deepspeed**、**accelerate**和**hf**的加载方式、**降低了模型适配的难度和项目使用的难度**，从而更容易的部署自己最新的模型。最终的目的是用户只要了解怎么样使用HF进行模型加载和生成内容就可以上手。

（仓库初步构建中，欢迎提出改进或者适配模型的建议。）

## 背景

待补充

## 支持的模型

支持且不限于以下模型 ，原则上支持transformer 全系列

| 模型          | 16bit | 4bit | ptv2 | deepspeed | accelerate | hf |
| ------------- | ----- | ---- | ---- | --------- | ---------- | -- |
| chatglm-6b    | √    | √   | ×   | √        | √         | √ |
| all-embedding | √    | √   | ×   | √        | ×         | √ |
| Qwen-14B      |       |      |      |           |            |    |

## 使用方式

待补充

## 致谢

    FastChat     https://github.com/lm-sys/FastChat
