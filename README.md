# Fastchat的模型适配器

本项目的目的是简化对**FastChat**项目中模型适配的难度，从而更容易的部署自己最新的模型，而不需要研究fastchat的实现过程。最终的目的是用户只要了解怎么样使用HF加载模型和生成内容就可以上手。

（仓库初步构建中，欢迎提出改进或者适配模型的建议。）

## 背景

## 支持的模型

支持且不限于以下模型 ，原则上支持transformer 全系列

| 模型          | 16bit | 4bit | ptv2 | deepspeed | accelerate | hf |
| ------------- | ----- | ---- | ---- | --------- | ---------- | -- |
| chatglm-6b    | √    | √   | ×   | √        | √         | √ |
| all-embedding | √    | √   | ×   | √        | ×         | √ |
|               |       |      |      |           |            |    |

## 使用方式


## 致谢

[FastChat](https://github.com/lm-sys/FastChat)     https://github.com/lm-sys/FastChat
