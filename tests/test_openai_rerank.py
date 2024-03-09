import openai


# 新版本
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8082/v1"
model = "bge-reranker-base"
data = {
    "model": model,
    "query":"你多大了",
    "input": [
        "你是谁",
        "今年几岁",
    ],
}
data = openai.Embedding().create(**data)
print(data["data"])
