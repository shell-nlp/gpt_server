import openai


# 新版本
openai.api_key = "EMPTY"
openai.api_base = "http://192.168.102.19:8081/v1"
model = "piccolo-base-zh"
data = {
    "model": model,
    "input": [
        "你是谁",
        "你是谁",
    ],
}
data = openai.Embedding().create(**data)
print(data["data"])
