from openai import OpenAI
from rich import print
import base64


## 测试只对 文本嵌入
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8082/v1")
data = client.embeddings.create(model="bge-vl", input=["你是谁", "你是谁"])

print(data.data)
## 测试只对 图片嵌入


def image_to_base64(image_path):
    """将图片转换为Base64字符串"""
    base64_prefix = "data:image/png;base64,"

    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_prefix + base64_string


image_path = "../assets/logo.png"
# 使用本地的图片
url = image_to_base64(image_path)
data = client.embeddings.create(model="bge-vl", input=[url, url])

print(data.data)
## 测试 图文一起嵌入
data = client.embeddings.create(
    model="bge-vl", input=[{"text": "你好", "image": url}] * 2
)

print(data.data)
