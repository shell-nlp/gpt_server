import base64
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8082/v1")
# 两种响应方式
## response_format = "url"    默认为 url
img = client.images.generate(model="flux", prompt="A red pig", response_format="url")
print(img.data[0])
## response_format = "b64_json"
img = client.images.generate(
    model="flux", prompt="A red pig", response_format="b64_json"
)
image_bytes = base64.b64decode(img.data[0].b64_json)
with open("output.png", "wb") as f:
    f.write(image_bytes)
