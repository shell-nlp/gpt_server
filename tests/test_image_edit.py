import base64
from pathlib import Path
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8082/v1")
# 两种响应方式
## response_format = "url"    默认为 url
model = "image-edit"
image_path = Path(__file__).parent.parent / "assets/logo.png"
img = client.images.edit(
    model=model, prompt="变成红色", image=open(image_path, "rb"), response_format="url"
)
print(img.data[0])
## response_format = "b64_json" 使用这个请打开下面的注释
# img = client.images.edit(
#     model=model,
#     prompt="变成红色",
#     response_format="b64_json",
#     image=open(image_path, "rb"),
# )
# image_bytes = base64.b64decode(img.data[0].b64_json)
# with open("output.png", "wb") as f:
#     f.write(image_bytes)
