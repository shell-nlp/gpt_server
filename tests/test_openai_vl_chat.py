import base64
from openai import OpenAI


def image_to_base64(image_path):
    """将图片转换为Base64字符串"""
    base64_prefix = "data:image/png;base64,"

    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_prefix + base64_string


image_path = "../assets/logo.png"
# 使用本地的图片
url = image_to_base64(image_path)
# 使用网络图片
url = "https://opencompass.oss-cn-shanghai.aliyuncs.com/image/compass-hub/botchat_banner.png"

# 新版本 opnai
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8082/v1")

stream = True
output = client.chat.completions.create(
    model="internvl2",  # internlm chatglm3  qwen  llama3 chatglm4
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "请描述这个图片",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": url,
                    },
                },
            ],
        }
    ],
    stream=stream,
)
if stream:
    for chunk in output:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
else:
    print(output.choices[0].message.content)
print()
