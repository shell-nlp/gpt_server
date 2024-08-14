from openai import OpenAI

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
                        "url": "https://opencompass.oss-cn-shanghai.aliyuncs.com/image/compass-hub/botchat_banner.png",
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
