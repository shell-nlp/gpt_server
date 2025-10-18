from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8082/v1")

stream = True
response = client.responses.create(
    model="minicpmv",
    stream=True,
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "请描述这个图片"},
                {
                    "type": "input_image",
                    "image_url": "https://opencompass.oss-cn-shanghai.aliyuncs.com/image/compass-hub/botchat_banner.png",
                },
            ],
        }
    ],
)

for i in response:
    print(i)
