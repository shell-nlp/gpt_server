import base64
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8082/v1")
# 两种响应方式
## response_format = "url"    默认为 url
prompt = "身着粉色汉服、精致刺绣的中国年轻女子。无可挑剔的妆容，额头上的红色花卉图案。精致的高髻，金凤头饰，红花，珠子。持有圆形折扇，上面有女士、树木、鸟。霓虹灯闪电灯（⚡️），明亮的黄色光芒，位于伸出的左手掌上方。室外夜景柔和，剪影的西安大雁塔，远处的七彩灯光模糊。"
model = "z_image"
img = client.images.generate(
    model=model, prompt=prompt, response_format="url", size="1664x928"
)
print(img.data[0])
# response_format = "b64_json"
img = client.images.generate(model=model, prompt=prompt, response_format="b64_json")
image_bytes = base64.b64decode(img.data[0].b64_json)
with open("output.png", "wb") as f:
    f.write(image_bytes)
