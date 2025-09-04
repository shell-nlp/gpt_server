from openai import OpenAI
from pydantic import BaseModel, Field

# 新版本 opnai
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8082/v1")
model = "qwen"
# 方式一
output = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "南京到北京多远"}],
    response_format={"type": "text"},
)
print(output.choices[0].message.content)
print("-" * 100)
# 方式二
output = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "用json进行回答"},
        {"role": "user", "content": "南京到北京多远"},
    ],
    response_format={"type": "json_object"},
)
print(output.choices[0].message.content)
print("-" * 100)


# 方式三
class Distance(BaseModel):
    距离: int = Field()
    单位: str


output = client.beta.chat.completions.parse(
    model=model,
    messages=[{"role": "user", "content": "南京到北京多远"}],
    response_format=Distance,
)

print(output.choices[0].message.parsed.dict())
print()
