from openai import OpenAI
from pydantic import BaseModel, Field

# 新版本 opnai
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8082/v1")
model = "qwen3"
# 方式一
output = client.responses.create(
    model=model,
    input=[{"role": "user", "content": "南京到北京多远"}],
)
print(output.output_text)
print("-" * 100)
# 方式二
output = client.responses.create(
    model=model,
    input=[
        {"role": "system", "content": "用json进行回答"},
        {"role": "user", "content": "南京到北京多远"},
    ],
    text={"format": {"type": "json_object"}},
)
print(output.output_text)
print("-" * 100)


# 方式三
class Distance(BaseModel):
    距离: int = Field()
    单位: str


output = client.responses.create(
    model=model,
    input=[{"role": "user", "content": "南京到北京多远"}],
    text={
        "format": {
            "type": "json_schema",
            "name": "test",
            "schema": Distance.model_json_schema(),
        }
    },
)

print(output.output_text)
print()
