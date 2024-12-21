from openai import OpenAI
from rich import print

# 新版本 opnai
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8082/v1")
moderation = client.moderations.create(
    input="I want to kill them.", model="text-moderation"
)
print(moderation)
