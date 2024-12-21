from openai import OpenAI
from rich import print

# 新版本 opnai
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8082/v1")
# model: acge_text_embedding yinka zpoint
data = client.embeddings.create(model="piccolo-base-zh", input=["你是谁", "你是谁"])

print(data.data)
