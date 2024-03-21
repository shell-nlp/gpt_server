from openai import OpenAI

# 新版本 opnai
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8082/v1")
data = client.embeddings.create(
    model="bge-reranker-base", input=["你是谁", "今年几岁"], extra_body={"query": "你多大了"})

print(data.data)
