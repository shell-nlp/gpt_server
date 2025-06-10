from openai import OpenAI
from rich import print
import numpy as np

# 新版本 opnai
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8082/v1")
# model: acge_text_embedding yinka zpoint
response = client.embeddings.create(model="bge-m3", input=["我喜欢你", "我也喜欢你"])
print(response.data)
embeddings = [np.array(item.embedding) for item in response.data]  # 转为NumPy数组
v_a = embeddings[0].reshape(1, -1)  # 向量a
v_b = embeddings[1].reshape(-1, 1)  # 向量b
# 计算余弦相似度
similarity = np.dot(v_a, v_b)[0][0]
print(f"余弦相似度: {similarity:.4f}")
