from openai import OpenAI
import time

# 新版本 opnai
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8082/v1")

t1 = time.time()
output = client.completions.create(
    model="qwen", prompt=["从1数到10。开始:1,2,"] * 8, max_tokens=1000
)


for completion_choice in output.choices:
    print(completion_choice.index + 1, "--->", completion_choice.text)
print("cost time:", time.time() - t1)
