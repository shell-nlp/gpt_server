"""支持 dify 等开源项目"""

import requests
from rich import print


def rerank():
    url = f"http://localhost:8082/v1/rerank"
    documents = [
        "A man is eating food.",
        "A man is eating a piece of bread.",
        "The girl is carrying a baby.",
        "A man is riding a horse.",
        "A woman is playing violin.",
    ]
    query = "A man is eating pasta."
    request_body = {
        "model": "bge-reranker-base",
        "documents": documents,
        "query": query,
        "return_documents": True,
    }

    response = requests.post(url, json=request_body)

    response_data = response.json()
    return response_data


print(rerank())
