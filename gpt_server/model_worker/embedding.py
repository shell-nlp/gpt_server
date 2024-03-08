from typing import List
from gpt_server.model_worker.base import ModelWorkerBase
import sentence_transformers


class EmbeddingWorker(ModelWorkerBase):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        conv_template: str = None,  # type: ignore
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template,
            model_type="embedding",
        )
        model_name = model_path
        model_kwargs = {"device": "cuda"}
        self.encode_kwargs = {"normalize_embeddings": True, "batch_size": 64}
       
        self.client = sentence_transformers.SentenceTransformer(
            model_name,**model_kwargs
        )

    def generate_stream_gate(self, params):
        pass

    def get_embeddings(self, params):
        print("params", params)
        print("worker_id:", self.worker_id)
        self.call_ct += 1
        ret = {"embedding": [], "token_num": 0}
        texts = params["input"]
        outputs = self.client.tokenize(texts)
        token_num = outputs["input_ids"].size(0) * outputs["input_ids"].size(1)
        
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embedding = self.client.encode(texts,**self.encode_kwargs)
        ret["embedding"] = embedding.tolist()
        ret["token_num"] = token_num
        return ret


if __name__ == "__main__":
    EmbeddingWorker.run()
