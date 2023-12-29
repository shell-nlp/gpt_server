import uuid
import os
from typing import List
from fastchat.serve.base_model_worker import BaseModelWorker, app
from langchain.embeddings import HuggingFaceEmbeddings

from gpt_server.utils import get_free_tcp_port


class EmbeddingWorker(BaseModelWorker):
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
        )
        model_name = model_path
        model_kwargs = {"device": "cuda"}
        encode_kwargs = {"normalize_embeddings": True, "batch_size": 64}
        self.embedding = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        self.init_heart_beat()

    def get_embeddings(self, params):
        print("params", params)
        print("worker_id:", self.worker_id)
        self.call_ct += 1
        ret = {"embedding": [], "token_num": 0}
        texts = params["input"]
        outputs = self.embedding.client.tokenize(texts)
        token_num = outputs["input_ids"].size(0) * outputs["input_ids"].size(1)
        embedding = self.embedding.embed_documents(texts=texts)
        ret["token_num"] = token_num
        ret["embedding"] = embedding
        return ret


def get_worker(
    model_path: str,
    controller_addr: str = "http://localhost:21001",
    worker_addr: str = "http://localhost:21002",
    worker_id: str = str(uuid.uuid4())[:8],
    model_names: List[str] = ["piccolo-base-zh"],
    limit_worker_concurrency: int = 6,
    conv_template: str = None,  # type: ignore
):
    worker = EmbeddingWorker(
        controller_addr,
        worker_addr,
        worker_id,
        model_path,
        model_names,
        limit_worker_concurrency,
        conv_template=conv_template,
    )
    return worker


def main():
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="gpus")

    parser.add_argument("--local_rank", type=str, default="local-rank")  # 必传
    parser.add_argument("--master_port", type=str, default="master_port")
    parser.add_argument("--model_name_or_path", type=str, default="model_name_or_path")
    parser.add_argument(
        "--model_names", type=lambda s: s.split(","), default="model_names"
    )

    args = parser.parse_args()

    print("local-rank", args.local_rank)
    print("master_port", args.master_port)

    os.environ["MASTER_PORT"] = args.master_port
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    host = "localhost"
    port = get_free_tcp_port()
    worker_addr = f"http://{host}:{port}"
    worker = get_worker(
        worker_addr=worker_addr,
        model_path=args.model_name_or_path,
        model_names=args.model_names,
    )
    if args.local_rank == "0":
        print("=======================================")
        print(f"{args.model_names[0]} 启动成功!")
        print("=======================================")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
