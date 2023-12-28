from typing import List
from fastchat.serve.base_model_worker import BaseModelWorker, app
from langchain.embeddings import HuggingFaceEmbeddings
from fastchat.model.model_adapter import add_model_args
import uuid
import argparse


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
    model_path: str = "/home/dev/model/assets/embeddings/sensenova/piccolo-base-zh/",
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


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21003)
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    add_model_args(parser)
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument(
        "--limit-worker-concurrency",
        type=int,
        default=5,
        help="Limit the model concurrency to prevent OOM.",
    )
    parser.add_argument("--embed-in-truncate", action="store_true")
    args = parser.parse_args()
    worker_addr = f"http://{args.host}:{args.port}"
    worker = get_worker(worker_addr=worker_addr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
