from gpt_server.model_worker.utils import patch
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
patch()


def patch_infinity_embedder():
    import infinity_emb.transformer.embedder.sentence_transformer as embedder_module

    def patched_embedder_tokenize_lengths(self, sentences: list[str]) -> list[int]:
        """修复 SentenceTransformerPatched.tokenize_lengths 方法"""
        # 使用 tokenizer 的现代 API
        tks = self._infinity_tokenizer(
            sentences,
            add_special_tokens=False,
            truncation="longest_first",
            padding=False,
            return_length=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        # 提取长度信息
        if isinstance(tks, dict) and "length" in tks:
            return tks["length"].tolist()
        elif hasattr(tks, "encodings"):
            return [len(t.tokens) for t in tks.encodings]
        else:
            return [len(seq) for seq in tks["input_ids"]]

    embedder_module.SentenceTransformerPatched.tokenize_lengths = (
        patched_embedder_tokenize_lengths
    )


def patch_infinity_crossencoder():
    import infinity_emb.transformer.crossencoder.torch as crossencoder_module

    def patched_tokenize_lengths(self, sentences: list[str]) -> list[int]:
        """修复版本的 tokenize_lengths 方法，使用现代 transformers API"""
        # 使用 tokenizer 的 __call__ 方法
        tks = self._infinity_tokenizer(
            sentences,
            add_special_tokens=False,
            truncation="longest_first",
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_length=True,
            return_tensors=None,
        )
        # 根据 transformers 版本返回长度
        if isinstance(tks, dict) and "length" in tks:
            # 新版本返回字典，包含 length 字段
            return tks["length"].tolist()
        elif hasattr(tks, "encodings"):
            # 旧版本可能有 encodings 属性
            return [len(t.tokens) for t in tks.encodings]
        else:
            # 通用方法：计算每个序列的 token 数量
            return [len(seq) for seq in tks["input_ids"]]

    crossencoder_module.CrossEncoderPatched.tokenize_lengths = patched_tokenize_lengths


patch_infinity_embedder()
patch_infinity_crossencoder()
