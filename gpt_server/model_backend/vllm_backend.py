from vllm import LLM

llm = LLM(
    "/home/dev/model/chatglm3-6b/", tensor_parallel_size=1, trust_remote_code=True
)
output = llm.generate("你是谁")
print(output)


def get_vllm_model(model_path, model_class):
    pass
