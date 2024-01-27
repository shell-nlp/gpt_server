from fastchat.conversation import get_conv_template
from langchain.chat_models import ChatOpenAI
import asyncio
import os

os.system("clear")

conv = get_conv_template("chatglm3")
conv.append_message(conv.roles[0], "你是谁")
prompt = conv.get_prompt() + "<|assistant|>"

print(prompt)
print("---------------------------")
# llm = ChatOpenAI(
#     model="chatglm3",
#     openai_api_key="x",
#     openai_api_base="http://192.168.102.19:8082/v1",
#     max_tokens=5120,
# )
# for i in llm.stream("你是谁"):
#     print(i.content, end="", flush=True)
# print()
# assert 0
from vllm import LLM, SamplingParams, AsyncLLMEngine, AsyncEngineArgs
from transformers import AutoTokenizer

# 异步方法
model_path = "/home/dev/model/chatglm3-6b/"
engine_args = AsyncEngineArgs(
    model_path, tensor_parallel_size=1, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
input_ids = tokenizer.build_chat_input("你是谁", history=[], role="user")[
    "input_ids"
].tolist()[0]
print("完美", input_ids)  # 完美
print(tokenizer.decode(input_ids))
print("*************")
# inputs = tokenizer.encode(prompt, return_tensors="pt").tolist()[0]
inputs = tokenizer.encode_plus(prompt, return_tensors="pt", is_split_into_words=True)[
    "input_ids"
].tolist()[0]
print("探索", inputs)
print(tokenizer.decode(inputs))

assert 0
# input_ids = inputs


engine = AsyncLLMEngine.from_engine_args(engine_args)


# ps -ef |grep ray | awk '{print $2}' |xargs -I{} kill -9 {}
# ray start --head
# ray start --address='192.168.102.19:6379'
async def main(request_id="0"):
    sampling = SamplingParams(
        use_beam_search=False, top_p=0.8, temperature=0, max_tokens=2048
    )

    results_generator = engine.generate(
        "你是谁",
        sampling_params=sampling,
        request_id=request_id,
        prompt_token_ids=input_ids,
    )
    # get the results
    async for request_output in results_generator:
        print(request_output.outputs[0].text)


if __name__ == "__main__":
    asyncio.run(main())

# --------------------------------------
# llm = LLM(
#     "/home/dev/model/chatglm3-6b/", tensor_parallel_size=1, trust_remote_code=True
# )
# tokenizer = llm.llm_engine.tokenizer
# input_ids = tokenizer.build_chat_input("你是谁？", history=[], role="user")[
#     "input_ids"
# ].tolist()
# print(tokenizer.decode(input_ids[0]))
# # print(input_ids)
# output = llm.generate(sampling_params=sampling, prompt_token_ids=input_ids)

# print(output)
