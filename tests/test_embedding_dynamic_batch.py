import asyncio
from openai import AsyncOpenAI
import time


async def f():
    batch = 1
    client = AsyncOpenAI(api_key="EMPTY", base_url="http://localhost:8082/v1")
    data = await client.embeddings.create(
        model="zpoint",
        input=["你是谁"] * batch,
    )
    return data.data


async def main():
    t1 = time.time()
    coro_list = []
    thread_num = 50
    for i in range(thread_num):
        coro_list.append(f())
    res = await asyncio.gather(*coro_list)
    t2 = time.time()
    print(f"耗时： {(t2-t1)*1000:.2f} ms")


# without dynamic_batch
# batch   thread
# 1        1      223.36  ms
# 1        10     615.48 ms
# 1        50     2041.31 ms
# 1        100    4369.68 ms
# 1        1000   36s
# 100      1      2219.71 ms


# with dynamic_batch   1 core
# batch   thread
# 1        1      310.21 ms
# 1        10     578.45ms
# 1        50     1800.96 ms
# 1        100    2901.79 ms
# 1        1000   26.6 s
# 100      1      2228.17 ms



if __name__ == "__main__":

    asyncio.run(main())
