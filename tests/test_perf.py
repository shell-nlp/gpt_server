from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark
from rich import print

if __name__ == "__main__":
    args = Arguments(
        url="http://localhost:8082/v1/chat/completions",  # 请求的URL地址
        parallel=20,  # 并行请求的任务数量
        model="qwen",  # 使用的模型名称
        number=20,  # 请求数量
        api="openai",  # 使用的API服务
        dataset="openqa",  # 数据集名称
        stream=True,  #  是否启用流式处理
    )
    run_perf_benchmark(args)
    print(
        "想要了解指标的含义,请访问: https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/quick_start.html"
    )
