"""用于对 Embedding 模型进行评估的 MTEB 任务
指标文档: https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/mteb.html
"""

from evalscope import TaskConfig
from evalscope.run import run_task

# 待测试模型的列表
test_model_list = [
    {
        "model_name": "bge-m3",
        "dimensions": 1024,
    },
]

for test_model in test_model_list[:]:
    task_cfg = TaskConfig(
        eval_backend="RAGEval",
        eval_config={
            "tool": "MTEB",
            "model": [
                {
                    "model_name": test_model["model_name"],  # piccolo-base-zh bge-m3
                    "api_base": "http://localhost:8082/v1",
                    "api_key": "EMPTY",
                    "dimensions": test_model["dimensions"],
                    "encode_kwargs": {
                        "batch_size": 50,
                    },
                }
            ],
            "eval": {
                "tasks": [
                    "MedicalRetrieval",
                ],
                "verbosity": 2,
                "top_k": 10,
                "overwrite_results": True,
                # "limits": 100,
            },
        },
    )

    # Run task
    run_task(task_cfg=task_cfg)
# or
# run_task(task_cfg=two_stage_task_cfg)
