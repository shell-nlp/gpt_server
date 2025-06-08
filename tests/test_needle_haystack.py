"""大海捞针评测"""

import os
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model="qwen",
    api_url="http://localhost:8082/v1",
    api_key="123",
    eval_type="service",  # 使用API模型服务
    datasets=["needle_haystack"],
    eval_batch_size=20,
    dataset_args={
        "needle_haystack": {
            "subset_list": ["chinese", "english"][:1],  # 可选，指定使用中文或英文子集
            # 支持配置的参数
            "extra_params": {
                # 问题
                "retrieval_question": "What is the best thing to do in San Francisco?",
                # 插入的文本（可以设置为多个）
                "needles": [
                    "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
                ],
                # 语料的最小长度
                "context_lengths_min": 1000,
                # 语料的最大长度
                "context_lengths_max": 64 * 1024,  # 64K
                # 语料的区间数
                "context_lengths_num_intervals": 20,
                # 插入文本最小位置（百分数）
                "document_depth_percent_min": 0,
                # 插入文本最大位置（百分数）
                "document_depth_percent_max": 100,
                # 插入文本位置区间数
                "document_depth_percent_intervals": 10,
                # tokenizer的路径(可以指定modelscope的id)
                "tokenizer_path": "/home/dev/model/Qwen/Qwen2___5-32B-Instruct-AWQ/",
                "show_score": True,  # 是否在heatmap上显示分数
            },
        }
    },
    generation_config={
        "max_tokens": 512,  # 最大生成token数
    },
    judge_worker_num=5,
    judge_model_args={
        "model_id": "qwen",
        "api_url": "http://localhost:8082/v1",
        "api_key": "123",
    },
)
run_task(task_cfg=task_cfg)
