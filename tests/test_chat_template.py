from transformers import AutoTokenizer

url = "https://opencompass.oss-cn-shanghai.aliyuncs.com/image/compass-hub/botchat_banner.png"
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "请描述这个图片",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": url,
                },
            },
        ],
    }
]

chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
tokenizer = AutoTokenizer.from_pretrained(
    "/home/dev/model/IntervitensInc/InternVL3-38B-AWQ"
)
# chat_template = None
prompt = tokenizer.apply_chat_template(
    conversation=messages,
    chat_template=chat_template,
    tokenize=False,
    add_generation_prompt=True,
)

print(prompt)
