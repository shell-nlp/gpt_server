import argparse
import gradio as gr
from openai import OpenAI
import os
import sys
import yaml

# 配置根目录
root_dir = os.path.dirname(os.path.dirname(__file__))
root_dir = os.path.abspath(root_dir)

original_pythonpath = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = original_pythonpath + ":" + root_dir
sys.path.append(root_dir)
support_models = []
config_path = os.path.join(root_dir, "gpt_server/script/config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
# TODO 没有添加别名
for model_config_ in config["models"]:
    for model_name, model_config in model_config_.items():
        # 启用的模型
        if model_config["enable"]:
            if model_config["model_type"] != "embedding":
                support_models.append(model_name)

# Argument parser setup
parser = argparse.ArgumentParser(
    description="Chatbot Interface with Customizable Parameters"
)
parser.add_argument(
    "--model-url", type=str, default="http://localhost:8082/v1", help="Model URL"
)
parser.add_argument(
    "-m", "--model", type=str, default="chatglm4", help="Model name for the chatbot"
)
parser.add_argument(
    "--temp", type=float, default=0.8, help="Temperature for text generation"
)
parser.add_argument(
    "--stop-token-ids", type=str, default="", help="Comma-separated stop token IDs"
)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=8083)

# Parse the arguments
args = parser.parse_args()

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = args.model_url

# Create an OpenAI client to interact with the API server
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


models = [i.id for i in client.models.list() if i.id in support_models]


def predict(
    user_input: str,
    chatbot: list,
    model: str,
):
    chatbot.append((user_input, ""))
    # Convert chat history to OpenAI format
    history_openai_format = [
        {"role": "system", "content": "You are a great ai assistant."}
    ]
    for human, assistant in chatbot:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": assistant})
    history_openai_format.append({"role": "user", "content": user_input})

    # Create a chat completion request and send it to the API server
    stream = client.chat.completions.create(
        model=model,  # Model name to use
        messages=history_openai_format,  # Chat history
        temperature=args.temp,  # Temperature for text generation
        stream=True,  # Stream response
        extra_body={
            "repetition_penalty": 1,
            "stop_token_ids": (
                [int(id.strip()) for id in args.stop_token_ids.split(",") if id.strip()]
                if args.stop_token_ids
                else []
            ),
        },
    )

    # Read and return generated text from response stream
    partial_message = ""
    for chunk in stream:
        partial_message += chunk.choices[0].delta.content or ""
        chatbot[-1] = (user_input, partial_message)
        yield chatbot


def reset_state():
    return []


def reset_user_input():
    return gr.update(value="")


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">GPT_SERVER</h1>""")
    model = (
        gr.Dropdown(
            choices=models,
            label="选择模型",
            value=models[0],
            type="value",
            interactive=True,
        ),
    )
    chatbot = gr.Chatbot()
    with gr.Column():
        user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10)
        with gr.Row():
            submitBtn = gr.Button("Submit", variant="primary")
            emptyBtn = gr.Button("Clear History")
    submitBtn.click(
        predict,
        [user_input, chatbot, model[0]],
        [chatbot],
        show_progress=True,
    )
    submitBtn.click(reset_user_input, [], [user_input])
    emptyBtn.click(reset_state, outputs=[chatbot], show_progress=True)
demo.queue().launch(
    share=False, inbrowser=True, server_name="0.0.0.0", server_port=8083
)
