import streamlit as st
from openai import OpenAI
import os
import sys
import yaml

if "config" not in st.session_state:
    # 配置根目录
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
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
                if (
                    model_config["model_type"] != "embedding"
                    and model_config["model_type"] != "embedding_infinity"
                ):
                    support_models.append(model_name)
    port = config["serve_args"]["port"]
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{port}/v1",
    )


def clear_chat_history():
    del st.session_state.messages


def init_chat_history():
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown("您好，很高兴为您服务！🥰")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []

    return st.session_state.messages


def main():
    st.title(f"GPT_SERVER")
    models = [i.id for i in client.models.list() if i.id in support_models]
    model = st.sidebar.selectbox(label="选择模型", options=models)
    temperature = st.sidebar.slider(
        label="temperature", min_value=0.0, max_value=2.0, value=0.8, step=0.1
    )
    top_p = st.sidebar.slider(
        label="top_p", min_value=0.0, max_value=1.0, value=1.0, step=0.1
    )
    messages = init_chat_history()

    if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        messages.append({"role": "user", "content": prompt})
        stream = client.chat.completions.create(
            model=model,  # Model name to use
            messages=messages,  # Chat history
            temperature=temperature,  # Temperature for text generation
            top_p=top_p,
            stream=True,  # Stream response
        )
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            partial_message = ""
            for chunk in stream:
                partial_message += chunk.choices[0].delta.content or ""
                placeholder.markdown(partial_message)
        messages.append({"role": "assistant", "content": partial_message})

        st.button("清空对话", on_click=clear_chat_history)


if __name__ == "__main__":
    main()
