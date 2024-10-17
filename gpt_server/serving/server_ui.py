import streamlit as st
import yaml
import os
import sys
from loguru import logger
from copy import deepcopy

if "config" not in st.session_state:
    # 配置根目录
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    root_dir = os.path.abspath(root_dir)
    sys.path.append(root_dir)
    original_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = original_pythonpath + ":" + root_dir
    sys.path.append(root_dir)
    config_path = os.path.join(root_dir, "gpt_server/script/config2.yaml")
    st.session_state["config_path"] = config_path
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        st.session_state["config"] = config
        st.session_state["init_config"] = deepcopy(config)


def update_config(config: dict):
    config_path = st.session_state["config_path"]
    yaml_config = yaml.dump(config, allow_unicode=True, sort_keys=False)
    with open(config_path, "w", encoding="utf8") as f:
        f.write(yaml_config)
    logger.info(f"yaml写入成功!")
    st.session_state["config"] = config


st.title("GPT_SERVER")

tab = st.sidebar.radio(
    "配置选项卡", ("OpenAI 服务配置", "Controller 配置", "Model_worker 配置")
)


# Function for Serve Args
def serve_args():
    config = st.session_state["init_config"]
    st.header("OpenAI服务配置")
    serve_host = st.text_input("host", config["serve_args"]["host"], key="serve_host")
    serve_port = st.text_input(
        "port",
        config["serve_args"]["port"],
        key="serve_port",
    )
    serve_controller_address = st.text_input(
        "controller_address",
        config["serve_args"]["controller_address"],
        key="serve_controller_address",
    )
    return serve_host, serve_port, serve_controller_address


# Function for Controller Args
def controller_args():
    config = st.session_state["init_config"]
    st.header("Controller 配置")
    controller_host = st.text_input(
        "host", config["controller_args"]["host"], key="controller_host"
    )
    controller_port = st.text_input(
        "port", config["controller_args"]["port"], key="controller_port"
    )
    dispatch_method = st.selectbox(
        "dispatch_method",
        options := ["shortest_queue", "lottery"],
        index=options.index(config["controller_args"]["dispatch_method"]),
        key="dispatch_method",
    )
    return controller_host, controller_port, dispatch_method


# Function for Model Worker Args
def model_worker_args():
    init_config = st.session_state["init_config"]
    new_config = st.session_state["config"]
    config = deepcopy(st.session_state["config"])
    st.header("Model_worker 配置")
    config["model_worker_args"]["host"] = model_worker_host = st.text_input(
        "host", init_config["model_worker_args"]["host"], key="model_worker_host"
    )
    config["model_worker_args"]["controller_address"] = model_controller_address = (
        st.text_input(
            "controller_address",
            init_config["model_worker_args"]["controller_address"],
            key="model_controller_address",
        )
    )
    # --------------------------------
    model_tab_dict = {}
    for i, model_config_ in enumerate(new_config["models"]):
        for model_name, model_config in model_config_.items():
            model_tab_dict[model_name] = model_config["enable"]

    model_tab_options = [
        (f"{model_name} | 开启状态: {':heavy_check_mark:' if enable_state else ':x:'}")
        for model_name, enable_state in model_tab_dict.items()
    ]

    model_tab = st.radio(
        "模型：",
        options=model_tab_options,
        horizontal=True,
        key="model_tab",
    )

    for i, model_config_ in enumerate(config["models"]):  # list
        for model_name, model_config in model_config_.items():
            if model_tab.split("|")[0].strip() == model_name:
                enable_state = model_config["enable"]
                left, right = st.columns(2)
                with left:

                    def on_change():
                        new_config["models"][i] = {
                            st.session_state[f"model_name_{i}"]: {
                                "alias": st.session_state[f"alias_{i}"],
                                "enable": st.session_state[f"enable_{i}"],
                                "model_name_or_path": st.session_state[
                                    f"model_name_or_path_{i}"
                                ],
                                "model_type": st.session_state[f"model_type_{i}"],
                                "work_mode": st.session_state[f"work_mode_{i}"],
                                "enable_prefix_caching": st.session_state[
                                    f"enable_prefix_caching_{i}"
                                ],
                                "device": st.session_state[f"device_{i}"],
                                "workers": yaml.safe_load(
                                    st.session_state[f"workers_{i}"]
                                ),
                            }
                        }
                        del_model = st.session_state[f"del_model_{i}"]
                        new_model = st.session_state[f"new_model_{i}"]

                        start_server = st.session_state[f"start_server_{i}"]
                        stop_server = st.session_state[f"stop_server_{i}"]
                        if start_server:
                            from gpt_server.utils import run_cmd

                            start_server_cmd = "nohup python -m gpt_server.serving.main > gpt_server.log &"
                            run_cmd(start_server_cmd)
                        if stop_server:
                            from gpt_server.utils import stop_server

                            stop_server()
                            logger.warning("服务已停止成功！")
                        if new_model:
                            new_config["models"].append(
                                {
                                    "new_model_name": {
                                        "alias": st.session_state[f"alias_{i}"],
                                        "enable": False,
                                        "model_name_or_path": st.session_state[
                                            f"model_name_or_path_{i}"
                                        ],
                                        "model_type": st.session_state[
                                            f"model_type_{i}"
                                        ],
                                        "work_mode": st.session_state[f"work_mode_{i}"],
                                        "enable_prefix_caching": st.session_state[
                                            f"enable_prefix_caching_{i}"
                                        ],
                                        "device": st.session_state[f"device_{i}"],
                                        "workers": yaml.safe_load(
                                            st.session_state[f"workers_{i}"]
                                        ),
                                    }
                                }
                            )
                        if del_model:
                            del new_config["models"][i]
                        update_config(new_config)

                    model_name_input = st.text_input(
                        "model_name",
                        model_name,
                        key=f"model_name_{i}",
                        on_change=on_change,
                    )
                    enable = st.selectbox(
                        "enable",
                        options := [True, False],
                        index=options.index(enable_state),
                        key=f"enable_{i}",
                        on_change=on_change,
                    )
                    enable_prefix_caching = st.selectbox(
                        "enable_prefix_caching",
                        options := [True, False],
                        index=options.index(
                            model_config.get("enable_prefix_caching", False)
                        ),
                        key=f"enable_prefix_caching_{i}",
                        on_change=on_change,
                    )
                    device = st.selectbox(
                        "device",
                        options := ["gpu", "cpu"],
                        index=options.index(model_config["device"]),
                        key=f"device_{i}",
                        on_change=on_change,
                    )
                with right:
                    model_alias = st.text_input(
                        "alias",
                        model_config["alias"],
                        placeholder="输入别名，例如gpt4",
                        key=f"alias_{i}",
                        on_change=on_change,
                    )
                    model_type = st.selectbox(
                        "model_type",
                        options := [
                            "qwen",
                            "yi",
                            "internlm",
                            "chatglm",
                            "llama",
                            "embedding_infinity",
                            "embedding",
                            "internvl2",
                            "baichuan",
                            "deepseek",
                            "minicpmv",
                            "mixtral",
                        ],
                        index=options.index(model_config["model_type"]),
                        key=f"model_type_{i}",
                        on_change=on_change,
                    )
                    work_mode = st.selectbox(
                        "work_mode",
                        options := [
                            "vllm",
                            "lmdeploy-turbomind",
                            "lmdeploy-pytorch",
                            "hf",
                        ],
                        index=options.index(model_config["work_mode"]),
                        key=f"work_mode_{i}",
                        on_change=on_change,
                    )

                model_name_or_path = st.text_input(
                    "model_name_or_path",
                    model_config["model_name_or_path"],
                    key=f"model_name_or_path_{i}",
                    on_change=on_change,
                )
                workers = model_config["workers"]
                # workers_str = json.dumps(workers, ensure_ascii=False, indent=2)
                workers_str = yaml.dump(workers)
                workers_value = st.text_area(
                    label="workers",
                    value=workers_str,
                    key=f"workers_{i}",
                    on_change=on_change,
                )
                workers_value_dict = yaml.safe_load(workers_value)
                c1, c2, c3, c4 = st.columns(4, gap="large")
                c1.button(label="启动服务", key=f"start_server_{i}", on_click=on_change)
                c2.button(label="停止服务", key=f"stop_server_{i}", on_click=on_change)
                c3.button(
                    label="删除这个模型", key=f"del_model_{i}", on_click=on_change
                )
                c4.button(label="添加新模型", key=f"new_model_{i}", on_click=on_change)

                config["models"][i] = {
                    model_name_input: {
                        "alias": model_alias,
                        "enable": enable,
                        "model_name_or_path": model_name_or_path,
                        "model_type": model_type,
                        "work_mode": work_mode,
                        "enable_prefix_caching": enable_prefix_caching,
                        "device": device,
                        "workers": workers_value_dict,
                    }
                }

                return config


config = st.session_state["config"]

if tab == "OpenAI 服务配置":
    (
        config["serve_args"]["host"],
        config["serve_args"]["port"],
        config["serve_args"]["controller_address"],
    ) = serve_args()
elif tab == "Controller 配置":

    (
        config["controller_args"]["host"],
        config["controller_args"]["port"],
        config["controller_args"]["dispatch_method"],
    ) = controller_args()
elif tab == "Model_worker 配置":

    config = model_worker_args()
update_config(config=config)
