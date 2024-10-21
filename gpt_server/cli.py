import argparse
import subprocess
import os

root_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(root_dir)
chat_ui_path = os.path.join(root_dir, "serving", "chat_ui.py")
server_ui_path = os.path.join(root_dir, "serving", "server_ui.py")


def run():
    parser = argparse.ArgumentParser(description="GPT Server CLI")
    parser.add_argument("--chat_ui", action="store_true", help="启动问答UI界面")
    parser.add_argument("--server_ui", action="store_true", help="启动服务UI界面")
    args = parser.parse_args()
    print(args)
    if args.chat_ui:
        cmd = f"streamlit run {chat_ui_path}"
        subprocess.run(cmd, shell=True)
    if args.server_ui:
        cmd = f"streamlit run {server_ui_path}"
        subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    run()
