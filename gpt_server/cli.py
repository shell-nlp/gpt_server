import subprocess
import os
import typer

app = typer.Typer()
root_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(root_dir)
chat_ui_path = os.path.join(root_dir, "serving", "chat_ui.py")
server_ui_path = os.path.join(root_dir, "serving", "server_ui.py")


@app.command(help="启动 GPT Server UI")
def ui(
    server: bool = typer.Option(False, help="启动服务UI界面"),
    chat: bool = typer.Option(False, help="启动问答UI界面"),
):
    if server:
        cmd = f"streamlit run {server_ui_path}"
        subprocess.run(cmd, shell=True)
    if chat:
        cmd = f"streamlit run {chat_ui_path}"
        subprocess.run(cmd, shell=True)


def main():
    app()


if __name__ == "__main__":
    main()
