import os
import sys
import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install


pwd = os.path.dirname(__file__)
version_file = "gpt_server/version.py"


# 自定义安装类
class CustomInstallCommand(install):
    """自定义安装命令类，用于在安装过程中执行额外的脚本"""

    def run(self):
        # 调用父类的 run 方法
        install.run(self)

        # 运行 Bash 脚本
        script_path = os.path.join(os.path.dirname(__file__), "install.sh")
        if os.path.exists(script_path):
            print("Running install_script.sh...")
            try:
                subprocess.check_call(["/bin/bash", script_path])
            except subprocess.CalledProcessError as e:
                print(f"Error executing script {script_path}: {e}")
            sys.exit(1)
        else:
            print(f"Script {script_path} not found!")


def readme():
    with open(os.path.join(pwd, "README.md"), encoding="utf-8") as f:
        content = f.read()
    return content


def get_version():
    with open(os.path.join(pwd, version_file), "r") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__version__"]


setup(
    name="gpt_server",
    version=get_version(),
    license="Apache 2.0",
    description="gpt_server是一个用于生产级部署LLMs或Embedding的开源框架。",
    long_description=readme(),
    long_description_content_type="text/markdown",
    author="Yu Liu",
    author_email="506610466@qq.com",
    packages=find_packages(exclude=()),
    # ... 其他 setup 参数 ...
    cmdclass={
        "install": CustomInstallCommand,  # 关联自定义安装类
    },
)
