import os
from setuptools import setup, find_packages


pwd = os.path.dirname(__file__)
version_file = "gpt_server/version.py"


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
    packages=find_packages(),
    include_package_data=True,  # 确保包含 MANIFEST.in 中的文件
    # ... 其他 setup 参数 ...
)
