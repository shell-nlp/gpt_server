pip install -r requirements_v2.txt
pip install --force-reinstall lmdeploy==0.5.1 --no-deps

# 防止Python c库没有加载导致lmdeploy pytorch后端报错
export C_INCLUDE_PATH=/usr/include/python3.8:$C_INCLUDE_PATH
export LUS_INCLUDE_PATH=/usr/include/python3.8:$CPLUS_INCLUDE_PATH
