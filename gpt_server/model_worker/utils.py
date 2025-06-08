import httpx
from loguru import logger
from fastapi import HTTPException
import base64
import io


def extract_base64(data_url: str):
    """从Data URL中提取纯Base64数据"""
    return data_url.split(",", 1)[-1]  # 从第一个逗号后分割


async def get_bytes_from_url(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="无法从指定 URL 下载数据")
        return response.content


async def load_base64_or_url(base64_or_url):
    # 根据 reference_audio 内容判断读取方式
    if base64_or_url.startswith("http://") or base64_or_url.startswith("https://"):
        audio_bytes = await get_bytes_from_url(base64_or_url)
    else:
        try:
            if "data:" in base64_or_url:
                base64_or_url = extract_base64(data_url=base64_or_url)
            audio_bytes = base64.b64decode(base64_or_url)
        except Exception as e:
            logger.warning("无效的 base64 数据: " + str(e))
            raise HTTPException(status_code=400, detail="无效的 base64 数据: " + str(e))
    # 利用 BytesIO 包装字节数据，然后使用 soundfile 读取为 numpy 数组
    try:
        bytes_io = io.BytesIO(audio_bytes)
    except Exception as e:
        logger.warning("读取数据失败: " + str(e))
        raise HTTPException(status_code=400, detail="读取数据失败: " + str(e))
    return bytes_io


if __name__ == "__main__":

    # 示例用法
    data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg..."
    pure_base64 = extract_base64(data_url)
    print(pure_base64)  # 输出: iVBORw0KGgoAAAANSUhEUg...
