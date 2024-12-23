from pathlib import Path
from openai import OpenAI
import asyncio
import edge_tts
from rich import print


async def main():
    list_voices = await edge_tts.list_voices()
    zh_list_voices = [i["ShortName"] for i in list_voices if "zh-CN" in i["ShortName"]]
    print(f"支持以下中文voice: \n{zh_list_voices}")
    # 新版本 opnai
    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8082/v1")
    speech_file_path = Path(__file__).parent / "speech.mp3"
    response = client.audio.speech.create(
        model="edge_tts",
        voice="zh-CN-YunxiNeural",
        input="你好啊，我是人工智能。",
        speed=1.0,
    )
    response.write_to_file(speech_file_path)


if __name__ == "__main__":
    asyncio.run(main())
