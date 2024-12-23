from pathlib import Path
from openai import OpenAI

# 新版本 opnai
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8082/v1")
speech_file_path = Path(__file__).parent / "speech.mp3"
response = client.audio.speech.create(
    model="edge_tts",
    voice="zh-CN-YunxiNeural",
    input="你好啊，我是人工智能。",
)
response.write_to_file(speech_file_path)
