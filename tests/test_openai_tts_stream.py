from pathlib import Path
from openai import OpenAI

speech_file_path = Path(__file__).parent / "speech.mp3"
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8082/v1")
with client.audio.speech.with_streaming_response.create(
    model="tts",
    voice="alloy",
    input="本期节目主要内容： 一.习近平在参加北京市区人大代表换届选举投票时强调 不断发展全过程人民民主 加强选举全过程监督",
) as response:
    response.stream_to_file(speech_file_path)
