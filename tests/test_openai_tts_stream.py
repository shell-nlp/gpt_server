import base64
from pathlib import Path
from openai import OpenAI

speech_file_path = Path(__file__).parent / "speech.mp3"


with open("../assets/audio_data/roles/余承东/reference_audio.wav", "rb") as f:
    audio_bytes = f.read()
audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
clone_voice = False  # 是否使用声音克隆
# 雷军声音
url = "https://s1.aigei.com/src/aud/mp3/59/59b47e28dbc14589974a428180ef338d.mp3?download/%E9%9B%B7%E5%86%9B%E8%AF%AD%E9%9F%B3%E5%8C%85_%E7%88%B1%E7%BB%99%E7%BD%91_aigei_com.mp3&e=1745911680&token=P7S2Xpzfz11vAkASLTkfHN7Fw-oOZBecqeJaxypL:RvcXPTseOqkvy2f_ppELez7d8jY="
if clone_voice:
    voice = audio_base64
    # voice = url
else:
    voice = "新闻联播女声"

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8082/v1")
with client.audio.speech.with_streaming_response.create(
    model="tts",
    voice=voice,  # 内置 新闻联播女声， 支持声音克隆，voice 可以是base64  或者 一个 url
    input="本期节目主要内容： 一.习近平在参加北京市区人大代表换届选举投票时强调 不断发展全过程人民民主 加强选举全过程监督",
    speed="very_high",  # ["very_low", "low", "moderate", "high", "very_high"]
    extra_body={
        "pitch": "high"
    },  # ["very_low", "low", "moderate", "high", "very_high"]
) as response:
    with open(speech_file_path, mode="wb") as f:
        for chunk in response.iter_bytes():
            f.write(chunk)  # 这个 chunk 可以直接通过播放器进行流式的 实时播放
