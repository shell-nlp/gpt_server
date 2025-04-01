from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8082/v1")

audio_file = open("/home/dev/liuyu/project/gpt_server/test/asr_example_zh.wav", "rb")
transcript = client.audio.transcriptions.create(model="asr", file=audio_file)
print(transcript)
