import base64
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8082/v1")

img = client.images.generate(model="flux", prompt="A red pig")

image_bytes = base64.b64decode(img.data[0].b64_json)
with open("output.png", "wb") as f:
    f.write(image_bytes)
