from openai import OpenAI

# 新版本 opnai
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8082/v1")

stream = True
input_ = [{"role": "user", "content": "南京天气怎么样"}]
tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state, e.g., 'San Francisco, CA'",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
]
response = client.responses.create(
    model="qwen", input=input_, stream=stream, tools=tools
)


if stream:
    for event in response:
        print(event)
else:
    print(response, end="\n\n")
