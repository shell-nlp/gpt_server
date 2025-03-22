from openai import OpenAI
import json

# 新版本 opnai
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8082/v1")


def get_weather(location: str, unit: str):
    return f"Getting the weather for {location} in {unit}..."


tool_functions = {"get_weather": get_weather}
tools = [
    {
        "type": "function",
        "function": {
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
    }
]
# 方式一
response = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "南京的天气怎么样"}],
    tools=tools,
    tool_choice="auto",
)
tool_call = response.choices[0].message.tool_calls[0].function
print(response.choices[0].message.tool_calls)
print(f"Function called: {tool_call.name}")
print(f"Arguments: {tool_call.arguments}")
print(f"Result: {get_weather(**json.loads(tool_call.arguments))}")
