import json

from openai import OpenAI


def get_weather(location: str, unit: str = "2") -> str:
    """
    Get the current weather in a given location
    """
    return "暴雨"


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

input_messages = [{"role": "user", "content": "南京天气怎么样？"}]


def main():
    base_url = "http://0.0.0.0:8082/v1"
    model = "qwen3"
    client = OpenAI(base_url=base_url, api_key="empty")
    response = client.responses.create(
        model=model, input=input_messages, tools=tools, tool_choice="required"
    )
    tool_call = response.output[0]
    args = json.loads(tool_call.arguments)
    result = get_weather(**args)

    input_messages.append(tool_call)  # append model's function call message
    input_messages.append(
        {  # append result message
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": str(result),
        }
    )
    print(input_messages)
    response_2 = client.responses.create(
        model=model,
        input=input_messages,
        tools=tools,
    )
    print(response_2.output_text)


if __name__ == "__main__":
    main()
