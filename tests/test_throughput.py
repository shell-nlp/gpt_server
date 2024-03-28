import argparse
import threading
import time


def main(args):
    from openai import OpenAI

    server_address = args.server_address

    client = OpenAI(api_key="EMPTY", base_url=f"{server_address}/v1")

    print(f"server_address: {server_address}")

    content = "给我讲一个100字以上的故事"
    # content = "你是谁"
    data = {
        "model": args.model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": args.max_new_tokens,
        "stream": True,
    }

    def send_request(results, i):
        thread_server_address = server_address
        print(f"thread {i} goes to {thread_server_address}")
        output = client.chat.completions.create(
            model=args.model_name, messages=[
                {"role": "user", "content": content}],
            stream=True)

        text = ""
        for chunk in output:
            token = chunk.choices[0].delta.content
            if token:
                text += token
        print(f"完成 threads {i}")
        response_new_words = text
        # print(f"=== Thread {i} ===, words: {1}, error code: {error_code}")
        # results[i] = len(response_new_words) - len(content)
        results[i] = len(response_new_words)

    # use N threads to prompt the backend
    tik = time.time()
    threads = []
    results = [None] * args.n_thread
    for i in range(args.n_thread):
        t = threading.Thread(target=send_request, args=(results, i))
        t.start()
        # t.join()
        threads.append(t)

    for t in threads:
        t.join()

    print(f"Time (POST): {time.time() - tik} s")
    n_words = sum(results)
    time_seconds = time.time() - tik
    print(
        f"Time (Completion): {time_seconds}, n threads: {args.n_thread}, "
        f"throughput: {n_words / time_seconds} words/s."
        f"RPS: {args.n_thread / time_seconds} req/s."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-address", type=str,
                        default="http://localhost:8082")
    parser.add_argument("--model-name", type=str, default="qwen")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--n-thread", type=int, default=6)
    parser.add_argument("--test-dispatch", action="store_true")
    args = parser.parse_args()

    main(args)
