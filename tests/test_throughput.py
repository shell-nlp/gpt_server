import argparse
import threading
import time


def main(args):
    from openai import OpenAI

    server_address = args.server_address

    client = OpenAI(api_key="EMPTY", base_url=f"{server_address}/v1")

    print(f"server_address: {server_address}")

    content = "给我讲一个100字以上的故事"

    def send_request(results, i, prefill_times):
        thread_server_address = server_address
        print(f"Thread {i} goes to {thread_server_address}")
        t1 = time.time()
        output = client.chat.completions.create(
            model=args.model_name,
            messages=[{"role": "user", "content": content}],
            stream=True,
            temperature=0.0,
            max_tokens=args.max_new_tokens,
        )
        text = ""
        i_ = 0
        for chunk in output:
            token = chunk.choices[0].delta.content
            if token:
                i_ += 1
                if i_ == 1:
                    prefill_time = time.time() - t1
                    prefill_times[i] = prefill_time
                    print(f"线程 {i} - Prefill Time: {prefill_time:.2f}s")
                text += token
        # print(f"完成 Threads {i}")
        response_new_words = text
        # print(f"=== Thread {i} ===, words: {1}, error code: {error_code}")
        # results[i] = len(response_new_words) - len(content)
        results[i] = len(response_new_words)
        return prefill_time

    # use N threads to prompt the backend
    tik = time.time()
    threads = []
    results = [None] * args.n_thread
    prefill_times = [None] * args.n_thread
    for i in range(args.n_thread):
        t = threading.Thread(target=send_request, args=(results, i, prefill_times))
        t.start()
        # t.join()
        threads.append(t)

    for t in threads:
        t.join()
    QPS = len([i for i in prefill_times if i <= 1])
    n_words = sum(results)
    time_seconds = time.time() - tik
    print("*" * 60)
    print(
        f"Threads: {args.n_thread}\n" f"Time (POST): {time_seconds:.2f}\n",
        f"Throughput: {n_words / time_seconds:.2f} words/s\n",
        f"RPS: {args.n_thread / time_seconds:.2f} req/s\n",
        f"QPS: {QPS}\n",
        sep="",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-address", type=str, default="http://localhost:8082")

    parser.add_argument("--model-name", type=str, default="qwen")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--n-thread", type=int, default=20)
    parser.add_argument("--test-dispatch", action="store_true")
    args = parser.parse_args()
    threads = []
    for i in range(1):
        t = threading.Thread(target=main, args=(args,))
        t.start()
        threads.append(t)
        time.sleep(1)
    for t in threads:
        t.join()
    # main(args)
