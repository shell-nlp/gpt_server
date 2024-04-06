import argparse
from gpt_server.utils import start_server

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="gpt_server RESTful API server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host name")
    parser.add_argument("--port", type=int, default=8082, help="port number")
    args = parser.parse_args()
    start_server(host=args.host, port=args.port)
