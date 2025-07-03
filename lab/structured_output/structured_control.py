import json
import socket
from pathlib import Path

from utils.observations import LocalObs
from lmstudio import Client

CONFIG_PATH = Path(__file__).with_name("structured_config.json")
ACTION_ADDR = ("127.0.0.1", 5005)


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def send_action(action_idx: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(str(action_idx).encode(), ACTION_ADDR)
    sock.close()


def query_action(client: Client, config: dict, frame) -> int:
    prompt = config["user_prompt"]
    params = {
        "messages": [
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object", "schema": config["schema"]},
    }
    response = client.llm.remote_call("chat", params)
    data = json.loads(response["choices"][0]["message"]["content"])
    return int(data["action"])


def main() -> None:
    obs = LocalObs(source=1)
    client = Client()
    config = load_config()
    frame = obs.get_frame_224()
    action = query_action(client, config, frame)
    send_action(action)
    obs.close()
    client.close()


if __name__ == "__main__":
    main()
