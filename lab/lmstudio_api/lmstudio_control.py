import socket
from utils.observations import LocalObs
from lmstudio import Client

# Address of the combined action server
ACTION_ADDR = ("127.0.0.1", 5005)


def send_action(action_idx: int) -> None:
    """Send the chosen action index over UDP."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(str(action_idx).encode(), ACTION_ADDR)
    sock.close()


def query_action(client: Client, frame) -> int:
    """Request the next action from LM Studio.

    The LM is expected to reply with a plain integer in the range 0‑6.
    """
    prompt = "Given the current Undertale frame, respond with the next action index (0-6)."
    params = {
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "text"},
    }
    response = client.llm.remote_call("chat", params)
    text = response["choices"][0]["message"]["content"].strip()
    try:
        return int(text)
    except ValueError:
        raise RuntimeError(f"Unexpected LM response: {text!r}")


def main() -> None:
    obs = LocalObs(source=1)
    client = Client()
    frame = obs.get_frame_224()
    action = query_action(client, frame)
    send_action(action)
    obs.close()
    client.close()


if __name__ == "__main__":
    main()
