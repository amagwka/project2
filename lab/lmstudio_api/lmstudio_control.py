import socket
import json
import cv2
from utils.observations import LocalObs
from lmstudio import Client
from lmstudio.history import Chat

# Address of the combined action server
ACTION_ADDR = ("127.0.0.1", 5005)

# Prompts and JSON schema for structured output
SYSTEM_PROMPT = (
    "You control Undertale by sending numeric action indices from 0 to 6."
)
USER_PROMPT = (
    "Given the current frame, respond with JSON {\"action\": <index>} where"
    " <index> is the next action to take."
)
SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "integer", "minimum": 0, "maximum": 6},
    },
    "required": ["action"],
}


def send_action(action_idx: int) -> None:
    """Send the chosen action index over UDP."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(str(action_idx).encode(), ACTION_ADDR)
    sock.close()


def query_action(client: Client, frame) -> int:
    """Request the next action from LM Studio using structured output."""

    _, buffer = cv2.imencode(".png", frame)
    chat = Chat()
    chat.add_system_prompt(SYSTEM_PROMPT)
    handle = client.prepare_image(buffer.tobytes(), name="frame.png")
    chat.add_user_message(USER_PROMPT, images=[handle])

    model = client.llm.model()
    result = model.respond(
        chat,
        response_format={"type": "json_object", "schema": SCHEMA},
    )
    data = json.loads(result.content)
    try:
        return int(data["action"])
    except (KeyError, ValueError, TypeError) as exc:
        raise RuntimeError(f"Unexpected LM response: {result.content!r}") from exc


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
