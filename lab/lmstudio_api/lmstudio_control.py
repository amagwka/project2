import socket
import json
import cv2
import lmstudio

from utils.observations import LocalObs
from lmstudio import Client

# Address of the combined action server
ACTION_ADDR = ("127.0.0.1", 5005)

# Prompts and JSON schema for structured output
ACTION_NAMES = [
    "up",
    "down",
    "left",
    "right",
    "enter",
    "shift",
    #"wait",
]
NAME_TO_INDEX = {name: idx for idx, name in enumerate(ACTION_NAMES)}

SYSTEM_PROMPT = (
    "You control Undertale by sending actions from the following list: "
    + ", ".join(ACTION_NAMES)
    + "."
)
USER_PROMPT = (
    'Given the current frame, respond with JSON {"action": <action>} where'
    " <action> is exactly one of: " + ", ".join(ACTION_NAMES) + "."
)
SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ACTION_NAMES},
    },
    "required": ["action"],
}


def send_action(action_idx: int) -> None:
    """Send the chosen action index over UDP."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(str(action_idx).encode(), ACTION_ADDR)
    sock.close()


def query_action(
    client: Client, frame, chat: lmstudio.Chat | None = None
) -> tuple[int, lmstudio.Chat]:
    """Request the next action from LM Studio and return the action index.

    The provided ``chat`` context is updated in place so that subsequent calls
    retain the conversation history.  A new chat is created if ``chat`` is ``None``.
    """

    _, buffer = cv2.imencode(".png", frame)
    handle = client.prepare_image(buffer.tobytes(), name="frame.png")

    if chat is None:
        chat = lmstudio.Chat()
        chat.add_system_prompt(SYSTEM_PROMPT)

    chat.add_user_message(USER_PROMPT, images=[handle])

    model = client.llm.model()
    result = model.respond(chat)
    content = result.content
    #print(content)
    data = json.loads(content)
    try:
        action_name = str(data["action"]).lower()
        return NAME_TO_INDEX[action_name]
    except (KeyError, ValueError, TypeError, LookupError) as exc:
        raise RuntimeError(f"Unexpected LM response: {content!r}") from exc


def main() -> None:
    import time
    obs = LocalObs(source=1)
    client = Client()
    chat = lmstudio.Chat()
    chat.add_system_prompt(SYSTEM_PROMPT)
    frame = obs.get_frame_224()
    action, chat = query_action(client, frame, chat)
    print(f"Action: {ACTION_NAMES[action]}")
    send_action(action)
    obs.close()
    client.close()


if __name__ == "__main__":
    main()
