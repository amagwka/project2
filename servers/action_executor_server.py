import time
from typing import Optional

from .nats_base import PatternNatsServer

try:
    from pynput.keyboard import Controller, Key
    keyboard = Controller()
    ACTION_KEYS = [Key.up, Key.down, Key.left, Key.right, "z", "x"]
except Exception:
    keyboard = None
    ACTION_KEYS = [None] * 6


class ActionExecutorServer(PatternNatsServer):
    """Execute keyboard actions published on multiple queues."""

    def __init__(self, *, subject: str = "actions.>", url: str = "nats://127.0.0.1:4222"):
        # Subscribe to all action subjects including the base "actions"
        super().__init__(subject, url)

    async def handle(self, subject: str, data: bytes) -> Optional[bytes]:
        msg = data.decode().strip().split()
        if not msg:
            return b"ERR"
        try:
            idx = int(msg[0])
            duration_ms = int(msg[1]) if len(msg) > 1 else 10
        except Exception:
            return b"ERR"
        if keyboard is None or idx < 0 or idx >= len(ACTION_KEYS):
            return b"OK"
        key = ACTION_KEYS[idx]
        keyboard.press(key)
        time.sleep(max(0.01, min(duration_ms / 1000.0, 0.1)))
        keyboard.release(key)
        return b"OK"


start_action_executor_server = ActionExecutorServer
