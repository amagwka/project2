import time
try:
    from pynput.keyboard import Controller, Key
    keyboard = Controller()
except Exception:  # pragma: no cover - handle headless environments
    Controller = None
    Key = None
    keyboard = None

from servers.reward_server import ExternalRewardTracker
from servers.tracker import RewardTracker
from servers.constants import (
    ARROW_DELAY,
    WAIT_DELAY,
    NON_ARROW_DELAY,
    ARROW_IDX,
    WAIT_IDX,
)
from servers.nats_base import NatsServer

# Keyboard setup
if Key is not None:
    ACTION_KEYS = [Key.up, Key.down, Key.left, Key.right, "z", "x", None]
else:
    ACTION_KEYS = [None] * 7


# Action sender
def send_action(action_idx):
    try:
        if action_idx == WAIT_IDX:
            time.sleep(WAIT_DELAY)
            return

        if keyboard is None:
            return
        key = ACTION_KEYS[action_idx]
        keyboard.press(key)
        if action_idx in ARROW_IDX:
            time.sleep(ARROW_DELAY)
        else:
            time.sleep(NON_ARROW_DELAY)
        keyboard.release(key)
    except IndexError:
        print(f"[Error] Invalid action index: {action_idx}")



class NatsActionRewardServer(NatsServer):
    """NATS server handling actions and reward queries."""

    def __init__(self, tracker: RewardTracker, subject: str = "actions", url: str = "nats://127.0.0.1:4222"):
        super().__init__(subject, url)
        self.tracker = tracker

    async def handle(self, data: bytes) -> bytes | None:
        m = data.decode().strip().upper()
        if m == "GET":
            r = self.tracker.compute_reward()
            return f"{r:.6f}".encode()
        if m == "RESET":
            self.tracker.reset()
            return b"OK"
        try:
            action_idx = int(m)
            send_action(action_idx)
            return b"DONE"
        except ValueError:
            return b"ERR"


def start_nats_combined_server(tracker: RewardTracker, subject: str = "actions", url: str = "nats://127.0.0.1:4222") -> None:
    """Convenience helper starting ``NatsActionRewardServer``."""

    server = NatsActionRewardServer(tracker, subject=subject, url=url)
    server.serve(cleanup=tracker.close)


if __name__ == "__main__":
    tracker = ExternalRewardTracker()
    start_nats_combined_server(tracker)
