import time
from pynput.keyboard import Controller, Key

from servers.reward_server import ExternalRewardTracker
from servers.tracker import RewardTracker
from servers.constants import (
    ARROW_DELAY,
    WAIT_DELAY,
    NON_ARROW_DELAY,
    ARROW_IDX,
    WAIT_IDX,
)
from servers.base import UdpServer

# Keyboard setup
ACTION_KEYS = [Key.up, Key.down, Key.left, Key.right, "z", "x", None]
keyboard = Controller()


# Action sender
def send_action(action_idx):
    try:
        if action_idx == WAIT_IDX:
            time.sleep(WAIT_DELAY)
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


class ActionRewardServer(UdpServer):
    """UDP server handling both actions and reward queries."""

    def __init__(self, tracker: RewardTracker, host: str = "0.0.0.0", port: int = 5005):
        super().__init__(host, port, buffer_size=1024)
        self.tracker = tracker

    # --------------------------------------------------------------
    def handle(self, data: bytes, addr):
        m = data.decode().strip().upper()
        if m == "GET":
            r = self.tracker.compute_reward()
            return f"{r:.6f}"
        if m == "RESET":
            self.tracker.reset()
            return b"OK"
        try:
            action_idx = int(m)
            send_action(action_idx)
            return b"DONE"
        except ValueError:
            return b"ERR"


def start_combined_udp_server(tracker: RewardTracker, host: str = "0.0.0.0", port: int = 5005):
    """Convenience helper starting ``ActionRewardServer``."""

    server = ActionRewardServer(tracker, host=host, port=port)
    server.serve(cleanup=tracker.close)


if __name__ == "__main__":
    tracker = ExternalRewardTracker()
    start_combined_udp_server(tracker)
