import time
import threading

import psutil
from pymem import Pymem
from pymem.exception import MemoryReadError

from servers.base import UdpServer
from servers.nats_base import NatsServer

from servers.tracker import RewardTracker


class ExternalRewardTracker(RewardTracker):
    def __init__(self, process_id=None, reward_config=None):
        # Auto-find Undertale.exe if no PID provided
        if process_id is None:
            process_id = self._wait_for_undertale()
        self.pm = Pymem()
        self.pm.open_process_from_id(process_id)

        self.previous_values = {}
        self.room_set = set()
        self.reward_config = reward_config or {
            "Experience Points": 0.1,
            "Health Points": 1.0,
            "Fight Metric": 0.1
        }

        # (base, offsets, label) triples
        self.base_addr_and_offsets = [
            (self.pm.base_address + 0x00408950, [0x0, 0x44, 0x10, 0x364, 0x3F0], "Experience Points"),
            (self.pm.base_address + 0x00408950, [0x0, 0x44, 0x10, 0x1F0, 0x620], "Health Points"),
            (self.pm.base_address + 0x00408950, [0x0, 0x44, 0x10, 0x190, 0x10], "Fight Metric"),
            (self.pm.base_address + 0x618EA0,              [0x0],   "Room"),
            (self.pm.base_address + 0x00408950, [0x0, 0x44, 0x10, 0x13C, 0x4A0], "Maximum Health Points"),
        ]

    def _wait_for_undertale(self):
        """Block until UNDERTALE.exe shows up in process list, return its PID."""
        print("Waiting for UNDERTALE.exe to launch...")
        while True:
            for proc in psutil.process_iter(attrs=["name", "pid"]):
                if proc.info["name"].lower() == "undertale.exe":
                    pid = proc.info["pid"]
                    print(f"Found UNDERTALE.exe (PID={pid})")
                    return pid
            time.sleep(1)

    def resolve_pointer(self, base_addr, offsets):
        addr = base_addr
        try:
            for off in offsets[:-1]:
                addr = self.pm.read_int(addr + off)
            return addr + offsets[-1]
        except MemoryReadError:
            return None

    def track_metrics(self):
        rewards = {}
        for base, offs, label in self.base_addr_and_offsets:
            resolved = self.resolve_pointer(base, offs)
            if resolved is None:
                continue
            # Room uses int, others double
            if label == "Room":
                val = self.pm.read_int(resolved)
            else:
                val = self.pm.read_double(resolved)
            prev = self.previous_values.get(label, val)
            delta = val - prev
            # special new-room bonus
            if label == "Room":
                bonus = 10 if val not in self.room_set else 0
                if bonus > 0:
                    self.room_set.add(val)
                delta = bonus
            rewards[label] = delta
            self.previous_values[label] = val
        return rewards

    def compute_reward(self):
        rwds = self.track_metrics()
        tot = 0.0
        for k, delta in rwds.items():
            w = self.reward_config.get(k, 1.0)
            tot += w * delta
        return tot

    def reset(self):
        self.previous_values.clear()
        self.room_set.clear()
        print("[Tracker] Reset.")

    def close(self):
        self.pm.close_process()


class RewardServer(UdpServer):
    """UDP server exposing the reward tracker."""

    def __init__(self, tracker: RewardTracker, host: str = "0.0.0.0", port: int = 5006):
        super().__init__(host, port, buffer_size=64)
        self.tracker = tracker

    def handle(self, data: bytes, addr):
        cmd = data.decode().strip().upper()
        if cmd == "GET":
            r = self.tracker.compute_reward()
            return f"{r:.6f}"
        if cmd == "RESET":
            self.tracker.reset()
            return b"OK"
        return b"ERR"


class NatsRewardServer(NatsServer):
    """NATS server exposing the reward tracker."""

    def __init__(self, tracker: RewardTracker, subject: str = "rewards", url: str = "nats://127.0.0.1:4222"):
        super().__init__(subject, url)
        self.tracker = tracker

    async def handle(self, data: bytes) -> bytes | None:
        cmd = data.decode().strip().upper()
        if cmd == "GET":
            r = self.tracker.compute_reward()
            return f"{r:.6f}".encode()
        if cmd == "RESET":
            self.tracker.reset()
            return b"OK"
        return b"ERR"


def start_udp_reward_server(tracker: RewardTracker, host: str = "0.0.0.0", port: int = 5006):
    """Convenience helper for ``RewardServer``."""

    server = RewardServer(tracker, host=host, port=port)
    server.serve(cleanup=tracker.close)


def start_nats_reward_server(tracker: RewardTracker, subject: str = "rewards", url: str = "nats://127.0.0.1:4222") -> None:
    """Convenience helper for ``NatsRewardServer``."""

    server = NatsRewardServer(tracker, subject=subject, url=url)
    server.serve(cleanup=tracker.close)


if __name__ == "__main__":
    # Example usage: if you know pid, pass it; otherwise None to auto-find
    tracker = ExternalRewardTracker(process_id=None)
    start_udp_reward_server(tracker)
