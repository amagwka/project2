import time
import threading

import psutil
from pymem import Pymem
from pymem.exception import MemoryReadError

from servers.base import UdpServer


class ExternalRewardTracker:
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


def start_udp_reward_server(tracker, host="0.0.0.0", port=5006):
    """Start a UDP server that replies to GET and RESET commands."""

    def handler(msg: str, addr):
        cmd = msg.strip().upper()
        if cmd == "GET":
            r = tracker.compute_reward()
            return f"{r:.6f}".encode()
        if cmd == "RESET":
            tracker.reset()
            return b"OK"
        return b"ERR"

    server = UdpServer(host, port, buffer_size=64)
    server.serve_forever(handler, cleanup=tracker.close)


if __name__ == "__main__":
    # Example usage: if you know pid, pass it; otherwise None to auto-find
    tracker = ExternalRewardTracker(process_id=None)
    start_udp_reward_server(tracker)
