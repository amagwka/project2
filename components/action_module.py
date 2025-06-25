import socket
import time
import threading
import psutil
from pymem import Pymem
from pymem.exception import MemoryReadError
from pynput.keyboard import Controller, Key

# Keyboard setup
ACTION_KEYS = [Key.up, Key.down, Key.left, Key.right, 'z', 'x', Key.space]
ARROW_IDX = {0, 1, 2, 3}
keyboard = Controller()

# Reward Tracker
class ExternalRewardTracker:
    def __init__(self, process_id=None, reward_config=None):
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
        self.base_addr_and_offsets = [
            (self.pm.base_address + 0x00408950, [0x0, 0x44, 0x10, 0x364, 0x3F0], "Experience Points"),
            (self.pm.base_address + 0x00408950, [0x0, 0x44, 0x10, 0x1F0, 0x620], "Health Points"),
            (self.pm.base_address + 0x00408950, [0x0, 0x44, 0x10, 0x190, 0x10],  "Fight Metric"),
            (self.pm.base_address + 0x618EA0,              [0x0],                  "Room"),
            (self.pm.base_address + 0x00408950, [0x0, 0x44, 0x10, 0x13C, 0x4A0], "Maximum Health Points"),
        ]

    def _wait_for_undertale(self):
        print("Waiting for UNDERTALE.exe to launch...")
        while True:
            for proc in psutil.process_iter(attrs=["name", "pid"]):
                if proc.info["name"].lower() == "undertale.exe":
                    print(f"Found UNDERTALE.exe (PID={proc.info['pid']})")
                    return proc.info["pid"]
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
            val = self.pm.read_int(resolved) if label == "Room" else self.pm.read_double(resolved)
            prev = self.previous_values.get(label, val)
            delta = val - prev
            if label == "Room":
                delta = 10 if val not in self.room_set else 0
                if delta > 0:
                    self.room_set.add(val)
            rewards[label] = delta
            self.previous_values[label] = val
        return rewards

    def compute_reward(self):
        rwds = self.track_metrics()
        return sum(self.reward_config.get(k, 1.0) * v for k, v in rwds.items())

    def reset(self):
        self.previous_values.clear()
        self.room_set.clear()
        print("[Tracker] Reset.")

    def close(self):
        self.pm.close_process()


# Action sender
def send_action(action_idx):
    try:
        key = ACTION_KEYS[action_idx]
        keyboard.press(key)
        if action_idx in ARROW_IDX:
            time.sleep(0.08)
        keyboard.release(key)
    except IndexError:
        print(f"[Error] Invalid action index: {action_idx}")

# Unified UDP server
def start_combined_udp_server(tracker, host='0.0.0.0', port=5005):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    print(f"[Combined UDP Server] Listening on {host}:{port}")

    try:
        while True:
            data, addr = sock.recvfrom(1024)
            msg = data.decode().strip().upper()

            if msg == "GET":
                r = tracker.compute_reward()
                reply = f"{r:.6f}".encode()
            elif msg == "RESET":
                tracker.reset()
                reply = b"OK"
            else:
                try:
                    action_idx = int(msg)
                    send_action(action_idx)
                    reply = b"DONE"
                except ValueError:
                    reply = b"ERR"
            sock.sendto(reply, addr)
    except KeyboardInterrupt:
        print("\n[Server] Shutdown.")
    finally:
        sock.close()
        tracker.close()


if __name__ == "__main__":
    tracker = ExternalRewardTracker()
    start_combined_udp_server(tracker)