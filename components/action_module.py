import socket
import time
from pynput.keyboard import Controller, Key
from components.external_reward import ExternalRewardTracker

# Keyboard setup
ACTION_KEYS = [Key.up, Key.down, Key.left, Key.right, 'z', 'x', Key.space]
ARROW_IDX = {0, 1, 2, 3}
keyboard = Controller()


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
