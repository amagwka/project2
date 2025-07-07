import socket
from typing import Tuple

class UdpClient:
    """Simple UDP helper for sending actions and requesting rewards."""
    def __init__(self,
                 action_addr: Tuple[str, int],
                 reward_addr: Tuple[str, int],
                 combined_server: bool = False,
                 timeout: float = 0.2):
        self.action_addr = action_addr
        self.reward_addr = reward_addr if not combined_server else action_addr
        self.combined_server = combined_server

        self.action_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if combined_server:
            self.reward_socket = self.action_socket
        else:
            self.reward_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.reward_socket.settimeout(timeout)

    def send_action(self, action_idx: int) -> None:
        msg = str(action_idx).encode()
        self.action_socket.sendto(msg, self.action_addr)
        if self.combined_server:
            try:
                self.reward_socket.recvfrom(32)
            except Exception:
                pass

    def send_reset(self) -> None:
        try:
            self.reward_socket.sendto(b"RESET", self.reward_addr)
            self.reward_socket.recvfrom(32)
        except Exception:
            pass

    def get_reward(self) -> float:
        try:
            self.reward_socket.sendto(b"GET", self.reward_addr)
            data, _ = self.reward_socket.recvfrom(32)
            return float(data.decode().strip())
        except Exception:
            return 0.0

    def close(self) -> None:
        self.action_socket.close()
        if not self.combined_server:
            self.reward_socket.close()

