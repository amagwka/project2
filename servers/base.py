import socket
from typing import Callable, Optional


class UdpServer:
    """Simple UDP server that dispatches messages to a handler."""

    def __init__(self, host: str = "0.0.0.0", port: int = 0, buffer_size: int = 65535):
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.port = self.sock.getsockname()[1]
        print(f"[UdpServer] Listening on udp://{self.host}:{self.port}")

    def serve_forever(
        self,
        handler: Callable[[str, tuple], bytes],
        cleanup: Optional[Callable[[], None]] = None,
    ) -> None:
        """Run the server until interrupted."""
        try:
            while True:
                try:
                    data, addr = self.sock.recvfrom(self.buffer_size)
                except ConnectionResetError:
                    # Ignore spurious resets on some platforms
                    continue
                msg = data.decode("utf-8", errors="ignore").strip()
                reply = handler(msg, addr)
                if isinstance(reply, str):
                    reply = reply.encode()
                if reply is not None:
                    try:
                        self.sock.sendto(reply, addr)
                    except ConnectionResetError:
                        continue
        except KeyboardInterrupt:
            print("\n[UdpServer] Shutdown.")
        finally:
            self.sock.close()
            if cleanup is not None:
                cleanup()
