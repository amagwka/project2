import socket
import threading
from typing import Callable, Optional


class UdpServer:
    """Simple UDP server that dispatches messages to ``handle``."""

    def __init__(self, host: str = "0.0.0.0", port: int = 0, buffer_size: int = 65535):
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.port = self.sock.getsockname()[1]
        self._shutdown = threading.Event()
        print(f"[UdpServer] Listening on udp://{self.host}:{self.port}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def handle(self, data: bytes, addr) -> Optional[bytes]:
        """Handle one incoming message and return an optional reply."""
        raise NotImplementedError

    def serve(
        self,
        cleanup: Optional[Callable[[], None]] = None,
    ) -> None:
        """Run the server until :meth:`shutdown` is called."""
        try:
            while not self._shutdown.is_set():
                try:
                    data, addr = self.sock.recvfrom(self.buffer_size)
                except ConnectionResetError:
                    continue
                except OSError:
                    break
                reply = self.handle(data, addr)
                if isinstance(reply, str):
                    reply = reply.encode()
                if reply:
                    try:
                        self.sock.sendto(reply, addr)
                    except ConnectionResetError:
                        continue
        except KeyboardInterrupt:
            print("\n[UdpServer] Shutdown.")
        finally:
            self.sock.close()
            if cleanup:
                cleanup()

    def shutdown(self) -> None:
        """Stop the server loop."""
        self._shutdown.set()
        try:
            self.sock.sendto(b"", (self.host, self.port))
        except Exception:
            pass
