import asyncio
from typing import Optional, Callable
from nats.aio.client import Client as NATS


class NatsServer:
    """Simple NATS request/reply server."""

    def __init__(self, subject: str, url: str = "nats://127.0.0.1:4222"):
        self.subject = subject
        self.url = url
        self.loop = asyncio.new_event_loop()
        self.nc = NATS()
        self._shutdown = asyncio.Event()
        print(f"[NatsServer] Listening on {subject} at {url}")

    async def handle(self, data: bytes) -> Optional[bytes]:
        """Handle one request and return an optional reply."""
        raise NotImplementedError

    async def _cb(self, msg):
        reply = await self.handle(msg.data)
        if isinstance(reply, str):
            reply = reply.encode()
        if reply is not None and msg.reply:
            await self.nc.publish(msg.reply, reply)

    async def _serve(self, cleanup: Optional[Callable[[], None]] = None):
        await self.nc.connect(self.url, loop=self.loop)
        await self.nc.subscribe(self.subject, cb=self._cb)
        await self._shutdown.wait()
        await self.nc.drain()
        if cleanup:
            cleanup()

    def serve(self, cleanup: Optional[Callable[[], None]] = None) -> None:
        try:
            self.loop.run_until_complete(self._serve(cleanup))
        except KeyboardInterrupt:
            pass
        finally:
            if not self.loop.is_closed():
                self.loop.close()

    def shutdown(self) -> None:
        self.loop.call_soon_threadsafe(self._shutdown.set)
