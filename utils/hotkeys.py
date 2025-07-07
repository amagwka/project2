from typing import Callable
from time import sleep
from pynput import keyboard


def start_listener(toggle: Callable[[], None]) -> keyboard.Listener:
    """Start an F9 hot-key listener that invokes ``toggle`` when pressed."""

    def on_press(key: keyboard.Key) -> None:
        if key == keyboard.Key.f9:
            toggle()
            sleep(0.1)

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    return listener
