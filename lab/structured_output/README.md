# LM Studio Structured Output

This folder contains a small experiment using LM Studio's structured output
feature. The configuration file defines a JSON schema for an object with a
single field `action` which maps to the action index expected by the Undertale
UDP server.

The script `structured_control.py` captures one frame from OBS Studio (video
source `1`), sends the structured request to LM Studio and forwards the
predicted action to the action server.
