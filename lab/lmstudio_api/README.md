# LM Studio API Experiment

This experiment demonstrates controlling the Undertale action server using the
LM Studio Python SDK. The script captures frames from OBS Studio
(video source `1`) via OpenCV and queries a locally running LM Studio instance
for the next action index.

## Files
 - `lmstudio_control.py` – captures a frame and queries LM Studio for the next
   action. The helper preserves conversation history so repeated calls can
   provide additional context to the model before sending the predicted action
   to the UDP server.

LM Studio must be running locally for the script to connect.
