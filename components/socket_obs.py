import socket
import cv2
import numpy as np
import torch
import logging
from transformers import AutoProcessor, AutoModel

class FrameServer:
    def __init__(self, source=1, mode="dino", model_name=None, device="cpu", port=5555):
        self.device = device
        self.mode = mode
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.video_capture = cv2.VideoCapture(source)
        if not self.video_capture.isOpened():
            logging.warning(f"Could not open video source {source}. Using dummy frames.")
        if self.mode == "dino" and model_name is not None:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device, dtype=torch.bfloat16).eval()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(("0.0.0.0", port))
        self.server_socket.listen(1)
        print(f"FrameServer running on port {port}")

    def read_frame(self) -> np.ndarray:
        ret, frame = self.video_capture.read()
        if not ret:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return frame

    def get_embedding(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - self.mean) / self.std
        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.amp.autocast(self.device, dtype=torch.bfloat16):
            emb = self.model(pixel_values=tensor).last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        return emb

    def serve(self):
        while True:
            client, addr = self.server_socket.accept()
            print(f"Client connected: {addr}")
            with client:
                while True:
                    try:
                        data = client.recv(16)
                        if not data:
                            break
                        cmd = data.decode().strip()
                        if cmd == "frame":
                            frame = self.read_frame()
                            resized = cv2.resize(frame, (224, 224))
                            # Send as raw bytes (RGB)
                            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                            client.sendall(rgb.tobytes())
                        elif cmd == "embed" and self.mode == "dino":
                            frame = self.read_frame()
                            emb = self.get_embedding(frame)
                            client.sendall(emb.astype(np.float32).tobytes())
                        else:
                            client.sendall(b"invalid")
                    except Exception as e:
                        print("Error:", e)
                        break

    def close(self):
        if self.video_capture.isOpened():
            self.video_capture.release()
        cv2.destroyAllWindows()
        self.server_socket.close()

# Usage:
server = FrameServer(source=1, mode="dino", model_name="facebook/dinov2-with-registers-small", device="cuda", port=5555)
server.serve()
