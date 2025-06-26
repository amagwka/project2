import cv2
import numpy as np
import torch
import logging
from transformers import AutoProcessor, AutoModel

class LocalObs:
    def __init__(self, source=1, mode="dino", model_name=None, device="cpu", embedding_dim=384):
        self.device = device
        self.mode = mode
        self.embedding_dim = embedding_dim
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.USE_GPU_PREPROCESS = True
        self.video_capture = cv2.VideoCapture(source)
        if not self.video_capture.isOpened():
            logging.warning(f"Could not open video source {source}. Using dummy frames.")
        if self.mode == "dino" and model_name is not None:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device, dtype=torch.bfloat16).eval()
        else:
            self.processor = None
            self.model = None

    def read_frame(self) -> np.ndarray:
        ret, frame = self.video_capture.read()
        if not ret:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return frame

    def get_frame_224(self) -> np.ndarray:
        frame = self.read_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return cv2.resize(frame, (224, 224))
    

    def get_embedding(self) -> np.ndarray:
        # 1) Read + color + resize (all CPU & OpenCV)
        frame = self.read_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))

        # 2) To tensor (on CPU), then optionally GPU preprocessing
        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()  # (1,3,224,224) CPU float32

        if self.USE_GPU_PREPROCESS:
            tensor = tensor.to(self.device, non_blocking=True)
            tensor = tensor.div(255.0)
            mean = torch.as_tensor(self.mean, device=self.device).view(1,3,1,1)
            std  = torch.as_tensor(self.std,  device=self.device).view(1,3,1,1)
            tensor = tensor.sub(mean).div(std)
        else:
            # Original CPU-based normalization
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - self.mean) / self.std
            tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(self.device)

        if self.model is None:
            # Return a dummy embedding when the model is unavailable
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # 3) Inference (with AMP)
        with torch.no_grad(), torch.amp.autocast(self.device, dtype=torch.bfloat16):
            emb = (
                self.model(pixel_values=tensor)
                    .last_hidden_state
                    .mean(dim=1)
                    .squeeze(0)
                    .cpu()
                    .to(torch.float32)
            )

        # 4) Final scaling + numpy
        return emb.numpy() / 10.0

    
    def close(self):
        if self.video_capture.isOpened():
            self.video_capture.release()
        cv2.destroyAllWindows()

# Usage example:
if __name__ == "__main__":
    obs = LocalObs(source=1, mode="dino", model_name="facebook/dinov2-with-registers-small", device="cuda", embedding_dim=384)
    emb = obs.get_embedding()
    print("Embedding shape:", emb.shape)
    obs.close()
