import threading
import time
import logging
import cv2
import numpy as np
import torch
from torch import Tensor
from transformers import AutoProcessor, AutoModel
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FrameCollector:
    def __init__(self, source: int = 1):
        self.video_capture = cv2.VideoCapture(source)
        if not self.video_capture.isOpened():
            logging.warning(f"Could not open video source {source}. Using dummy frames.")

    def read(self) -> np.ndarray:
        ret, frame = self.video_capture.read()
        if not ret:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return frame

    def close(self):
        if self.video_capture.isOpened():
            self.video_capture.release()
        cv2.destroyAllWindows()


class FrameProcessor:
    def __init__(self, mode: str, model_name: str, device: str):
        self.mode = mode
        self.device = device
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        if mode == "dino":
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device, dtype=torch.bfloat16).eval()

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        if self.mode == "frame":
            return cv2.resize(frame, (224, 224))
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - self.mean) / self.std
            tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad(), torch.amp.autocast(self.device, dtype=torch.bfloat16):
                emb = self.model(pixel_values=tensor).last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
            return emb


class FrameFilter:
    @staticmethod
    def get_filtered_frames(frames: np.ndarray, steps: int = 70) -> torch.Tensor:
        if len(frames) == 0:
            return torch.tensor([])
        lin_steps = np.round(np.linspace(1, 100, steps)).astype(int)
        indices = np.cumsum(np.insert(lin_steps, 0, 0))[:-1]
        indices = indices[indices < len(frames)]
        filtered_frames = frames[indices]
        # Convert from HWC to CHW format and create torch tensor
        if len(filtered_frames.shape) == 4:  # Batch of frames
            filtered_frames = np.transpose(filtered_frames, (0, 3, 1, 2))
        return torch.from_numpy(filtered_frames)


class ObservationGetter:
    def __init__(self, source=1, model_name="facebook/dinov2-small", interval=0.1, mode="dino"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.collector = FrameCollector(source)
        self.processor = FrameProcessor(mode, model_name, self.device)
        self.frames = []
        self.timestamps = []
        self.interval = interval
        self.mode = mode
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._update_thread = None
        self._show_thread = None

    def _update_loop(self):
        while not self._stop_event.is_set():
            start = time.time()
            frame = self.collector.read()
            processed = self.processor.preprocess(frame)
            with self._lock:
                self.frames.append(processed)
                self.timestamps.append(time.time())
            time.sleep(max(0.0, self.interval - (time.time() - start)))

    def _show_loop(self):
        while not self._stop_event.is_set():
            frame = self.collector.read()
            cv2.imshow("Live Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self._stop_event.set()
                break
        cv2.destroyAllWindows()

    def start(self, show_preview=False):
        if self._update_thread is None or not self._update_thread.is_alive():
            self._stop_event.clear()
            self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self._update_thread.start()
            logging.info("Started frame update thread.")
        if show_preview and (self._show_thread is None or not self._show_thread.is_alive()):
            self._show_thread = threading.Thread(target=self._show_loop, daemon=True)
            self._show_thread.start()
            logging.info("Started live frame preview thread.")

    def stop(self):
        self._stop_event.set()
        if self._update_thread:
            self._update_thread.join()
        if self._show_thread:
            self._show_thread.join()
        self.collector.close()
        logging.info("Stopped threads and released resources.")

    def get_all(self) -> np.ndarray:
        with self._lock:
            return np.array(self.frames)

    def get_filtered(self, steps=70) -> torch.Tensor:
        return FrameFilter.get_filtered_frames(self.get_all(), steps)

    def analyze_intervals(self) -> List[float]:
        with self._lock:
            intervals = np.diff(self.timestamps)
            logging.info(f"Interval stats — Min: {np.min(intervals):.3f}s, Max: {np.max(intervals):.3f}s, Mean: {np.mean(intervals):.3f}s")
            return intervals.tolist()


if __name__ == "__main__":
    obs = ObservationGetter(source=1, mode="frame")
    obs.start(show_preview=False)
    time.sleep(15)
    obs.stop()
    filtered = obs.get_filtered()
    logging.info(f"Filtered frames: {filtered.shape}")
