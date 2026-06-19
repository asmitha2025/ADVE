import cv2
import threading
import queue
import numpy as np
from typing import Optional, Callable


class RTSPStream:
    """
    Connects to a live RTSP camera feed.
    Feeds frames into ADVEPipeline continuously.
    Outputs embeddings via callback or WebSocket.
    
    Usage:
        stream = RTSPStream("rtsp://192.168.1.100:554/stream")
        stream.start(callback=lambda emb, ts: store_embedding(emb, ts))
    """

    def __init__(self, url: str, buffer_size: int = 30):
        self.url          = url
        self.buffer       = queue.Queue(maxsize=buffer_size)
        self._stop        = threading.Event()
        self._capture_thr = None
        self._process_thr = None
        self.pipeline     = None
        self.callback: Optional[Callable] = None

    def _capture_loop(self):
        cap = cv2.VideoCapture(self.url)
        if not cap.isOpened():
            raise ConnectionError(f"Cannot connect to: {self.url}")

        print(f"Connected: {self.url}")
        frame_idx = 0

        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Stream interrupted, reconnecting...")
                cap.release()
                import time; time.sleep(2)
                cap = cv2.VideoCapture(self.url)
                continue

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            try:
                self.buffer.put_nowait((frame_idx, frame, timestamp))
            except queue.Full:
                pass  # drop frame if buffer full — live stream, don't lag

            frame_idx += 1

        cap.release()

    def _process_loop(self):
        from adve.core.pipeline import ADVEPipeline
        from adve.core.config   import Config

        pipeline = ADVEPipeline(Config())

        while not self._stop.is_set():
            try:
                frame_idx, frame, timestamp = self.buffer.get(timeout=1.0)
            except queue.Empty:
                continue

            result = pipeline.process_frame(frame, frame_idx)

            if self.callback and result:
                self.callback({
                    "frame_idx":     frame_idx,
                    "timestamp":     timestamp,
                    "embedding":     result["embedding"].tolist(),
                    "is_anchor":     result["is_anchor"],
                    "encoder_saved": not result["encoder_called"],
                })

    def start(self, callback: Optional[Callable] = None):
        self.callback     = callback
        self._capture_thr = threading.Thread(target=self._capture_loop, daemon=True)
        self._process_thr = threading.Thread(target=self._process_loop, daemon=True)
        self._capture_thr.start()
        self._process_thr.start()
        print(f"ADVE stream started: {self.url}")

    def stop(self):
        self._stop.set()
        if self._capture_thr:
            self._capture_thr.join()
        if self._process_thr:
            self._process_thr.join()
        print("Stream stopped")


# Multi-camera manager
class MultiCameraManager:
    """
    Manages N simultaneous RTSP camera streams.
    Each runs its own ADVE pipeline independently.
    Embeddings from all cameras feed into a shared FAISS index.
    """

    def __init__(self, index_writer=None):
        self.streams     = {}
        self.index_writer = index_writer

    def add_camera(self, camera_id: str, rtsp_url: str):
        stream = RTSPStream(rtsp_url)
        stream.start(callback=lambda r: self._on_embedding(camera_id, r))
        self.streams[camera_id] = stream
        print(f"Camera added: {camera_id} → {rtsp_url}")

    def _on_embedding(self, camera_id: str, result: dict):
        if self.index_writer:
            self.index_writer.add(
                video_path  = camera_id,
                camera_id   = camera_id,
                timestamp   = result["timestamp"],
                frame_idx   = result["frame_idx"],
                embedding   = np.array(result["embedding"]),
                is_anchor   = result["is_anchor"],
            )

    def remove_camera(self, camera_id: str):
        if camera_id in self.streams:
            self.streams[camera_id].stop()
            del self.streams[camera_id]

    def status(self) -> dict:
        return {
            cid: {"url": s.url, "buffer_size": s.buffer.qsize()}
            for cid, s in self.streams.items()
        }
