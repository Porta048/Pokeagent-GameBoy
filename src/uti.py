import threading
from collections import deque
import hashlib
import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
from .hyperparameters import HYPERPARAMETERS
class ImageCache:
    def __init__(self, max_size: int = HYPERPARAMETERS['CACHE_SIZE']):
        self.cache = {}
        self.access_queue = deque()
        self.max_size = max_size
        self.access_counters = {}
        self.cleanup_threshold = int(max_size * 0.8)
        self._lock = threading.Lock()

    def _get_image_hash(self, img: np.ndarray) -> str:
        return hashlib.md5(img.tobytes()).hexdigest()

    def get(self, img: np.ndarray) -> 'np.ndarray':
        h = self._get_image_hash(img)
        with self._lock:
            if h in self.cache:
                try:
                    self.access_queue.remove(h)
                    self.access_queue.append(h)
                except ValueError:
                    pass
                self.access_counters[h] = self.access_counters.get(h, 0) + 1
                return self.cache[h]
        return None

    def save(self, img: np.ndarray, img_proc: np.ndarray):
        h = self._get_image_hash(img)
        with self._lock:
            if h in self.cache:
                return
            if len(self.cache) >= self.cleanup_threshold:
                self._cleanup_unlocked()
            if len(self.cache) >= self.max_size:
                h_old = self.access_queue.popleft()
                self.cache.pop(h_old, None)
                self.access_counters.pop(h_old, None)
            self.cache[h] = img_proc
            self.access_queue.append(h)
            self.access_counters[h] = 1

    def _cleanup_unlocked(self):
        """Internal cleanup - must be called with lock held."""
        if len(self.access_counters) > 10:
            sorted_items = sorted(self.access_counters.items(), key=lambda x: x[1])
            n_remove = int(len(sorted_items) * 0.2)
            for h, _ in sorted_items[:n_remove]:
                if h in self.cache:
                    del self.cache[h]
                    try:
                        self.access_queue.remove(h)
                    except ValueError:
                        pass
                    del self.access_counters[h]
class AsyncSaver:
    def __init__(self):
        self.thread = None
        self.queue = deque()
        self.active = False
        self.lock = threading.Lock()
    def save_async(self, func, data):
        with self.lock:
            self.queue.append((func, data))
            if not self.active:
                self.thread = threading.Thread(target=self._process_queue)
                self.thread.daemon = True
                self.thread.start()
                self.active = True
    def _process_queue(self):
        while True:
            with self.lock:
                if not self.queue:
                    self.active = False
                    break
                func, data = self.queue.popleft()
            try:
                func(data)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error in async saver: {str(e)}")
class FrameStack:
    def __init__(self, stack_size=4):
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)
    def reset(self, initial_frame):
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(initial_frame)
    def add(self, frame):
        self.frames.append(frame)
    def get_stack(self):
        if TORCH_AVAILABLE and (torch.cuda.is_available() or torch.backends.mps.is_available()):  
            return torch.cat(list(self.frames), dim=0)
        return np.concatenate(list(self.frames), axis=0)
