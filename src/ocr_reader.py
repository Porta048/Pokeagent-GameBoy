import hashlib
import logging
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
logger = logging.getLogger(__name__)
_OCR_AVAILABLE = False
_easyocr = None
try:
    import easyocr
    _easyocr = easyocr
    _OCR_AVAILABLE = True
except ImportError:
    logger.warning(
        "EasyOCR non installato. Installa con: pip install easyocr\n"
        "L'OCR sarà disabilitato, il sistema funzionerà comunque."
    )
class GameBoyOCR:
    REGIONS = {
        'dialogue': (104, 144, 8, 152),
        'battle_menu': (104, 144, 80, 160),
        'enemy_name': (0, 16, 8, 80),
        'player_name': (64, 80, 80, 152),
        'enemy_hp': (16, 32, 8, 80),
        'player_hp': (80, 96, 80, 152),
        'moves_list': (104, 144, 0, 80),
        'start_menu': (0, 144, 80, 160),
        'full_screen': (0, 144, 0, 160),
    }
    def __init__(self, gpu: bool = True, lang: List[str] = None):
        self._reader = None  
        self._gpu = gpu
        self._lang = lang or ['en']
        self._cache = {}  
        self._cache_maxsize = 100
    @property
    def is_available(self) -> bool:
        return _OCR_AVAILABLE
    def _get_reader(self):
        if not _OCR_AVAILABLE:
            return None
        if self._reader is None:
            logger.info("Caricamento modello EasyOCR (prima volta, ~2-5s)...")
            try:
                self._reader = _easyocr.Reader(
                    self._lang,
                    gpu=self._gpu,
                    verbose=False
                )
                logger.info("Modello EasyOCR caricato.")
            except Exception as e:
                logger.error(f"Errore caricamento EasyOCR: {e}")
                return None
        return self._reader
    def _preprocess_gameboy(self, img: np.ndarray, upscale: int = 4) -> np.ndarray:
        if img is None or img.size == 0:
            return img
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
        h, w = img.shape[:2]
        upscaled = cv2.resize(
            img,
            (w * upscale, h * upscale),
            interpolation=cv2.INTER_NEAREST
        )
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrasted = clahe.apply(upscaled)
        binary = cv2.adaptiveThreshold(
            contrasted,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )
        denoised = cv2.medianBlur(binary, 3)
        return denoised
    def _get_cache_key(self, img: np.ndarray, region: str) -> str:
        img_hash = hashlib.md5(img.tobytes()).hexdigest()[:16]
        return f"{region}_{img_hash}"
    def _cache_get(self, key: str) -> Optional[List]:
        return self._cache.get(key)
    def _cache_set(self, key: str, value: List):
        if len(self._cache) >= self._cache_maxsize:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = value
    def _extract_region(self, screen: np.ndarray, region_name: str) -> np.ndarray:
        if region_name not in self.REGIONS:
            return screen
        y1, y2, x1, x2 = self.REGIONS[region_name]
        h, w = screen.shape[:2]
        y1, y2 = max(0, y1), min(h, y2)
        x1, x2 = max(0, x1), min(w, x2)
        return screen[y1:y2, x1:x2]
    def read_text(
        self,
        screen: np.ndarray,
        region: str = 'full_screen',
        preprocess: bool = True
    ) -> List[Tuple[str, float]]:
        if not self.is_available:
            return []
        reader = self._get_reader()
        if reader is None:
            return []
        if screen is None or screen.size == 0:
            return []
        cache_key = self._get_cache_key(screen, region)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        try:
            roi = self._extract_region(screen, region)
            if preprocess:
                roi = self._preprocess_gameboy(roi)
            results = reader.readtext(roi, detail=1)
            texts = [
                (text.strip().upper(), conf)
                for (_, text, conf) in results
                if text.strip()
            ]
            texts.sort(key=lambda x: x[1], reverse=True)
            self._cache_set(cache_key, texts)
            return texts
        except Exception as e:
            logger.warning(f"Errore OCR regione '{region}': {e}")
            return []
    def read_dialogue(self, screen: np.ndarray) -> str:
        texts = self.read_text(screen, region='dialogue')
        if texts:
            return ' '.join(t[0] for t in texts if t[1] > 0.3)
        return ''
    def read_menu_options(self, screen: np.ndarray) -> List[str]:
        texts = self.read_text(screen, region='battle_menu')
        if len(texts) < 2:
            texts = self.read_text(screen, region='start_menu')
        return [t[0] for t in texts if t[1] > 0.4]
    def read_battle_info(self, screen: np.ndarray) -> Dict[str, str]:
        info = {
            'enemy_name': '',
            'player_name': '',
            'moves': []
        }
        enemy_texts = self.read_text(screen, region='enemy_name')
        if enemy_texts:
            info['enemy_name'] = enemy_texts[0][0]
        player_texts = self.read_text(screen, region='player_name')
        if player_texts:
            info['player_name'] = player_texts[0][0]
        move_texts = self.read_text(screen, region='moves_list')
        info['moves'] = [t[0] for t in move_texts if t[1] > 0.3]
        return info
    def read_all(self, screen: np.ndarray) -> Dict[str, any]:
        return {
            'dialogue': self.read_dialogue(screen),
            'menu': self.read_menu_options(screen),
            'battle': self.read_battle_info(screen),
            'raw_texts': self.read_text(screen, region='full_screen')
        }
    def clear_cache(self):
        self._cache.clear()
_global_ocr: Optional[GameBoyOCR] = None
def get_ocr(gpu: bool = True) -> GameBoyOCR:
    global _global_ocr
    if _global_ocr is None:
        _global_ocr = GameBoyOCR(gpu=gpu)
    return _global_ocr
