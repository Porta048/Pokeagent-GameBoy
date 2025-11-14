import hashlib
from functools import lru_cache
from typing import Tuple
import numpy as np
import cv2
from .hyperparameters import HYPERPARAMETERS
from .screen_regions import SCREEN_REGIONS


class GameStateDetector:
    @lru_cache(maxsize=50)
    def _calc_feat(self, h: str, b: bytes) -> Tuple[float, float]:
        img = np.frombuffer(b, dtype=np.uint8).reshape((144, 160))
        hp_var = np.var(img[SCREEN_REGIONS['HP_BAR']])
        menu_edges = cv2.Canny(img[SCREEN_REGIONS['MENU_AREA']].astype(np.uint8), 50, 150)
        return hp_var, np.sum(menu_edges > 0) / menu_edges.size

    def detect_battle(self, scr: np.ndarray) -> bool:
        if scr is None or scr.size == 0 or scr.shape != (144, 160): return False
        try:
            h = hashlib.md5(scr.tobytes()).hexdigest()
            hp_var, edge_dens = self._calc_feat(h, scr.tobytes())
            return hp_var > HYPERPARAMETERS['HP_THRESHOLD'] and edge_dens > 0.1
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error in battle detection: {str(e)}")
            return False

    def detect_menu(self, scr: np.ndarray) -> bool:
        if scr is None or scr.size == 0 or scr.shape != (144, 160): return False
        try:
            edges = cv2.Canny(scr.astype(np.uint8), 50, 150)
            edge_dens = np.sum(edges > 0) / edges.size
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
            return edge_dens > HYPERPARAMETERS['MENU_THRESHOLD'] and (len(lines) if lines is not None else 0) > 5
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error in menu detection: {str(e)}")
            return False

    def detect_dialogue(self, scr: np.ndarray) -> bool:
        if scr is None or scr.size == 0 or scr.shape != (144, 160): return False
        try:
            dlg = scr[SCREEN_REGIONS['DIALOG_BOX']]  # Using SCREEN_REGIONS instead of hardcoded values
            edges = cv2.Canny(dlg.astype(np.uint8), 50, 150)
            return np.std(dlg) > HYPERPARAMETERS['DIALOGUE_THRESHOLD'] and np.sum(edges[0, :]) > 20 and np.sum(edges[-1, :]) > 20
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error in dialogue detection: {str(e)}")
            return False