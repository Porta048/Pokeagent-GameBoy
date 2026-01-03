import hashlib
import logging
from functools import lru_cache
from typing import Dict, Optional, Tuple
import numpy as np
import cv2
from .hyperparameters import HYPERPARAMETERS
from .screen_regions import SCREEN_REGIONS
logger = logging.getLogger(__name__)
class GameStateDetector:
    @lru_cache(maxsize=200)  
    def _calc_feat(self, h: str, b: bytes) -> Tuple[float, float]:
        img = np.frombuffer(b, dtype=np.uint8).reshape((144, 160))
        hp_var = np.var(img[SCREEN_REGIONS['HP_BAR']])
        menu_edges = cv2.Canny(img[SCREEN_REGIONS['MENU_AREA']].astype(np.uint8), 50, 150)
        edge_density = np.sum(menu_edges > 0) / menu_edges.size
        return hp_var, edge_density
    def detect_battle(self, scr: np.ndarray) -> bool:
        if scr is None or scr.size == 0 or scr.shape != (144, 160):
            return False
        try:
            h = hashlib.md5(scr.tobytes()).hexdigest()
            hp_var, edge_dens = self._calc_feat(h, scr.tobytes())
            is_battle = hp_var > HYPERPARAMETERS['HP_THRESHOLD'] and edge_dens > 0.1
            return is_battle
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Errore rilevamento battaglia: {str(e)}")
            return False
    def detect_menu(self, scr: np.ndarray) -> bool:
        if scr is None or scr.size == 0 or scr.shape != (144, 160):
            return False
        try:
            edges = cv2.Canny(scr.astype(np.uint8), 50, 150)
            edge_dens = np.sum(edges > 0) / edges.size
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=50,
                minLineLength=30,
                maxLineGap=10
            )
            num_lines = len(lines) if lines is not None else 0
            is_menu = (edge_dens > HYPERPARAMETERS['MENU_THRESHOLD'] and num_lines > 5)
            return is_menu
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Errore rilevamento menu: {str(e)}")
            return False
    def detect_dialogue(self, scr: np.ndarray) -> bool:
        if scr is None or scr.size == 0 or scr.shape != (144, 160):
            return False
        try:
            dlg = scr[SCREEN_REGIONS['DIALOG_BOX']]
            text_variance = np.std(dlg)
            edges = cv2.Canny(dlg.astype(np.uint8), 50, 150)
            top_border_edges = np.sum(edges[0, :])
            bottom_border_edges = np.sum(edges[-1, :])
            is_dialogue = (
                text_variance > HYPERPARAMETERS['DIALOGUE_THRESHOLD'] and
                top_border_edges > 20 and
                bottom_border_edges > 20
            )
            return is_dialogue
        except Exception as e:
            logger.warning(f"Errore rilevamento dialogo: {str(e)}")
            return False
    def get_ocr_reader(self):
        if not hasattr(self, '_ocr_reader'):
            try:
                from .ocr_reader import get_ocr
                self._ocr_reader = get_ocr()
                if not self._ocr_reader.is_available:
                    logger.info("OCR non disponibile (EasyOCR non installato)")
                    self._ocr_reader = None
            except ImportError:
                logger.info("Modulo OCR non trovato, funzionalità OCR disabilitata")
                self._ocr_reader = None
        return self._ocr_reader
    def read_screen_text(self, scr: np.ndarray) -> Optional[Dict]:
        ocr = self.get_ocr_reader()
        if ocr is None:
            return None
        if scr is None or scr.size == 0 or scr.shape != (144, 160):
            return None
        try:
            return ocr.read_all(scr)
        except Exception as e:
            logger.warning(f"Errore OCR: {e}")
            return None
    def read_dialogue_text(self, scr: np.ndarray) -> str:
        ocr = self.get_ocr_reader()
        if ocr is None:
            return ''
        if scr is None or scr.size == 0 or scr.shape != (144, 160):
            return ''
        try:
            return ocr.read_dialogue(scr)
        except Exception as e:
            logger.warning(f"Errore lettura dialogo: {e}")
            return ''
    def read_menu_options(self, scr: np.ndarray) -> list:
        ocr = self.get_ocr_reader()
        if ocr is None:
            return []
        if scr is None or scr.size == 0 or scr.shape != (144, 160):
            return []
        try:
            return ocr.read_menu_options(scr)
        except Exception as e:
            logger.warning(f"Errore lettura menu: {e}")
            return []
    def read_battle_info(self, scr: np.ndarray) -> Optional[Dict]:
        ocr = self.get_ocr_reader()
        if ocr is None:
            return None
        if scr is None or scr.size == 0 or scr.shape != (144, 160):
            return None
        try:
            return ocr.read_battle_info(scr)
        except Exception as e:
            logger.warning(f"Errore lettura info battaglia: {e}")
            return None