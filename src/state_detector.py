"""
State Detector per Pokemon Rosso/Blu.
Rileva automaticamente lo stato corrente del gioco analizzando lo schermo.

STATI RILEVATI:
1. Battaglia: Barra HP visibile, interfaccia combattimento
2. Menu: Linee rette, bordi netti, layout griglia
3. Dialogo: Box testo in basso schermo con bordi
4. Esplorazione: Tutto il resto (overworld, camminata)

TECNICHE:
- Computer Vision con OpenCV (Canny edge detection, Hough transform)
- Analisi varianza pixel in regioni specifiche (barra HP)
- Cache LRU per performance (evita ricalcoli su frame identici)
- Soglie adattive basate su hyperparameters

NOTA PERFORMANCE: Cache LRU aumentata a 200 (era 50) per gestire
meglio sequenze lunghe di frame simili durante esplorazione.
"""
import hashlib
from functools import lru_cache
from typing import Tuple
import numpy as np
import cv2
from .hyperparameters import HYPERPARAMETERS
from .screen_regions import SCREEN_REGIONS


class GameStateDetector:
    """
    Rilevatore stato gioco tramite analisi visiva dello schermo.

    Usa tecniche di Computer Vision per distinguere tra:
    - Battaglia (HP bar + interfaccia combattimento)
    - Menu (linee rette + layout griglia)
    - Dialogo (text box in basso)
    - Esplorazione (default)
    """
    @lru_cache(maxsize=200)  # AUMENTATO da 50 a 200 per migliore caching
    def _calc_feat(self, h: str, b: bytes) -> Tuple[float, float]:
        """
        Calcola features visive dallo schermo con caching.

        Args:
            h: Hash MD5 del frame (per cache key)
            b: Bytes del frame (grayscale 144x160)

        Returns:
            (hp_variance, edge_density):
            - hp_variance: Varianza pixel nella zona HP bar (alta durante battaglia)
            - edge_density: Densità edge nella zona menu (alta quando menu aperto)

        NOTA: LRU cache evita ricalcoli costosi su frame identici.
        """
        img = np.frombuffer(b, dtype=np.uint8).reshape((144, 160))

        # Varianza HP bar: alta quando barra HP visibile (battaglia)
        hp_var = np.var(img[SCREEN_REGIONS['HP_BAR']])

        # Densità edge zona menu: alta quando menu aperto (linee rette)
        menu_edges = cv2.Canny(img[SCREEN_REGIONS['MENU_AREA']].astype(np.uint8), 50, 150)
        edge_density = np.sum(menu_edges > 0) / menu_edges.size

        return hp_var, edge_density

    def detect_battle(self, scr: np.ndarray) -> bool:
        """
        Rileva se siamo in battaglia analizzando HP bar e interfaccia.

        LOGICA:
        1. Controlla varianza zona HP bar (alta se barra visibile)
        2. Controlla densità edge (interfaccia combattimento ha molti bordi)

        SOGLIE:
        - HP variance > HP_THRESHOLD (default 500)
        - Edge density > 0.1 (10% pixel sono edge)

        Returns:
            True se in battaglia, False altrimenti
        """
        # Validazione input
        if scr is None or scr.size == 0 or scr.shape != (144, 160):
            return False

        try:
            # Calcola hash per cache
            h = hashlib.md5(scr.tobytes()).hexdigest()
            hp_var, edge_dens = self._calc_feat(h, scr.tobytes())

            # Battaglia: HP bar visibile (alta varianza) + edge density moderata
            is_battle = hp_var > HYPERPARAMETERS['HP_THRESHOLD'] and edge_dens > 0.1

            return is_battle

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Errore rilevamento battaglia: {str(e)}")
            return False

    def detect_menu(self, scr: np.ndarray) -> bool:
        """
        Rileva se siamo in un menu analizzando linee rette e bordi.

        LOGICA:
        1. Estrae edge con Canny (bordi netti)
        2. Calcola densità edge totale
        3. Rileva linee rette con Hough transform

        SOGLIE:
        - Edge density > MENU_THRESHOLD (default 0.15 = 15%)
        - Numero linee rette > 5 (menu hanno layout a griglia)

        Returns:
            True se in menu, False altrimenti
        """
        # Validazione input
        if scr is None or scr.size == 0 or scr.shape != (144, 160):
            return False

        try:
            # Estrae edge con Canny
            edges = cv2.Canny(scr.astype(np.uint8), 50, 150)
            edge_dens = np.sum(edges > 0) / edges.size

            # Rileva linee rette con Hough transform (menu hanno molte linee)
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=50,
                minLineLength=30,
                maxLineGap=10
            )
            num_lines = len(lines) if lines is not None else 0

            # Menu: alta densità edge + molte linee rette
            is_menu = (edge_dens > HYPERPARAMETERS['MENU_THRESHOLD'] and num_lines > 5)

            return is_menu

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Errore rilevamento menu: {str(e)}")
            return False

    def detect_dialogue(self, scr: np.ndarray) -> bool:
        """
        Rileva se siamo in dialogo analizzando text box in basso schermo.

        LOGICA:
        1. Estrae regione dialog box (bottom screen)
        2. Calcola std deviation (testo ha alta varianza)
        3. Controlla edge su bordi superiore/inferiore (cornice box)

        SOGLIE:
        - Std deviation > DIALOGUE_THRESHOLD (default 30)
        - Edge su bordo superiore > 20 pixel
        - Edge su bordo inferiore > 20 pixel

        Returns:
            True se in dialogo, False altrimenti
        """
        # Validazione input
        if scr is None or scr.size == 0 or scr.shape != (144, 160):
            return False

        try:
            # Estrae zona dialog box (parte bassa schermo)
            dlg = scr[SCREEN_REGIONS['DIALOG_BOX']]

            # Calcola std deviation (testo ha alta varianza pixel)
            text_variance = np.std(dlg)

            # Estrae edge per rilevare cornice del box
            edges = cv2.Canny(dlg.astype(np.uint8), 50, 150)

            # Conta edge su bordo superiore e inferiore (cornice dialog box)
            top_border_edges = np.sum(edges[0, :])
            bottom_border_edges = np.sum(edges[-1, :])

            # Dialogo: alta varianza testo + cornice visibile su entrambi i bordi
            is_dialogue = (
                text_variance > HYPERPARAMETERS['DIALOGUE_THRESHOLD'] and
                top_border_edges > 20 and
                bottom_border_edges > 20
            )

            return is_dialogue

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Errore rilevamento dialogo: {str(e)}")
            return False