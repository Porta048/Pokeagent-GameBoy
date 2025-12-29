"""
OCR Reader per Pokemon Rosso/Blu (Game Boy).

Estrae testo dallo schermo del Game Boy usando EasyOCR.
Progettato per leggere:
- Dialoghi NPC
- Opzioni menu (FIGHT, ITEM, POKeMON, RUN)
- Nomi Pokemon e mosse durante battaglie
- HP e statistiche visive

DESIGN:
- Lazy loading del modello OCR (caricato solo al primo uso)
- Cache LRU per evitare ricalcoli su frame identici
- Preprocessing ottimizzato per font 8x8 pixel del Game Boy
- Fallback graceful se OCR non disponibile

NOTA PERFORMANCE:
- Prima chiamata: ~2-5s (caricamento modello)
- Chiamate successive: ~50-150ms con cache miss
- Con cache hit: <1ms
"""
import hashlib
import logging
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Flag per disponibilità OCR
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
    """
    Lettore OCR specializzato per schermi Game Boy.

    Caratteristiche:
    - Preprocessing per font 8x8 pixel (upscale + contrast)
    - Regioni specifiche per dialoghi, menu, battaglie
    - Cache LRU per performance
    - Lazy loading del modello EasyOCR

    Uso:
        ocr = GameBoyOCR()
        text = ocr.read_dialogue(screen_array)
        menu = ocr.read_menu_options(screen_array)
        battle = ocr.read_battle_info(screen_array)
    """

    # Regioni schermo Game Boy (144x160 pixel)
    # Formato: (y_start, y_end, x_start, x_end)
    REGIONS = {
        # Dialogo: box testo in basso (ultimi ~40 pixel)
        'dialogue': (104, 144, 8, 152),

        # Menu battaglia in basso a destra (FIGHT, ITEM, POKeMON, RUN)
        'battle_menu': (104, 144, 80, 160),

        # Nome Pokemon avversario (in alto)
        'enemy_name': (0, 16, 8, 80),

        # Nome Pokemon giocatore (in basso, sopra menu)
        'player_name': (64, 80, 80, 152),

        # HP avversario (barra e numeri)
        'enemy_hp': (16, 32, 8, 80),

        # HP giocatore
        'player_hp': (80, 96, 80, 152),

        # Lista mosse durante selezione attacco
        'moves_list': (104, 144, 0, 80),

        # Menu START (ITEM, POKeMON, etc.)
        'start_menu': (0, 144, 80, 160),

        # Schermo intero (per OCR generico)
        'full_screen': (0, 144, 0, 160),
    }

    def __init__(self, gpu: bool = True, lang: List[str] = None):
        """
        Inizializza OCR reader.

        Args:
            gpu: Usa GPU se disponibile (default True)
            lang: Lingue per OCR (default ['en'] per Pokemon inglese)
        """
        self._reader = None  # Lazy loading
        self._gpu = gpu
        self._lang = lang or ['en']
        self._cache = {}  # Cache manuale per risultati OCR
        self._cache_maxsize = 100

    @property
    def is_available(self) -> bool:
        """Controlla se OCR è disponibile."""
        return _OCR_AVAILABLE

    def _get_reader(self):
        """Lazy loading del modello EasyOCR."""
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
        """
        Preprocessing ottimizzato per font Game Boy 8x8 pixel.

        Steps:
        1. Upscale (x4 default) per migliorare riconoscimento
        2. Contrasto aumentato (font GB ha basso contrasto)
        3. Binarizzazione adattiva per separare testo da sfondo
        4. Denoising leggero

        Args:
            img: Immagine grayscale (144x160 o regione)
            upscale: Fattore di upscaling (default 4)

        Returns:
            Immagine preprocessata per OCR
        """
        if img is None or img.size == 0:
            return img

        # Assicura uint8
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)

        # Upscale con interpolazione nearest (preserva pixel art)
        h, w = img.shape[:2]
        upscaled = cv2.resize(
            img,
            (w * upscale, h * upscale),
            interpolation=cv2.INTER_NEAREST
        )

        # Aumenta contrasto con CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrasted = clahe.apply(upscaled)

        # Binarizzazione adattiva (funziona meglio del threshold globale per GB)
        binary = cv2.adaptiveThreshold(
            contrasted,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )

        # Denoising leggero
        denoised = cv2.medianBlur(binary, 3)

        return denoised

    def _get_cache_key(self, img: np.ndarray, region: str) -> str:
        """Genera chiave cache per immagine + regione."""
        img_hash = hashlib.md5(img.tobytes()).hexdigest()[:16]
        return f"{region}_{img_hash}"

    def _cache_get(self, key: str) -> Optional[List]:
        """Recupera risultato da cache."""
        return self._cache.get(key)

    def _cache_set(self, key: str, value: List):
        """Salva risultato in cache con eviction LRU semplificata."""
        if len(self._cache) >= self._cache_maxsize:
            # Rimuovi primo elemento (FIFO semplificato)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = value

    def _extract_region(self, screen: np.ndarray, region_name: str) -> np.ndarray:
        """Estrae regione specifica dallo schermo."""
        if region_name not in self.REGIONS:
            return screen

        y1, y2, x1, x2 = self.REGIONS[region_name]

        # Valida bounds
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
        """
        Legge testo da una regione dello schermo.

        Args:
            screen: Schermo Game Boy (144x160 grayscale)
            region: Nome regione da REGIONS o 'full_screen'
            preprocess: Applica preprocessing (default True)

        Returns:
            Lista di (testo, confidenza) ordinata per confidenza
            Lista vuota se OCR non disponibile o errore
        """
        if not self.is_available:
            return []

        reader = self._get_reader()
        if reader is None:
            return []

        # Validazione input
        if screen is None or screen.size == 0:
            return []

        # Controlla cache
        cache_key = self._get_cache_key(screen, region)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            # Estrai regione
            roi = self._extract_region(screen, region)

            # Preprocessing
            if preprocess:
                roi = self._preprocess_gameboy(roi)

            # OCR
            results = reader.readtext(roi, detail=1)

            # Estrai (testo, confidenza)
            texts = [
                (text.strip().upper(), conf)
                for (_, text, conf) in results
                if text.strip()
            ]

            # Ordina per confidenza decrescente
            texts.sort(key=lambda x: x[1], reverse=True)

            # Cache risultato
            self._cache_set(cache_key, texts)

            return texts

        except Exception as e:
            logger.warning(f"Errore OCR regione '{region}': {e}")
            return []

    def read_dialogue(self, screen: np.ndarray) -> str:
        """
        Legge il testo del dialogo corrente.

        Args:
            screen: Schermo Game Boy (144x160 grayscale)

        Returns:
            Testo del dialogo (stringa vuota se non rilevato)
        """
        texts = self.read_text(screen, region='dialogue')
        if texts:
            # Unisci tutti i testi trovati
            return ' '.join(t[0] for t in texts if t[1] > 0.3)
        return ''

    def read_menu_options(self, screen: np.ndarray) -> List[str]:
        """
        Legge le opzioni del menu corrente (battaglia o START).

        Args:
            screen: Schermo Game Boy (144x160 grayscale)

        Returns:
            Lista di opzioni menu rilevate (es. ['FIGHT', 'ITEM', 'POKeMON', 'RUN'])
        """
        # Prova prima menu battaglia
        texts = self.read_text(screen, region='battle_menu')

        # Se pochi risultati, prova menu START
        if len(texts) < 2:
            texts = self.read_text(screen, region='start_menu')

        # Filtra per confidenza e restituisci solo testi
        return [t[0] for t in texts if t[1] > 0.4]

    def read_battle_info(self, screen: np.ndarray) -> Dict[str, str]:
        """
        Estrae informazioni dalla schermata di battaglia.

        Returns:
            Dict con chiavi:
            - enemy_name: Nome Pokemon avversario
            - player_name: Nome Pokemon giocatore
            - moves: Lista mosse disponibili
        """
        info = {
            'enemy_name': '',
            'player_name': '',
            'moves': []
        }

        # Nome avversario
        enemy_texts = self.read_text(screen, region='enemy_name')
        if enemy_texts:
            info['enemy_name'] = enemy_texts[0][0]

        # Nome giocatore
        player_texts = self.read_text(screen, region='player_name')
        if player_texts:
            info['player_name'] = player_texts[0][0]

        # Mosse
        move_texts = self.read_text(screen, region='moves_list')
        info['moves'] = [t[0] for t in move_texts if t[1] > 0.3]

        return info

    def read_all(self, screen: np.ndarray) -> Dict[str, any]:
        """
        Estrazione completa di tutto il testo visibile.

        Utile per debugging e logging.

        Returns:
            Dict con tutte le informazioni estratte:
            - dialogue: Testo dialogo
            - menu: Opzioni menu
            - battle: Info battaglia (se applicabile)
            - raw_texts: Lista grezza (testo, confidenza)
        """
        return {
            'dialogue': self.read_dialogue(screen),
            'menu': self.read_menu_options(screen),
            'battle': self.read_battle_info(screen),
            'raw_texts': self.read_text(screen, region='full_screen')
        }

    def clear_cache(self):
        """Svuota cache OCR."""
        self._cache.clear()


# Singleton per uso globale (evita ricaricare modello)
_global_ocr: Optional[GameBoyOCR] = None


def get_ocr(gpu: bool = True) -> GameBoyOCR:
    """
    Ottiene istanza singleton OCR.

    Args:
        gpu: Usa GPU se disponibile

    Returns:
        Istanza GameBoyOCR (singleton)
    """
    global _global_ocr
    if _global_ocr is None:
        _global_ocr = GameBoyOCR(gpu=gpu)
    return _global_ocr
