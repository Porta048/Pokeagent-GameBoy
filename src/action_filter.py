from typing import List, TYPE_CHECKING
import torch
from .config import config
if TYPE_CHECKING:
    from torch import Tensor
class ContextAwareActionFilter:
    @staticmethod
    def get_action_mask(game_state: str) -> List[float]:
        mask = [1.0] * len(config.ACTIONS)
        if game_state == "battle":
            """
            BATTAGLIA: Priorità A (attacco/conferma), frecce (selezione mosse), B (fuga/indietro)
            Start/Select sono inutili durante combattimento
            """
            mask[config.ACTIONS.index(None)] = 0.3    
            mask[config.ACTIONS.index('start')] = 0.3  
            mask[config.ACTIONS.index('select')] = 0.3 
        elif game_state == "menu":
            """
            MENU: Priorità A (conferma), B (indietro), frecce (navigazione)
            Start può chiudere menu, NOOP inutile
            """
            mask[config.ACTIONS.index(None)] = 0.3    
            mask[config.ACTIONS.index('select')] = 0.5 
        elif game_state == "dialogue":
            """
            DIALOGO: Priorità A (avanza testo), B (velocizza/skippa)
            Frecce/Start/Select raramente utili
            """
            mask[config.ACTIONS.index(None)] = 0.5    
            mask[config.ACTIONS.index('up')] = 0.3     
            mask[config.ACTIONS.index('down')] = 0.3
            mask[config.ACTIONS.index('left')] = 0.3
            mask[config.ACTIONS.index('right')] = 0.3
            mask[config.ACTIONS.index('start')] = 0.3  
            mask[config.ACTIONS.index('select')] = 0.3 
        elif game_state == "exploring":
            """
            ESPLORAZIONE: Priorità frecce (movimento), A (interazione)
            START (menu) ridotto per evitare aperture ripetitive
            NOOP ridotto per incoraggiare movimento attivo
            """
            mask[config.ACTIONS.index(None)] = 0.5
            mask[config.ACTIONS.index('start')] = 0.4  # Reduced from 1.0 to discourage menu spam
            mask[config.ACTIONS.index('select')] = 0.5 
        return mask
    @staticmethod
    def apply_mask_to_logits(logits: 'Tensor', mask: List[float]) -> 'Tensor':
        try:
            mask_tensor = torch.tensor(mask, device=logits.device, dtype=logits.dtype)
            masked_logits = logits + torch.log(mask_tensor + 1e-8)
            return masked_logits
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Errore applicazione maschera azioni: {str(e)}")
            return logits