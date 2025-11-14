from typing import List, TYPE_CHECKING
import torch
from config import config

if TYPE_CHECKING:
    from torch import Tensor


class ContextAwareActionFilter:
    """
    Filtro intelligente per azioni basato su contesto di gioco.
    Riduce azioni irrilevanti basandosi sullo stato corrente.

    Strategia pattern: strategie di filtraggio differenti per ogni stato di gioco.
    """

    @staticmethod
    def get_action_mask(game_state: str) -> List[float]:
        """
        Ritorna maschera di probabilità per azioni valide nel contesto.
        1.0 = azione utile, 0.7 = azione meno utile (molto più permissivo).

        Questo non blocca azioni, ma riduce LEGGERMENTE la loro probabilità di selezione.
        """
        mask = [1.0] * len(config.ACTIONS)  # Default: tutte le azioni hanno peso normale

        if game_state == "battle":
            # In battaglia: leggera preferenza per A, frecce, B
            mask[config.ACTIONS.index(None)] = 0.7  # NOOP
            mask[config.ACTIONS.index('start')] = 0.7
            mask[config.ACTIONS.index('select')] = 0.7

        elif game_state == "menu":
            # In menu: leggera preferenza per A, B, frecce
            mask[config.ACTIONS.index(None)] = 0.8  # NOOP
            mask[config.ACTIONS.index('start')] = 0.9
            mask[config.ACTIONS.index('select')] = 0.7

        elif game_state == "dialogue":
            # In dialogo: leggera preferenza per A e B
            mask[config.ACTIONS.index('up')] = 0.7
            mask[config.ACTIONS.index('down')] = 0.7
            mask[config.ACTIONS.index('left')] = 0.7
            mask[config.ACTIONS.index('right')] = 0.7
            mask[config.ACTIONS.index('start')] = 0.7
            mask[config.ACTIONS.index('select')] = 0.7

        elif game_state == "exploring":
            # In esplorazione: tutte le azioni quasi uguali
            mask[config.ACTIONS.index(None)] = 0.8  # NOOP

        return mask

    @staticmethod
    def apply_mask_to_logits(logits: 'Tensor', mask: List[float]) -> 'Tensor':
        """
        Applica maschera ai logit della policy.
        Riduce logit per azioni meno utili nel contesto.
        """
        try:
            mask_tensor = torch.tensor(mask, device=logits.device, dtype=logits.dtype)
            # Maschera log-space: logits * mask (equivalente a prob^mask in spazio lineare)
            return logits + torch.log(mask_tensor + 1e-8)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error applying action mask: {str(e)}")
            return logits