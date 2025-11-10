from typing import List, TYPE_CHECKING
import torch
from config import config

if TYPE_CHECKING:
    from torch import Tensor


class ContextAwareActionFilter:
    """
    Intelligent filter for actions based on game context.
    Reduces irrelevant actions based on current state.

    Pattern Strategy: different filtering strategies for each game state.
    """
    
    @staticmethod
    def get_action_mask(game_state: str) -> List[float]:
        """
        Returns probability mask for valid actions in context.
        1.0 = useful action, 0.7 = less useful action (much more permissive).

        This doesn't block actions, but reduces their selection probability LIGHTLY.
        """
        mask = [1.0] * len(config.ACTIONS)  # Default: all actions have normal weight

        if game_state == "battle":
            # In battle: slight preference for A, arrows, B
            mask[config.ACTIONS.index(None)] = 0.7  # NOOP
            mask[config.ACTIONS.index('start')] = 0.7
            mask[config.ACTIONS.index('select')] = 0.7

        elif game_state == "menu":
            # In menu: slight preference for A, B, arrows
            mask[config.ACTIONS.index(None)] = 0.8  # NOOP
            mask[config.ACTIONS.index('start')] = 0.9
            mask[config.ACTIONS.index('select')] = 0.7

        elif game_state == "dialogue":
            # In dialogue: slight preference for A and B
            mask[config.ACTIONS.index('up')] = 0.7
            mask[config.ACTIONS.index('down')] = 0.7
            mask[config.ACTIONS.index('left')] = 0.7
            mask[config.ACTIONS.index('right')] = 0.7
            mask[config.ACTIONS.index('start')] = 0.7
            mask[config.ACTIONS.index('select')] = 0.7

        elif game_state == "exploring":
            # In exploration: all actions almost equal
            mask[config.ACTIONS.index(None)] = 0.8  # NOOP

        return mask

    @staticmethod
    def apply_mask_to_logits(logits: 'Tensor', mask: List[float]) -> 'Tensor':
        """
        Applies mask to policy logits.
        Reduces logits for less useful actions in context.
        """
        try:
            mask_tensor = torch.tensor(mask, device=logits.device, dtype=logits.dtype)
            # Log-space mask: logits * mask (equivalent to prob^mask in linear space)
            return logits + torch.log(mask_tensor + 1e-8)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error applying action mask: {str(e)}")
            return logits