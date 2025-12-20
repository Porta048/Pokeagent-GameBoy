"""
Action Filter Contestuale per Pokemon Rosso/Blu.
Filtra azioni in base allo stato corrente del gioco per migliorare efficienza.

FILOSOFIA:
Invece di permettere tutte le 9 azioni sempre, il filtro RIDUCE la probabilità
di azioni inutili nel contesto corrente. Questo accelera l'apprendimento perché
l'agente non spreca tempo provando azioni che non hanno senso.

ESEMPI:
- Durante battaglia: riduce probabilità Start/Select (inutili in combattimento)
- Durante dialogo: riduce probabilità frecce direzionali (non servono per avanzare testo)
- Durante esplorazione: riduce probabilità NOOP (vogliamo movimento attivo)

TECNICA: Mascheramento soft (0.3-1.0) invece di hard (0.0/1.0)
- Permette comunque esplorazione casuale (importante per RL)
- Ma guida l'agente verso azioni sensate nel contesto
"""
from typing import List, TYPE_CHECKING
import torch
from .config import config

if TYPE_CHECKING:
    from torch import Tensor


class ContextAwareActionFilter:
    """
    Filtro intelligente per azioni basato su contesto di gioco.

    Maschera soft: Riduce probabilità azioni inutili ma non le blocca totalmente.
    Questo mantiene esplorazione (importante per RL) ma guida verso azioni sensate.

    Stati gestiti:
    - battle: Durante combattimento Pokemon
    - menu: Nei menu (Pokemon, Zaino, ecc.)
    - dialogue: Durante dialoghi con NPC
    - exploring: Overworld, camminata libera
    """

    @staticmethod
    def get_action_mask(game_state: str) -> List[float]:
        """
        Ritorna maschera di probabilità per azioni valide nel contesto.

        VALORI MASCHERA:
        - 1.0: Azione utile nel contesto (priorità normale)
        - 0.5: Azione meno utile (ridotta ma non bloccata)
        - 0.3: Azione raramente utile (fortemente ridotta)

        NOTE DESIGN:
        - Mascheramento SOFT (non hard 0.0) per mantenere esplorazione
        - Valori più aggressivi (0.3-0.5) rispetto a prima (0.7-0.9)
          per guidare meglio l'agente
        - Ogni stato ha strategia specifica basata su gameplay Pokemon

        Args:
            game_state: Stato corrente ('battle', 'menu', 'dialogue', 'exploring')

        Returns:
            Lista di 9 float (uno per azione) che modulano probabilità
        """
        # Inizializza tutte le azioni con peso normale
        mask = [1.0] * len(config.ACTIONS)

        if game_state == "battle":
            """
            BATTAGLIA: Priorità A (attacco/conferma), frecce (selezione mosse), B (fuga/indietro)
            Start/Select sono inutili durante combattimento
            """
            mask[config.ACTIONS.index(None)] = 0.3    # NOOP - quasi mai utile in battaglia
            mask[config.ACTIONS.index('start')] = 0.3  # Start - inutile in battaglia
            mask[config.ACTIONS.index('select')] = 0.3 # Select - inutile in battaglia
            # A, B, frecce restano 1.0 (utili)

        elif game_state == "menu":
            """
            MENU: Priorità A (conferma), B (indietro), frecce (navigazione)
            Start può chiudere menu, NOOP inutile
            """
            mask[config.ACTIONS.index(None)] = 0.3    # NOOP - inutile in menu
            mask[config.ACTIONS.index('select')] = 0.5 # Select - raramente usato
            # A, B, frecce, Start restano >= 0.5 (utili)

        elif game_state == "dialogue":
            """
            DIALOGO: Priorità A (avanza testo), B (velocizza/skippa)
            Frecce/Start/Select raramente utili
            """
            mask[config.ACTIONS.index(None)] = 0.5    # NOOP
            mask[config.ACTIONS.index('up')] = 0.3     # Frecce - inutili in dialogo
            mask[config.ACTIONS.index('down')] = 0.3
            mask[config.ACTIONS.index('left')] = 0.3
            mask[config.ACTIONS.index('right')] = 0.3
            mask[config.ACTIONS.index('start')] = 0.3  # Start - inutile in dialogo
            mask[config.ACTIONS.index('select')] = 0.3 # Select - inutile in dialogo
            # A e B restano 1.0 (utili per avanzare testo)

        elif game_state == "exploring":
            """
            ESPLORAZIONE: Priorità frecce (movimento), A (interazione), Start (menu)
            NOOP ridotto per incoraggiare movimento attivo
            """
            mask[config.ACTIONS.index(None)] = 0.5    # NOOP - vogliamo movimento attivo
            mask[config.ACTIONS.index('select')] = 0.5 # Select - raramente usato
            # Frecce, A, B, Start restano 1.0 (tutti utili in overworld)

        return mask

    @staticmethod
    def apply_mask_to_logits(logits: 'Tensor', mask: List[float]) -> 'Tensor':
        """
        Applica maschera ai logit della policy network.

        MATEMATICA:
        Logit mascherati = logit_originali + log(mask)

        Questo modifica le probabilità finali dopo softmax:
        - mask=1.0 → log(1.0)=0.0 → nessun cambiamento
        - mask=0.5 → log(0.5)=-0.69 → probabilità ridotta ~50%
        - mask=0.3 → log(0.3)=-1.20 → probabilità ridotta ~70%

        IMPORTANTE: Opera in log-space per stabilità numerica.
        Non usa moltiplicazione diretta sui logit perché distorcerebbe
        le proporzioni relative tra azioni.

        Args:
            logits: Logit dalla policy network (shape: [batch, n_actions])
            mask: Lista probabilità relative per ogni azione

        Returns:
            Logit mascherati (stessa shape dell'input)
        """
        try:
            # Converti mask a tensor sullo stesso device dei logit
            mask_tensor = torch.tensor(mask, device=logits.device, dtype=logits.dtype)

            # Applica maschera in log-space: logits + log(mask)
            # epsilon (1e-8) previene log(0) se mask contiene zeri
            masked_logits = logits + torch.log(mask_tensor + 1e-8)

            return masked_logits

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Errore applicazione maschera azioni: {str(e)}")
            # Fallback: ritorna logit originali senza maschera
            return logits