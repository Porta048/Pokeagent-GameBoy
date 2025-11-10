import torch

class Config:
    # Percorso del file ROM del gioco Pokemon
    ROM_PATH = "pokemon_red.gb"

    # Impostazioni dell'emulatore
    HEADLESS = False  # Se True, esegue l'emulatore senza interfaccia grafica
    EMULATION_SPEED = 0  # 0 = illimitata, 1 = velocità normale
    RENDER_ENABLED = True  # Se True, visualizza il gioco
    RENDER_EVERY_N_FRAMES = 2  # Renderizza ogni N frame per fluidità

    # Impostazioni dell'agente AI
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_DIR_PREFIX = "pokemon_ai_saves"
    MODEL_FILENAME = "model_ppo.pth"
    STATS_FILENAME = "stats_ppo.json"
    GAME_STATE_FILENAME = "game_state.state"

    # Stack di frame per l'input della rete neurale
    FRAME_STACK_SIZE = 4

    # Frequenza di salvataggio del checkpoint
    SAVE_FREQUENCY = 10000

    # Intervallo per il logging delle performance
    PERFORMANCE_LOG_INTERVAL = 1000

    # Azioni disponibili per l'agente
    ACTIONS = [
        None,      # No-op
        'up',      # Freccia su
        'down',    # Freccia giù
        'left',    # Freccia sinistra
        'right',   # Freccia destra
        'a',       # Pulsante A
        'b',       # Pulsante B
        'start',   # Pulsante Start
        'select'   # Pulsante Select
    ]

    # Mappatura per il frameskip adattivo
    FRAMESKIP_MAP = {
        "dialogue": 6,
        "battle": 12,
        "menu": 8,
        "exploring": 10,
        "base": 8
    }

    # Soglie per il rilevamento dello stato del gioco
    HP_THRESHOLD = 500
    MENU_THRESHOLD = 0.15
    DIALOGUE_THRESHOLD = 30

    # Impostazioni anti-loop (attualmente disabilitate)
    ANTI_LOOP_ENABLED = False
    
# Crea un'istanza della configurazione da importare in altri moduli
config = Config()
