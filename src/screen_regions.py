# Screen regions for detection (144x160 Game Boy screen)
# Formato: (slice(y_start, y_end), slice(x_start, x_end))

SCREEN_REGIONS = {
    # Regioni per State Detector (CV-based)
    'HP_BAR': (slice(100, 120), slice(90, 150)),
    'MENU_AREA': (slice(110, 140), slice(0, 80)),
    'DIALOG_BOX': (slice(100, 140), slice(10, 150)),
    'CENTER_REGION': (slice(60, 100), slice(70, 90)),

    # Regioni per OCR (coordinate y1,y2,x1,x2 per estrazione diretta)
    # NOTA: OCR usa formato diverso (tuple) rispetto a CV (slice)
}

# Regioni OCR separate per evitare conflitti con slice
# Formato: (y_start, y_end, x_start, x_end) - compatibile con numpy indexing diretto
OCR_REGIONS = {
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

    # Schermo intero
    'full_screen': (0, 144, 0, 160),
}