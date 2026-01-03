SCREEN_REGIONS = {
    'HP_BAR': (slice(100, 120), slice(90, 150)),
    'MENU_AREA': (slice(110, 140), slice(0, 80)),
    'DIALOG_BOX': (slice(100, 140), slice(10, 150)),
    'CENTER_REGION': (slice(60, 100), slice(70, 90)),
}
OCR_REGIONS = {
    'dialogue': (104, 144, 8, 152),
    'battle_menu': (104, 144, 80, 160),
    'enemy_name': (0, 16, 8, 80),
    'player_name': (64, 80, 80, 152),
    'enemy_hp': (16, 32, 8, 80),
    'player_hp': (80, 96, 80, 152),
    'moves_list': (104, 144, 0, 80),
    'start_menu': (0, 144, 80, 160),
    'full_screen': (0, 144, 0, 160),
}