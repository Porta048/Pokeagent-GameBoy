from dataclasses import dataclass

from pyboy import PyBoy

# Indirizzi WRAM di Pokemon Rosso ITA (= pret/pokered USA + 5), verificati sull'emulatore
IN_BATTLE = 0xD05C
PARTY_COUNT = 0xD168
PARTY_MON1 = 0xD170
PARTY_MON_SIZE = 44
DEX_OWNED = 0xD2FC
DEX_BYTES = 19
BADGES = 0xD35B
MAP_ID = 0xD363
PLAYER_Y = 0xD366
PLAYER_X = 0xD367
EVENT_FLAGS = 0xD74C
EVENT_FLAGS_BYTES = 0x140

# Offset dentro la struttura di un Pokemon della squadra
MON_HP = 1
MON_LEVEL = 33
MON_MAX_HP = 34


@dataclass(frozen=True)
class GameState:
    map_id: int
    x: int
    y: int
    in_battle: int
    badges: int
    dex_owned: int
    event_flags: int
    party_levels: tuple[int, ...]
    party_hp: tuple[int, ...]
    party_max_hp: tuple[int, ...]


def _u16(pyboy: PyBoy, addr: int) -> int:
    return pyboy.memory[addr] << 8 | pyboy.memory[addr + 1]


def _bit_count(pyboy: PyBoy, start: int, length: int) -> int:
    return sum(b.bit_count() for b in pyboy.memory[start:start + length])


def read_state(pyboy: PyBoy) -> GameState:
    mem = pyboy.memory
    party = [PARTY_MON1 + i * PARTY_MON_SIZE for i in range(min(mem[PARTY_COUNT], 6))]
    return GameState(
        map_id=mem[MAP_ID],
        x=mem[PLAYER_X],
        y=mem[PLAYER_Y],
        in_battle=mem[IN_BATTLE],
        badges=mem[BADGES].bit_count(),
        dex_owned=_bit_count(pyboy, DEX_OWNED, DEX_BYTES),
        event_flags=_bit_count(pyboy, EVENT_FLAGS, EVENT_FLAGS_BYTES),
        party_levels=tuple(mem[a + MON_LEVEL] for a in party),
        party_hp=tuple(_u16(pyboy, a + MON_HP) for a in party),
        party_max_hp=tuple(_u16(pyboy, a + MON_MAX_HP) for a in party),
    )
