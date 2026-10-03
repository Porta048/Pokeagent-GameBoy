from types import SimpleNamespace

from agent import game_state as gs
from agent.env import LEVEL_SOFT_CAP, REWARD_LEVEL, heal_amount, score
from agent.game_state import GameState, read_state


def make_state(hp: tuple[int, ...], levels: tuple[int, ...], max_hp: tuple[int, ...] = (100, 100)) -> GameState:
    return GameState(0, 0, 0, 0, 0, 0, 0, levels, hp, max_hp)


def test_read_state_from_ram() -> None:
    mem = bytearray(0x10000)
    mem[gs.PARTY_COUNT] = 2
    mem[gs.PARTY_MON1 + gs.MON_HP:gs.PARTY_MON1 + gs.MON_HP + 2] = (300).to_bytes(2, "big")
    mem[gs.PARTY_MON1 + gs.MON_LEVEL] = 12
    mem[gs.PARTY_MON1 + gs.PARTY_MON_SIZE + gs.MON_LEVEL] = 5
    mem[gs.BADGES] = 0b101
    mem[gs.EVENT_FLAGS] = 0xFF
    mem[gs.MAP_ID], mem[gs.PLAYER_X], mem[gs.PLAYER_Y] = 3, 7, 9

    s = read_state(SimpleNamespace(memory=mem))

    assert (s.map_id, s.x, s.y) == (3, 7, 9)
    assert s.badges == 2
    assert s.event_flags == 8
    assert s.party_levels == (12, 5)
    assert s.party_hp == (300, 0)


def test_heal_rewarded_only_without_level_up_or_wipe() -> None:
    assert heal_amount(make_state((50, 50), (5, 5)), make_state((100, 100), (5, 5))) == 0.5
    assert heal_amount(make_state((50, 50), (5, 5)), make_state((100, 100), (6, 5))) == 0.0
    assert heal_amount(make_state((0, 0), (5, 5)), make_state((100, 100), (5, 5))) == 0.0
    assert heal_amount(make_state((100, 100), (5, 5)), make_state((50, 50), (5, 5))) == 0.0


def test_levels_above_soft_cap_worth_less() -> None:
    capped = score(make_state((1,), (LEVEL_SOFT_CAP,), (1,)), 0, 0, 0.0)
    above = score(make_state((1,), (LEVEL_SOFT_CAP + 4,), (1,)), 0, 0, 0.0)
    assert above - capped == REWARD_LEVEL
