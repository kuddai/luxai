import numpy as np

from lux.game_constants import GAME_CONSTANTS
from lux.game_objects import Unit, CityTile
from lux.game import Game


NONE = 0

NORTH = 1
EAST = 2
SOUTH = 3
WEST = 4
BUILD_CITY = 5
BUILD_UNIT = 6

MOVEMENT_ACTIONS = {NORTH, EAST, SOUTH, WEST}
UNIT_ACTIONS = MOVEMENT_ACTIONS | {NONE, BUILD_CITY}
CITY_ACTIONS = {NONE, BUILD_UNIT}
ALL_ACTIONS = UNIT_ACTIONS | CITY_ACTIONS

DIRECTION_MAP = {
    GAME_CONSTANTS['DIRECTIONS']['NORTH']: NORTH,
    GAME_CONSTANTS['DIRECTIONS']['EAST']: EAST,
    GAME_CONSTANTS['DIRECTIONS']['SOUTH']: SOUTH,
    GAME_CONSTANTS['DIRECTIONS']['WEST']: WEST,
    GAME_CONSTANTS['DIRECTIONS']['CENTER']: NONE,
}
REVERSE_DIRECTION_MAP = {v: k for k, v in DIRECTION_MAP.items()}

assert len(REVERSE_DIRECTION_MAP) == len(DIRECTION_MAP)


def is_movement_possible(game: Game, unit: Unit, direction: str):
    assert direction in GAME_CONSTANTS['DIRECTIONS'].values()
    assert direction != GAME_CONSTANTS['DIRECTIONS']['CENTER']

    next_pos = unit.pos.translate(direction, 1)
    x, y = next_pos.x, next_pos.y

    if x < 0 or x >= game.map_width:
        return False

    if y < 0 or y >= game.map_height:
        return False

    cell = game.map.get_cell(x, y)
    if cell.citytile:
        if cell.citytile.team != unit.team:
            return False

    # We can't check there for collision with other units, because they can move on this turn as well
    return True


def get_unit_actions_mask(game: Game, unit: Unit):
    mask = np.zeros((len(ALL_ACTIONS),), dtype=np.float32)
    mask[NONE] = 1.0

    if not unit.can_act():
        return mask

    for i in range(len(ALL_ACTIONS)):
        if i not in UNIT_ACTIONS:
            continue

        if i in MOVEMENT_ACTIONS:
            direction = REVERSE_DIRECTION_MAP[i]
            if is_movement_possible(game, unit, direction):
                mask[i] = 1.0

        if i == BUILD_CITY:
            cell = game.map.get_cell_by_pos(unit.pos)
            has_enough_resources = unit.cargo.wood + unit.cargo.coal + unit.cargo.uranium >= 100
            is_cell_empty = cell.citytile is None and cell.resource is None
            if has_enough_resources and is_cell_empty:
                mask[i] = 1.0

    return mask


def get_city_actions_mask(game: Game, city: CityTile):
    mask = np.zeros((len(ALL_ACTIONS),), dtype=np.float32)
    mask[NONE] = 1.0

    if not city.can_act():
        return mask

    for i in range(len(ALL_ACTIONS)):
        if i not in CITY_ACTIONS:
            continue

        if i == BUILD_UNIT:
            owner = city.team
            total_city_tiles = sum(len(c.citytiles) for c in game.players[owner].cities.values())
            if len(game.players[owner].units) < total_city_tiles:
                mask[i] = 1.0

    return mask
