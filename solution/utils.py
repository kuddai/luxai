from collections import namedtuple
import os
import sys

import numpy as np
import tensorflow as tf

import solution.constants as constants
from solution.constants import DEBUG
from lux.constants import Constants
from lux.game import Game
from lux.game_constants import GAME_CONSTANTS
from lux.game_map import GameMap
from lux.game_objects import Unit


RESOURCE_TYPES = [
    Constants.RESOURCE_TYPES.WOOD,
    Constants.RESOURCE_TYPES.COAL,
    Constants.RESOURCE_TYPES.URANIUM,
]
UNIT_TYPES = [
    Constants.UNIT_TYPES.WORKER,
    Constants.UNIT_TYPES.CART,
]
MAP_DEPTH = len(RESOURCE_TYPES) + 2

_print = print

StateRepresentation = namedtuple(
    'StateRepresentation',
    ['map', 'game_vector', 'units_spatial', 'units_vector', 'units_mask']
)

STATE_SPEC = StateRepresentation(
    tf.TensorSpec((32, 32, 5), tf.float32, 'map'),
    tf.TensorSpec((57,), tf.float32, 'game_vector'),
    # tf.TensorSpec((64, 5, 5, 5), tf.float32, 'units_spatial'),
    # tf.TensorSpec((64, 6), tf.float32, 'units_vector'),
    tf.TensorSpec((15, 15, 5), tf.float32, 'units_spatial'),
    tf.TensorSpec((6,), tf.float32, 'units_vector'),
    tf.TensorSpec((64,), tf.bool, 'units_mask'),
)


def print(*args, **kwargs):
    return _print(*args, **kwargs, file=sys.stderr)


def is_movement_possible(unit: Unit, direction):
    if direction == Constants.DIRECTIONS.CENTER:
        return True

    new_pos = unit.pos.translate(direction, 1)
    if new_pos.x < 0 or new_pos.x >= constants.MAP_SIZE:
        return False
    if new_pos.y < 0 or new_pos.y >= constants.MAP_SIZE:
        return False

    # A unit can't go in a cell if an another unit is there, but this unit could move as well
    return True


def resource_type_to_one_hot(resource_type) -> tf.Tensor:
    if resource_type is None:
        return tf.zeros(len(RESOURCE_TYPES))
    return tf.one_hot(RESOURCE_TYPES.index(resource_type), len(RESOURCE_TYPES))


def value_to_one_hot(value, min_value, max_value, steps) -> tf.Tensor:
    assert steps >= 3

    result = np.zeros((steps,), dtype=np.float32)

    if value >= max_value:
        result[-1] = 1.0
        return tf.constant(result)

    if value <= min_value:
        result[0] = 1.0
        return tf.constant(result)

    max_value -= min_value
    value -= min_value
    idx = max(min(int(value / max_value * steps), steps - 2), 1)
    result[idx] = 1.0
    return tf.constant(result)


def unit_type_to_one_hot(unit_type) -> tf.Tensor:
    return tf.one_hot(UNIT_TYPES.index(unit_type), len(UNIT_TYPES))


def extract_map(game_map: GameMap) -> tf.Tensor:
    # map_mask = np.zeros((constants.MAX_MAP_SIZE, constants.MAX_MAP_SIZE), dtype=np.float32)
    # map_mask[0:game_map.height, 0:game_map.width] = 1.0

    # TODO: add amount of resources as one-hot
    map_data = np.zeros((constants.MAX_MAP_SIZE, constants.MAX_MAP_SIZE, MAP_DEPTH))

    for row in game_map.map:
        for cell in row:
            if cell is None:
                continue

            x, y = cell.pos.x, cell.pos.y
            data = np.zeros((MAP_DEPTH,))

            if cell.resource is not None:
                data[0:3] = resource_type_to_one_hot(cell.resource.type)

            if cell.citytile is not None:
                data[3:5] = [
                    cell.citytile.team == constants.MY_PLAYER_ID,
                    cell.citytile.team == constants.ENEMY_PLAYER_ID,
                ]

            map_data[y][x] = data

    return tf.constant(map_data, dtype=tf.float32)


def get_full_day_length():
    day_length = GAME_CONSTANTS['PARAMETERS']['DAY_LENGTH']
    night_length = GAME_CONSTANTS['PARAMETERS']['NIGHT_LENGTH']
    return day_length + night_length


def get_is_day(game_turn: int) -> tf.Tensor:
    day_length = GAME_CONSTANTS['PARAMETERS']['DAY_LENGTH']
    full_day_length = get_full_day_length()

    return (game_turn % full_day_length) < day_length


def extract_unit(unit: Unit, extracted_map: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    inventory = tf.constant([unit.cargo.wood, unit.cargo.coal, unit.cargo.uranium], dtype=tf.float32)
    unit_type = unit_type_to_one_hot(unit.type)

    x = unit.pos.x
    y = unit.pos.y
    r = constants.UNIT_SPATIAL_DATA_RADIUS
    s = constants.UNIT_SPATIAL_DATA_SIZE

    mx = max(0, x - r)
    my = max(0, y - r)
    ux = max(0, r - x)
    uy = max(0, r - y)
    sx = min(s - ux, constants.MAP_SIZE - mx)
    sy = min(s - uy, constants.MAP_SIZE - my)

    # mask = np.zeros((s, s), dtype=np.float32)
    # mask[uy:uy + sy, ux:ux + sx] = 1

    unit_map_shape = (s, s, *extracted_map.shape[2:])
    unit_map = np.zeros(unit_map_shape, dtype=np.float32)
    unit_map[uy:uy + sy, ux:ux + sx] = extracted_map[my:my + sy, mx:mx + sx]

    assert unit_map.ndim == extracted_map.shape.rank

    return tf.constant(unit_map), tf.concat((inventory, unit_type, [float(unit.can_act())]), 0)


def extract_state(game: Game) -> StateRepresentation:
    my_player = game.players[constants.MY_PLAYER_ID]

    map_data = extract_map(game.map)

    is_day = get_is_day(game.turn)
    fuel = sum(city.fuel for city in my_player.cities.values())
    fuel_one_hot = value_to_one_hot(fuel, 0, 800, 15)
    if my_player.city_tile_count == 0:
        fuel_one_hot = tf.zeros((15,), dtype=tf.float32)

    day_vector = np.zeros((get_full_day_length(),), dtype=np.float32)
    day_vector[game.turn % get_full_day_length()] = 1.0

    game_vector = tf.constant([
        is_day,
        game.turn / GAME_CONSTANTS['PARAMETERS']['MAX_DAYS'],
    ], dtype=tf.float32)
    game_vector = tf.concat((game_vector, fuel_one_hot, day_vector), -1)

    units_mask = np.zeros((constants.MAX_UNITS,), dtype=np.float32)
    units_mask[0:len(my_player.units)] = 1

    units_spatial = np.zeros((
        # constants.MAX_UNITS,
        constants.UNIT_SPATIAL_DATA_SIZE,
        constants.UNIT_SPATIAL_DATA_SIZE,
        MAP_DEPTH,
    ))
    units_vector = np.zeros((
        # constants.MAX_UNITS,
        6,
    ))
    units_mask = np.zeros((constants.MAX_UNITS,), dtype=np.bool)

    for i, unit in enumerate(my_player.units):
        if i >= constants.MAX_UNITS:
            raise ValueError('GG LIVAEM')
        if i > 0:
            raise ValueError('GG LIVAEM 2')

        spatial, vector = extract_unit(unit, map_data)
        #units_spatial[i] = spatial
        #units_vector[i] = vector
        units_mask[i] = True

        units_spatial = spatial
        units_vector = vector

    return StateRepresentation(
        map=map_data,
        game_vector=game_vector,
        units_spatial=tf.constant(units_spatial, dtype=tf.float32),
        units_vector=tf.constant(units_vector, dtype=tf.float32),
        units_mask=tf.constant(units_mask, dtype=tf.bool)
    )


if __name__ == '__main__':
    def test():
        tf.debugging.assert_equal(resource_type_to_one_hot(Constants.RESOURCE_TYPES.COAL), tf.constant([0.0, 1.0, 0.0]))
        tf.debugging.assert_equal(resource_type_to_one_hot(None), tf.constant([0.0, 0.0, 0.0]))
        tf.debugging.assert_equal(unit_type_to_one_hot(Constants.UNIT_TYPES.WORKER), tf.constant([1.0, 0.0]))

        constants.UNIT_SPATIAL_DATA_RADIUS = 2
        constants.UNIT_SPATIAL_DATA_SIZE = 5
        constants.MAP_SIZE = 5
        constants.MAX_MAP_SIZE = 10
        u = Unit(0, Constants.UNIT_TYPES.WORKER, 0, 0, 0, 0, 0, 0, 0)
        m = tf.constant([
            [0.5932822, 0.13756526, 0.5275594, 0.568472, 0.37355626],
            [0.07532156, 0.5515177, 0.88979673, 0.50862825, 0.30911565],
            [0.37483418, 0.3375548, 0.61602676, 0.4462092, 0.7143347],
            [0.7349111, 0.49183202, 0.4517169, 0.0125227, 0.7782315],
            [0.49945366, 0.47839952, 0.39023733, 0.5751648, 0.21957862],
        ])

        # Test 1
        u.pos.x = 0
        u.pos.y = 0
        expected = tf.constant([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0.5932822, 0.13756526, 0.5275594],
            [0, 0, 0.07532156, 0.5515177, 0.88979673],
            [0, 0, 0.37483418, 0.3375548, 0.61602676],
        ])
        expected_mask = tf.constant([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
        ], dtype=tf.float32)
        data, *_ = extract_unit(u, m)
        tf.debugging.assert_equal(expected, data)
        # tf.debugging.assert_equal(expected_mask, mask)

        # Test 2
        u.pos.x = 4
        u.pos.y = 3
        expected = tf.constant([
            [0.88979673, 0.50862825, 0.30911565, 0, 0],
            [0.61602676, 0.4462092, 0.7143347, 0, 0],
            [0.4517169, 0.0125227, 0.7782315, 0, 0],
            [0.39023733, 0.5751648, 0.21957862, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        expected_mask = tf.constant([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=tf.float32)
        data, *_ = extract_unit(u, m)
        tf.debugging.assert_equal(expected, data)
        # tf.debugging.assert_equal(expected_mask, mask)

        # Test 3
        u.pos.x = 2
        u.pos.y = 2
        expected = tf.constant([
            [0.5932822, 0.13756526, 0.5275594, 0.568472, 0.37355626],
            [0.07532156, 0.5515177, 0.88979673, 0.50862825, 0.30911565],
            [0.37483418, 0.3375548, 0.61602676, 0.4462092, 0.7143347],
            [0.7349111, 0.49183202, 0.4517169, 0.0125227, 0.7782315],
            [0.49945366, 0.47839952, 0.39023733, 0.5751648, 0.21957862],
        ])
        expected_mask = tf.constant([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ], dtype=tf.float32)
        data, *_ = extract_unit(u, m)
        tf.debugging.assert_equal(expected, data)
        # tf.debugging.assert_equal(expected_mask, mask)

    test()
