import numpy as np

import solution.constants as constants
from solution.utils import print
from lux.game import Game, Player


UNIT_SCORE_SCALE = 100
CITY_SCORE_SCALE = 1000
CARGO_SCALE = 0.1
FUEL_SCALE = 0.5


def _sum_cargo(player: Player):
    wood = coal = uranium = 0
    for u in player.units:
        wood += u.cargo.wood
        coal += u.cargo.coal
        uranium += u.cargo.uranium

    return np.array([wood, coal, uranium])


def _sum_cities_fuel(player: Player):
    return sum(city.fuel for city in player.cities.values())


def calculate_reward(prev_state: Game, next_state: Game):
    prev_player = prev_state.players[constants.MY_PLAYER_ID]
    next_player = next_state.players[constants.MY_PLAYER_ID]

    reward = 0

    unit_diff = len(next_player.units) - len(prev_player.units)
    unit_reward = unit_diff * UNIT_SCORE_SCALE
    reward += unit_reward

    prev_cargo = _sum_cargo(prev_player)
    next_cargo = _sum_cargo(next_player)
    cargo_reward = np.sum(np.maximum(next_cargo - prev_cargo, 0)) * CARGO_SCALE
    reward += cargo_reward

    prev_cities_fuel = _sum_cities_fuel(prev_player)
    next_cities_fuel = _sum_cities_fuel(next_player)
    fuel_reward = max(0, next_cities_fuel - prev_cities_fuel) * FUEL_SCALE
    reward += fuel_reward

    city_diff = next_player.city_tile_count - prev_player.city_tile_count
    city_reward = city_diff * CITY_SCORE_SCALE
    reward += city_reward

    return reward
