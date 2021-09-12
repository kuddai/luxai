import copy
import json
import math
import os
import pickle
import signal
import sys
import time
import uuid

import tensorflow as tf

import solution.constants
from solution.reward import calculate_reward, CITY_SCORE_SCALE, UNIT_SCORE_SCALE
import solution.utils as utils
import solution.actions as s_actions
from solution.model import Model
from solution.sequence_saver import SequenceSaver


if __package__ == "":
    # for kaggle-environments
    from lux.game import Game
    from lux.game_map import Cell, RESOURCE_TYPES
    from lux.constants import Constants
    from lux.game_constants import GAME_CONSTANTS
    from lux import annotate
else:
    # for CLI tool
    from .lux.game import Game
    from .lux.game_map import Cell, RESOURCE_TYPES
    from .lux.constants import Constants
    from .lux.game_constants import GAME_CONSTANTS
    from .lux import annotate

DIRECTIONS = Constants.DIRECTIONS
game_state = None
prev_action = None
total_time = 0
total_reward = 0
WAS_DEAD = False


sequence_saver = SequenceSaver(utils.STATE_SPEC, 6, 1)

model = Model()
path = os.path.realpath(solution.constants.LATEST_MODEL_SYMLINK_PATH)
model_version = path.split('_')[-1]
model.load_weights(path)

print('Loading model from', path, model_version, file=sys.stderr)

use_nn_model = bool(os.environ.get('USE_NN_MODEL', False))
print('Using nn model:', use_nn_model, file=sys.stderr)


def dump():
    if not os.path.exists(solution.constants.REPLAY_PATH):
        os.mkdir(solution.constants.REPLAY_PATH)

    file_name = '%s' % uuid.uuid4()
    tmp_file_path = os.path.join(solution.constants.TMP_PATH, file_name)
    final_file_path = os.path.join(solution.constants.REPLAY_PATH, file_name)

    with open(tmp_file_path, 'wb') as f:
        pickle.dump(sequence_saver.get(), f)

    os.rename(tmp_file_path, final_file_path)

    if not os.path.exists(solution.constants.GAMES_DATA_PATH):
        os.mkdir(solution.constants.GAMES_DATA_PATH)

    game_data = {
        'reward': total_reward,
    }
    with open(tmp_file_path, 'w') as f:
        json.dump(game_data, f)

    os.rename(tmp_file_path, os.path.join(solution.constants.GAMES_DATA_PATH, file_name))


def will_be_dead_next_turn(game_state, direction):
    is_night = not utils.get_is_day(game_state.turn)
    my_player = game_state.players[solution.constants.MY_PLAYER_ID]

    # TODO: precise calculating
    city_reward = 0
    city_is_dead = True
    if my_player.city_tile_count > 0:
        city_is_dead = False
        city_fuel = sum(city.fuel for city in my_player.cities.values())

        if solution.constants.DEBUG:
            print('cf', city_fuel, is_night, file=sys.stderr)

        if is_night and city_fuel < GAME_CONSTANTS['PARAMETERS']['LIGHT_UPKEEP']['CITY']:
            city_reward = -CITY_SCORE_SCALE
            city_is_dead = True

    unit_reward = 0
    unit_is_dead = True
    if my_player.units:
        unit_is_dead = False
        unit = my_player.units[0]
        next_pos = unit.pos.translate(direction, 1)
        wood, coal, uranium = unit.cargo.wood, unit.cargo.coal, unit.cargo.uranium

        researched_coal = my_player.researched_coal()
        researched_uranium = my_player.researched_uranium()

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if abs(dx) == 1 and abs(dy) == 1:
                    continue
                nx, ny = next_pos.x + dx, next_pos.y + dy
                if nx < 0 or nx >= game_state.map.width:
                    continue
                if ny < 0 or ny >= game_state.map.height:
                    continue

                cell = game_state.map.get_cell(nx, ny)
                if cell.resource:
                    if cell.resource.type == GAME_CONSTANTS['RESOURCE_TYPES']['WOOD']:
                        wood += min(cell.resource.amount, GAME_CONSTANTS['PARAMETERS']['WORKER_COLLECTION_RATE']['WOOD'])
                    if researched_coal and cell.resource.type == GAME_CONSTANTS['RESOURCE_TYPES']['COAL']:
                        coal += min(cell.resource.amount, GAME_CONSTANTS['PARAMETERS']['WORKER_COLLECTION_RATE']['COAL'])
                    if researched_uranium and cell.resource.type == GAME_CONSTANTS['RESOURCE_TYPES']['URANIUM']:
                        uranium += min(cell.resource.amount, GAME_CONSTANTS['PARAMETERS']['WORKER_COLLECTION_RATE']['URANIUM'])

        fuel = wood * GAME_CONSTANTS['PARAMETERS']['RESOURCE_TO_FUEL_RATE']['WOOD']
        fuel += coal * GAME_CONSTANTS['PARAMETERS']['RESOURCE_TO_FUEL_RATE']['COAL']
        fuel += uranium * GAME_CONSTANTS['PARAMETERS']['RESOURCE_TO_FUEL_RATE']['URANIUM']
        if is_night and fuel < GAME_CONSTANTS['PARAMETERS']['LIGHT_UPKEEP']['WORKER']:
            unit_reward = -UNIT_SCORE_SCALE
            unit_is_dead = True

    if solution.constants.DEBUG:
        print(unit_is_dead, city_is_dead, unit_reward, city_reward, file=sys.stderr)

    if unit_is_dead and city_is_dead:
        return True, city_reward + unit_reward

    return False, 0


def agent(observation, configuration):
    global game_state
    global prev_action
    global total_time
    global total_reward
    global WAS_DEAD

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player

        solution.constants.MY_PLAYER_ID = observation.player
        solution.constants.ENEMY_PLAYER_ID = 1 - observation.player
        solution.constants.MAP_SIZE = game_state.map_width
    else:
        # TODO: this operation is really slow (about 1.5 seconds overall)
        # delete it if reward calculating changes
        prev_game_state = copy.deepcopy(game_state)
        game_state._update(observation["updates"])

    actions = []

    if WAS_DEAD:
        print('DEAD FUCK', file=sys.stderr)
        return actions

    ### AI Code goes down here! ###
    player = game_state.players[observation.player]

    if game_state.turn >= 1:
        reward = calculate_reward(prev_game_state, game_state)
        total_reward += reward
        start_time = time.time()
        prev_state = utils.extract_state(prev_game_state)
        total_time += time.time() - start_time
        action = prev_action
        sequence_saver.add(prev_state, action, reward, None, {'model_version': model_version})

    opponent = game_state.players[(observation.player + 1) % 2]
    width, height = game_state.map.width, game_state.map.height

    resource_tiles: list[Cell] = []
    for y in range(height):
        for x in range(width):
            cell = game_state.map.get_cell(x, y)
            if cell.has_resource():
                resource_tiles.append(cell)

    # we iterate over all our units and do something with them
    assert len(player.units) <= 1
    prev_action = s_actions.NONE

    if not use_nn_model:
        for unit in player.units:
            #actions.append(unit.move('n'))
            #continue

            if unit.is_worker() and unit.can_act():
                closest_dist = math.inf
                closest_resource_tile = None
                if unit.get_cargo_space_left() > 0:
                    # if the unit is a worker and we have space in cargo, lets find the nearest resource tile and try to mine it
                    for resource_tile in resource_tiles:
                        if resource_tile.resource.type == Constants.RESOURCE_TYPES.COAL and not player.researched_coal(): continue
                        if resource_tile.resource.type == Constants.RESOURCE_TYPES.URANIUM and not player.researched_uranium(): continue
                        dist = resource_tile.pos.distance_to(unit.pos)
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_resource_tile = resource_tile
                    if closest_resource_tile is not None:
                        direction = unit.pos.direction_to(closest_resource_tile.pos)
                        actions.append(unit.move(direction))
                        prev_action = s_actions.DIRECTION_MAP[direction]
                else:
                    # if unit is a worker and there is no cargo space left, and we have cities, lets return to them
                    if len(player.cities) > 0:
                        closest_dist = math.inf
                        closest_city_tile = None
                        for k, city in player.cities.items():
                            for city_tile in city.citytiles:
                                dist = city_tile.pos.distance_to(unit.pos)
                                if dist < closest_dist:
                                    closest_dist = dist
                                    closest_city_tile = city_tile
                        if closest_city_tile is not None:
                            move_dir = unit.pos.direction_to(closest_city_tile.pos)
                            actions.append(unit.move(move_dir))
                            prev_action = s_actions.DIRECTION_MAP[move_dir]

    # you can add debug annotations using the functions in the annotate object
    # actions.append(annotate.circle(0, 0))

    #if game_state.turn >= 1:
    #    print(game_state.turn, reward, file=sys.stderr)

    if use_nn_model and len(player.units) == 1:
        unit = player.units[0]
        if unit.can_act():
            state_repr = utils.extract_state(game_state)
            nn_action = model([
                tf.expand_dims(s, 0) for s in state_repr
            ])
            masked_actions = [
                nn_action[0][i].numpy().item()
                if utils.is_movement_possible(unit, s_actions.REVERSE_MAP[i])
                else -1e9
                for i in range(5)
            ]
            masked_actions = tf.expand_dims(tf.constant(masked_actions), 0)
            best_action_index = tf.argmax(nn_action, -1).numpy().item()
            best_action = s_actions.REVERSE_MAP[best_action_index]
            #actions.append(annotate.text(unit.pos.x, unit.pos.y, best_action))

            random_action_index = tf.random.categorical(masked_actions, 1).numpy().item()
            random_action = s_actions.REVERSE_MAP[random_action_index]

            if solution.constants.DEBUG:
                print(
                    game_state.turn,
                    player.city_tile_count,
                    nn_action,
                    best_action_index,
                    best_action,
                    random_action_index,
                    random_action,
                    file=sys.stderr
                )
                print(state_repr[1][2:17], file=sys.stderr)

            if not solution.constants.EVAL:
                prev_action = random_action_index
                actions.append(unit.move(random_action))
            else:
                prev_action = best_action_index
                actions.append(unit.move(best_action))

    is_dead, reward = will_be_dead_next_turn(game_state, s_actions.REVERSE_MAP[prev_action])

    if is_dead and game_state.turn < 359:
        WAS_DEAD = True
        total_reward += reward
        print('Dead before end', game_state.turn, total_reward, file=sys.stderr)
        sequence_saver.add(utils.extract_state(game_state), prev_action, reward, None, {'model_version': model_version})
        dump()

    if game_state.turn == 359:
        print('Made it to the end', total_reward, player.city_tile_count, file=sys.stderr)
        # TODO: add reward and save (?)
        #print(total_time, total_time / 360, file=sys.stderr)
        dump()

    return actions
