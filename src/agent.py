import collections
import json
import os
import pickle
import random
import signal
import sys
import time
import uuid

import numpy as np
import tensorflow as tf

import solution.constants
import solution.utils as utils
import solution.actions as s_actions
from solution.model import Model
from solution.sequence_saver import SequenceSaver
from solution.rl import n_step_return
from solution.utils import print


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
total_time = collections.defaultdict(int)

prev_nn_state = None
prev_nn_actions = None

#if solution.constants.DEBUG:
#    tf.config.run_functions_eagerly(True)

#if solution.constants.SEED is not None:
#    print('Setting seed from env:', solution.constants.SEED)
#    tf.random.set_seed(solution.constants.SEED)
#    random.seed(solution.constants.SEED)
#    np.random.seed(solution.constants.SEED)


def timeit(key):
    def decorator(f):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = f(*args, **kwargs)
            total_time[key] += time.time() - start_time
            return result
        return wrapper
    return decorator


@timeit('on_seq_close')
def on_seq_close(sequence):
    n_step_target = n_step_return(
        [tf.expand_dims(s, 0) for s in sequence.states],
        tf.expand_dims(sequence.actions, 0),
        tf.expand_dims(sequence.rewards, 0),
        tf.expand_dims(sequence.is_not_done, 0),
        gamma=solution.constants.GAMMA,
        n=solution.constants.N_STEPS,
        online_network=online_network,
        target_network=target_network,
    )

    unit_mask = tf.expand_dims(tf.cast(sequence.states[4][0], dtype=tf.float32), 0)

    first_state = [tf.expand_dims(s[0], 0) for s in sequence.states]
    first_action = tf.expand_dims(sequence.actions[0], 0)

    online_q_values = online_network(first_state)
    state_value = tf.gather(online_q_values, first_action, axis=-1, batch_dims=2)
    state_value = tf.reduce_sum(state_value * unit_mask, axis=-1)

    td_error = n_step_target - state_value
    sequence.td_error = abs(td_error.numpy().item())


sequence_saver = SequenceSaver(utils.STATE_SPEC, solution.constants.N_STEPS + 1, 1, on_close=on_seq_close)

input_shape = [(1,) + tuple(i.shape) for i in utils.STATE_SPEC]
online_network = Model()
online_network.build(input_shape)
target_network = Model()
target_network.build(input_shape)

online_path = os.environ.get('ONLINE_MODEL_PATH', os.path.realpath(solution.constants.LATEST_MODEL_SYMLINK_PATH))
model_version = online_path.split('_')[-1]
online_network.load_weights(online_path)

target_path = os.environ.get('TARGET_MODEL_PATH', os.path.realpath(solution.constants.TARGET_MODEL_SYMLINK_PATH))
target_network.load_weights(target_path)

print('Online model', online_path, model_version)
print('Target model', target_path)

# Just capture sigterm to not get killed by parent process before dump
signal.signal(signal.SIGTERM, lambda *args, **kwargs: None)


def dump(game_data):
    if 'AGENT_ID' not in os.environ:
        print('!!! No AGENT_ID env is provided, the results of the game will not be saved')
        return
    print('kuddai dumping game data')

    sequence_saver.close()

    if not os.path.exists(solution.constants.REPLAY_PATH):
        os.makedirs(solution.constants.REPLAY_PATH)

    file_name = '%s' % uuid.uuid4()
    tmp_file_path = os.path.join(solution.constants.TMP_PATH, file_name)
    final_file_path = os.path.join(solution.constants.REPLAY_PATH, file_name)

    
    print('kuddai pickling', tmp_file_path,  final_file_path)
    with open(tmp_file_path, 'wb') as f:
        pickle.dump(sequence_saver.get(), f)
    os.rename(tmp_file_path, final_file_path)

    if not os.path.exists(solution.constants.GAMES_DATA_PATH):
        os.mkdir(solution.constants.GAMES_DATA_PATH)

    game_data['agent_id'] = os.environ['AGENT_ID']
    game_data['eps'] = solution.constants.EPS

    with open(tmp_file_path, 'w') as f:
        json.dump(game_data, f)
    os.rename(tmp_file_path, os.path.join(solution.constants.GAMES_DATA_PATH, file_name))


def is_game_over(game: Game):
    assert game.turn <= 360

    if game.turn == 360:
        return True
    print('kuddai turn', game.turn)

    my_player = game.players[solution.constants.MY_PLAYER_ID]
    enemy_player = game.players[solution.constants.ENEMY_PLAYER_ID]

    my_units, my_cities = len(my_player.units), my_player.city_tile_count
    enemy_units, enemy_cities = len(enemy_player.units), enemy_player.city_tile_count
    if my_units + my_cities == 0:
        return True

    if enemy_units + enemy_cities == 0:
        return True

    return False


def get_reward(game: Game) -> float:
    if not is_game_over(game):
        return 0.0

    my_player = game.players[solution.constants.MY_PLAYER_ID]
    enemy_player = game.players[solution.constants.ENEMY_PLAYER_ID]

    my_units, my_cities = len(my_player.units), my_player.city_tile_count
    enemy_units, enemy_cities = len(enemy_player.units), enemy_player.city_tile_count

    if my_cities > enemy_cities:
        return 1.0

    if my_cities < enemy_cities:
        return -1.0

    if my_units > enemy_units:
        return 1.0

    if my_units < enemy_units:
        return -1.0

    # Draw
    return 0.0


@timeit('select_action')
def select_action(q_values, actions_mask):
    assert len(q_values) == len(s_actions.ALL_ACTIONS)
    masked_q_values = np.where(actions_mask, q_values, np.full(q_values.shape, -1e9))

    best_action_index = np.argmax(masked_q_values, -1)

    if random.random() >= solution.constants.EPS:
        return best_action_index

    mask_sum = np.sum(actions_mask)
    p = actions_mask / mask_sum
    indices = np.arange(0, len(s_actions.ALL_ACTIONS))
    random_action_index = np.random.choice(indices, 1, p=p)[0]

    return random_action_index


@timeit('full_agent_time')
def agent(observation, configuration):
    global game_state
    global prev_nn_state
    global prev_nn_actions

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
        game_state._update(observation["updates"])


    ### AI Code goes down here! ###
    # you can add debug annotations using the functions in the annotate object
    # actions.append(annotate.circle(0, 0))

    if game_state.turn > 0:
        reward = get_reward(game_state)
        sequence_saver.add(prev_nn_state, prev_nn_actions, reward, None, {'model_version': model_version})

        if is_game_over(game_state):
            print('Game ended. Last turn: %s, reward: %s' % (game_state.turn - 1, reward))
            print('Total time:', total_time)
            dump({
                'reward': reward,
                'game_length': game_state.turn - 1,
            })
            sys.exit(0)

    player = game_state.players[observation.player]

    start_time = time.time()
    nn_state = utils.extract_state(game_state)
    total_time['extract_state'] += time.time() - start_time

    nn_actions = []
    actions = []

    start_time = time.time()
    q_values = online_network([
        tf.expand_dims(s, 0) for s in nn_state
    ])
    total_time['main_nn_call'] += time.time() - start_time

    counter = 0

    occupied = set()
    start_time = time.time()
    for unit in player.units:
        unit_q_values = q_values[0][counter].numpy()
        counter += 1

        actions_mask = s_actions.get_unit_actions_mask(game_state, unit)

        # Additional check for cells that are occupied already in this turn
        for i in s_actions.MOVEMENT_ACTIONS:
            if actions_mask[i] == 0.0:
                continue

            direction = s_actions.REVERSE_DIRECTION_MAP[i]
            next_pos = unit.pos.translate(direction, 1)
            x, y = next_pos.x, next_pos.y
            if (x, y) in occupied:
                actions_mask[i] = 0.0

        game_action_index = select_action(unit_q_values, actions_mask)
        nn_actions.append(game_action_index)

        if game_action_index in s_actions.MOVEMENT_ACTIONS:
            direction = s_actions.REVERSE_DIRECTION_MAP[game_action_index]

            # "Pre-move" in this cell to make sure that other units cannot move in it in this turn
            next_pos = unit.pos.translate(direction, 1)
            occupied.add((next_pos.x, next_pos.y))

            actions.append(unit.move(direction))
        elif game_action_index == s_actions.BUILD_CITY:
            actions.append(unit.build_city())

    for city_key in sorted(player.cities.keys()):
        for city_tile in player.cities[city_key].citytiles:
            city_q_values = q_values[0][counter].numpy()
            counter += 1

            actions_mask = s_actions.get_city_actions_mask(game_state, city_tile)

            game_action_index = select_action(city_q_values, actions_mask)
            nn_actions.append(game_action_index)

            if game_action_index == s_actions.BUILD_UNIT:
                # TODO: 'pre-increment' units count to stop other cities from building units this turn
                actions.append(city_tile.build_worker())

    total_time['unit_loop'] += time.time() - start_time

    if solution.constants.DEBUG:
        print(game_state.turn, 'act', actions)

    prev_nn_state = nn_state
    prev_nn_actions = nn_actions

    return actions
