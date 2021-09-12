import argparse
import os
import random
import subprocess
import time

import constants


AGENTS_LOG_DIR = constants.AGENTS_LOG_PATH


def run_single_agent(agent_id, env=None):
    agent_env = os.environ.copy()
    agent_env['CUDA_VISIBLE_DEVICES'] = ''  # force CPU agent inference
    agent_env['TF_CPP_MIN_LOG_LEVEL'] = '3'  # do not print tensorflow logs
    agent_env['USE_NN_MODEL'] = '1'
    agent_env['AGENT_ID'] = str(agent_id)

    if env:
        agent_env.update(env)

    players = [
        constants.MAIN_PATH,
        constants.MAIN_BASELINE_PATH,
    ]
    random.shuffle(players)

    cmd = [
        constants.LUX_BINARY,
        *players,
        '--python=python3',
        '--maxtime=5000',
    ]

    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=agent_env)


def main(args):
    if not os.path.exists(AGENTS_LOG_DIR):
        os.makedirs(AGENTS_LOG_DIR)

    for file_name in os.listdir(AGENTS_LOG_DIR):
        file_path = os.path.join(AGENTS_LOG_DIR, file_name)
        os.unlink(file_path)

    processes = []
    start_times = []
    games = [0] * args.n

    for agent_id in range(args.n):
        processes.append(run_single_agent(agent_id))
        start_times.append(time.time())

    while any(map(lambda p: p.returncode is None, processes)):
        for i, p in enumerate(processes):
            if p.returncode is not None:
                print(i, 'is done')
                continue

            result = p.poll()

            is_timeout = time.time() - start_times[i] > args.timeout

            if result is None and not is_timeout:
                # Still running
                continue

            if result is not None:
                # Game ended normally
                print('Game %s for agent %s has ended' % (games[i], i))

                if result != 0:
                    print('Agent %s has died with code %s' % (i, result))

                for index, out_stream in enumerate(p.communicate()):
                    name = 'stdout' if index == 0 else 'stderr'
                    data = out_stream.decode('utf-8').strip()

                    if data:
                        with open(os.path.join(AGENTS_LOG_DIR, '%s_%s_%s' % (i, games[i], name)), 'w') as f:
                            f.write(data)
            else:
                # Timeout
                print('Agent %s timeout, terminating' % i)
                p.kill()

            # Create a new game
            games[i] += 1
            processes[i] = run_single_agent(i)
            start_times[i] = time.time()

        time.sleep(0.5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int)
    parser.add_argument('--timeout', type=int, default=45)
    args = parser.parse_args()
    main(args)
