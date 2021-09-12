import os

DEBUG = bool(os.environ.get('DEBUG', False))
EVAL = bool(os.environ.get('EVAL', False))

ROOT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
MAIN_PATH = os.path.join(ROOT_PATH, 'main.py')
MAIN_BASELINE_PATH = os.path.join(ROOT_PATH, 'baseline.py')
LUX_BINARY = 'lux-ai-2021'

TMP_PATH = '/tmp/lux_ai/'
AGENTS_LOG_PATH = os.path.join(TMP_PATH, 'agents')
REPLAY_PATH = os.path.join(TMP_PATH, 'replays')
GAMES_DATA_PATH = os.path.join(TMP_PATH, 'games')
MODELS_PATH = os.path.join(TMP_PATH, 'models')
LATEST_MODEL_SYMLINK_PATH = os.path.join(MODELS_PATH, 'latest')

UNIT_SPATIAL_DATA_RADIUS = 7
UNIT_SPATIAL_DATA_SIZE = UNIT_SPATIAL_DATA_RADIUS * 2 + 1
MAX_UNITS = 64

MAX_MAP_SIZE = 32

# This will be changed on the first tick of the game
MAP_SIZE = None

MY_PLAYER_ID = None
ENEMY_PLAYER_ID = None
