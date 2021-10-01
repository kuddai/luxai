import os

DEBUG = bool(os.environ.get('DEBUG', False))
EPS = float(os.environ.get('EPS', 0))
DRAW = bool(os.environ.get('DRAW', False))
SEED = int(os.environ['SEED']) if 'SEED' in os.environ else None

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
TARGET_MODEL_SYMLINK_PATH = os.path.join(MODELS_PATH, 'target')

UNIT_SPATIAL_DATA_RADIUS = 11
UNIT_SPATIAL_DATA_SIZE = UNIT_SPATIAL_DATA_RADIUS * 2 + 1
MAX_UNITS = 32

MAX_MAP_SIZE = 32

# RL
N_STEPS = 3
GAMMA = 0.99

# This will be changed on the first tick of the game
MAP_SIZE = None
MY_PLAYER_ID = None
ENEMY_PLAYER_ID = None
