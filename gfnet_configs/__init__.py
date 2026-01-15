import os
from types import SimpleNamespace

cfg = SimpleNamespace()

# training record
cfg.DEBUG_MODE = False
cfg.RANK = int(os.environ.get('RANK', default = 0))
cfg.GLOBAL_STEP = 0
cfg.STEP_SIZE = 1
cfg.LOCAL_RANK = -1

# data path
cfg.DATA_PATH = 'data' ### your own data path