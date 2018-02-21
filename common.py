# edit settings here
from pathlib import Path

ROOT_DIR = Path(__file__).parent.absolute()

DATA_DIR = ROOT_DIR / 'data'
RESULTS_DIR = ROOT_DIR / 'results'

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

##---------------------------------------------------------------------
import os
from datetime import datetime

# PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_PATH = ROOT_DIR

IDENTIFIER = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# numerical libs

import numpy as np
import random

import matplotlib

matplotlib.use('TkAgg')

from torch.utils.data.sampler import *

# ---------------------------------------------------------------------------------
print('@%s:  ' % os.path.basename(__file__))

if 1:
    SEED = 2016  # 1510302253  #int(time.time()) #
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print('\tset random seed')
    print('\t\tSEED=%d' % SEED)

if 1:
    torch.backends.cudnn.benchmark = True  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled = True
    print('\tset cuda environment')
    print('\t\ttorch.__version__              =', torch.__version__)
    print('\t\ttorch.version.cuda             =', torch.version.cuda)
    print('\t\ttorch.backends.cudnn.version() =', torch.backends.cudnn.version())
    try:
        print('\t\tos[\'CUDA_VISIBLE_DEVICES\']  =', os.environ['CUDA_VISIBLE_DEVICES'])
        NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    except Exception:
        print('\t\tos[\'CUDA_VISIBLE_DEVICES\']  =', 'None')
        NUM_CUDA_DEVICES = 1

    print('\t\ttorch.cuda.device_count()   =', torch.cuda.device_count())
    print('\t\ttorch.cuda.current_device() =', torch.cuda.current_device())

print('')

# ---------------------------------------------------------------------------------
