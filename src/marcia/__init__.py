import warnings

import matplotlib as mpl

from .core import *
from .fitting import *
from .io import *
from .plotting import *

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

mpl.rcParams['image.cmap'] = 'cividis'
