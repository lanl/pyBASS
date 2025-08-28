"""
A python package for Bayesian Adaptive Spline Surfaces
"""

__all__ = ["utils", "sobol", "BASS"]

import sys

if sys.version_info[0] == 3 and sys.version_info[1] < 6:
    raise ImportError("Python Version 3.6 or above is required for pyBASS.")
else:  # Python 3
    pass
    # Here we can also check for specific Python 3 versions, if needed

del sys

from .BASS import *
from .sobol import *
from .utils import *
