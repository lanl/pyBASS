import numpy as np


def rootmeansqerror(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
