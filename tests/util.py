import numpy as np


def rootmeansqerror(predictions, targets):
    sqdiff = (predictions - targets) ** 2
    return np.sqrt(sqdiff.mean())
