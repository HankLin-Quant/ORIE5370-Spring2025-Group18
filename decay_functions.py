import numpy as np

def EW_decay(T):
    # T is look-back window length
    return np.array([1/T]*T)

def Exp_decay(T, base):
    # the returning weight vector, from left to right, represents weights from most recent to most distant data

    weights = np.array([base**(t) for t in range(T)])
    return weights / weights.sum()

def Lin_decay(T,end_weight):
    # Generate T equally spaced values from 1 (for most recent) to end_weight (for oldest).
    weights = np.linspace(1, end_weight, T)
    # Normalize the weights so that they sum to 1.
    return weights / weights.sum()