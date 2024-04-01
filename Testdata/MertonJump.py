import numpy as np


def merton_jump_paths(v=0.035, lam=8, steps=1000, sigma=0.35):
    ''' The function calculates a path of a merton jump model based on the transferred parameters.
    :param T: time to maturity
    :param r: risk free rate
    :param m: meean of jump size
    :param v: standard deviation of jump
    :param lam:    intensity of jump i.e. number of jumps per annum
    :param steps:  time steps
    :param sigma:  annaul standard deviation , for weiner process
    :return: merton-jump-process [list],signed jumps [list] , contamination [float]
    '''
    S = 1.0    # current stock price
    T = 1      # time to maturity
    r = 0.02   # risk free rate
    Npaths = 1 # number of paths to simulate
    m = 0      # meean of jump size

    size = (steps, Npaths)
    dt = T / steps

    # jump rate (i.e) contamination parameter for IF
    contamin = lam * dt

    # poisson-distributed jumps
    jumps = np.random.poisson(lam * dt, size=size)

    poi_rv = np.multiply(jumps,
                         np.random.normal(m, v, size=size)).cumsum(axis=0)
    geo = np.cumsum(((r - sigma ** 2 / 2 - lam * (m + v ** 2 * 0.5)) * dt +
                     sigma * np.sqrt(dt) *
                     np.random.normal(size=size)), axis=0)

    return np.exp(geo + poi_rv) * S, jumps, contamin
