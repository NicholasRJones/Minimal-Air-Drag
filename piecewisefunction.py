"""""""""
Project 2 Function File:
This file contains the function for computing the objective value and gradient to optimize.
This function is a piecewise linear function from the [0,1] interval modeling the drag of the shape for some initial input.
The first and last inputs to the initial input is fixed to force our desired scale.
This function implements a penalty system within the function to avoid non monotonic solutions.
The parameters of this function are:
    - Number of intervals of equal length
The p input determines which function value you'd like to calculate.
    0 - function value
    1 - gradient
    2 - both
"""""""""

import numpy as np


def piecewise(x, para, p):
    x_offset = np.delete(x, 0, axis=None)
    x_offset = np.append(x_offset, 0, axis=None)
    s = (x_offset - x) * para.parameter
    s = np.delete(s, para.parameter, axis=None)
    diff = x_offset ** 2 - x ** 2
    diff = np.delete(diff, para.parameter, axis=None)
    max = np.maximum(-diff, np.zeros(para.parameter))
    D = s ** 2 / (1 + s ** 2) * diff / 2 + np.exp(max) - 1
    f = D.sum()
    if p == 0:
        return f
    if p > 0:
        g = []
        for n in range(len(x) - 1):
            if n == 0:
                g.append(0)
            else:
                xg = np.array(x)
                xg[n] = x[n] + 10 ** (-8)
                xg_offset = np.delete(xg, 0, axis=None)
                xg_offset = np.append(xg_offset, 0, axis=None)
                s = (xg_offset - xg) * para.parameter
                s = np.delete(s, para.parameter, axis=None)
                gdiff = xg_offset ** 2 - xg ** 2
                gdiff = np.delete(gdiff, para.parameter, axis=None)
                max = np.maximum(-gdiff, np.zeros(para.parameter))
                D_g = s ** 2 / (1 + s ** 2) * gdiff / 2 + np.exp(max) - 1
                g.append((D_g.sum() - f) / (10 ** (-8)))
        g.append(0)
        if p > 1:
            return f, g
        return g
