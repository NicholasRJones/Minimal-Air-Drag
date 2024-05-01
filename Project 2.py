from Optimization.Functions import piecewisefunction as pf
from Optimization.Algorithm import classy, optisolve as op
import numpy as np
import matplotlib.pyplot as plt

# Project 2:
para = classy.para(0.0001, 0.19, 0, 500, 0, 0, 0)
pr = classy.funct(pf.piecewise, 'LBFGS', 'strongwolfe',
                  np.arange(0, para.parameter + 1) / para.parameter, para, 1)

a = np.arange(0, para.parameter + 1) / para.parameter
b = op.optimize(pr)
plt.plot(a, b.input, color='maroon', marker='o', markerfacecolor='black')
plt.show()
