from casadi import *
import numpy as np

# This is the file where I tested the casadi because the doc is totally unreadable..
# For exp: if you wanna use sum1 the <only> info you could get is that you could
# pass an arg into it, that's it. It's so so painful.
opti = casadi.Opti()
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
x = opti.variable(3, 3, 'full')
print(eig_symbolic(a))
"""p = opti.parameter(3, 3)
opti.set_value(p, a)
print(sum1(a))
helper_1 = np.eye(3)
helper_2 = MX.eye(3)
helper_2 = helper_2 @ a
print(sum2(a))
print(repmat(helper_2, (1, 2)))
s = opti.parameter(3, 3)
print(cumsum(np.array([1,1,1])))
print(p)
"""


