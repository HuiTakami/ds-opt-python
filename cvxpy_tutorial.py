import cvxpy
import cvxpy as cx
import numpy as np

a = np.array([[1,1],[2,2]])
b = np.array([[1,1],[2,2]])
# b1 = cx.Variable(1)
# b2 = cx.Variable(1)
# b3 = cx.promote(b1, (1000,1))
# b4 = cx.promote(b2, (1000,1))
# b5 = cx.hstack([b3, b4])
# c1 = np.zeros((1000,2)) + b5
c = []
b1 = cx.Variable(1)
b2 = cx.reshape(cx.hstack([b1, b1, b1]), (3, 1))
c.append(a)
c.append(b)
c = np.array(c)
a = cx.Variable((3, 3))
b = cx.Variable((3, 3))
c = cx.Variable((2, 2))
d = cx.Variable((3, 1))
d = np.repeat(d, 788, axis=0)
s = cvxpy.bmat(d)
f = a + np.eye(3) + cvxpy.vstack((d, d, d)).T
sb = c + np.eye(2)
print( np.array([np.eye(2)[:, 0]]) + d.T)
sb = 1