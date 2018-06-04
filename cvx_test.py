import cvxpy as cp
import numpy as np

# Problem data.
N = 100
n = 500
c = 2
l2_dim = 10
len_w = n + n*l2_dim + l2_dim*c + c
np.random.seed(1)
wk = np.random.randn(len_w)*100 + np.ones(len_w)*2
g = np.random.randn(len_w)
ind = np.random.randint(0,1,N)
y = np.random.rand(N,c)

# Construct the problem.
w = cp.Variable(len_w)
yhat = cp.Variable((N,c))
objective = cp.Minimize(cp.sum(-yhat[np.arange(N),ind] + cp.log_sum_exp(yhat, axis=1)) + cp.norm(w,2))
constraints = [yhat == y + g @ (w - wk)]
prob = cp.Problem(objective, constraints)

# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
# The optimal value for x is stored in `x.value`.
# The optimal Lagrange multiplier for a constraint is stored in
# `constraint.dual_value`.
# print(constraints[0].dual_value)