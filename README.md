# InteriorPointMethod-for-NonlinearSDP
Implemented IP Method for NSDP proposed by Yamashita, Yabe, and Harada (2012) and Yamashita, Yabe (2009).

We implemented an interior point method for solving the following nonlinear semidefinite programming problem (NSDP):

> minimize    f(x) (x is n-dimensional real vector)
> subject to  g(x)≧0, h(x)=0,
>             X(x)\succeq O (i.e., X(x) is positive semidefinite)

Here, all functions f, g, h, and X are twice continuously differentiable, and the matrix-valued function X is the mapping from R^n to the space of m-dimensional symmetric matrices.

The program is originally coded for our latest study, Distributionally Robust Expected Residual Minimization for Stochastic Variational Inequality Problems (currently available on arXiv, https://arxiv.org/abs/2111.07500); hence, it might be inconvenient for general form of NSDP like the definition above, we coded that is useful for many users who are seeking NSDP solvers as much as possible though.

Perhaps our code may have inefficient steps, please contact me when you noticed them.

Thank you!
