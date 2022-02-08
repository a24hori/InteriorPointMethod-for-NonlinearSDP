# InteriorPointMethod-for-NonlinearSDP
Implementation of a primal-dual interior point method for nonlinear semidefinite programming problems with second-order cone constraints proposed by Yamashita, Yabe, and Harada [1] and Yamashita, Yabe [2].

We implemented an interior point method for solving the following nonlinear semidefinite programming problem (NSDP):

```
minimize    f(x) (x is n-dimensional real vector) 
subject to  g(x)≧0, h(x)=0,
            X(x)\succeq 0 (i.e., X(x) is positive semidefinite)
            t(x) \in K (K is a second-order cone)
```

Here, all functions f, g, h, t and X are twice continuously differentiable, and the matrix-valued function X is the mapping from R^n to the space of m-dimensional symmetric matrices.

The program is originally coded for our latest study, Distributionally Robust Expected Residual Minimization for Stochastic Variational Inequality Problems (currently available on arXiv, https://arxiv.org/abs/2111.07500); hence, it might be inconvenient for general form of NSDP like the definition above, we coded that is useful for many users who are seeking NSDP solvers as much as possible though.

Perhaps our code may have inefficient steps, please contact me when you noticed them.

The test program is now under development, we will release soon.

Thank you!


---

### References

[1] Yamashita, H., Yabe, H. & Harada, K. A primal–dual interior point method for nonlinear semidefinite programming. Math. Program. 135, 89–121 (2012). 

[2] Yamashita, H. & Yabe, H. A primal–dual interior point method for nonlinear optimization over second-order cones. Optim. Method. Softw. 24, 407-426 (2009).
