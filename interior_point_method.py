'''interior_point_method.py

    This is a program to solve the following nonlinear
    semidefinite programming problem with single second-order cone constraint (P):\
        min_x   f(x), x \in R^m
        s.t.    ceq(x) = 0,\
                cieq(x) >= 0 (can be transformed to cieq(x)-s=0 and s>=0 by adding slack variable),
                coneg(x) \in K (can be transformed to coneg(x)-t=0 by adding the slack variable t\in K),
                matX(x) >> O (can be transformed to matX(x)-U=O and U>>O by adding slack variable),
    where ceq:R^n→R^m, cieq:R^n→R^l, coneg:R^n→R^q and matX:R^n→S^p.
    The cone K is defined by\
        K := { x=(x0,x')\in R^q | \|x'\|^2 <= x0 }.
    Here, S^p is a set of symmetric matrices on R^{p*p}, and '>>' denotes
    that the l.h.s. matrix is positive semidefinite.

    By adding slack variables to inequality constraints, they can be designed to
    generate a sequence of feasible points even starting from infeasible point:
        cieq(x) - s = 0, s >= 0.
    Likewise, we have\
        coneg(x) - t = 0, t \in K,
        matX(x) - U = O, U >> O.

    We code to solve the NSDP with second-order cone constraints 
    above in the primal-dual interior point method proposed by [1] and [2],
    and we combined these algorithms.

    - Variables and Lagrange multipliers:
        x: primal variable whose dimension is n.
        s: slack variable for cieq(x)>=0 whose dimension is l (# of constraints).
        t: slack variable for coneg(x) \in K whose dimension is q.
        U: slack (matrix) variable assumed to be positive semidefinite (p.s.d.)\
            whose dimension is p*p.
        eta: Lagrange multiplier for ceq = 0 whose dimension is m.
        lmd: Lagrange multiplier for cieq(x) >= 0 whose dimension is l.
        xi: Lagrange multiplier for coneg(x) \in K whose dimension is q.
        Z: Lagrange (symmetric p.s.d. matrix) multiplier for matX >> O whose dimension is p*p.
    
    - Lagrange function:
        L(x,s,t,U,eta,lmd,xi,Z):=f(x)-eta@ceq(x)-lmd@(cieq(x)-s)-xi@(cone(x)-t)-matIP(Z,matX(x)-U),
    where matIP(.,.) is Frobenius matrix product defined by trace(A@B) for symmetric matrices A, B.

    Coded by Atsushi Hori (Kyoto University), Oct. 20th, 2020.

    References:
        [1] H. Yamashita and H. Yabe, A Primal-dual Interior Point Method for
            Nonlinear Optimization Over Second Order Cones, technical report,
            Mathematical Systems Inc., Tokyo, 2005 (revised 2006).
        [2] Yamashita, H., Yabe, H. & Harada, K. A primal–dual interior point
            method for nonlinear semidefinite programming. Math. Program. 135,
            89-121 (2012). https://doi.org/10.1007/s10107-011-0449-z

    WARNING:
        This program is not optimized for speed.
        It is designed for educational purpose only.

'''

# ******************************************************
# ****************** Import modules ********************
# ******************************************************
from numpy import *
import numpy as np
import time
from applied_linalg import *
import warnings
warnings.simplefilter('error', ComplexWarning)
warnings.simplefilter('ignore')
# ******************************************************



# =========================================
# Program for interior point method
# =========================================
class IPSolve():
    """Solve nonlinear SDP by interior point method

    Args:
    # Functions
        - f: Objective function
        - grad_f: Gradient of f
        - hess_f: Hessian of f
        - ceq: Equality function (dimension is m)
        - jac_ceq: Transposed Jacobian of ceq
        - hess_ceq: Hessian of ceq
        - cieq: Inequality function (dimension is l)
        - jac_cieq: Transposed Jacobian of cieq
        - hess_cieq: Hessian of cieq
        - cone: Cone constraint function (dimension is q)
        - jac_cone: Transposed Jacobian of cone
        - hess_cone: Hessian of cone
        - matX: Matrix constraint function (dimension is p*p)
        - A: Matrix gradient of matX
        - hess_matX: Hessian of matX
    # Dimension 
        - n (type: int): dim. of decision vector
        - m (type: int): dim. of equality constraint function
        - l (type: int): dim. of inequality constraint function
        - p (type: int): dim. of matrix constraint function
        - q (type: int): dim. of cone constraint function
    # Initial point
        - init_x: Initial point so that it is interior point of a feasible set
    # Other specific parameters 
        - grad_check (type: boolean, default: False): Check whether user specified gradient is sufficiently close to finite difference approximation.
        
    Output:
        - sol_x: Solution and Lagrange multipliers
        - iter: Number of iteration
        - comput_time: Computation time
    """

    def __init__(self, init_x, verbose=True, grad_check=False, **probdata):
        
        # Init. point
        self.init_x = init_x
        # Gradient check
        self.grad_check = grad_check
        # Verbose
        self.verbose = verbose
        
        # ---
        # Functions
        # ---
        self.f = probdata['f']
        self.grad_f = probdata['grad_f']
        self.hess_f = probdata['hess_f']
        self.ceq = probdata['ceq']
        self.B0 = probdata['jac_ceq']
        self.hess_ceq = probdata['hess_ceq']
        self.cieq = probdata['cieq']
        self.A0 = probdata['jac_cieq']
        self.hess_cieq = probdata['hess_cieq']
        self.cone= probdata['cone']
        self.grad_cone= probdata['jac_cone']
        self.hess_cone= probdata['hess_cone']
        self.matX= probdata['matX']
        self.A= probdata['A']
        self.hess_matX= probdata['hess_matX']
        
        # ---
        # Dimensions
        # ---
        self.n = probdata['n']
        self.m = probdata['m']
        self.l = probdata['l']
        self.p = probdata['p']
        self.q = probdata['q']

        # ---
        # Solve NLP
        # ---
        self.sol_x, self.iter, self.comput_time = self.sdpip(self.init_x)

    # ---
    # Gradient of Lagrange function
    # ---
    def grad_lag(self, x, eta, lmd, xi, Z):

        # Gradient for Inequality constraints
        if not self.l == 0:
            gradieqlag = self.B0(x) @ lmd
        else:
            gradieqlag = zeros(self.n)
           
        # Gradient for Equality constraints
        if not self.m == 0:
            gradeqlag = self.A0(x) @ eta
        else:
            gradeqlag = zeros(self.n)

        # Gradient for Second-order cone constraints
        if not self.q == 0:
            gradconelag = self.grad_cone(x) @ xi
        else:
            gradconelag = zeros(self.n)

        # Gradient for Nonlinear matrix constraints
        if not self.p == 0:
            gradmatlag = self.adjAx_Z(x, Z)
        else:
            gradmatlag = zeros(self.n)

        return self.grad_f(x) - gradieqlag - gradmatlag -\
            gradconelag - gradeqlag 


    # ---
    # Hessian of Lagrange function
    # ---
    def hess_lag(self, x, eta, lmd, xi, Z):
       
        # Hessian matrix for Equality constraints
        if not self.m == 0:
            hesseqlag = zeros((self.n, self.n))
            for i in range(self.m):
                hesseqlag += eta[i] * self.hess_ceq(x, i)
        else:
            hesseqlag = zeros((self.n, self.n))

        # Hessian matrix for Inequality constraints
        if not self.l == 0:
            hessieqlag = zeros((self.n, self.n))
            for i in range(self.l):
                hessieqlag += lmd[i] * self.hess_cieq(x, i)
        else:
            hessieqlag = zeros((self.n, self.n))

        # Hessian matrix for nonlinear second-order cone constraints
        if not self.q == 0:
            hessconelag = zeros((self.n, self.n))
            for i in range(self.q):
                hessconelag += xi[i] * self.hess_cone(x, i)
        else:
            hessconelag = zeros((self.n, self.n))

        # Hessian matrix for Nonlinear matrix constraints
        if not self.p == 0:
            hessmatlag = self.adjBx_Z(x, Z)
        else:
            hessmatlag = zeros((self.n, self.n))

        return self.hess_f(x) - hessieqlag - hessmatlag -\
            hessconelag - hesseqlag 


    # ---
    # Adjoint operator <Ax,z>=<x,A'z>
    # ---
    def adjAx_Z(self, x, Z):
        vec = empty(self.n)
        for i in range(self.n):
            vec[i] = matIP(self.A(x, i), Z)
        return vec


    # ---
    # Differential of Adjoint operator
    # ---
    def adjBx_Z(self, x, Z):
        mat = empty((self.n, self.n))
        for i in range(self.n):
            for j in range(i, self.n):
                mat[i, j] = matIP(self.hess_matX(x, i, j), Z)
                mat[j, i] = mat[i, j]
        return mat


    # Delta X
    def delta_X(self, x, dx):
        return matsum([dx[i] * self.A(x, i) for i in range(self.n)])

    # ---
    # KKT residual function and its norm
    # ---
    def residual0(self, w):
        x, eta, lmd, xi, Z = w
        if not self.m == 0:
            r0_g = self.ceq(x)
        else:
            r0_g = array([])
        if not self.l == 0:
            r0_h = block([self.cieq(x) * lmd])
        else:
            r0_h = array([])
        if not self.q == 0:
            r0_K = block([JProd(self.cone(x), xi)])
        else:
            r0_K = array([])
        if not self.p == 0:
            r0_X = self.matX(x) @ Z
        else:
            r0_X = array([])

        return block([self.grad_lag(x, eta, lmd, xi, Z), r0_g, r0_h, r0_K]), r0_X


    # Norm
    def r0_norm(self, w):
        r0_vec, r0_mat = self.residual0(w)

        r01_norm = linalg.norm(r0_vec)
        r02_norm = mat_norm(r0_mat)

        return sqrt(r01_norm ** 2 + r02_norm ** 2)


    # ---
    # Barrier KKT (BKKT) condition perturbed by mu>0 and its norm
    # ---
    def r(self, w, mu):
        x, eta, lmd, xi, Z = w

        if not self.m == 0:
            r_g = self.ceq(x)
        else:
            r_g = array([])
        if not self.l == 0:
            r_h = block([self.cieq(x) * lmd - mu * ones(self.l)])
        else:
            r_h = array([])
        if not self.q == 0:
            r_K = block([JProd(self.cone(x), xi) - mu * unitvec(self.q)])
        else:
            r_K = array([])
        if not self.p == 0:
            r_X1 = self.matX(x) @ Z - mu * eye(self.p)
        else:
            r_X1 = array([])
            
        r_ghK = block([self.grad_lag(x, eta, lmd, xi, Z), r_g, r_h, r_K])
        #print(f'r_ghK: {r_ghK}')

        return r_ghK, r_X1
        
    # Norm
    def r_norm(self, w, mu):
        r_vec, r_mat = self.r(w, mu)
        r1_norm = linalg.norm(r_vec)
        r2_norm = mat_norm(r_mat)

        return sqrt(r1_norm ** 2 + r2_norm ** 2)


    # ---
    # Numerical directional derivative of r_norm
    # ---
    def approx_diff(self, wk, mu, i, dir, eps=1e-4):
        tmp = wk[i]
        wk[i] = tmp + eps * dir[i]
        f_pd = self.r_norm(wk, mu)
        wk[i] = tmp
        f_ = self.r_norm(wk, mu)
        return (f_pd - f_) / eps


    # ---
    # Transform matrix T for the variables 
    # ---
    @staticmethod
    def Tvec(vec):
        return 2 * Arw(vec) @ Arw(vec) - Arw(JProd(vec, vec))


    # ---
    # Tvec: generate Nesterov-Todd direction matrix Tp
    # ---
    def TvecNT(self, t, xi):
        T_half = self.Tvec(vec_power(t, 1 / 2))
        pp = vec_power(T_half @ vec_power(T_half @ xi, -1 / 2), -1 / 2)
        Tp = self.Tvec(pp)
        return Tp

    # ---
    # Scaling matrix T, W and H by applying Nesterov-Todd direction
    # ---
    @staticmethod
    def Tmat(W):
        '''
        The NT direction scaling matrix T is obtained by \
            T = mat_minus1half(W),
        but it is not used in main alg.
        Thus, we omit the operation.
        '''
        return mat_minus1half(W)

    @staticmethod
    def Wmat(U, Z):
        U_half = mat_1half(U)
        W = U_half @ mat_minus1half(U_half @ Z @ U_half) @ U_half
        invW = linalg.inv(W)
        return W, invW


    def Hmat(self, x, invX, Z):
        ''' In general the matrix H is not symmetric. However,
            if we choose the direction appropriately such that
            U_@Z_ = Z_@U_, we have that H is symmetric. '''

        H = empty((self.n, self.n))
        if not all(isclose(invX.imag, zeros_like(invX))):
            raise ValueError('invX is not real matrix.')
        for i in range(self.n):
            for j in range(i, self.n):
                # Choose HRVW/KSH/M direction
                H[i, j] = trace(self.A(x, i) @ invX @ self.A(x, j) @ Z)
                H[j, i] = H[i, j]
        
        return H

    # ==================================================
    # Check specified gradient if the user provided parameter `grad_check` is True.
    # ==================================================
    
    # ---
    # Check gradient
    # ---
    def check_grad_f(self, x):
        if not all(isclose(self.grad_f(x), self.finite_diff_approx(x))):
            print('\n============ ERROR Information ==============')
            finiteapprox = self.finite_diff_approx(x)
            truegradf = self.grad_f(x)
            print('finite_diff_approx(x):\n{}'.format(finiteapprox))
            print('given gradient:\n{}'.format(truegradf))
            print('diff:\n{}'.format(finiteapprox - truegradf))
            print('\|diff\|: {}'.format(linalg.norm(finiteapprox - truegradf)))
            raise Exception('Check gradient failed.')


    # ---
    # Check Jacobian of cone(x)
    # ---
    def check_jac_cone(self, x, eps=1.0e-04):
        for i in range(self.n):
            if not all(isclose(self.num_jac_fun(self.cone, x), self.grad_cone(x))):
                print('\n============ ERROR Information ==============')
                print('x:{}'.format(x))
                print('cone(x):\n{}'.format(self.cone(x)))
                print('num_jac_fun(cone, x):\n{}'.format(self.num_jac_fun(self.cone, x)))
                print(f'Specified Jacobian: {self.grad_cone(x)}')
                print('diff:\n{}'.format(self.num_jac_fun(self.cone, x) - self.grad_cone(x)))
                raise Exception('The specified Jacobian of cone(x) is different from approx.')
        return True


    # ---
    # Check matrix gradient
    # ---
    def check_grad_matX(self, x): # for unittest
        for i in range(self.n):
            if not all(isclose(self.matrix_num_grad(x, i), self.A(x, i))):
                print('\n============ ERROR Information ==============')
                print('i: {}'.format(i))
                print('x:{}'.format(x))
                print('matX:\n{}'.format(self.matX(x)))
                print('matrixnumgrad(x, i):\n{}'.format(self.matrix_num_grad(x, i)))
                print('A(x,i):\n{}'.format(self.A(x, i)))
                print('diff:\n{}'.format(self.matrix_num_grad(x, i) - self.A(x, i)))
                print('\|diff\|: {}'.format(linalg.norm(self.matrix_num_grad(x, i) - self.A(x, i))))
                raise ValueError('The specified gradient of matX(x) is different from approx.')
            
        return True


    # ---
    # Check Hessian of matX
    # ---
    def check_hess_matX(self, x):
        for j in range(self.n):
            for k in range(self.n):
                if not all(isclose(self.matrix_num_hess(x, j, k), self.hess_matX(x, j, k), atol=1e-4)):
                    print('\n============ ERROR Information ==============')
                    print('j, k: {}, {}'.format(j, k))
                    print('matrixnumhess(x, j, k):\n{}'.format(self.matrix_num_hess(x, j, k)))
                    print('HessmatX(x,j,k):\n{}'.format(self.hess_matX(x,j,k)))
                    print('diff:\n{}'.format(self.matrix_num_hess(x,j,k) - self.hess_matX(x,j,k)))
                    print('\|diff\|: {}'.format(linalg.norm(self.matrix_num_hess(x,j,k) - self.hess_matX(x,j,k))))
                    raise ValueError('Gradient Check Failed.')
                
        return True


    # ---
    # Finite difference approximation of the objective function f
    # ---
    def finite_diff_approx(self, x, eps=1.0e-04):
        grad = empty(self.n)
        for i in range(self.n):
            tmp = x[i]
            x[i] = tmp + eps
            f_p = self.f(x)
            x[i] = tmp - eps
            f_m = self.f(x)
            x[i] = tmp
            grad[i] = (f_p - f_m) / (2 * eps)
        return grad


    # ---
    # Finite difference approximation of vector-valued function
    # ---
    def num_jac_fun(self, fun, x, eps=1.0e-04):
        jac_val = empty((self.n, self.q))
        for i in range(self.n):
            tmp = x[i]
            x[i] = tmp + eps
            f_p = fun(x)
            x[i] = tmp - eps
            f_m = fun(x)
            x[i] = tmp
            jac_val[i] = array([(f_p - f_m) / (2 * eps)])
        return jac_val


    # ---
    # Finite difference approximation of the matrix function matX
    # ---
    def matrix_num_grad(self, x, i, eps=1.0e-4):
        # Numerical gradient
        tmp = x[i]
        x[i] = tmp + eps
        f_p = self.matX(x)
        x[i] = tmp - eps
        f_m = self.matX(x)
        # modify x
        x[i] = tmp
        return (f_p - f_m) / (2 * eps)


    # ---
    # Finite difference approximation of the difference of the matrix function matX
    # ---
    def matrix_num_hess(self, x, j, k, eps=1e-4):
        # Numerical Hessian
        tmp = x[j]
        x[j] = tmp + eps
        f_p = self.A(x, k)
        x[j] = tmp - eps
        f_m = self.A(x, k)
        # modify x
        x[j] = tmp
        return (f_p - f_m) / (2 * eps)



    # ================================
    # Interior point method algorithm
    # ================================


    # ---
    # Check whether the initial point is an interior point of constraints
    # ---
    def check_interior_point(self, init_x):

        # Inequality constraint
        if not self.l == 0:
            test_cieq_interior = all(self.cieq(init_x) > zeros_like(self.cieq(init_x)))
        else:
            test_cieq_interior = 1
        # Cone constraint
        if not self.q == 0:
            test_cone_interior = self.cone(init_x)[0] > linalg.norm(self.cone(init_x)[1:])
        else:
            test_cone_interior = 1
        # Matrix constraint
        if not self.p == 0:
            test_mat_interior = min(linalg.eigvals(self.matX(init_x))) > 0
        else:
            test_mat_interior = 1
        if test_cieq_interior and test_cone_interior and test_mat_interior:
            return True
        else:
            return False
        

    # ---
    # Initialization
    # ---
    def init_wk(self, init_x):
        xk = init_x
        etak = array([0])
        lmdk = array([0])
        xik = array([0])
        Zk = array([[0]])
        if not self.m == 0:
            etak = ones(self.m)
        if not self.l == 0:
            lmdk = ones(self.l)
        if not self.q == 0:
            xik = block([2 * self.q, ones(self.q - 1)])
        if not self.p == 0:
            Zk = eye(self.p)

        return xk, etak, lmdk, xik, Zk

        

    # ---
    # Interior point algorithm (outer iter.)
    # ---
    def sdpip(self, init_x):
        """Interior point method for nonlinear programming

        Args:
            init_x (numpy.array): Initial point of the algorithm. Note that\
                the initial point init_x is not necessarily an interior point of the constraints.

        Returns:
            wk (numpy.array): Solution to NSDP.
            k+1 (int): Outer iteration number
            comput_time (float64): Computation time (sec.)
        """

        # Initialize
        maxiter = 1000  # Maximum iteration
        epsilon = 1.0e-6  # Termination tolerance for r0
        Mc = 1
        muk = 1  # Centralization parameter
        muk_coeff = 0.1
        rho = 1 # Penalty
        # Line search parameters
        beta = 0.90 # Stepsize 
        gamma = 0.95 # Coeff. of initial stepsize
        epsilon0 = 1.0e-3 # Slope in Armijo rule
        nu = 1
        sigma = 1

        # wk initialization
        wk = self.init_wk(init_x)

        if self.check_interior_point(init_x) == False:
            raise ValueError('The initial point given is not interior point of the constraints.')

        is_failed = False
        start_time = time.time()
        for k in range(maxiter):

            if is_failed:
                is_failed = False
                start_time = time.time()
            
            if self.verbose == True:
                print(f'------ Mc: {Mc}, muk: {muk}, Mc*muk: {Mc*muk} ------')

            ''' Step 1. Solve Approximate BKKT point satisfying\
                \|r(wk,muk)\|\leq Mc*muk by the line search algorithm '''
            try:
                wk, detak = self.sdpls(wk, Mc, muk, beta, gamma, epsilon0, nu, rho, sigma)
                xk, etak, lmdk, xik, Zk = wk
                # Choose penalty parameter for the violation of equality constr.
                if not self.m == 0:
                    rho = max(linalg.norm(etak+detak, ord=inf)) + 1
            except (ValueError, np.core._exceptions.UFuncTypeError):
                # Stop timer to recount
                time.time()
                is_failed = True
                print(f'beta={beta}, Mc={Mc}, muk_coeff={muk_coeff}, epsilon={epsilon} failed.')
                # Mildening parameter
                muk_coeff = 0.2
                muk = 1
                beta -= 0.1
                epsilon *= 10
                # Reset initial point
                wk = self.init_wk(init_x)
                if beta < 0.20:
                    raise Exception(f"Failed to solve even beta is sufficiently small.")
                print(f'Changed: beta={beta}, muk_coeff={muk_coeff}, epsilon={epsilon}')
                continue

            ''' Step 2. (Termination): Check r0 feasibility '''
            r0_norm_val = self.r0_norm(wk)
            if self.verbose:
                print(f'||r0(wk)||: {r0_norm_val}')
            if r0_norm_val <= epsilon:
                break

            ''' Step 3. (Update muk): . '''
            muk = muk * muk_coeff

        end_time = time.time()
        comput_time = end_time - start_time

        return wk, k + 1, comput_time


    # ---
    # Line search algorithm (inner iter.)
    # ---
    def sdpls(self, wk, Mc, mu, beta, gamma, epsilon0, nu, rho, sigma):

        # Initialization
        xk, etak, lmdk, xik, Zk = wk
        detak = array([0])
        r_mu_recent = array([])

        k = 0

        # Step 1,2: 
        r_mu_value_new = self.r_norm(wk, mu)
        while r_mu_value_new > Mc * mu:
            
            ''' Check gradient and Hessian '''
            if self.grad_check:
                # gradf(x)
                self.check_grad_f(xk)
                # dcone(x)
                self.check_jac_cone(xk)
                # A(x, i)
                self.check_grad_matX(xk)
                # HessmatX(x, i, j)
                self.check_hess_matX(xk)


            ''' Step. 2: Solve Newton direction '''
            dwk = self.newton_direction(wk, mu)
            dxk, detak, dlmdk, dxik, dZk = dwk

            ''' Step 3. Determine step size alphak '''
            alphak = self.determine_step_size(
                wk, dwk, beta, epsilon0, mu, gamma, nu, rho, sigma)

            ''' (Step 4'. Update variables) '''
            xk = xk + alphak * dxk
            if self.m != 0:
                etak = etak + detak
            if self.l != 0:
                lmdk = lmdk + alphak * dlmdk
            if self.q != 0:
                xik = xik + alphak * dxik
            if self.p != 0:
                Zk = Zk + alphak * dZk

            r_mu_value_old = r_mu_value_new
            wk = (xk, etak, lmdk, xik, Zk)
            r_mu_value_new = self.r_norm(wk, mu)
            r_mu_recent = append(r_mu_recent, r_mu_value_new - r_mu_value_old)

            if self.verbose:
                print(f'alphak (step size): {alphak}')
                print(f'r_mu(wk) (after): {r_mu_value_new}')

            if all(r_mu_recent[-3:] > 1.0e-03):
                raise ValueError(f"Updated BKKT is increasing.")

            ''' If the recent r_mu values does not sufficiently decrease, raise exception. '''
            if (k > 5) and (sum(abs(r_mu_recent[-5:])) < 1):
                raise ValueError(f"r_mu does not change by SDPLS. Change beta.")

            k += 1
            
        if self.verbose:
            print('inner iter.: {}'.format(k))

        return wk, detak


    # ---
    # Solve Newton Equation
    # ---
    def newton_direction(self, wk, mu):
        '''
            Solve Newton equation to obtain the next direction
            of the iterate. 

            Args:
                wk: current point wk=(xk, etak, lmdk, xik, Zk)
                mu: perturbation parameter

            Returns:
                dwk: Newton direction
        '''

        xk, etak, lmdk, xik, Zk = wk

        # ---
        # Prepare coefficient matrix and constant vector
        # ---
        if not self.l == 0:
            cieq_val = self.cieq(xk)
            Sigmak = diag(lmdk / cieq_val)
            inv_cieq = 1 / cieq_val
            nabla_cieq = self.B0(xk)
            
            # (Use in this form)
            # Transformation matrix
            Sigmak_B0 = nabla_cieq @ Sigmak @ nabla_cieq.T
            B0_mod_term = mu * nabla_cieq @ inv_cieq
        else:
            Sigmak_B0 = zeros((self.n, self.n))
            B0_mod_term = zeros(self.n)
            
        if not self.q == 0:
            conek = self.cone(xk)
            invconek = vec_power(conek, -1)
            dconek = self.grad_cone(xk)
            Tp = self.TvecNT(conek, xik)
            invTp = linalg.inv(Tp)
            conek_ = Tp @ conek # Must be unitvec(q) for NT direction
            xik_ = invTp @ xik

            # Transformation matrix
            Trans_vec = Tp @ linalg.inv(Arw(conek_)) @ Arw(xik_) @ Tp
            Trans_dcone = dconek @ Trans_vec @ dconek.T
            dcone_mod_term = dconek @ (xik + mu * invconek - Trans_vec @ conek)
        else:
            Trans_dcone = zeros((self.n, self.n))
            dcone_mod_term = zeros(self.n)
            
        if not self.p == 0:
            matXk = self.matX(xk)
            invXk = linalg.inv(matXk)
            # Transformation matrix
            Hk = self.Hmat(xk, invXk, Zk)
            adjAx_mod_term = mu * self.adjAx_Z(xk, invXk)
        else:
            Hk = zeros((self.n, self.n))
            adjAx_mod_term = zeros(self.n)

        # ---
        # Solve Newton's equation
        # ---

        # Eigenvalue modification of the Hessian of Lagrange function
        # (ref. Nocedal, Wright: Numerical Optimization p.50, 51)
        Gk = self.eigmod(self.hess_lag(xk, etak, lmdk, xik, Zk)).real

        # Define equation
        if not self.m == 0:
            # The case where eq. constraints exists in original problem
            JF = block([[Gk + Sigmak_B0 + Trans_dcone + Hk, -self.A0(xk).T],\
                        [self.A0(xk), zeros((self.m, self.m))]])
            F = block([self.grad_f(xk) - self.A0(xk) @ etak - B0_mod_term -
                       dcone_mod_term - adjAx_mod_term, self.ceq(xk)])
        else:
            # No Eq. constraints exists in original problem
            JF = Gk + Sigmak_B0 + Trans_dcone + Hk
            F = self.grad_f(xk) - B0_mod_term - dcone_mod_term - adjAx_mod_term

        # Solve equation
        dk = linalg.solve(JF, -F)

        # If the solution is not found, stop the algorithm.
        '''
        if not all(isclose(JF @ dk, -F)):
            raise ValueError(f'Newton Equation failed to solve. Residual: {JF@dk+F}')
        '''

        # ---
        # Assigning each variable
        # ---

        # xk
        dxk = dk[:self.n]

        if not self.m == 0:  # eq. constr. only
            detak = dk[self.n:self.n + self.m]
        else:
            detak = array([0])
        if not self.l == 0:
            dlmdk = mu * inv_cieq - lmdk - Sigmak @ nabla_cieq.T @ dxk
        else:
            dlmdk = array([0])
        if not self.q == 0:
            dxik = mu * invconek - xik - Trans_vec @ dconek.T @ dxk
        else:
            dxik = array([0])
        if not self.p == 0:
            dXk = self.delta_X(xk, dxk)
            dZk = mu * invXk - Zk - 1/2*(invXk @ dXk @ Zk + Zk @ dXk @ invXk)
        else:
            dXk = array([0])
            dZk = array([0])

        dwk = (dxk, detak, dlmdk, dxik, dZk)

        return dwk


    # ---
    # Determine step size
    # ---
    def determine_step_size(self, wk, dwk, beta, epsilon0, mu, gamma, nu, rho, sigma):
        '''\
            Determine step size by Armijo's rule.

            Args:
                wk: current point wk=(xk, etak, lmdk, xik, Zk)
                dwk: Newton direction
                beta: backtracking parameter
                epsilon0: slope parameter
                mu: perturbation parameter
                gamma: initial step size parameter
                nu: barrier penalty parameter of semidefinite/conic/inequality\
                    constraints for merit functions
                rho: barrier penalty parameter of equality constraints for merit functions
                sigma: primal-dual barrier parameter for conic constraints
                
            Returns:
                alpha: step size
        '''

        xk, etak, lmdk, xik, Zk = wk
        dxk, detak, dlmdk, dxik, dZk = dwk

        if not self.l == 0:
            # Step size initial
            alphalmdk = - gamma / (min(dlmdk / lmdk))
            if alphalmdk < 0:
                alphalmdk = 1
        else:
            alphalmdk = 1

        if not self.q == 0:
            # Prepare
            conek = self.cone(xk)
            invconek = vec_power(conek, -1)
            invxik = vec_power(xik, -1)
            # Step size initial 
            alphaxik = gamma *\
                roots(array([detvec(dxik), 2 * xik @ R(self.q) @ dxik, detvec(xik)]))[1]
            if alphaxik < 0:
                alphaxik = 1
        else:
            alphaxik = 1
            invxik = array([])
        
        if not self.p == 0:
            # Prepare
            invZk = linalg.inv(Zk)
            # Step size initial
            alphaZk = - gamma / (min(linalg.eigvals(invZk @ dZk)).real)
            if alphaZk < 0:
                alphaZk = 1
        else:
            invZk = array([])
            alphaZk = 1

        # Initial step size
        alphak_bar = min(alphalmdk, alphaxik, alphaZk, 1)

        lk = 0
        while True:

            Fvalue = self.F_merit(xk, lmdk, xik, Zk, mu, nu, rho, sigma)
            dF = self.dFl(xk, dxk, lmdk, dlmdk, xik, dxik, Zk, dZk, invxik, invZk, mu, nu, rho, sigma)
            
            '''
            dF_approx = direc_F(xk, dxk, lmdk, dlmdk, xik, dxik, Zk, dZk, mu, nu, rho, sigma)       
            if not isclose(dF, dF_approx, atol=1.0e-2):
                print(f'dF: {dF}')
                print(f'approx.: {dF_approx}')
                raise ValueError('dF is not accurate.')
            '''
            ''' dF check does not work appropriately sometimes in the case where the sequence converge rapidly or sufficiently strong convex.'''
            '''
            if dF > 0: 
                #print(f'dF_approx: {dF_approx}')
                print(f'dF: {dF}')
                raise ValueError(f'directional derivative of the merit function is positive.')
            '''

            stepsize = alphak_bar * (beta ** lk)
            x_new = xk + stepsize * dxk
            if not self.l == 0:
                lmd_new = lmdk + stepsize * dlmdk
            else:
                lmd_new = 0
            if not self.q == 0:
                xi_new = xik + stepsize * dxik
            else:
                xi_new = 0
            if not self.p == 0:
                Z_new = Zk + stepsize * dZk
            else:
                Z_new = zeros([0])
            
            # Check whether new variables are not complex
            if (not all(isclose(x_new.imag, zeros_like(x_new))) or
            not all(isclose(lmd_new.imag, zeros_like(lmd_new))) or
            not all(isclose(xi_new.imag, zeros_like(xi_new))) or
            not all(isclose(Z_new.imag, zeros_like(Z_new)))):
                raise ValueError('Imaginary number exists')

            Fvalue_new = self.F_merit(x_new, lmd_new, xi_new, Z_new, mu, nu, rho, sigma)
            min_eig = min(linalg.eigvals(self.matX(x_new)))
            test1 = (Fvalue_new <= Fvalue + epsilon0 * stepsize * dF)
            test2 = (min_eig > 0)
            if not self.q == 0:
                test3 = (is_int_cone(self.cone(xk)))
            else:
                test3 = True
            if not self.l == 0:
                test4 = (all(self.cieq(xk) > 0))
            else:
                test4 = True
            if test1 and test2 and test3 and test4:
                break
            else:
                verbose = False
                if verbose:
                    if test1 == False:
                        print(f'[Almijo condition]: {test1}')
                        print(f'Fvalue_new: {Fvalue_new}')
                        print(f'{Fvalue + epsilon0 * stepsize * dF}')
                    if test2 == False:
                        print(f'[is_int_matrix_inequality]: {test2}')
                        print(f'eigmin: {min_eig}')
                    if test3 == False:
                        print(f'[is_int_cone]: {test3}')
                    if test4 == False:
                        print(f'[is_cieq_positive]: {test4}')
                        print(f'cieq(xk): {self.cieq(xk)}')

            lk += 1
            if stepsize < 1.0e-15:
                raise ValueError(
                    'Armijo does not work appropriately for the case where\n\
                    muk={}, nu={}, beta={}.'.format(mu,nu,beta))

        return stepsize


    # ---
    # Eigenvalue modification of Hessan
    # ---
    def eigmod(self, hess):
        if min(linalg.eigvals(hess)) <= 0:
            eigs, matEig = linalg.eig(hess)
            hess = matEig @ (diag(abs(eigs)) + 1.0e-4*eye(self.n)) @ matEig.T
        return hess


    # ---
    # Primal-dual merit function
    # ---
    def F_merit(self, x, lmd, xi, Z, mu, nu, rho, sigma):
        return self.Fbp(x, mu, rho) +\
            nu * self.Fpd(x, lmd, xi, Z, mu, sigma, num_cones=1)


    # ---
    # Primal barrier penalty function Fbp
    # ---
    def Fbp(self, x, mu, rho):

        if not self.m == 0:
            norm1_ceq = linalg.norm(self.ceq(x), ord=1)
        else:
            norm1_ceq = 0
        if not self.l == 0:
            cieq_val = self.cieq(x)
            logbar_cieq = sum(log(cieq_val))
        else:
            cieq_val = 0
            logbar_cieq = 0
        if not self.q == 0:
            conek = self.cone(x)
            logbar_cone = 1/2 * log(detvec(conek))
        else:
            conek = 0
            logbar_cone = 0
        if not self.p == 0:
            matXk = self.matX(x)
            logbar_matX = log(linalg.det(matXk))
        else:
            matXk = 0
            logbar_matX = 0


        return self.f(x) - mu * (logbar_cieq + logbar_cone + logbar_matX) +\
            rho * norm1_ceq


    # ---
    # Primal-dual barrier function
    # ---
    def Fpd(self, x, lmd, xi, Z, mu, sig, num_cones=1):
        '''
            Primal-dual barrier function for NSDP     
            
            Args:
                - primal variables:
                    x: primal variables
                - dual variables:
                    lmd: dual variables
                    xi: dual variables
                    Z: dual variables
                - others:
                    mu: barrier parameter
                    sig: barrier parameter
                    num_cones: number of cones
                    
            Returns:
                (float): Primal-dual barrier function value
        '''

        if not self.l == 0:
            cieq_val = self.cieq(x)
            pd_cieq = cieq_val @ lmd - mu * sum(log(cieq_val * lmd))
        else:
            cieq_val = 0
            pd_cieq = 0
        if not self.q == 0:
            conek = self.cone(x)
            scaling_term = (conek @ xi) / num_cones
            pd_cone = (num_cones + sig) * log(scaling_term + abs(scaling_term - mu)) -\
                1/2 * log(detvec(conek) * detvec(xi))
        else:
            conek = 0
            pd_cone = 0
        if not self.p == 0:
            matXk = self.matX(x)
            pd_matX = matIP(matXk, Z) - mu * log(linalg.det(matXk) * linalg.det(Z))
        else:
            matXk = 0
            pd_matX = 0
        
        return pd_cieq + pd_cone + pd_matX


    # ---
    # Directional derivative of approx. Fl of the merit function F_merit
    # ---
    def dFl(self, x, dx, lmd, dlmd, xi, dxi, Z, dZ, invxi, invZ, mu, nu, rho, sigma):
        '''
            Directional derivative of the merit function F_merit
            
            Args:
                - primal variables:
                    x: primal variables
                    dx: direction
                - dual variables:
                    lmd: dual variables
                    dlmd: direction of lmd
                    xi: dual variables
                    dxi: direction of xi
                    Z: dual variables
                    dZ: direction of Z
                    invxi: inverse of xi in the sense of Jordan algebra
                    invZ: inverse of Z
                    mu: barrier parameter
                    nu: barrier parameter
                    rho: barrier parameter
                    sigma: barrier parameter relating of conic constraints
        '''
        return self.dFbpl(x, dx, dlmd, mu, rho) +\
            nu * self.dFpdl(x, dx, lmd, dlmd, xi, dxi, Z, dZ,
               invxi, invZ, mu, sigma, num_cones=1)


    def dFbpl(self, x, dx, dlmdk, mu, rho):
        '''
            Directional derivative of barrier parameter function Fbp in (dx, dlmdk)
            
            Args:
                x: primal variables
                dx: direction of x
                dlmdk: direction of dual variables
                mu: barrier parameter
                rho: barrier parameter
                
            Returns:
                (float): Directional derivative of Fbp
        '''

        if not self.m == 0:
            norm1_dceq = linalg.norm(self.ceq(x) + self.A0(x).T @ dx, ord=1) - linalg.norm(self.ceq(x), ord=1)
        else:
            norm1_dceq = 0
            
        if not self.l == 0:
            cieq_val = self.cieq(x)
            inv_cieq = 1 / cieq_val
            nabla_cieq = self.B0(x)
            
            dlogbar_dcieq = (nabla_cieq.T @ dx) @ inv_cieq
        else:
            dlogbar_dcieq = 0
            
        if not self.q == 0:
            conek = self.cone(x)
            invconek = vec_power(conek, -1)

            dlogbar_dcone = invconek @ self.grad_cone(x).T @ dx
        else:
            dlogbar_dcone = 0

        if not self.p == 0:
            matXk = self.matX(x)
            invXk = linalg.inv(matXk)
            dXk = self.delta_X(x, dx)

            dlogbar_matX = trace(invXk @ dXk)
        else:
            dlogbar_matX = 0
       
        return self.grad_f(x) @ dx - mu * (dlogbar_dcieq + dlogbar_dcone + dlogbar_matX) - rho * norm1_dceq


    def dFpdl(self, x, dx, lmd, dlmd, xi, dxi, Z, dZ, invxi, invZ, mu, sigma, num_cones=1):
        '''
            Directional derivative of primal-dual barrier function Fpd in (dx, dlmdk, dxi, dZ)
            
            Args:
                x: primal variables
                dx: direction of x
                lmd: dual variables
                dlmd: direction of dual variables
                xi: dual variables
                dxi: direction of dual variables
                Z: dual variables
                dZ: direction of dual variables
                invxi: inverse of xi in the sense of Jordan algebra
                invZ: inverse of Z
                mu:
                sigma: 
                num_cones:

            Returns:
                (float): Directional derivative of Fpd
        '''

        if not self.l == 0:
            cieq_val = self.cieq(x)
            nabla_cieq = self.B0(x)
            dpl_cieq = (nabla_cieq.T @ dx) @ lmd - mu * lmd @ (nabla_cieq / cieq_val).T @ dx +\
            cieq_val @ dlmd - mu * dlmd @ (1 / lmd)
        else:
            dpl_cieq = 0

        if not self.q == 0:
            conek = self.cone(x)
            invconek = vec_power(conek, -1)
            dconek = self.grad_cone(x)
            dcone_xk = dconek.T @ dx
            conexi = conek @ xi
            dpl_cone = (num_cones + sigma) *\
                ((xi @ dcone_xk + conek @ dxi) / num_cones +\
                abs((conexi + xi @ dcone_xk + conek @ dxi) / num_cones - mu) - \
                abs(conexi / num_cones - mu)) / (conexi / num_cones + abs(conexi / num_cones - mu))\
                - (invconek @ dcone_xk + invxi @ dxi)
        else:
            dpl_cone = 0
        
        if not self.p == 0:
            matXk = self.matX(x)
            invXk = linalg.inv(matXk)
            dXk = self.delta_X(x, dx)
            
            dpl_matX = trace(dXk @ Z + matXk @ dZ - mu * (invXk @ dXk + invZ @ dZ))
        else:
            dpl_matX = 0

        return dpl_cieq + dpl_cone + dpl_matX


    # ---
    # Directional derivative of the merit function F_merit by the central difference method
    # ---
    def direc_F(self, x, dx, lmd, dlmd, xi, dxi, Z, dZ, mu, nu, rho, sigma):
        h = 1.0e-04
        Fval_ = self.F_merit(x+h*dx, lmd+h*dlmd, xi+h*dxi, Z+h*dZ, mu, nu, rho, sigma)
        Fval = self.F_merit(x, lmd, xi, Z, mu, nu, rho, sigma)
        return (Fval_ - Fval) / h


    @staticmethod
    def pause():
        print(f'Press Enter to Continue:')
        input()

