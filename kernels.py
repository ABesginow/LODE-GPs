import itertools
import torch
from torch.distributions import constraints
import torch
from functools import reduce
from gpytorch.lazy import *
from gpytorch.lazy.non_lazy_tensor import lazify
from gpytorch.kernels.kernel import Kernel
from pyro.nn.module import PyroParam
from pyro.nn.module import PyroModule
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
from sage.arith.misc import factorial
import numpy as np
import pdb

DEBUG = False
class SageExpression(Kernel):

    def __init__(self, input_dim, base_fkt :sage.symbolic.expression.Expression, hyperparameters:dict=None, var1=var('x1'), var2=var('x2'), active_dims=None):
        super().__init__(input_dim, active_dims)
        self.base_cov = base_fkt
        var_values = {}
        for param in self.base_cov.variables():
            if param == var1 or param == var2:
                continue
            setattr(self, str(param), PyroParam(torch.tensor(float(hyperparameters[str(param)]), requires_grad=True) if str(param) in hyperparameters.keys() else torch.tensor(float(1.), requires_grad=True), constraints.positive))
            var_values[str(param)] = getattr(self, str(param))
        self.params = hyperparameters
        self.var1 = var1
        self.var2 = var2
        add_vars = [str(key) for key in self.params.keys()]
        add_vars.extend([str(self.var1), str(self.var2)])
        RPN = shunting(str(self.base_cov), additional_variables = add_vars)
        print(f"str(self.base_cov): {str(self.base_cov)}")
        print(f"add_vars:{add_vars}")
        print(f"RPN:{RPN}")
        print(f"str(self.var1):{str(self.var1)}")
        print(f"str(self.var2):{str(self.var2)}")
        print(f"var_values:{var_values}")
        func_str = reconstruct(RPN, additional_variables = add_vars, x1_var=str(self.var1), x2_var=str(self.var2), var_values=var_values)
        print(func_str)
        # This is incredibly unsafe
        # TODO: Write a proper (and fast) function generator that returns a callable (lambda?) expression
        exec(f"self.evaluate = lambda self, v1, v2: {func_str}")
        self.last_X = None
        self.last_prepared_X = None

    #TODO: If I train the hyperparameters of the kernel and then diff it, it will lose all progress!
    # TODO: Rechtsseitige Ableitung beachten / einbauen (In der "Oberklasse" beachten)
    def derive(self, d_poly : sage.symbolic.expression.Expression=var('d'), d_var=var('d'), e_var=var('x')):
        result = None
        # if the derivatives are just integers
        if type(d_poly) == sage.rings.integer.Integer or d_poly == 0:
            return SageExpression(d_poly*self.base_cov, self.params)
        if not type(d_poly) == sage.symbolic.expression.Expression:
            assert "Derivative expression is neither sage.symbolic.expression.Expression nor sage.rings.integer.Integer"
        # Catching the case of derivatives being d^n which causes derivatives.operands() to be the list [d, n] instead of [d^n]
        w0 = SR.wild()
        if (not all(op.has(d_var) for op in d_poly.operands()) and len(d_poly.operands()) == 2) or (d_poly.has(d_var) and len(d_poly.operands()) == 0):
            temp = self.base_cov
            while d_poly.has(d_var):
                if type(temp) == sage.rings.integer.Integer:
                    return SageExpression(0, self.params)
                temp = temp.diff(e_var)
                # decrease by one deriv
                d_poly = d_poly/d_var
            # If this is the first entry, replace (otherwise getting Type-Error)
            if result == None:
                result = temp
            else:
                result += temp
            return SageExpression(result, self.params)

        # Iterate over all the operands
        for operand in d_poly.operands():
            temp = self.base_cov
            # If the operand does not contain a "d" it is a constant
            if not operand.has(d_var):
                temp = temp*operand
            # If it contains a "d" it is a derivative
            while operand.has(d_var):
                if type(temp) == sage.rings.integer.Integer:
                    return SageExpression(0, self.params)
                temp = temp.diff(e_var)
                operand = operand/d_var
            if result == None:
                result = temp
            else:
                result += temp
        return SageExpression(result, self.params)

    def base_cov(self):
        return self.base_cov

    def set_hyperparameters(self, hyperparameters:dict):
        self.params = hyperparameters

    def forward(self, X, Z=None):
        if Z == None:
            Z = X
        #pdb.set_trace()
        #K_0 = X-Z


    """
    # For Testing out the speed with a profiler (exec-based approach) ==Only for debug purposes==
    def forward(self, X, Z=None):
        if Z == None:
            Z = X
        if not self.last_X is None and torch.all(torch.eq(X, self.last_X)):#id(X) == id(self.last_X):
            prepared_X = self.last_prepared_X
        else:
            \"""
            Result in matrix of the form
            [[(1, 11), (1, 12), (1, 13)],
             [(2, 11), (2, 12), (2, 13)],
             [(3, 11), (3, 12), (3, 13)]]
            with
            X = [1, 2, 3]
            Z = [11, 12, 13]
            \"""
            self.last_X = X
            prepared_X = [(entry_z, entry_x) for entry_x, entry_z in product(Z,X)]
            self.last_prepared_X = prepared_X
        #result = [[0 for i in range(len(Z))] for j in range(len(X))]
        #pdb.set_trace()
        result  = []
        # For debug purposes
        for v1, v2 in prepared_X:
            unused_var = torch.tensor(float(24))
            garbage = torch.pow(unused_var, torch.tensor(float(2)))
            temp = torch.sub((float(v1)), (float(v2)))
            temp = torch.pow(temp, float(2))
            part1 = torch.pow(self.l, (float(2)))
            temp = torch.div(temp, part1)
            temp = torch.mul(torch.div(float(-1), float(2)), temp)
            temp = torch.exp(temp)
            part2 = torch.pow(self.sigma, float(2))
            r = torch.mul(part2, temp)
            result.append(r)
        # Calculate the matrix directly instead of entrywise
        #result = [self.evaluate(self, elem[0], elem[1]) for elem in prepared_X]
        result = torch.reshape(torch.Tensor(result), (len(Z), len(X)))
        return result

    # loop-based approach to calculate the individual entries of the list (sagemath.subs-based)
    def forward(self, X, Z=None):
        if Z == None:
            Z = X
        result = [[0 for i in range(len(Z))] for j in range(len(X))]
        #pdb.set_trace()
        param_dict = {}
        for param in self.base_cov.variables():
            if param == self.var1 or param == self.var2:
                continue
            param_dict[param] = float(getattr(self, str(param)))
        for i, a in enumerate(X):
            for j, b in enumerate(Z):
                #pdb.set_trace()
                param_dict[self.var1] = float(a.float())
                param_dict[self.var2] = float(b.float())
                result[i][j] = float(self.base_cov.subs(param_dict))
                if not len(param_dict) == len(self.base_cov.variables()):
                    assert "Number of parameters doesn't match required variables"
        return torch.tensor(result)
   """



class Diff_SE_kernel(Kernel):

    asym_sign_matr = [[int(1), int(1), int(-1), int(-1)], [int(-1), int(1), int(1), int(-1)], [int(-1), int(-1), int(1), int(1)], [int(1), int(-1), int(-1), int(1)]]


    def __init__(self,  var=None, length=None, active_dims=None):
        super().__init__(active_dims=active_dims)
        self.is_diffable = True
        setattr(self, 'var', torch.nn.Parameter(torch.tensor(float(var))
                                                if not var is None else
                                                torch.tensor(float(1.)),
                                                requires_grad=True))
        setattr(self, 'length', torch.nn.Parameter(torch.tensor(float(length))
                                                   if not length is None else
                                                   torch.tensor(float(1.)),
                                                   requires_grad=True))
        self.K_0 = None
        self.K_1 = None
        self.K_4 = None


    # Written for the asymmetric (general) case
    def single_term_extract(self, d_poly, d_var=var('d')):
        """
        Returns the degree and the coefficient (either as tensor or as a parameter)
        """
        deriv_dict = {}
        degree = int(d_poly.degree(d_var))
        # See if it's of the form a*x^n
        if (not len(d_poly.operands()) == 0) and ('^' in str(d_poly.operands()[0]) or '^' in str(d_poly.operands()[1])):
            # 1 if the coefficient is in [1], else it must be in [0]
            coeff_index = int('^' in str(d_poly.operands()[0]))
            coeff = None
            if not d_poly.operands()[coeff_index].is_numeric():
                # If it doesn't exist, a trainable parameter with initial value 1 is created
                if not hasattr(self, str(d_poly.operands()[coeff_index])):
                    setattr(self,  str(d_poly.operands()[coeff_index]),
                            torch.nn.Parameter(torch.tensor(float(1.)),
                            requires_grad=True))
                coeff = getattr(self, str(d_poly.operands()[coeff_index]))
            else:
                coeff = torch.tensor(float(d_poly.operands()[coeff_index]))
        # Else it's of the form x^n
        else:
            coeff = torch.tensor(float(1.))
        return degree, coeff


    def prepare_asym_deriv_dict(self, left_poly, right_poly, left_d_var=var('dx1'), right_d_var=var('dx2')):
        # Will be filled as follows: [{'d^o': 4, 'd^p': 2, 'coeff':[a, b]}, {'d^o': 3, 'd^p': 3, 'coeff':[1, c]}, ...]
        deriv_list = []
        # simulate multiplication of expressions
        # TODO vllt mal mit product() versuchen und vorher die Listen vorbereiten?

        # Check if either left or right 'polynomial' is just an Integer
        # since this is treated differently by sage
        if type(left_poly) in [sage.rings.integer.Integer,
                               sage.rings.real_mpfr.RealLiteral]:
            left_iteration_list = [left_poly]
        # If it has len == 0 it is a single element expression and produces
        # empty .operands() list
        elif len(left_poly.operands()) == 0:
            left_iteration_list = [left_poly]
        else:
            left_iteration_list = left_poly.operands()
        for left in left_iteration_list:
            if type(right_poly) in [sage.rings.integer.Integer,
                                    sage.rings.real_mpfr.RealLiteral]:
                right_iteration_list = [right_poly]
            elif len(right_poly.operands()) == 0:
                right_iteration_list = [right_poly]
            else:
                right_iteration_list = right_poly.operands()

            for right in right_iteration_list:
                left_right = [left, right]
                # Check if the derivatives are just numbers
                # ---
                # Note: Difference between is_numeric() and this typecheck:
                # Only if a number is of type Expression do they have the
                # 'is_numeric()' function therefore I can't use that here, but
                # I can in the single term extraction function
                # ---
                left_right_number_bool = [type(left) in
                                          [sage.rings.real_mpfr.RealLiteral,
                                           sage.rings.integer.Integer],
                                          type(right) in
                                          [sage.rings.real_mpfr.RealLiteral,
                                           sage.rings.integer.Integer]]
                if all(left_right_number_bool):
                    deriv_list.append({'d^o':0, 'd^p':0, 'coeff':[float(left), float(right)]})
                if not type(left) == sage.symbolic.expression.Expression and not type(right) == sage.symbolic.expression.Expression:
                    assert "Derivative expression is neither sage.symbolic.expression.Expression nor number"
                # Catching the case of derivatives being d^n which causes derivatives.operands() to be the list [d, n] instead of [d^n]
                # The below steps will also recognize exponential expressions of the form n^x
                # this will result in unexpected behaviour or exit with an error
                deriv_entry = {'d^o':0, 'd^p':0, 'coeff':[]}
                for d_poly, d_var in zip(left_right, [left_d_var, right_d_var]):
                    # Check if either left or right is number
                    # -> make coefficient
                    if type(d_poly) in [sage.rings.integer.Integer,
                                        sage.rings.real_mpfr.RealLiteral]:
                        degr = 0
                        coeff = torch.tensor(float(d_poly))
                    elif ((not all(op.has(d_var) for op in d_poly.operands())
                           and len(d_poly.operands()) == 2)
                              or (d_poly.has(d_var)
                                  and len(d_poly.operands()) == 0)):
                        degr, coeff = self.single_term_extract(d_poly, d_var)
                    # If the operand does not contain a "d" it is a constant
                    # TODO: This should never happen since the first if
                    # catches this case
                    elif not d_poly.has(d_var):
                        degr = 0
                        coeff = torch.tensor(float(d_poly))
                    else:
                        # If it contains a "d" it is a derivative
                        # "append" the new dict to the previous dict
                        degr, coeff = self.single_term_extract(d_poly, d_var)
                    if d_var == left_d_var:
                        deriv_entry['d^o'] = degr
                    else:
                        deriv_entry['d^p'] = degr
                    deriv_entry['coeff'].append(coeff)
                deriv_list.append(deriv_entry)
        return deriv_list


    def diff(self, left_poly, right_poly, left_d_var=var('dx1'), right_d_var=var('dx2')):

        derivation_term_dict = self.prepare_asym_deriv_dict(left_poly, right_poly, left_d_var, right_d_var)
        class diffed_SE_kernel(Kernel):
            asym_sign_matr = [[int(1), int(1), int(-1), int(-1)], [int(-1), int(1), int(1), int(-1)], [int(-1), int(-1), int(1), int(1)], [int(1), int(-1), int(-1), int(1)]]

            def __init__(self,  var=None, length=None, active_dims=None):
                super().__init__(active_dims=active_dims)
                setattr(self, 'var', torch.nn.Parameter(torch.tensor(float(var))
                                                        if not var is None else
                                                        torch.tensor(float(1.)),
                                                        requires_grad=True))
                setattr(self, 'length', torch.nn.Parameter(torch.tensor(float(length))
                                                           if not length is None else
                                                           torch.tensor(float(1.)),
                                                           requires_grad=True))
                self.K_0 = None
                self.K_1 = None
                self.K_4 = None

            def coeffs(self, given_n):
                # See http://oeis.org/A096713
                real_n = int(given_n/2)
                m, k = var('m, k')
                # even
                # T(2*m, k) = (-1)^(m+k)*(2*m)!*2^(k-m)/((m-k)!*(2*k)!), k = 0..m.
                if given_n % 2 == 0:
                    # This notation is only valid in iPython
                    #T(m,k) = factorial(2*m)*2^(k-m)/(factorial(m-k)*factorial(2*k))
                    # As an actual Python file I need to use:
                    T = lambda m, k : factorial(2*m)*2**(k-m)/(factorial(m-k)*factorial(2*k))
                # odd
                # T(2*m+1, k) = (-1)^(m+k)*(2*m+1)!*2^(k-m)/((m-k)!*(2*k+1)!), k = 0..m. (End)
                else:
                    # See above
                    #T(m,k) = factorial(2*m+1)*2^(k-m)/(factorial(m-k)*factorial(2*k+1))
                    T = lambda m, k: factorial(1*m+1)*2**(k-m)/(factorial(m-k)*factorial(2*k+1))

                return [int(T(real_n, k)) for k in range(real_n+1)]


            def _slice_input(self, X):
                r"""
                Slices :math:`X` according to ``self.active_dims``. If ``X`` is 1D then returns
                a 2D tensor with shape :math:`N \times 1`.
                :param torch.Tensor X: A 1D or 2D input tensor.
                :returns: a 2D slice of :math:`X`
                :rtype: torch.Tensor
                """
                if X.dim() == 2:
                    return X[:, self.active_dims]
                elif X.dim() == 1:
                    return X.unsqueeze(1)
                else:
                    raise ValueError("Input X must be either 1 or 2 dimensional.")

            def _square_scaled_dist(self, X, Z=None):
                r"""
                Returns :math:`\|\frac{X-Z}{l}\|^2`.
                """
                if Z is None:
                    Z = X
                if not X.shape[1] == 1:
                    X = self._slice_input(X)
                if not Z.shape[1] == 1:
                    Z = self._slice_input(Z)
                if X.size(int(1)) != Z.size(int(1)):
                    raise ValueError("Inputs must have the same number of features.")

                #scaled_X = X / self.length
                #scaled_Z = Z / self.length
                X2 = (X ** 2).sum(1, keepdim=True)
                Z2 = (Z ** 2).sum(1, keepdim=True)
                XZ = X.matmul(Z.t())
                self.K_0 = X-Z.t()
                r2 = X2 - 2 * XZ + Z2.t()
                self.K_1 = r2.clamp(min=int(0))



            def forward(self, x1, x2, diag=False, **params):
                self.result_term = lambda self, l_, coefficients, i, sign, l_exponents, K_1_exponents: \
                coefficients[i]*(sign*(int(-1)**i))*(l_**l_exponents[i])*(self.K_0**K_1_exponents[i])

                self._square_scaled_dist(x1, x2)
                self.K_4 = torch.mul(self.var, torch.exp(float(-0.5) * self.K_1*(float(1)/self.length**float(2))))

                #return self.K_1, self.K_4, self.length

                result = None
                for term in derivation_term_dict:
                    degr_o = term['d^o']
                    degr_p = term['d^p']
                    poly_coeffs = term['coeff']
                    sign = self.asym_sign_matr[int(degr_o)%int(4)][int(degr_p)%int(4)]
                    l_exponents = [np.ceil((degr_o+degr_p)/int(2)) + i for i in range(int((degr_o+degr_p)/2)+int(1))]
#                artificial_degree = np.ceil((degr_o+degr_p)/int(2))
                    K_1_exponents = [int(i*2) if int(degr_o+degr_p)%2 == 0 else int(i*int(2)+int(1)) for i in range(int((degr_o+degr_p)/2)+int(1))]
                    coefficients = self.coeffs(int(degr_o+degr_p))
                    if DEBUG:
                        print(f"(x1-x2)^i : {K_1_exponents}")
                        print(f"Coefficients: {coefficients}")
                        print(f"Starting sign: {sign}")
                        print(f"l^(2*N) : {l_exponents}")
                    l_ = float(1)/self.length**(float(2))
                    if result is None:
                        temp = [self.result_term(self, l_, coefficients, i, sign, l_exponents, K_1_exponents=K_1_exponents) for i in range(int((degr_o+degr_p)/2)+int(1))]
                        result = sum(temp)*poly_coeffs[int(0)]*poly_coeffs[int(1)]
                    else:
                        #int(degr_o+degr_p) if int(degr_o+degr_p)%2 == 0 else int(degr_o+degr_p-1)
                        result += sum([self.result_term(self, l_, coefficients, i, sign, l_exponents, K_1_exponents=K_1_exponents) for i in range(int((degr_o+degr_p)/2)+int(1))])*poly_coeffs[0]*poly_coeffs[int(1)]
                return self.K_4*result
        return diffed_SE_kernel(var=self.var, length=self.length, active_dims=self.active_dims)


    def _square_scaled_dist(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|^2`.
        """
        if Z is None:
            Z = X
        X = self._slice_input(X)
        Z = self._slice_input(Z)
        if X.size(int(1)) != Z.size(int(1)):
            raise ValueError("Inputs must have the same number of features.")

        #scaled_X = X / self.length
        #scaled_Z = Z / self.length
        X2 = (X ** 2).sum(1, keepdim=True)
        Z2 = (Z ** 2).sum(1, keepdim=True)
        XZ = X.matmul(Z.t())
        self.K_0 = X-Z.t()
        r2 = X2 - 2 * XZ + Z2.t()
        self.K_1 = r2.clamp(min=int(0))


    def _diag(self, X):
        """
        Calculates the diagonal part of covariance matrix on active features.
        """
        return self.var.expand(X.size(0))


    def _slice_input(self, X):
        r"""
        Slices :math:`X` according to ``self.active_dims``. If ``X`` is 1D then returns
        a 2D tensor with shape :math:`N \times 1`.
        :param torch.Tensor X: A 1D or 2D input tensor.
        :returns: a 2D slice of :math:`X`
        :rtype: torch.Tensor
        """
        if X.dim() == 2:
            return X[:, self.active_dims]
        elif X.dim() == 1:
            return X.unsqueeze(1)
        else:
            raise ValueError("Input X must be either 1 or 2 dimensional.")

    def forward(self, x1, x2, diag=False, **params):
        var = torch.nn.functional.relu(self.var)
        length = torch.nn.functional.relu(self.length)
        if x2 == None:
            x2 = x1

        if diag:
            return self._diag(X)

        self._square_scaled_dist(x1, x2)
        self.K_4 = var * torch.exp(-0.5 * self.K_1/(length**2))

        #if all(torch.eq(X, Z)) and not all(entry < float(0.00001) for entry in (self.K_4-torch.transpose(self.K_4, int(0), int(1))).flatten()):
        #    print(self.K_4)
        #    assert "Cov. Matr is not symmetric"
        #if X.shape == Z.shape and all(torch.eq(X, Z)):
            #print(f"X:\n {X}")
            #print(f"Z:\n {Z}")
            #print(f"K_4:\n {self.K_4}")
            #print(torch.eig(self.K_4)[0])
        #    assert all(entry < float(0.00001) for entry in (self.K_4-torch.transpose(self.K_4, int(0), int(1))).flatten()), "Covariance matrix is not symmetric"
        #    assert all(entry[0] >= -0.0001 for entry in torch.eig(self.K_4)[0]), "Eigenvalues contain negative values"

        return self.K_4 + torch.eye(len(self.K_4)) * 1e-4



class MatrixKernel(Kernel):


    def __init__(self, matrix, active_dims=None):
        super().__init__(active_dims=active_dims)
        self.num_tasks = np.shape(matrix)[0]
        if not np.shape(matrix)[0] == np.shape(matrix)[1]:
            assert "Kernel matrix is not square"
        self.matrix = matrix
        # check if matrix is symmetrical (after init throw in random values and
        # check for symmetry & eigenvalues)
        for i, row in enumerate(self.matrix):
            for j, kernel in enumerate(row):
                if kernel is None or kernel == 0:
                    pass
                else:
                    setattr(self, f'kernel_{i}{j}', kernel)

    # TODO aktualisieren
    def _diag(self, X):
        """
        Calculate the diagonal part for each of the kernels and construct a diagonal matrix
        """
        #TODO debug
        if X.ndim == 1:
            H_x = np.shape(X)[0]

        result = None
        for i, kernel in enumerate(self.matrix):
            result1 = kernel.forward(X, diag=True)

            # append horizontally
            if result is None:
                result = result1
            else:
                # append vertically
                result = torch.cat((result, result1), int(0))
        return result

    def forward(self, x1, x2, diag=False, **params):
        if x2 == None:
            x2 = x1
        if not x1.ndim == 1:
            x1 = x1.flatten()
        if not x2.ndim == 1:
            x2 = x2.flatten()
        H_z = np.shape(x2)[0]
        H_x = np.shape(x1)[0]

        if diag:
            return self._diag(x1)
        zero_matrix = torch.zeros(H_x, H_z)
        #zero_matrix = torch.tensor([[int(0) for i in range(H_z)] for j in range(H_x)])
        result = None
        for i, row in enumerate(self.matrix) :
            temp = None
            for j, kernel in enumerate(row):
            # Create the result matrix
                if kernel is None or kernel == 0:
                    result1 = zero_matrix
                else:
                    result1 = kernel.forward(x1, x2)
                if temp is None:
                    temp = result1
                else:
                    #temp = CatLazyTensor(*[temp, result1])
                    if type(temp) == torch.Tensor and type(result1) == torch.Tensor:
                        temp = torch.hstack([temp, result1])
                    else:
                        temp = torch.hstack([temp.evaluate(), result1.evaluate()])

            # append vertically
            if result is None:
                result = temp
            else:
                #result = CatLazyTensor(*[result, temp], dim=1)
                if type(temp) == torch.Tensor and type(result) == torch.Tensor:
                    result = torch.vstack([result, temp])
                else:
                    result = torch.vstack([result.evaluate(), temp.evaluate()])

        print(result)
        result = torch.vstack([torch.hstack([result[k::H_x, l::H_x] for l in range(H_x)]) for k in range(H_x)])
        return result

    def num_outputs_per_input(self, x1, x2):
        return self.num_tasks


class DiffMatrixKernel(MatrixKernel):

    def __init__(self, matrix, active_dims=None):
        if not all([k == 0 or k is None or k.is_diffable for row in matrix for k in row]):
            assert "Not all kernels are differentiable"
        super().__init__(matrix, active_dims=active_dims)

    def calc_cell_diff(self, L, M, R):
        len_M = len(M)
        temp = None
        M_transpose = list(
            map(list, itertools.zip_longest(*M, fillvalue=None)))
        for j in range(len_M):
            for r_elem in R:
                for l_elem in L:
                    if l_elem is None or l_elem == 0:
                        import pdb
                        pdb.set_trace()
                    if temp is None:
                        if M_transpose[int(j/len_M)][j % len_M] is not None:
                            temp = M_transpose[int(j/len_M)][j % len_M].diff(left_poly=l_elem, right_poly=r_elem)
                        else:
                            pass
                    else:
                        if M_transpose[int(j/len_M)][j % len_M] is not None:
                            temp += M_transpose[int(j/len_M)][j % len_M].diff(left_poly=l_elem, right_poly=r_elem)
                        else:
                            pass

#            if temp is None:
#                # https://stackoverflow.com/questions/6473679/transpose-list-
#                # of-lists
#                M_transpose = list(
#                    map(list, itertools.zip_longest(*M, fillvalue=None)))
#                # temp is the derivative applied on the j-th element
#                temp = [M_transpose[int(j/len_M)][j % len_M].diff(
#                    left_poly=L[k], right_poly=R[j])
#                        for k in range(len(L))]
#            else:
#                temp += [M_transpose[int(j/len_M)][j % len_M].diff(
#                    left_poly=L[k], right_poly=R[j])
#                         for k in range(len(L))]
        return temp

    def diff(self, left_matrix=None, right_matrix=None):
        # iterate left matrix by rows and right matrix by columns and call the
        # respective diff command of the kernels with the row/cols as params
        output_matrix = [[0 for i in range(np.shape(self.matrix)[1])] for j in range(np.shape(self.matrix)[0])]
        for i, (l, r) in enumerate(zip(left_matrix.rows(), right_matrix.columns())):
            res = self.calc_cell_diff(l, self.matrix, r)
            output_matrix[int(i/np.shape(self.matrix)[0])][
                        int(i % np.shape(self.matrix)[0])]  = res

        return MatrixKernel(output_matrix)
