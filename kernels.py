import itertools
from itertools import zip_longest
import torch
from torch.distributions import constraints
import torch
from functools import reduce
from gpytorch.lazy import *
from gpytorch.lazy.non_lazy_tensor import  lazify
from gpytorch.kernels.kernel import Kernel
from pyro.nn.module import PyroParam
from pyro.nn.module import PyroModule
from sage.all import *
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
from sage.arith.misc import factorial
import numpy as np
import pdb
from gpytorch.constraints import Positive


DEBUG =False
def make_symmetric(matrix):
    # matrix can either be list or torch.Tensor
    if not type(matrix) in [torch.Tensor, list]:
        assert "Can only deal with 'torch.Tensor' and 'list'"
    if type(matrix) is list:
        # Check that it is 2D
        if not len(np.shape(matrix)) == 2 and not np.shape(matrix)[0] == np.shape(matrix)[1]:
            assert "List is not 2D or dimensions are not of equal size"
        row_len = np.shape(matrix)[0]
    if type(matrix) is torch.Tensor:
        if not matrix.dim() == 2 and not matrix.shape[0] == matrix.shape[1]:
            assert "Tensor is not 2D or dimensions are not of equal size"
        row_len = matrix.shape[0]
    # Size for iteration to make it in single loop
    size = row_len**2
    for i in range(size):
        matrix[i % row_len][int(i/row_len)] = matrix[int(i/row_len)][i % row_len]

    return matrix




    # Written for the asymmetric (general) case
def single_term_extract(d_poly, context, d_var=var('d')):
    """
    Returns the degree and the coefficient (either as tensor or as a parameter)
    """
    assert context is not None, "Context must be specified"
    # Check the polynomial is just a number
    if not type(d_poly) in [sage.rings.integer.Integer, sage.rings.real_mpfr.RealLiteral, sage.symbolic.expression.Expression]:
        return 0, [torch.tensor(float(d_poly))]
    degree = int(d_poly.degree(d_var))
    coeff = []
    # It's of the form x^n or x
    if ((len(d_poly.operands()) == 2 and ('^' in str(d_poly) or '**' in str(d_poly))) or ((len(d_poly.operands()) == 0) and d_poly.has(d_var))) and not '*' in str(d_poly):
        coeff.append(torch.tensor(float(1.)))
    # It's of the form a*b*...*x^n or a*b*...*x -> extract the coefficients
    elif (not len(d_poly.operands()) == 0):
        for item in d_poly.operands():
            # Check if power or d_var is in item and skip that
            if ('^' in str(item) or '**' in str(item)) or item.has(d_var):
                continue
            # if coefficient is a variable, create torch parameter
            # (if it doesn't exist already)
            if not item.is_numeric():
                # If it doesn't exist, a trainable parameter with initial value 1 is created
                if not hasattr(context, str(item)):
                    setattr(context,  str(item),
                            torch.nn.Parameter(torch.tensor(float(1.)),
                            requires_grad=True))
                coeff.append(getattr(context, str(item)))
            # if coefficient is constant, float() it
            else:
                coeff.append(torch.tensor(float(item)))
    # Check if d_poly is just a variable/number
    elif len(d_poly.operands()) == 0 and not d_poly.has(d_var):
        if d_poly.is_numeric():
            coeff.append(torch.tensor(float(d_poly)))
        else:
            if not hasattr(context, str(d_poly)):
                setattr(context,  str(d_poly),
                        torch.nn.Parameter(torch.tensor(float(1.)),
                        requires_grad=True))
            coeff.append(getattr(context, str(d_poly)))
    return degree, coeff


def extract_operand_list(polynomial, d_var):
    """
    Cases to take care of
     - [x] Simply int/float values as either right or left poly
     - [x] Simply coefficients of sage number types
     - [x] Simply exponent term without a power (e.g. left =  x1)
     - [x] Coefficient (variable or number) times x (e.g. left = 4*x)
       (Note: left.operands() has the same result as for left=x^4)
     - [x] Simply exponent terms (e.g. left = x1^4)
     - [x] List of coefficients of sage number types
     - [x] Polynomials with addition in either left or right poly
       (e.g. left = a*x1^3 + b*x1^2)
     - [x] Polynomials without addition in either left or right poly
       (e.g. left= a*x1^3)
     - [x] Other cases should all cause errors
    """
    """
    Known cases that pass but produce wrong results:
    left =  (a+1)*dx1^3
    right = (a+b)dx2
    #TODO make them produce errors
    """
    # This is a hack to enable working with variables (due to matrices being
    # constructed from a polynomial ring with function fields etc.)
    from sage.symbolic.ring import SymbolicRing
    SR = SymbolicRing()
    polynomial = SR(str(polynomial))

    # Check for things like (a+b)*x^n and e^3*x^n
    if type(polynomial) in [sage.symbolic.expression.Expression] and any(not(not op in [sage.rings.integer.Integer, sage.rings.real_mpfr.RealLiteral] and op.has(d_var)) and ('+' in str(op) or '**' in str(op) or '^' in str(op)) for op in polynomial.operands()):
        raise Exception(f"Format unknown, polynomial required for differentiation. \n{str(polynomial)}")

    # Check if polynomial is an int/float -> List of operands is just the
    # number
    if type(polynomial) in [int, float]:
        list_of_operands = [polynomial]

    # Check if polynomial is just an
    # Integer/RealLiteral since this is treated differently by sage
    # -> List of operands is just the Integer/RealLiteral
    elif type(polynomial) in [sage.rings.integer.Integer,
                           sage.rings.real_mpfr.RealLiteral]:
        list_of_operands = [polynomial]

    # Check if it's just an exponent term without a power by checking length
    # If it has len == 0 it is a single element expression and produces
    # empty .operands() list
    elif type(polynomial) is sage.symbolic.expression.Expression and len(polynomial.operands()) == 0 and polynomial.has(d_var):
        list_of_operands = [polynomial]

    # IF polynomial.operands() is non empty AND every element has
    # left_d_var -> sum of deriv expressions
    elif '+' in str(polynomial) and type(polynomial) is sage.symbolic.expression.Expression:
    #elif type(polynomial) is sage.symbolic.expression.Expression and (all(op.has(d_var) for op in polynomial.operands()) and len(polynomial.operands()) >= 2):
        list_of_operands = polynomial.operands()



    # Check if there is a coefficient times/to the power of x
    # Note: Both 4*x and x^4 will produce the same output via
    # polynomial.operands() i.e. [4, x]
    # -> Pass as is and handle this on a higher level
    elif type(polynomial) is sage.symbolic.expression.Expression and len(polynomial.operands()) == 2 and polynomial.has(d_var):
        list_of_operands = [polynomial]

    # Handles things of the form a*x^n which are not sums of deriv expressions
    # which is handled above
    # -> Pass as is and handle this on a higher level
    elif type(polynomial) is sage.symbolic.expression.Expression and any(('^' in str(op) and op.has(d_var)) for op in polynomial.operands()) and len(polynomial.operands()) >= 2:
        list_of_operands = [polynomial]

    # Check if there are only coefficients of sage number types
    # -> return as is and handle this on a higher level
    elif type(polynomial) is sage.symbolic.expression.Expression and all(type(op) in [sage.rings.integer.Integer, sage.rings.real_mpfr.RealLiteral, sage.symbolic.expression.Expression] for op in polynomial.operands()) and not any(op.has(d_var) for op in polynomial.operands()):
        list_of_operands = [polynomial]

    else:
         print(type(polynomial))
         raise Exception(f"Format unknown, polynomial required for differentiation. \n{str(polynomial)}")

    return list_of_operands


def prepare_asym_deriv_dict(left_poly, right_poly, context, left_d_var=var('dx1'), right_d_var=var('dx2')):
    assert context is not None, "Context must be specified"
    # Will be filled as follows: [[[var_left, var_right], d^left, d^right], ...]
    # [[[0, a], 1, 3], [[b, 17], 0, 0], [[c, d], 17, 42], ...]
    deriv_list = []
    left_iteration_list = extract_operand_list(left_poly, left_d_var)
    right_iteration_list = extract_operand_list(right_poly, right_d_var)
    left_right = itertools.product(left_iteration_list, right_iteration_list)
    for left, right in left_right:
        left_exponent, left_coeffs = single_term_extract(left, context, left_d_var)
        right_exponent, right_coeffs = single_term_extract(right, context, right_d_var)
        all_coeffs = left_coeffs.copy()
        all_coeffs.extend(right_coeffs)
        deriv_list.append([all_coeffs, left_exponent, right_exponent])
    return deriv_list



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


class exp_kernel(Kernel):
    """
    Implements a kernel of the form exp^(a*(t1+t2)) where a is a number/variable
    :param [torch.nn.Parameter, torch.Tensor] factors: Coefficient 'a' in
    reverse polish notation
    """
    def __init__(self, factor, coeff_exponent,  active_dims=None):
        super().__init__(active_dims=active_dims)
        self.coeff = factor
        self.exp_coeff = coeff_exponent

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


    def forward(self, X, Z=None):
        if len(X.shape) == 1:
            X = self._slice_input(X)
        if Z is None:
            Z = X
        elif len(Z.shape) == 1:
            Z = self._slice_input(Z)
        if X.size(int(1)) != Z.size(int(1)):
            raise ValueError("Inputs must have the same number of features.")
        x1_plus_x2 = X+Z
        return torch.exp(x1_plus_x2)


    def diff(self, left_poly, right_poly, left_d_var=var('dx1'), right_d_var=var('dx2'), parent_context=None):
        # TODO check if id(self) is in parent_context.named_kernel_list and depending on yes/no add variance/lengthscale as parameters or not
        # If they already exist, take the adresses of the parent hyperparameters and make the diffed_SE_kernel parameters be references to these adresses
        #if not parent_context is None:
        diffed_kernel = diffed_exp_kernel(self.coeff, self.exp_coeff, active_dims=self.active_dims)
        diffed_kernel.set_l_poly(left_poly)
        diffed_kernel.set_r_poly(right_poly)
        diffed_kernel.set_base_kernel(self)
        if parent_context is None:
            parent_context = diffed_kernel
        derivation_term_list = prepare_asym_deriv_dict(left_poly, right_poly, parent_context, left_d_var, right_d_var)
        derived_form_list = []
        for term in derivation_term_list:
            # term will have the form [[coeff1, coeff2, ...], exponent of dx1, exponent of dx2]
            degr_x1 = term[1]
            degr_x2 = term[2]
            poly_coeffs = term[0]
            derived_form_list.append([poly_coeffs, degr_x1+degr_x2])
            # [coeffs, degr_x1+degr_x2]
        diffed_kernel.set_derivation_coefficients_list(derived_form_list)
        diffed_kernel.set_derivation_term_dict(derivation_term_list)
        return diffed_kernel


class diffed_exp_kernel(Kernel):
        def __init__(self, factor, coeff_exponent, active_dims=None):
            super().__init__(active_dims=active_dims)
            self.derivation_term_dict = None
            self.coeff = factor
            self.exp_coeff = coeff_exponent

        def is_equal(self, other):
            if not isinstance(other, self.__class__):
                return False
            elif other.l_poly == self.l_poly and other.r_poly == self.r_poly and other.base_kernel == self.base_kernel:
                return True

        def has_equal_basekernel(self, other):
            if not isinstance(other, self.__class__):
                return False
            elif other.base_kernel == self.base_kernel:
                return True

        def set_r_poly(self, r_poly):
            self.r_poly = r_poly

        def set_l_poly(self, l_poly):
            self.l_poly = l_poly

        def set_base_kernel(self, base_kernel):
            self.base_kernel = id(base_kernel)

        def set_derivation_term_dict(self, derivation_term_dict):
            self.derivation_term_dict = derivation_term_dict

        def set_derivation_coefficients_list(self, derivation_coefficients_list):
            self.derivation_coefficients_list = derivation_coefficients_list
            """
            The form of derivation_coefficients_list is:
            Is it a list of lists? Or just a list?
            [[coeff, coeff_exponent], [coeff, coeff_exponent2], ...]
            """

        def __str__(self):
            coeff_string = ""
            for i, summand in enumerate(self.derivation_coefficients_list):
                coeff_string += f" > Summand {i}:\ncoefficients:{str(summand[0])}\nexponent:{summand[1]}"
            string = f"Received derivation form: {self.derivation_term_dict}\nResulting list (including parameters):\n{coeff_string}"
            return string

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

        def forward(self, x1, x2=None, diag=False, **params):
            if x2 is None:
                x2 = x1
            if len(x1.shape) == 1:
                x1 = self._slice_input(x1)
            if len(x2.shape) == 1:
                x2 = self._slice_input(x2)
            if x1.size(int(1)) != x2.size(int(1)):
                raise ValueError("Inputs must have the same number of features.")
            result = None
            x1_plus_x2 = x1+x2.t()
            exp_of_add = torch.exp(x1_plus_x2)
            for summand in self.derivation_coefficients_list:
                poly_coeffs = summand[0]
                exp_coeff_power = summand[1]
                if result is None:
                    result = exp_of_add * torch.prod(torch.Tensor(poly_coeffs)) * (self.exp_coeff ** exp_coeff_power)
                else:
                    result += exp_of_add * torch.prod(torch.Tensor(poly_coeffs)) * (self.exp_coeff ** exp_coeff_power)
            return result



# TODO The kernel will likely explode whenever I have a torch.nn.parameter as
# a denominator
class diffed_SE_kernel(Kernel):
        asym_sign_matr = [[int(1), int(1), int(-1), int(-1)], [int(-1), int(1), int(1), int(-1)], [int(-1), int(-1), int(1), int(1)], [int(1), int(-1), int(-1), int(1)]]

        def __init__(self,  var=None, length=None, active_dims=None, variance_constraint=None, lengthscale_constraint=None, lengthscale_prior=None):
            super().__init__(active_dims=active_dims)
            if isinstance(var, torch.nn.Parameter):
                self.var = var
            else:
                self.register_parameter(name="var", parameter=torch.nn.Parameter(torch.tensor(float(var)) if not var is None else torch.tensor(float(1.)), requires_grad=True))
            if isinstance(length, torch.nn.Parameter):
                self.length = length
            else:
                self.register_parameter(name="length", parameter=torch.nn.Parameter(torch.tensor(float(length)) if not length is None else torch.tensor(float(1.)), requires_grad=True))

            self.K_0 = None
            self.K_1 = None
            self.K_4 = None
            self.derivation_term_dict = None

        def is_equal(self, other):
            if not isinstance(other, self.__class__):
                return False
            elif other.l_poly == self.l_poly and other.r_poly == self.r_poly and other.base_kernel == self.base_kernel:
                return True

        def has_equal_basekernel(self, other):
            if not isinstance(other, self.__class__):
                return False
            elif other.base_kernel == self.base_kernel:
                return True

        def set_r_poly(self, r_poly):
            self.r_poly = r_poly

        def set_l_poly(self, l_poly):
            self.l_poly = l_poly

        def set_base_kernel(self, base_kernel):
            self.base_kernel = id(base_kernel)

        def set_derivation_term_dict(self, derivation_term_dict):
            self.derivation_term_dict = derivation_term_dict

        def set_derivation_coefficients_list(self, derivation_coefficients_list):
            self.derivation_coefficients_list = derivation_coefficients_list

        def __str__(self):
            coeff_string = ""
            for i, summand in enumerate(self.derivation_coefficients_list):
                for j, op in enumerate(summand):
                    coeff_string += f" > Summand {i}, entry {j}:\npolynom coefficients:{str(op[0])}\nderivation coefficient:{op[1]}\nl exponent:{op[2]}\n(x1-x2) exponent:{op[-1]}\n"
            string = f"Received derivation form: {self.derivation_term_dict}\nResulting list (including parameters):\n{coeff_string}"
            return string

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
            if len(X.shape) == 1:
                X = self._slice_input(X)
            if len(Z.shape) == 1:
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
            #var = torch.nn.functional.relu(self.var)
            var = torch.exp(self.var)
            #length = torch.nn.functional.relu(self.length)
            length = torch.exp(self.length)
            self.result_term = lambda self, l_, coefficients, i, sign, l_exponents, K_1_exponents: \
            coefficients[i]*(sign*(int(-1)**i))*(l_**l_exponents[i])*(self.K_0**K_1_exponents[i])

            l_ = float(1)/length**(float(2))

            self._square_scaled_dist(x1, x2)
            self.K_4 = torch.mul(var, torch.exp(float(-0.5) * self.K_1*(float(1)/length**float(2))))

            #return self.K_1, self.K_4, self.length
            result = None
            # [[[coeff, l_exp, exp], [coeff, l_exp, exp], ...], [[coeff, l_exp, exp], ...], ...]
            for term in self.derivation_coefficients_list:
                for summand in term:
                    K_0_exp = summand[3]
                    l_exp = summand[2]
                    coeff = summand[1]
                    poly_coeffs = summand[0]

                    if result is None:
                        temp = coeff*(l_**l_exp)*(self.K_0**K_0_exp)*torch.prod(torch.Tensor(poly_coeffs))
                        result = temp
                    else:
                        #int(degr_o+degr_p) if int(degr_o+degr_p)%2 == 0 else int(degr_o+degr_p-1)
                        #TODO: This as well
                        temp = coeff*(l_**l_exp)*(self.K_0**K_0_exp)*torch.prod(torch.Tensor(poly_coeffs))
                        result += temp
            return self.K_4*result



class Diff_SE_kernel(Kernel):

    asym_sign_matr = [[int(1), int(1), int(-1), int(-1)],
                      [int(-1), int(1), int(1), int(-1)],
                      [int(-1), int(-1), int(1), int(1)],
                      [int(1), int(-1), int(-1), int(1)]]

    def __init__(self,  var=None, length=None, active_dims=None, variance_constraint=None, lengthscale_constraint=None, lengthscale_prior=None):
        super().__init__(active_dims=active_dims)
        self.is_diffable = True
        self.register_parameter(name="var", parameter=torch.nn.Parameter(torch.tensor(float(var)) if not var is None else torch.tensor(float(1.)), requires_grad=True))
        self.register_parameter(name="length", parameter=torch.nn.Parameter(torch.tensor(float(length)) if not length is None else torch.tensor(float(1.)), requires_grad=True))
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
            T = lambda m, k: factorial(2*m+1)*2**(k-m)/(factorial(m-k)*factorial(2*k+1))

        return [int(T(real_n, k)) for k in range(real_n+1)]


    def diff(self, left_poly, right_poly, left_d_var=var('dx1'), right_d_var=var('dx2'), parent_context=None):
        # TODO check if id(self) is in parent_context.named_kernel_list and depending on yes/no add variance/lengthscale as parameters or not
        # If they already exist, take the adresses of the parent hyperparameters and make the diffed_SE_kernel parameters be references to these adresses
        #if not parent_context is None:
        #    if id(self) in parent_context.named_kernel_list:
        diffed_kernel = diffed_SE_kernel(var=self.var, length=self.length, active_dims=self.active_dims)
        diffed_kernel.set_l_poly(left_poly)
        diffed_kernel.set_r_poly(right_poly)
        diffed_kernel.set_base_kernel(self)
        if parent_context is None:
            parent_context = diffed_kernel
        derivation_term_list = prepare_asym_deriv_dict(left_poly, right_poly, parent_context, left_d_var, right_d_var)
        derived_form_list = []
        for term in derivation_term_list:
            # term will have the form [[coeff1, coeff2, ...], exponent of dx1, exponent of dx2]
            degr_x1 = term[1]
            degr_x2 = term[2]
            poly_coeffs = term[0]
            sign = self.asym_sign_matr[int(degr_x1)%int(4)][int(degr_x2)%int(4)]
            K_0_exponents = [int(i*2) if int(degr_x1+degr_x2)%2 == 0 else int(i*int(2)+int(1)) for i in range(int((degr_x1+degr_x2)/2)+int(1))]
            coefficients = self.coeffs(int(degr_x1+degr_x2))
            coefficients = [c*sign*(-1)**i for i, c in enumerate(coefficients)]
            l_exponents = [np.ceil((degr_x1+degr_x2)/int(2)) + i for i in range(int((degr_x1+degr_x2)/2)+int(1))]
            derived_form_list.append([[poly_coeffs, coeff, int(l_exp), exp] for exp, coeff, l_exp in zip_longest(K_0_exponents, coefficients, l_exponents, fillvalue=0)])
        diffed_kernel.set_derivation_coefficients_list(derived_form_list)
        diffed_kernel.set_derivation_term_dict(derivation_term_list)
        return diffed_kernel

    def _square_scaled_dist(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|^2`.
        """
        if Z is None:
            Z = X
        if len(X.shape) == 1:
            X = self._slice_input(X)
        if len(Z.shape) == 1:
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
        #var = torch.nn.functional.relu(self.var)
        var = torch.exp(self.var)
        #length = torch.nn.functional.relu(self.length)
        length = torch.exp(self.length)
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
        # named_kernels is used during 'DiffMatrixKernel' which is why it's
        # defined before checking for the matrix
        self.base_kernels = []
        if matrix is None:
            return
        self.set_matrix(matrix, add_kernel_parameters=True)

    def set_matrix(self, matrix, add_kernel_parameters=False):
        self.num_tasks = np.shape(matrix)[0]
        if not np.shape(matrix)[0] == np.shape(matrix)[1]:
            assert "Kernel matrix is not square"
        # Set the lower triangle to be symmetric to the upper triangle
        #matrix = make_symmetric(matrix)
        self.matrix = matrix
        if add_kernel_parameters:
            for i, row in enumerate(self.matrix):
                for j, kernel in enumerate(row):
                    # Note: 'entry == kernel' only works because the kernels don't
                    # have a '__eq__' function, since then it checks the adresses
                    if not any([entry == kernel for entry in self.base_kernels]) and not (kernel is None or kernel == 0):
                        #if not hasattr(self, kernel) and (kernel is None or kernel == 0):
                        setattr(self, f'kernel_{i}{j}', kernel)
                        self.base_kernels.append(getattr(self, f"kernel_{i}{j}"))
            print(f"List of all kernels: {self.base_kernels}")

    def __str__(self):
        string = ""
        for i, row in enumerate(self.matrix):
            for j, kernel in enumerate(row):
                string = string + f"[{i},{j}]: " + str(kernel) + "\n\n"
        return string

    def add_named_kernel(self, kernel):
        setattr(self, f"{id(kernel)}", kernel)
        self.base_kernels.append(kernel)

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
        H_z = np.shape(x2)[0]
        H_x = np.shape(x1)[0]

        if diag:
            return self._diag(x1)
        zero_matrix = torch.zeros(H_x, H_z)
        #zero_matrix = torch.tensor([[int(0) for i in range(H_z)] for j in range(H_x)])
        result = None
        for i, row in enumerate(self.matrix) :
            temp = None
            for j, kernel in enumerate(row[i:]):
            # Create the result matrix
                if j < i:
                    temp = torch.hstack([zero_matrix for p in range(i-j)])
                if kernel is None or kernel == 0:
                    result1 = zero_matrix
                else:
                    result1 = kernel.forward(x1, x2)
                if temp is None:
                    temp = result1
                else:
                    #temp = CatLazyTensor(*[temp, result1])
                    temp = torch.hstack([delazify(temp), delazify(result1)])

            # append vertically
            if result is None:
                result = temp
            else:
                # If, at some point, 'CatLazyTensor' supports step_slicing
                # rewrite everything to use CatLazyTensors and lazy Tensors
                #result = CatLazyTensor(*[result, temp], dim=1)
                result = torch.vstack([delazify(result), delazify(temp)])
       # print(f"Result:\n{result}")
        result = make_symmetric(result)
       # print(f"Symmetric result:\n{result}")
        result = torch.vstack([torch.hstack([result[k::H_x, l::H_x] for l in range(H_x)]) for k in range(H_x)])
       # print(f"Interleaved result:\n{result}")
        DEBG = False
        if DEBG:
            eigs = torch.linalg.eigvals(result)
            if not all([True if 0 > e.real > -0.00001  else False for e in eigs]):
                print(eigs.real)
                print(eigs.imag)
                #assert "Not all Eigenvalues positive"
        #print(result.eig())
        return result

    def num_outputs_per_input(self, x1, x2):
        return self.num_tasks


class DiffMatrixKernel(MatrixKernel):

    def __init__(self, matrix, active_dims=None):
        if not all([k == 0 or k is None or k.is_diffable for row in matrix for k in row]):
            assert "Not all kernels are differentiable"
        super().__init__(matrix, active_dims=active_dims)


    def calc_cell_diff(self, L, M, R, context=None):
        result_kernel = None
        # https://stackoverflow.com/questions/6473679/transpose-list-
        # of-lists
        M_transpose = list(
           map(list, itertools.zip_longest(*M, fillvalue=None)))
        # Every row in 'M' is combined with each elem of the row given in 'R'
        # Or: For each elemtn in row 'R' combine with 'row_M'
        for r_elem, row_M in zip(R, M_transpose):
            # Each element in L gets exactly one element in 'row_M' to multiply
            # Or: Combine each element in row_M with exactly one element in 'L'
            for l_elem, m_elem in zip(L, row_M):
                if m_elem is not None:
                    current_kernel = m_elem.diff(left_poly=l_elem, right_poly=r_elem, parent_context=context)
                   #condition = any(e.has_equal_basekernel(current_kernel) for e in context.named_kernels) if hasattr(current_kernel, 'is_equal') else any(e is current_kernel for e in context.named_kernels)
                    #if condition:
                    #    index_condition = [e.has_equal_basekernel(current_kernel) if hasattr(current_kernel, 'is_equal') else e == current_kernel for e in context.named_kernels]
                    #    index = index_condition.index(True)
                    if result_kernel is None:
                        #if not condition:
                        result_kernel = current_kernel
                        context.add_named_kernel(current_kernel)
                        #else:
                        #    result_kernel = context.named_kernels[index]
                    else:
                        #if not condition:
                        result_kernel += current_kernel
                        context.add_named_kernel(current_kernel)
                        #else:
                        #    result_kernel += context.named_kernels[index]
                else:
                    pass
        return result_kernel

    def diff(self, left_matrix=None, right_matrix=None):
        # iterate left matrix by rows and right matrix by columns and call the
        # respective diff command of the kernels with the row/cols as params
        kernel = MatrixKernel(None)
        output_matrix = [[0 for i in range(np.shape(self.matrix)[1])] for j in range(np.shape(self.matrix)[0])]
        for i, (l, r) in enumerate(itertools.product(left_matrix.rows(), right_matrix.columns())):
            res = self.calc_cell_diff(l, self.matrix, r, context=kernel)
            output_matrix[int(i/np.shape(self.matrix)[0])][
                        int(i % np.shape(self.matrix)[0])]  = res
        kernel.set_matrix(output_matrix)
        print(output_matrix)
        return kernel
