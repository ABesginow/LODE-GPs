#=======================================================================
# Imports
#=======================================================================
import gpytorch 
from sage.all import *
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
from .kernels import LODE_Kernel, create_kernel_matrix_from_diagonal, differentiate_kernel_matrix, replace_sum_and_diff, translate_kernel_matrix_to_gpytorch_kernel 
import pprint
import torch


# Functions for standard ODE models
STANDARD_MODELS = {}

def register_LODEGP_model(name):
    def decorator(fn):
        STANDARD_MODELS[name] = fn
        return fn
    return decorator


def load_standard_model(name: str):
    try:
        return STANDARD_MODELS[name]()
    except KeyError:
        raise ValueError(f"No standard model found for: {name}")


def list_standard_models():
    return list(STANDARD_MODELS.keys())

#=======================================================================
# Base ODEs
#=======================================================================

@register_LODEGP_model("Bipendulum")
def bipendulum():
    model_parameters = torch.nn.ParameterDict()
    R = QQ['x']; (x,) = R._first_ngens(1)
    # Linearized bipendulum
    A = matrix(R, Integer(2), Integer(3), [x**2 + 9.81, 0, -1, 0, x**2+4.905, -1/2])
    return A, model_parameters

@register_LODEGP_model("Bipendulum first equation")
def bipendulum_first_eq():
    model_parameters = torch.nn.ParameterDict()
    R = QQ['x']; (x,) = R._first_ngens(1)
    # Linearized bipendulum
    A = matrix(R, Integer(1), Integer(3), [x**2 + 9.81, 0, -1])
    return A, model_parameters

@register_LODEGP_model("Bipendulum second equation")
def bipendulum_second_eq():
    model_parameters = torch.nn.ParameterDict()
    R = QQ['x']; (x,) = R._first_ngens(1)
    # Linearized bipendulum
    A = matrix(R, Integer(1), Integer(3), [0, x**2+4.905, -1/2])
    return A, model_parameters

@register_LODEGP_model("Bipendulum Parameterized")
def bipendulum_parameterized():
    F = FunctionField(QQ, names=('l1',)); (l1,) = F._first_ngens(1)
    F = FunctionField(F, names=('l2',)); (l2,) = F._first_ngens(1)
    R = F['x']; (x,) = R._first_ngens(1)
    # Linearized bipendulum
    A = matrix(R, Integer(2), Integer(3), [x**2 + 981/(100*l1), 0, -1/l1, 0, x**2+981/(100*l2), -1/l2])
    model_parameters = torch.nn.ParameterDict({
        "l1":torch.nn.Parameter(torch.tensor(0.0)),
        "l2":torch.nn.Parameter(torch.tensor(0.0))
    })
    return A, model_parameters




@register_LODEGP_model("Three tank")
def three_tank():
    model_parameters = torch.nn.ParameterDict()
    R = QQ['x']; (x,) = R._first_ngens(1)

    # 3 Tank system (5 dimensional uncontrollable system)
    A = matrix(R, Integer(3), Integer(5), [-x, 0, 0, 1, 0, 0, -x, 0, 1, 1, 0, 0, -x, 0, 1])
    return A, model_parameters


@register_LODEGP_model("Heating")
def heating_system():
    # Heating system with parameters
    F = FunctionField(QQ, names=('a',)); (a,) = F._first_ngens(1)
    F = FunctionField(F, names=('b',)); (b,) = F._first_ngens(1)
    R = F['x']; (x,) = R._first_ngens(1)

    A = matrix(R, Integer(2), Integer(3), [x+a, -a, -1, -b, x+b, 0])
    model_parameters = torch.nn.ParameterDict({
        "a":torch.nn.Parameter(torch.tensor(0.0)),
        "b":torch.nn.Parameter(torch.tensor(0.0))
    })

    return A, model_parameters


def unknown():
    model_parameters = torch.nn.ParameterDict()
    R = QQ['x']; (x,) = R._first_ngens(1)
    # System 1 (no idea)
    A = matrix(R, Integer(2), Integer(3), [x, -x**2+x-1, x-2, 2-x, x**2-x-1, -x])

    return A, model_parameters



#=======================================================================
# LODEGP Class
#=======================================================================
class LODEGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, **kwargs):
        super(LODEGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=num_tasks
        )

        ODE_name = kwargs["ODE_name"] if "ODE_name" in kwargs else None
        verbose = kwargs["verbose"] if "verbose" in kwargs else False
        if ODE_name is not None:
            A, self.model_parameters = load_standard_model(ODE_name)
        else:
            A = kwargs["A"]
            self.model_parameters = kwargs["parameter_dict"]

        D, U, V = A.smith_form()
        
        if verbose:
            print(f"D:{D}")
            print(f"V:{V}")
        x, a, b = var("x, a, b")
        V_temp = [list(b) for b in V.rows()]
        if verbose:
            print(V_temp)
        V = sage_eval(f"matrix({str(V_temp)})", locals={"x":x, "a":a, "b":b})
        Vt = V.transpose()
        kernel_matrix, self.kernel_translation_dict, parameter_dict = create_kernel_matrix_from_diagonal(D)
        self.ode_count = A.nrows()
        self.kernelsize = len(kernel_matrix)
        self.model_parameters.update(parameter_dict)
        if verbose:
            print(self.model_parameters)
        var(["x", "dx1", "dx2"] + ["t1", "t2"] + [f"LODEGP_kernel_{i}" for i in range(len(kernel_matrix[Integer(0)]))])
        k = matrix(Integer(len(kernel_matrix)), Integer(len(kernel_matrix)), kernel_matrix)
        V = V.substitute(x=dx1)
        Vt = Vt.substitute(x=dx2)

        #train_x = self._slice_input(train_x)

        self.common_terms = {
            "t_diff" : train_x-train_x.t(),
            "t_sum" : train_x+train_x.t(),
            "t_ones": torch.ones_like(train_x-train_x.t()),
            "t_zeroes": torch.zeros_like(train_x-train_x.t())
        }
        self.V = V
        self.matrix_multiplication = matrix(k.base_ring(), len(k[0]), len(k[0]), (V*k*Vt))
        self.diffed_kernel = differentiate_kernel_matrix(k, V, Vt, self.kernel_translation_dict)
        self.sum_diff_replaced = replace_sum_and_diff(self.diffed_kernel)
        self.covar_description = translate_kernel_matrix_to_gpytorch_kernel(self.sum_diff_replaced, self.model_parameters, common_terms=self.common_terms)
        self.covar_module = LODE_Kernel(self.covar_description, self.model_parameters)


    def __str__(self, substituted=False):
        if substituted:
            return pprint.pformat(str(self.sum_diff_replaced), indent=self.kernelsize)
        else:
            return pprint.pformat(str(self.diffed_kernel), indent=self.kernelsize)

    def __latexify_kernel__(self, substituted=False):
        if substituted:
            return pprint.pformat(latex(self.sum_diff_replaced), indent=self.kernelsize)
        else:
            return pprint.pformat(latex(self.diffed_kernel), indent=self.kernelsize)

    def __pretty_print_kernel__(self, substituted=False):
        if substituted:
            return pprint.pformat(pretty_print(self.matrix_multiplication), indent=self.kernelsize)
        else:
            pretty_print(self.matrix_multiplication)
            print(str(self.kernel_translation_dict))

    def _slice_input(self, X):
            r"""
            Slices :math:`X` according to ``self.active_dims``. If ``X`` is 1D then returns
            a 2D tensor with shape :math:`N \times 1`.
            :param torch.Tensor X: A 1D or 2D input tensor.
            :returns: a 2D slice of :math:`X`
            :rtype: torch.Tensor
            """
            if X.dim() == 2:
                #return X[:, self.active_dims]
                return X[:, 0]
            elif X.dim() == 1:
                return X.unsqueeze(1)
            else:
                raise ValueError("Input X must be either 1 or 2 dimensional.")

    def forward(self, X):
        if not torch.equal(X, self.train_inputs[0]):
            self.common_terms["t_diff"] = X-X.t()
            self.common_terms["t_sum"] = X+X.t()
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X, common_terms=self.common_terms)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x) 
