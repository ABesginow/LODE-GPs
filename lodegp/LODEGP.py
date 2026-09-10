#=======================================================================
# Imports
#=======================================================================
import gpytorch 
import numpy as np
from sage.all import *
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
from lodegp.kernels import LODE_Kernel, create_kernel_matrix_from_diagonal, differentiate_kernel_matrix, replace_sum_and_diff, translate_kernel_matrix_to_gpytorch_kernel 
import pprint
import torch


# Functions for standard ODE models
STANDARD_MODELS = {}

def register_LODEGP_model(name):
    def decorator(fn):
        STANDARD_MODELS[name] = fn
        return fn
    return decorator


def load_standard_model(name: str, kwargs: dict = None):
    try:
        return STANDARD_MODELS[name](**(kwargs or {}))
    except KeyError:
        raise ValueError(f"No standard model found for: {name}")


def list_standard_models():
    return list(STANDARD_MODELS.keys())

#=======================================================================
# Base ODEs
#=======================================================================


# ====
# Standard linearized Bipendulum
# ====

@register_LODEGP_model("Zero")
def bipendulum(**kwargs):
    l1 = kwargs.get("l1", 1.0)
    l2 = kwargs.get("l2", 2.0)
    model_parameters = torch.nn.ParameterDict()
    R = QQ['x']; (x,) = R._first_ngens(1)
    # Linearized bipendulum
    A = matrix(R, Integer(1), Integer(1), [-x])
    return A, model_parameters, {"x":var("x")}

@register_LODEGP_model("Bipendulum")
def bipendulum(**kwargs):
    l1 = kwargs.get("l1", 1.0)
    l2 = kwargs.get("l2", 2.0)
    model_parameters = torch.nn.ParameterDict()
    R = QQ['x']; (x,) = R._first_ngens(1)
    # Linearized bipendulum
    A = matrix(R, Integer(2), Integer(3), [x**2 + 9.81/l1, 0, -1/l1, 0, x**2+9.81/l2, -1/l2])
    return A, model_parameters, {"x":var("x")}

@register_LODEGP_model("Bipendulum first equation")
def bipendulum_first_eq(**kwargs):
    l1 = kwargs.get("l1", 1.0)
    model_parameters = torch.nn.ParameterDict()
    R = QQ['x']; (x,) = R._first_ngens(1)
    # Linearized bipendulum
    A = matrix(R, Integer(1), Integer(3), [x**2 + 9.81/l1, 0, -1/l1])
    return A, model_parameters, {"x":var("x")}

@register_LODEGP_model("Bipendulum second equation")
def bipendulum_second_eq(**kwargs):
    l2 = kwargs.get("l2", 2.0)
    model_parameters = torch.nn.ParameterDict()
    R = QQ['x']; (x,) = R._first_ngens(1)
    # Linearized bipendulum
    A = matrix(R, Integer(1), Integer(3), [0, x**2+9.81/l2, -1/l2])
    return A, model_parameters, {"x":var("x")}

@register_LODEGP_model("Bipendulum Parameterized")
def bipendulum_parameterized(**kwargs):
    # Think about using kwargs as parameter initizations for model_parameters
    F = FunctionField(QQ, names=('l1',)); (l1,) = F._first_ngens(1)
    F = FunctionField(F, names=('l2',)); (l2,) = F._first_ngens(1)
    R = F['x']; (x,) = R._first_ngens(1)
    # Linearized bipendulum
    A = matrix(R, Integer(2), Integer(3), [x**2 + 981/(100*l1), 0, -1/l1, 0, x**2+981/(100*l2), -1/l2])
    model_parameters = torch.nn.ParameterDict({
        "l1":torch.nn.Parameter(torch.tensor(0.0)),
        "l2":torch.nn.Parameter(torch.tensor(0.0))
    })
    x, l1, l2 = var(["x", "l1", "l2"])
    return A, model_parameters, {"x":x, "l1": l1, "l2": l2}

#====
# Awkward Bipendulum systems
#====

@register_LODEGP_model("Bipendulum Sum")
def bipendulum(**kwargs):
    l1 = kwargs.get("l1", 1.0)
    l2 = kwargs.get("l2", 2.0)
    model_parameters = torch.nn.ParameterDict()
    R = QQ['x']; (x,) = R._first_ngens(1)
    # Linearized bipendulum
    A = matrix(R, Integer(1), Integer(3), [x**2 + 9.81/l1, x**2+9.81/l2, -1/l1 -1/l2])
    return A, model_parameters, {"x":var("x")}

@register_LODEGP_model("Bipendulum Sum eq2 diffed")
def bipendulum(**kwargs):
    l1 = kwargs.get("l1", 1.0)
    l2 = kwargs.get("l2", 2.0)
    model_parameters = torch.nn.ParameterDict()
    R = QQ['x']; (x,) = R._first_ngens(1)
    # Linearized bipendulum
    A = matrix(R, Integer(1), Integer(3), [x**2 + 9.81/l1, x**3+x*9.81/l2, -1/l1 -x/l2])
    #A = matrix(R, Integer(2), Integer(3), [x**2 + 9.81/l1, 0, -1/l1, 0, x**2+9.81/l2, -1/l2])
    return A, model_parameters, {"x":var("x")}

@register_LODEGP_model("Bipendulum moon gravitation")
def bipendulum(**kwargs):
    l1 = kwargs.get("l1", 1.0)
    l2 = kwargs.get("l2", 2.0)
    model_parameters = torch.nn.ParameterDict()
    R = QQ['x']; (x,) = R._first_ngens(1)
    # Linearized bipendulum
    A = matrix(R, Integer(2), Integer(3), [x**2 + 1.62/l1, 0, -1/l1, 0, x**2+1.62/l2, -1/l2])
    return A, model_parameters, {"x":var("x")}

# ====
# Spring Mass Damper systems
# ====
# Helper function to create the continuous-time system matrices for a spring-mass-damper system with given parameters from https://arxiv.org/pdf/2407.17277 (MIT License)
def create_spring_mass_sys_ct(ms: np.ndarray, ks: np.ndarray, ds: np.ndarray, num_actuated: int):
    if type(ms) == list:
        ms = np.array(ms)
    elif type(ms) == np.ndarray:
        pass
    else:
        raise ValueError('ms should be a list or np.ndarray')
    assert ms.dtype in [np.float64,
                        np.int64], 'ms should be a list of floats or ints'
    num_masses = len(ms)
    assert num_masses > 2, 'ms should have at least 3 elements'
    assert num_actuated <= num_masses, 'num_actuated should be less than or equal to num_masses'
    assert num_masses == len(ks), 'ms and ks should have the same length'
    assert num_masses == len(ds), 'ms and ds should have the same length'

    ni = 2  # number of local states
    nx = ni * num_masses  # number of overall states
    nth = 4 * (num_masses-1) + 2 + num_actuated
    nu = num_actuated
    nw = num_masses
    if num_masses < 3:
        raise NotImplementedError()
    # Define continuous-time system matrices \dot{x}=A_c*x+B_c*u
    A_c = np.zeros((nx, nx))

    # i=1: also connected to ground
    A_c[0, :ni] = [0, 1]
    A_c[1, :2*ni] = np.array([-(ks[0]+ks[1]), -
                             (ds[0]+ds[1]), ks[1], ds[1]]) / ms[0]

    # 1<i<M-1
    for i in range(1, num_masses-1):
        A_c[i*ni, i*ni+1] = 1
        A_c[i*ni+1, (i-1)*ni:(i+2)*ni] = np.array([ks[i], ds[i], -
                                                   (ks[i]+ks[i+1]), -(ds[i]+ds[i+1]), ks[i+1], ds[i+1]]) / ms[i]


    A_c[-2, -1] = 1
    A_c[-1, -4:] = np.array([ks[-1], ds[-1], -ks[-1], -ds[-1]]) / ms[-1]

    B_c = np.zeros((nx, nu))
    B_c[1-2*nu::2] = np.diag([1/m for m in ms[-nu:]])

    A0 = A_c.copy()
    A0[1::ni] = 0


    return A_c, B_c

@register_LODEGP_model("HO")
def bipendulum_parameterized(**kwargs):
    R = QQ['x']; (x,) = R._first_ngens(1)
    model_parameters = torch.nn.ParameterDict()
    # Spring mass system with negative damping (therefore unstable)
    A = matrix(R, Integer(2), Integer(2), [-x, 1,
                                           -1, -x])
    return A, model_parameters, {"x":var("x")}

@register_LODEGP_model("SMD scaled")
def bipendulum_parameterized(**kwargs):
    R = QQ['x']; (x,) = R._first_ngens(1)
    model_parameters = torch.nn.ParameterDict()
    # Spring mass system with negative damping (therefore unstable)
    A = matrix(R, Integer(2), Integer(3), [-x, 1, 0,
                                           -1, 1 -x, 2.5])
    return A, model_parameters, {"x":var("x")}

@register_LODEGP_model("SM1")
def bipendulum_parameterized(**kwargs):
    R = QQ['x']; (x,) = R._first_ngens(1)
    model_parameters = torch.nn.ParameterDict()
    # Spring mass system with negative damping (therefore unstable)
    A = matrix(R, Integer(2), Integer(3), [-x, 1, 0,
                                           1, -x, 20])
    return A, model_parameters, {"x":var("x")}

@register_LODEGP_model("SMD1")
def bipendulum_parameterized(**kwargs):
    R = QQ['x']; (x,) = R._first_ngens(1)
    model_parameters = torch.nn.ParameterDict()
    # Spring mass system with negative damping (therefore unstable)
    A = matrix(R, Integer(2), Integer(3), [-x, 1, 0,
                                           -1,1 -x, 1])
    return A, model_parameters, {"x":var("x")}

@register_LODEGP_model("SMD1withAdditionalDerivative")
def bipendulum_parameterized(**kwargs):
    R = QQ['x']; (x,) = R._first_ngens(1)
    model_parameters = torch.nn.ParameterDict()
    # Linearized bipendulum
    A = matrix(R, Integer(3), Integer(4), [-x, 1, 0, 0,
                                           0, -x, 1, 0,
                                           1, -1 -x, 0, 1])
    return A, model_parameters, {"x":var("x")}

@register_LODEGP_model("SM2")
def bipendulum_parameterized(**kwargs):
    R = QQ['x']; (x,) = R._first_ngens(1)
    model_parameters = torch.nn.ParameterDict()
    # Linearized bipendulum
    k = [1, 1]
    m = [1, 1]
    A = matrix(R, Integer(4), Integer(6), [-x, 1, 0, 0, 0, 0,
                                           -(k[0] + k[1])/m[0], -x, k[1]/m[0], 0, 1., 0.,
                                           0, 0, -x, 1, 0, 0,
                                           k[1]/m[1], 0, -k[1]/m[1], -x, 0, 1.])
    return A, model_parameters, {"x":var("x")}

@register_LODEGP_model("SMD2")
def bipendulum_parameterized(**kwargs):
    R = QQ['x']; (x,) = R._first_ngens(1)
    model_parameters = torch.nn.ParameterDict()
    # Linearized bipendulum
    k = [2, 2]
    d = [1, 1]
    m = [1, 1]
    A = matrix(R, Integer(4), Integer(6), [-x, 1, 0, 0, 0, 0,
                                           -(k[0] + k[1])/m[0], +d[0]/m[0] - x, k[1]/m[0], 0, 5., 0.,
                                           0, 0, -x, 1, 0, 0,
                                           k[1]/m[1], 0, -k[1]/m[1], d[1]/m[1] - x, 0., 5.])
    return A, model_parameters, {"x":var("x")}

@register_LODEGP_model("SM3")
def bipendulum_parameterized(**kwargs):
    R = QQ['x']; (x,) = R._first_ngens(1)
    model_parameters = torch.nn.ParameterDict()
    # Linearized bipendulum
    k = [1, 1, 1]
    m = [1, 1, 1]
    A = matrix(R, Integer(6), Integer(7), [-x, 1, 0, 0, 0, 0, 0,
                                           -(k[0] + k[1])/m[0], -x, k[1]/m[0], 0, 0, 0, 0,
                                           0, 0, -x, 1, 0, 0, 0,
                                           0, 0, -(k[1] + k[2])/m[1], -x, k[2]/m[1], 0, 0,
                                           0, 0, 0, 0, -x, 1, 0,
                                           0, 0, 0, 0, -k[2]/m[2], -x, 1/m[2]])
    return A, model_parameters, {"x":var("x")}

@register_LODEGP_model("SMD3")
def bipendulum_parameterized(**kwargs):
    R = QQ['x']; (x,) = R._first_ngens(1)
    model_parameters = torch.nn.ParameterDict()
    # Linearized bipendulum
    k = 2*np.ones(3)
    d = np.ones(3)
    m = np.ones(3)
    A_c, B_c = create_spring_mass_sys_ct(m, k, d, num_actuated=3)
    A_sage = np.concatenate([A_c, 3.5 * B_c], axis=1) - x*np.eye(2*len(k), 2*len(k)+B_c.shape[1])
    A = matrix(R, Integer(2*len(k)), Integer(2*len(k)+B_c.shape[1]), A_sage.flatten().tolist())
    return A, model_parameters, {"x":var("x")}

@register_LODEGP_model("SMD4")
def bipendulum_parameterized(**kwargs):
    R = QQ['x']; (x,) = R._first_ngens(1)
    model_parameters = torch.nn.ParameterDict()
    # Linearized bipendulum
    k = 2 * np.ones(4)
    d = np.ones(4)
    m = np.ones(4)
    A_c, B_c = create_spring_mass_sys_ct(m, k, d, num_actuated=4)
    A_sage = np.concatenate([A_c, 3.5 * B_c], axis=1) - x*np.eye(2*len(k), 2*len(k)+B_c.shape[1])
    A = matrix(R, Integer(2*len(k)), Integer(2*len(k)+B_c.shape[1]), A_sage.flatten().tolist())
    return A, model_parameters, {"x":var("x")}

@register_LODEGP_model("SM5")
def bipendulum_parameterized(**kwargs):
    R = QQ['x']; (x,) = R._first_ngens(1)
    model_parameters = torch.nn.ParameterDict()
    # Linearized bipendulum
    k = [1, 1, 1, 1, 1]
    m = [1, 1, 1, 1, 1]
    
    A = matrix(R, Integer(10), Integer(11), [-x, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                             -2, -x, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, -x, 1, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, -2, -x, 1, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, -x, 1, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, -2, -x, 1, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, -x, 1, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, -2, -x, 1, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, -x, 1, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, -1, -x, 1])
    return A, model_parameters, {"x":var("x")}

@register_LODEGP_model("SMD5")
def bipendulum_parameterized(**kwargs):
    R = QQ['x']; (x,) = R._first_ngens(1)
    model_parameters = torch.nn.ParameterDict()
    # Linearized bipendulum
    k = 2 * np.ones(5)
    d = np.ones(5)
    m = np.ones(5)
    A_c, B_c = create_spring_mass_sys_ct(m, k, d, num_actuated=5)
    B_c = np.zeros((2*len(k), 1))
    B_c[-1, 0] = 3.5
    A_sage = (np.concatenate([A_c, B_c], axis=1) - x*np.eye(2*len(k), 2*len(k)+1)).flatten().tolist()
    A = matrix(R, Integer(2*len(k)), Integer(2*len(k)+1), A_sage)
    return A, model_parameters, {"x":var("x")}

@register_LODEGP_model("SMD6")
def bipendulum_parameterized(**kwargs):
    R = QQ['x']; (x,) = R._first_ngens(1)
    model_parameters = torch.nn.ParameterDict()
    # Linearized bipendulum
    k = np.ones(6)
    d = np.zeros(6)
    m = np.ones(6)
    A_c, B_c = create_spring_mass_sys_ct(m, k, d, num_actuated=1)
    A_sage = (np.concatenate([A_c, B_c], axis=1) - x*np.eye(2*len(k), 2*len(k)+1)).flatten().tolist()
    A = matrix(R, Integer(2*len(k)), Integer(2*len(k)+1), A_sage)
    return A, model_parameters, {"x":var("x")}

@register_LODEGP_model("SMD7")
def bipendulum_parameterized(**kwargs):
    R = QQ['x']; (x,) = R._first_ngens(1)
    model_parameters = torch.nn.ParameterDict()
    # Linearized bipendulum
    k = np.ones(7)
    d = np.zeros(7)
    m = np.ones(7)
    A_c, B_c = create_spring_mass_sys_ct(m, k, d, num_actuated=1)
    A_sage = (np.concatenate([A_c, B_c], axis=1) - x*np.eye(2*len(k), 2*len(k)+1)).flatten().tolist()
    A = matrix(R, Integer(2*len(k)), Integer(2*len(k)+1), A_sage)
    return A, model_parameters, {"x":var("x")}

@register_LODEGP_model("SMD8")
def bipendulum_parameterized(**kwargs):
    R = QQ['x']; (x,) = R._first_ngens(1)
    model_parameters = torch.nn.ParameterDict()
    # Linearized bipendulum
    k = np.ones(8)
    d = np.zeros(8)
    m = np.ones(8)
    A_c, B_c = create_spring_mass_sys_ct(m, k, d, num_actuated=1)
    A_sage = (np.concatenate([A_c, B_c], axis=1) - x*np.eye(2*len(k), 2*len(k)+1)).flatten().tolist()
    A = matrix(R, Integer(2*len(k)), Integer(2*len(k)+1), A_sage)
    return A, model_parameters, {"x":var("x")}

@register_LODEGP_model("SMD9")
def bipendulum_parameterized(**kwargs):
    R = QQ['x']; (x,) = R._first_ngens(1)
    model_parameters = torch.nn.ParameterDict()
    # Linearized bipendulum
    k = np.ones(9)
    d = np.zeros(9)
    m = np.ones(9)
    A_c, B_c = create_spring_mass_sys_ct(m, k, d, num_actuated=1)
    A_sage = (np.concatenate([A_c, B_c], axis=1) - x*np.eye(2*len(k), 2*len(k)+1)).flatten().tolist()
    A = matrix(R, Integer(2*len(k)), Integer(2*len(k)+1), A_sage)
    return A, model_parameters, {"x":var("x")}

@register_LODEGP_model("SMD10")
def smd(**kwargs):
    R = QQ['x']; (x,) = R._first_ngens(1)
    model_parameters = torch.nn.ParameterDict()
    # Linearized bipendulum
    k = np.ones(10)
    d = np.zeros(10)
    m = np.ones(10)
    A_c, B_c = create_spring_mass_sys_ct(m, k, d, num_actuated=1)
    A_sage = (np.concatenate([A_c, B_c], axis=1) - x*np.eye(2*len(k), 2*len(k)+1)).flatten().tolist()
    A = matrix(R, Integer(2*len(k)), Integer(2*len(k)+1), A_sage)
    return A, model_parameters, {"x":var("x")}


@register_LODEGP_model("Integrator3D")
def integrator_3d(**kwargs):
    R = QQ['x']; (x,) = R._first_ngens(1)
    model_parameters = torch.nn.ParameterDict()
    # Linearized bipendulum
    A = matrix(R, Integer(2), Integer(3), [-x, 1, 0, 0, -x, 1])
    return A, model_parameters, {"x":var("x")}


@register_LODEGP_model("No system")
def bipendulum_parameterized(**kwargs):
    R = QQ['x']; (x,) = R._first_ngens(1)
    model_parameters = torch.nn.ParameterDict()
    # Linearized bipendulum
    A = matrix(R, Integer(1), Integer(3), [0, 0, 0])
    return A, model_parameters, {"x":var("x")}




@register_LODEGP_model("Three tank")
def three_tank(**kwargs):
    model_parameters = torch.nn.ParameterDict()
    R = QQ['x']; (x,) = R._first_ngens(1)

    # 3 Tank system (5 dimensional uncontrollable system)
    A = matrix(R, Integer(3), Integer(5), [-x, 0, 0, 1, 0, 0, -x, 0, 1, 1, 0, 0, -x, 0, 1])

    return A, model_parameters, {"x":var("x")}


@register_LODEGP_model("Heating")
def heating_system(**kwargs):
    # Heating system with parameters
    F = FunctionField(QQ, names=('a',)); (a,) = F._first_ngens(1)
    F = FunctionField(F, names=('b',)); (b,) = F._first_ngens(1)
    R = F['x']; (x,) = R._first_ngens(1)

    A = matrix(R, Integer(2), Integer(3), [x+a, -a, -1, -b, x+b, 0])
    model_parameters = torch.nn.ParameterDict({
        "a":torch.nn.Parameter(torch.tensor(0.0)),
        "b":torch.nn.Parameter(torch.tensor(0.0))
    })
    x, a, b = var(["x", "a", "b"])
    return A, model_parameters, {"x":x, "a": a, "b": b}

@register_LODEGP_model("Minimal")
def unknown(**kwargs):
    model_parameters = torch.nn.ParameterDict()
    R = QQ['x']; (x,) = R._first_ngens(1)
    # System 1 (no idea)
    A = matrix(R, Integer(1), Integer(2), [x, -1])

    return A, model_parameters, {"x":var("x")}

@register_LODEGP_model("Minimal2")
def unknown(**kwargs):
    model_parameters = torch.nn.ParameterDict()
    R = QQ['x']; (x,) = R._first_ngens(1)
    # System 1 (no idea)
    A = matrix(R, Integer(1), Integer(2), [-x, 1])

    return A, model_parameters, {"x":var("x")}

@register_LODEGP_model("Minimal3")
def unknown(**kwargs):
    model_parameters = torch.nn.ParameterDict()
    R = QQ['x']; (x,) = R._first_ngens(1)
    # System 1 (no idea)
    A = matrix(R, Integer(1), Integer(1), [1 - x])

    return A, model_parameters, {"x":var("x")}

def unknown(**kwargs):
    model_parameters = torch.nn.ParameterDict()
    R = QQ['x']; (x,) = R._first_ngens(1)
    # System 1 (no idea)
    A = matrix(R, Integer(2), Integer(3), [x, -x**2+x-1, x-2, 2-x, x**2-x-1, -x])

    return A, model_parameters, {"x":var("x")}

#        elif ODE_type == "Minimal correct":
#            # \dot{x} = x
#            A = matrix(R, Integer(1), Integer(1), [1-x])
#        elif ODE_type == "Minimal":
#            # \dot{x} = u
#            A = matrix(R, Integer(1), Integer(2), [x, -1])
#        elif ODE_type == "Three Tank":
#            # 3 Tank system (5 dimensional uncontrollable system)
#            A = matrix(R, Integer(3), Integer(5), [-x, 0, 0, 1, 0, 0, -x, 0, 1, 1, 0, 0, -x, 0, 1])
#        elif ODE_type == "Three Tank 2":
#            # 3 Tank system (5 dimensional uncontrollable system)
#            A = matrix(R, Integer(3), Integer(5), [-x, 0, 0, 1, 0, 0, -x, 0, 1, 1, 0, 0, -x, 0, 1])
#        elif ODE_type == "Two Tank":
#            # 3 Tank system (5 dimensional uncontrollable system)
#            A = matrix(R, Integer(2), Integer(3), [-x, 0, 1, 0, -x, 1])
#        elif ODE_type == "Spring Mass":
#            # Spring mass damper system, easy to control
#            A = matrix(R, Integer(2), Integer(3), [-x, 1, 0, -1, -1-x, 1])
#        elif ODE_type == "Spring Mass unstable":
#            A = matrix(R, Integer(2), Integer(3), [-x, 1, 0, 1, -1-x, 1])


#=======================================================================
# LODEGP Class
#=======================================================================
class LODEGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, **kwargs):
        super(LODEGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=num_tasks
        )
        self.num_tasks = num_tasks
        base_kernel = kwargs["base_kernel"] if "base_kernel" in kwargs else "SE_kernel" # "Matern_kernel_52", "Matern_kernel_32", "SE_kernel"
        ODE_name = kwargs["ODE_name"] if "ODE_name" in kwargs else None
        verbose = kwargs["verbose"] if "verbose" in kwargs else False
        if ODE_name is not None:
            self.A, self.model_parameters, self.sage_locals = load_standard_model(ODE_name, kwargs["system_parameters"] if "system_parameters" in kwargs else None)
            self.ODE_name = ODE_name
        else:
            self.A = kwargs["A"]
            self.model_parameters = kwargs["parameter_dict"] if "parameter_dict" in kwargs else torch.nn.ParameterDict()
            self.sage_locals = kwargs["sage_locals"] if "sage_locals" in kwargs else {"x": QQ['x'].gen()}
            self.ODE_name = "Unknown"
        D, U, V = self.A.smith_form()
        if verbose:
            print(f"D:{D}")
            print(f"V:{V}")
            print(f"U:{U}")
        x, a, b = var("x, a, b")
        V_temp = [list(b) for b in V.rows()]
        if verbose:
            print(V_temp)
        V = sage_eval(f"matrix({str(V_temp)})", locals=self.sage_locals)
        #self.V = V
        Vt = V.transpose()
        kernel_matrix, self.kernel_translation_dict, parameter_dict = create_kernel_matrix_from_diagonal(D, base_kernel=base_kernel)
        self.ode_count = self.num_tasks
        self.kernelsize = len(kernel_matrix)
        self.model_parameters.update(parameter_dict)
        if verbose:
            print(self.model_parameters)
        #var(["x", "dx1", "dx2"] + ["t1", "t2"] + [f"LODEGP_kernel_{i}" for i in range(len(kernel_matrix[Integer(0)]))])
        dx1, dx2 = var(["dx1", "dx2"])
        self.k = matrix(Integer(len(kernel_matrix)), Integer(len(kernel_matrix)), kernel_matrix)
        V = V.substitute(x=dx1)
        Vt = Vt.substitute(x=dx2)
        self.V = V
        self.Vt = Vt
        #train_x = self._slice_input(train_x)

        self.sage_locals["t1"] = var("t1")
        self.sage_locals["t2"] = var("t2")

        self.common_terms = {
            "t_diff" : train_x-train_x.t(),
            "t_sum" : train_x+train_x.t(),
            "t_ones": torch.ones_like(train_x-train_x.t()),
            "t_zeroes": torch.zeros_like(train_x-train_x.t())
        }
        self.matrix_multiplication = matrix(self.k.base_ring(), len(self.k[0]), len(self.k[0]), (V*self.k*Vt))
        # Alias
        self.VkV = self.matrix_multiplication
        self.diffed_kernel = differentiate_kernel_matrix(self.k, V, Vt, self.kernel_translation_dict, dx1=dx2, dx2=dx2, base_kernel = base_kernel)
        self.sum_diff_replaced = replace_sum_and_diff(self.diffed_kernel)
        self.covar_description = translate_kernel_matrix_to_gpytorch_kernel(self.sum_diff_replaced, self.model_parameters, common_terms=self.common_terms)
        self.covar_module = LODE_Kernel(self.covar_description, self.model_parameters)


    def prepare_symbolic_ode_satisfaction_check(self, target, columnwise=True):
        """
        Create all the parameters required to run "calculate_differential_equation_error_symbolic"
        Note: If columnwise is True, you want to calculate the derivative for "t1", otherwise for "t2"
        Returns the following outputs:
        - model_diffed_kernel: the relevant row/column of the symbolic kernel matrix
        """
        if columnwise:
            model_diffed_kernel = [self.diffed_kernel[i][target] for i in range(len(self.diffed_kernel))]
        else:
            model_diffed_kernel = [self.diffed_kernel[target][i] for i in range(len(self.diffed_kernel))]
        return model_diffed_kernel

    def return_cov_fkt_row(self, target_row: int):
        """
        To be used in the row-wise verification that the ODE is satisfied.
        Corresponding equation: VkV' \cdot A^T
        Therefore it needs to be used on the second element of k.
        """
        if target_row >= len(self.diffed_kernel[0]):
            raise ValueError(f"target_row {target_row} is out of bounds for the kernel matrix with size {len(self.diffed_kernel[0])}")
        if target_row < 0:
            raise ValueError(f"target_row {target_row} cannot be negative")
        return [self.diffed_kernel[target_row][i] for i in range(len(self.diffed_kernel))]

    def return_cov_fkt_col(self, target_col: int):
        """
        To be used in the column-wise verification that the ODE is satisfied.
        Corresponding equation: A \cdot VkV'
        Therefore it needs to be used on the first element of k.
        """
        if target_col >= len(self.diffed_kernel[0]):
            raise ValueError(f"target_col {target_col} is out of bounds for the kernel matrix with size {len(self.diffed_kernel[0])}")
        if target_col < 0:
            raise ValueError(f"target_col {target_col} cannot be negative")
        return [self.diffed_kernel[i][target_col] for i in range(len(self.diffed_kernel))]



    def prepare_numeric_ode_satisfaction_check(self):
        """
        Create all the parameters required to run "calculate_differential_equation_error_numeric"
        Returns the following outputs:
        - local_values: a dictionary with the current parameter values 
        """
        local_values = {var(param_name): torch.exp(self.model_parameters[param_name]).item() for param_name in self.model_parameters}
        return local_values

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
