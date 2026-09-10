#=======================================================================
# Imports
#=======================================================================
import gpytorch 
from sage.all import *
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
from lodegp.kernels import *
import pprint
import torch
from lodegp.LODEGP import LODEGP
from GP_helpers.helpers.util_functions import (
    calculate_differential_equation_error_symbolic,
    calculate_differential_equation_error_numeric,
)
import pytest
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

def symmetry_check(model): 
    dx1, dx2 = var('dx1, dx2')
    symmetry_check_matrix = model.matrix_multiplication - matrix([[cell.substitute(dx1=dx2, dx2=dx1) for cell in row] for row in model.matrix_multiplication.T])
    # Symmetric up to numerical precision
    return all([all([cell < 1e-10 for cell in row]) for row in symmetry_check_matrix])

def eigval_check(model, train_x):
    covar_matrix = model(train_x).covariance_matrix
    eigvals = torch.linalg.eig(covar_matrix)[0]
    print("Eigenwerte (real):", eigvals)
    return all([eig.real > -1e-10 for eig in eigvals]), all([eig.imag < 1e-10 for eig in eigvals])

def test_eigenvalues():
    train_x = torch.linspace(0, 1, 10)
    train_y = torch.linspace(0, 1, 10)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(3, noise_constraint=gpytorch.constraints.Positive())
    model = LODEGP(train_x, train_y, likelihood, 3, ODE_name="Heating")
    eig_positive, eig_real = eigval_check(model, train_x)
    assert eig_positive 
    assert eig_real 
    
def test_symmetry_check():
    train_x = torch.linspace(0, 1, 10)
    train_y = torch.linspace(0, 1, 10)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(3, noise_constraint=gpytorch.constraints.Positive())
    model = LODEGP(train_x, train_y, likelihood, 3, ODE_name="Heating")
   # This matrix is suppossed to contain _only_ zeros
    assert symmetry_check(model)
    
   
def test_function():
    train_x = torch.linspace(0, 1, 10)
    train_y = torch.linspace(0, 1, 10)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(3, noise_constraint=gpytorch.constraints.Positive())
    A = matrix(SR, 1, 1, [1])
    x = var('x')
    model = LODEGP(train_x, train_y, likelihood, 1, A=A, sage_locals={'x': x})
    assert model.A == A
# Not running matern 32 due to highest order of derivative being greater than 1
@pytest.mark.parametrize("base_kernel", ["SE_kernel", "Matern_kernel_52"])
def test_heating(base_kernel):
    train_x = torch.linspace(0, 1, 10)
    train_y = torch.linspace(0, 1, 10)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(3)
    model = LODEGP(train_x, train_y, likelihood, 3, ODE_name="Heating", base_kernel=base_kernel)
    eig_positive, eig_real = eigval_check(model, train_x)
    assert eig_positive
    assert eig_real
    
# Not running matern kernels due to highest order of derivative being greater than 2 
@pytest.mark.parametrize("base_kernel", ["SE_kernel"])
def test_bipendulumkernel(base_kernel):
    train_x = torch.linspace(0, 1, 10)
    train_y = torch.linspace(0, 1, 10)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(3)
    model = LODEGP(train_x, train_y, likelihood, 3, ODE_name="Bipendulum", base_kernel=base_kernel)
    eig_positive, eig_real = eigval_check(model, train_x)
    assert eig_positive
    assert eig_real
    model.sage_locals = {'x': var('x')}
    
@pytest.mark.parametrize("base_kernel", ["SE_kernel", "Matern_kernel_32", "Matern_kernel_52"])
def test_three_tank(base_kernel):
    train_x = torch.linspace(0, 1, 10)
    train_y = torch.linspace(0, 1, 10)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(5)
    model = LODEGP(train_x, train_y, likelihood, 5, ODE_name="Three tank", base_kernel=base_kernel)
    eig_positive, eig_real = eigval_check(model, train_x)
    
    assert eig_positive
    assert eig_real


def test_calculate_differential_equation_error_symbolic():
    train_x = torch.linspace(0, 1, 10)
    train_y = torch.linspace(0, 1, 10)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(3)
    model = LODEGP(train_x, train_y, likelihood, 3, ODE_name="Bipendulum")
    ode_row_to_check = 1

    # Verify that the models output satisfies the given differential equation
    target_row = 2
    # Row is to be used with VkV' * A
    # Apply to second element of k
    model_cov_fkt_row = model.return_cov_fkt_row(target_row)
    diff_var = var("t2")
    differential_equation = [term.substitute(x=diff_var) for term in sage_eval(str(model.A[ode_row_to_check]), locals=model.sage_locals)]
    print(calculate_differential_equation_error_symbolic(model_cov_fkt_row, differential_equation, model.sage_locals, diff_var=diff_var)(t1=1, t2=1, signal_variance_2=1.0, lengthscale_2=1.0, a=1.0, b=1.0))


    # Column is to be used with A * VkV'
    # Apply to first element of k
    target_col = 0
    model_cov_fkt_col = model.return_cov_fkt_col(target_col)
    diff_var = var("t1")
    differential_equation = [term.substitute(x=diff_var) for term in sage_eval(str(model.A[ode_row_to_check]), locals=model.sage_locals)]
    print(calculate_differential_equation_error_symbolic(model_cov_fkt_col, differential_equation, model.sage_locals, diff_var=diff_var)(t1=1, t2=1, signal_variance_2=1.0, lengthscale_2=1.0, a=1.0, b=1.0))




def test_calculate_differential_equation_error_numeric():


    # Trainingsdaten basierend auf e^x
    train_x = torch.linspace(0, 1, 10)
    train_y = torch.exp(train_x)  # e^x
    # packen wir in 2D (z.B. y und y')
    train_y = torch.stack([train_y, train_y], -1)  

    # Likelihood und Modell
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        2, noise_constraint=gpytorch.constraints.Positive()
    )
    model = LODEGP(train_x, train_y, likelihood, 2, ODE_name="Minimal")

    # Lokale Werte
    local_values = model.prepare_numeric_ode_satisfaction_check()
    print("Lokale Werte:", local_values)

    # Differentialgleichung vorbereiten
    target_row = 1
    ode_row_to_check = 0
    diff_var = var("t2")
    differential_eq = [
        term.substitute(x=diff_var)
        for term in sage_eval(str(model.A[ode_row_to_check]), locals=model.sage_locals)
    ]

    model.eval()
    likelihood.eval()
    sage_locals = model.sage_locals
    data = torch.linspace(0, 3, 200)  # Testbereich etwas größer

    # Daten generierende Funktion: Modell-Ausgabe
    data_generating_functions = lambda x: model(x).mean

    # Numerischen ODE-Fehler berechnen
    with gpytorch.settings.prior_mode(True):
        result = calculate_differential_equation_error_numeric(
            differential_eq,
            sage_locals,
            data_generating_functions,        
            data,
            locals_values=local_values
        )

    print("Numerischer ODE-Fehler:", result)
    plt.plot(data, result.detach(), label="ODE Fehler")
    plt.axhline(0, color="black", linestyle="--")
    plt.legend()
    plt.show()
  


def test_regex():
    test_expressions = [
    "e^(x+y)",                
    "e^z",                    
    "sin(x)^2 + cos(y)",      
    "sqrt(x + y)",            
    "exp(x)"   ,  
    "x^2"           
]
    for expr in test_expressions:
        transformed = replace_basic_operations(expr)
        print(f"Ersetzt : {transformed}")




