import gpytorch
from LODEGP import LODEGP
from sage.all import *
import sage
import torch

torch.set_default_dtype(torch.float64)

# Generate a solution to the physical system
START = 2
END = 12
COUNT = 100
train_x = torch.linspace(START, END, COUNT)

y0_func = lambda x: float(781/8000)*torch.sin(x)/x - float(1/20)*torch.cos(x)/x**2 + float(1/20)*torch.sin(x)/x**3
y1_func = lambda x: float(881/8000)*torch.sin(x)/x - float(1/40)*torch.cos(x)/x**2 + float(1/40)*torch.sin(x)/x**3
y2_func = lambda x: float(688061/800000)*torch.sin(x)/x - float(2543/4000)*torch.cos(x)/x**2 + float(1743/4000)*torch.sin(x)/x**3 - float(3/5)*torch.cos(x)/x**4 + float(3/5)*torch.sin(x)/x**5 
y0 = y0_func(train_x)
y1 = y1_func(train_x)
y2 = y2_func(train_x)
train_y = torch.stack([y0, y1, y2], dim=-1)

# 1. Model definition
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)
model = LODEGP(train_x, train_y, likelihood, 3, ODE_name="Bipendulum", verbose=True, system_parameters={"l1": 1.0, "l2": 2.0}, base_kernel="Matern_kernel_52")
print(model(train_x))
