import gpytorch 
from gpytorch.kernels.kernel import Kernel
from sage.all import *
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
from kernels import *
import pprint
import time
import torch
import matplotlib.pyplot as plt


class LODEGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks):
        super(LODEGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=num_tasks
        )
        R = QQ['x']; (x,) = R._first_ngens(1)
        # System 1 (no idea)
        #A = matrix(R, Integer(2), Integer(3), [x, -x**2+x-1, x-2, 2-x, x**2-x-1, -x])
        # Heating system with parameters
        #A = matrix(R, Integer(2), Integer(3), [])
        # Linearized bipendulum
        #A = matrix(R, Integer(2), Integer(3), [x**2 + 9.81, 0, -1, 0, x**2+4.905, -1])
        # 3 Tank system (5 dimensional uncontrollable system)
        A = matrix(R, Integer(3), Integer(5), [-x, 0, 0, 1, 0, 0, -x, 0, 1, 1, 0, 0, -x, 0, 1])

        D, U, V = A.smith_form()
        print(f"D:{D}")
        print(f"V:{V}")
        Vt = V.transpose()
        kernel_matrix, self.kernel_translation_dict, parameter_dict = create_kernel_matrix_from_diagonal(D)
        self.kernelsize = len(kernel_matrix)
        self.model_parameters = parameter_dict
        PP = PolynomialRing(QQ, ["x", "dx1", "dx2"] + [f"LODEGP_kernel_{i}" for i in range(len(kernel_matrix[Integer(0)]))])
        var(["x", "dx1", "dx2"] + ["t1", "t2"] + [f"LODEGP_kernel_{i}" for i in range(len(kernel_matrix[Integer(0)]))])
        k = matrix(PP, Integer(len(kernel_matrix)), Integer(len(kernel_matrix)), kernel_matrix)
        V = V.change_ring(PP)
        Vt = Vt.change_ring(PP)
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
        self.covar_module = LODE_Kernel(self.covar_description, parameter_dict)


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
        #covar_x = covar_x.flatten()
        #print(list(torch.linalg.eigh(covar_x)[0])[::-1])
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x) 

torch.set_default_tensor_type(torch.DoubleTensor)

num_data = 15
train_x = torch.linspace(0, 15, num_data)
# System 1
#one = -0.25*torch.exp(train_x) + 2*(torch.cos(train_x) + torch.sin(train_x)) 
#two = 4*torch.sin(train_x) 
#three = -0.25*torch.exp(train_x) + 2*(torch.cos(train_x) - torch.sin(train_x)) 
## Heating system 
#one = float(0.5)*torch.cos(float(0.5)*train_x)*torch.exp(float(-0.1)*train_x) + float(0.9)*torch.exp(float(-0.1)*train_x)*torch.sin(float(0.5)*train_x)
#two = torch.sin(float(0.5)*train_x)*torch.exp(float(-0.1)*train_x)
#three = float(1.9)*torch.cos(float(0.5)*train_x)*torch.exp(float(-0.1)*train_x) - float(16/25)*torch.exp(float(-0.1)*train_x)*torch.sin(float(0.5)*train_x)
# Bipendulum
#one   = -float(41)/float(100)*torch.sin(float(3)*train_x)/torch.pow((train_x+int(1)), float(1))     - float(3)/float(5)*torch.cos(float(3)*train_x)/torch.pow((train_x+int(1)), float(2))           + float(1)/float(5)*torch.sin(float(3)*train_x)/torch.pow((train_x+int(1)), float(3))
#two   = float(81)/float(2000)*torch.sin(float(3)*train_x)/torch.pow((train_x+int(1)), float(1))     - float(3)/float(10)*torch.cos(float(3)*train_x)/torch.pow((train_x+int(1)), float(2))                   + float(1)/float(10)*torch.sin(float(3)*train_x)/torch.pow((train_x+int(1)), float(3))
#three = -float(3321)/float(10000)*torch.sin(float(3)*train_x)/torch.pow((train_x+int(1)), float(1)) + float(987)/float(500)*torch.cos(float(3)*train_x)/torch.pow((train_x+int(1)), float(2)) - float(3929)/float(500)*torch.sin(float(3)*train_x)/torch.pow((train_x+int(1)), float(3)) - float(36)/float(5)*torch.cos(float(3)*train_x)/torch.pow((train_x+int(1)), float(4)) + float(12)/float(5)*torch.sin(float(3)*train_x)/torch.pow((train_x+int(1)), float(5))
#
#train_y = torch.stack((one, two, three), -1)
# Three tank
one = float(1)*torch.exp(float(-0.5)*train_x)
two = float(1)*torch.exp(float(-0.25)*train_x)
three = float(1)*torch.exp(float(-0.25)*train_x) - float(1)*torch.exp(float(-0.5)*train_x)
four = -float(0.5)*torch.exp(float(-0.5)*train_x)
five = - float(0.25)*torch.exp(float(-0.25)*train_x) + float(0.5)*torch.exp(float(-0.5)*train_x)
train_y = torch.stack([one, two, three, four, five], int(-1))

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(5)
start = time.time()
model = LODEGP(train_x, train_y, likelihood, 5)
end = time.time()
model(train_x)

# Find optimal model hyperparameters
model.train()
likelihood.train()

training_iterations = 50
# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
print(list(model.named_parameters()))
for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
    optimizer.step()

print(list(model.named_parameters()))

test_x = torch.linspace(0, 1, 10)
model.eval()
likelihood.eval()

model(test_x)