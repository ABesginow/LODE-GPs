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
        # Heating system with parameters
        #F = FunctionField(QQ, names=('a',)); (a,) = F._first_ngens(1)
        #F = FunctionField(F, names=('b',)); (b,) = F._first_ngens(1)
        #R = F['x']; (x,) = R._first_ngens(1)

        #A = matrix(R, Integer(2), Integer(3), [x+a, -a, -1, -b, x+b, 0])
        #self.model_parameters = torch.nn.ParameterDict({
        #    "a":torch.nn.Parameter(torch.tensor(0.0)),
        #    "b":torch.nn.Parameter(torch.tensor(0.0))
        #})


        self.model_parameters = torch.nn.ParameterDict()
        R = QQ['x']; (x,) = R._first_ngens(1)
        # System 1 (no idea)
        #A = matrix(R, Integer(2), Integer(3), [x, -x**2+x-1, x-2, 2-x, x**2-x-1, -x])
        # Linearized bipendulum
        A = matrix(R, Integer(2), Integer(3), [x**2 + 9.81, 0, -1, 0, x**2+4.905, -1/2])
        # 3 Tank system (5 dimensional uncontrollable system)
        #A = matrix(R, Integer(3), Integer(5), [-x, 0, 0, 1, 0, 0, -x, 0, 1, 1, 0, 0, -x, 0, 1])

        D, U, V = A.smith_form()
        print(f"D:{D}")
        print(f"V:{V}")
        x, a, b = var("x, a, b")
        V_temp = [list(b) for b in V.rows()]
        print(V_temp)
        V = sage_eval(f"matrix({str(V_temp)})", locals={"x":x, "a":a, "b":b})
        Vt = V.transpose()
        kernel_matrix, self.kernel_translation_dict, parameter_dict = create_kernel_matrix_from_diagonal(D)
        self.ode_count = A.nrows()
        self.kernelsize = len(kernel_matrix)
        self.model_parameters.update(parameter_dict)
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
        #print(torch.linalg.eigvalsh(covar_x.evaluate()))
        #covar_x = covar_x.flatten()
        #print(list(torch.linalg.eigh(covar_x)[0])[::-1])
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x) 




torch.set_default_tensor_type(torch.DoubleTensor)

num_data = 50 
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
one   = -float(41)/float(100)*torch.sin(float(3)*train_x)/torch.pow((train_x+int(1)), float(1))     - float(3)/float(5)*torch.cos(float(3)*train_x)/torch.pow((train_x+int(1)), float(2))           + float(1)/float(5)*torch.sin(float(3)*train_x)/torch.pow((train_x+int(1)), float(3))
two   = float(81)/float(2000)*torch.sin(float(3)*train_x)/torch.pow((train_x+int(1)), float(1))     - float(3)/float(10)*torch.cos(float(3)*train_x)/torch.pow((train_x+int(1)), float(2))                   + float(1)/float(10)*torch.sin(float(3)*train_x)/torch.pow((train_x+int(1)), float(3))
three = -float(3321)/float(10000)*torch.sin(float(3)*train_x)/torch.pow((train_x+int(1)), float(1)) + float(987)/float(500)*torch.cos(float(3)*train_x)/torch.pow((train_x+int(1)), float(2)) - float(3929)/float(500)*torch.sin(float(3)*train_x)/torch.pow((train_x+int(1)), float(3)) - float(36)/float(5)*torch.cos(float(3)*train_x)/torch.pow((train_x+int(1)), float(4)) + float(12)/float(5)*torch.sin(float(3)*train_x)/torch.pow((train_x+int(1)), float(5))
#
train_y = torch.stack((one, two, three), -1)

# Three tank
#one = float(1)*torch.exp(float(-0.5)*train_x)
#two = float(1)*torch.exp(float(-0.25)*train_x)
#three = float(1)*torch.exp(float(-0.25)*train_x) - float(1)*torch.exp(float(-0.5)*train_x)
#four = -float(0.5)*torch.exp(float(-0.5)*train_x)
#five = - float(0.25)*torch.exp(float(-0.25)*train_x) + float(0.5)*torch.exp(float(-0.5)*train_x)
#train_y = torch.stack([one, two, three, four, five], int(-1))

num_tasks = 3

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks, noise_constraint=gpytorch.constraints.Positive())
start = time.time()
model = LODEGP(train_x, train_y, likelihood, num_tasks)
end = time.time()
model(train_x)

# Find optimal model hyperparameters
model.train()
likelihood.train()

training_iterations = 100
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
test_start = 1
test_end = 5 
test_count = 1000
eval_step_size = 1e-4
second_derivative=True 
divider = 3 if second_derivative else 2
number_of_samples = int(test_count/divider)
test_x = torch.linspace(test_start, test_end, number_of_samples)
if second_derivative:
    test_x = torch.cat([test_x, test_x+torch.tensor(eval_step_size), test_x+torch.tensor(2*eval_step_size)])
else:
    test_x = torch.cat([test_x, test_x+torch.tensor(eval_step_size)])
test_x = test_x.sort()[0]
model.eval()
likelihood.eval()

#output = model(test_x)
with torch.no_grad():
    output = likelihood(model(test_x))
# ODE solution precision evaluation
# Idea: Generate splines based on the mean/samples from the GP
# The sagemath spline function can be directly differentiated (up to grade 2)

# Then it's up to the user to manually write down the differential equation again 
# and sum up the (absolute) error terms

fkt = list()
for dimension in range(model.kernelsize):
    output_channel = output.mean[:, dimension]
    fkt.append(spline([(t, y) for t, y in zip(test_x, output_channel)]))


#ode_test_vals = np.linspace(test_start, test_end, 10)
ode_test_vals = test_x

# System 1 (no idea)
#A = matrix(R, Integer(2), Integer(3), [x, -x**2+x-1, x-2 | 2-x, x**2-x-1, -x])
#ode1 = lambda val: fkt[0].derivative(val, 1) - fkt[1].derivative(val, 2) + fkt[1].derivative(val, 1) - fkt[1](val) + fkt[2].derivative(val, 1) - fkt[2](val)
#ode2 = lambda val: 2*fkt[0](val) - fkt[0].derivative(val, 1) + fkt[1].derivative(val, 2) - fkt[1].derivative(val, 1) - fkt[1](val) - fkt[2].derivative(val, 1)

# Heating system with parameters
#A = matrix(R, Integer(2), Integer(3), [])
#a = torch.exp(model.model_parameters["a"].detach())
#b = torch.exp(model.model_parameters["b"].detach())
#print(f"a diff:{3 - a}")
#print(f"b diff:{1 - b}")
#ode1 = lambda val: fkt[0].derivative(val, 1) + fkt[0](val)*a - fkt[1](val)*a - fkt[2](val)
#ode2 = lambda val: -fkt[0](val)*b + fkt[1].derivative(val, 1) + fkt[1](val)*b


# Linearized bipendulum
#A = matrix(R, Integer(2), Integer(3), [x**2 + 9.81, 0, -1| 0, x**2+4.905, -1/2])
ode1 = lambda val: fkt[0].derivative(val, 2) + 9.81*fkt[0](val) - fkt[2](val)
ode2 = lambda val: fkt[1].derivative(val, 2) + 4.905*fkt[1](val) - 1/2*fkt[2](val)

# 3 Tank system (5 dimensional uncontrollable system)
#A = matrix(R, Integer(3), Integer(5), [-x, 0, 0, 1, 0| 0, -x, 0, 1, 1| 0, 0, -x, 0, 1])
#ode1 = lambda val: -fkt[0].derivative(val, 1) + fkt[3](val)
#ode2 = lambda val: -fkt[1].derivative(val, 1) + fkt[3](val) + fkt[4](val)
#ode3 = lambda val: -fkt[2].derivative(val, 1) + fkt[4](val)

ode_error_list = [[] for _ in range(model.ode_count)]
for val in ode_test_vals:
    for i in range(model.ode_count):
        ode_error_list[i].append(np.abs(globals()[f"ode{i+1}"](val)))

print(np.mean(ode_error_list[0]))
print(np.mean(ode_error_list[1]))
#print(np.mean(ode_error_list[2]))
print(np.mean(ode_error_list))




def calc_finite_differences(sample, point_step_size, skip=False, number_of_samples=0):
    """
    param skip: Decides whether to skip every second value of the sample.
                Useful for cases where original samples aren't equidistant
    """
    if sample.ndim == 2:
        NUM_CHANNELS = sample.shape[1]
    else:
        NUM_CHANNELS = 1
    if number_of_samples == 0:
        number_of_samples = sample.shape[0]

    gradients_list = list()
    if skip:
        step = 2
    for index in range(0, step*number_of_samples, step):
        gradients_list.append(list((-sample[index] + sample[index+1])/point_step_size))
    return gradients_list




sample = output.mean

sample_result = {}
#with gpytorch.settings.fast_pred_var():
#    sample = outputs.sample().float()
sample_result["sample"] = sample
#all_samples[sample_rep]["sample"] = sample
#step_size = float(float(END-START) / number_of_samples)
#sample = outputs.mean
dgl1_difference = []
dgl2_difference = []
c1_fin_diff = []
c2_fin_diff = []
c1_fin_d2 = []
c2_fin_d2 = []

import scipy.signal
b, a = scipy.signal.butter(3, 0.05, "lowpass")

# Makes sure the third entry (and following dividable by three) are skipped
mask_s1_s2 = [False if i%3 == 0 else True for i in range(1, len(sample)+1)]
# Skips the first and then every three
mask_s2_s3 = [False if i%3 == 0 else True for i in range(0, len(sample))]

s1_s2 = sample[mask_s1_s2]
s2_s3 = sample[mask_s2_s3]
s1_s2_fin_diffs = calc_finite_differences(s1_s2, eval_step_size, skip=True, number_of_samples=number_of_samples)

s2_s3_fin_diffs = calc_finite_differences(s2_s3, eval_step_size, skip=True, number_of_samples=number_of_samples)

# To calculate the second derivatives, I just need to calculate the finite differences
# of the first derivatives I have, then I have the second derivative for the point in
# the middle of them

# This smoothing block is stored in case I ever need it (e.g. if decreasing the step size too much)

s1_s2_fin_diff_smooth_c1 = scipy.signal.filtfilt(b, a, np.array(s1_s2_fin_diffs)[:,0])
s1_s2_fin_diff_smooth_c2 = scipy.signal.filtfilt(b, a, np.array(s1_s2_fin_diffs)[:,1])
s1_s2_fin_diff_smooth_c3 = scipy.signal.filtfilt(b, a, np.array(s1_s2_fin_diffs)[:,2])
s1_s2_fin_diffs_smooth = [[c1, c2, c3] for c1, c2, c3 in zip(s1_s2_fin_diff_smooth_c1, s1_s2_fin_diff_smooth_c2, s1_s2_fin_diff_smooth_c3)]

s2_s3_fin_diff_smooth_c1 = scipy.signal.filtfilt(b, a, np.array(s2_s3_fin_diffs)[:,0])
s2_s3_fin_diff_smooth_c2 = scipy.signal.filtfilt(b, a, np.array(s2_s3_fin_diffs)[:,1])
s2_s3_fin_diff_smooth_c3 = scipy.signal.filtfilt(b, a, np.array(s2_s3_fin_diffs)[:,2])
s2_s3_fin_diffs_smooth = [[c1, c2, c3] for c1, c2, c3 in zip(s2_s3_fin_diff_smooth_c1, s2_s3_fin_diff_smooth_c2, s2_s3_fin_diff_smooth_c3)]


import itertools
# Same as above
first_deriv_interleaved_smooth = torch.Tensor(list(itertools.chain(*zip(s1_s2_fin_diffs_smooth, s2_s3_fin_diffs_smooth))))
first_deriv_interleaved = torch.Tensor(list(itertools.chain(*zip(s1_s2_fin_diffs, s2_s3_fin_diffs))))
s2_fin_d2 = calc_finite_differences(first_deriv_interleaved, eval_step_size, skip=True, number_of_samples=number_of_samples)

# Same as above
#s2_fin_d2_smooth_c1 = scipy.signal.filtfilt(b, a, np.array(s2_fin_d2)[:,0])
#s2_fin_d2_smooth_c2 = scipy.signal.filtfilt(b, a, np.array(s2_fin_d2)[:,1])
#s2_fin_d2_smooth_c3 = scipy.signal.filtfilt(b, a, np.array(s2_fin_d2)[:,2])
#s2_fin_d2_smooth = [[c1, c2, c3] for c1, c2, c3 in zip(s2_fin_d2_smooth_c1, s2_fin_d2_smooth_c2, s2_fin_d2_smooth_c3)]

for index in range(number_of_samples):
    c1 = sample[index*3+1][0]
    c2 = sample[index*3+1][1]
    c3 = sample[index*3+1][2]

    #A = matrix(R, 2, 3,  [x^2+9.81, 0, -1, 0, x^2+4.9, -1/2])

    # Check the error in solving the first differential equation
    dgl1_diff = abs(s2_fin_d2[index][0]+float(9.81)*c1 -c3)
    dgl1_difference.append(dgl1_diff)

    # Check the error in solving the second differential equation
    dgl2_diff = abs(s2_fin_d2[index][1]+float(4.905)*c2 -float(0.5)*c3)
    dgl2_difference.append(dgl2_diff)

#experiment.print_log(f"model sample avg. error c1:{torch.mean(torch.Tensor(dgl1_difference))}")
#experiment.print_log(f"model sample avg. error c2:{torch.mean(torch.Tensor(dgl2_difference))}")
sample_result["avg_dgl_error"] = 0.5*(np.mean(dgl1_difference) + np.mean(dgl2_difference))
sample_result["dgl1_difference"] = dgl1_difference
sample_result["dgl2_difference"] = dgl2_difference
print(sample_result["avg_dgl_error"])
print([np.mean(dgl1_difference), np.mean(dgl2_difference)])