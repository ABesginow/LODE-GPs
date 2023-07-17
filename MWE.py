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
        self.ode_count = A.nrows()
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

test_start = 0
test_end = 10
test_count = 1000 
second_derivative = False
eval_step_size = 1e-6
divider = 2
number_of_samples = int(test_count/divider)

test_x = torch.linspace(test_start, test_end, test_count)
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

one = float(1)*torch.exp(float(-0.5)*test_x)
two = float(1)*torch.exp(float(-0.25)*test_x)
three = float(1)*torch.exp(float(-0.25)*test_x) - float(1)*torch.exp(float(-0.5)*test_x)
four = -float(0.5)*torch.exp(float(-0.5)*test_x)
five = - float(0.25)*torch.exp(float(-0.25)*test_x) + float(0.5)*torch.exp(float(-0.5)*test_x)
train_y = torch.stack([one, two, three, four, five], int(-1))

# Then it's up to the user to manually write down the differential equation again 
# and sum up the (absolute) error terms
sample_repititions = 1
all_samples = list()
for sample_rep in range(sample_repititions):
    sample_result = {}
    sample = output.mean
    sample = train_y
    #with gpytorch.settings.fast_pred_var():
    #    sample = outputs.sample()

    dgl1_difference = []
    dgl2_difference = []
    dgl3_difference = []
    c1_fin_diff = []
    c2_fin_diff = []
    c1_fin_d2 = []
    c2_fin_d2 = []

    import scipy.signal
    b, a = scipy.signal.butter(3, 0.05, "lowpass")

    if not second_derivative:
        s1_s2_fin_diffs = calc_finite_differences(sample, eval_step_size, skip=True, number_of_samples=number_of_samples)
    else:
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

    if second_derivative:
        s2_s3_fin_diff_smooth_c1 = scipy.signal.filtfilt(b, a, np.array(s2_s3_fin_diffs)[:,0])
        s2_s3_fin_diff_smooth_c2 = scipy.signal.filtfilt(b, a, np.array(s2_s3_fin_diffs)[:,1])
        s2_s3_fin_diff_smooth_c3 = scipy.signal.filtfilt(b, a, np.array(s2_s3_fin_diffs)[:,2])
        s2_s3_fin_diffs_smooth = [[c1, c2, c3] for c1, c2, c3 in zip(s2_s3_fin_diff_smooth_c1, s2_s3_fin_diff_smooth_c2, s2_s3_fin_diff_smooth_c3)]


    import itertools
    # Same as above
    #first_deriv_interleaved_smooth = torch.Tensor(list(itertools.chain(*zip(s1_s2_fin_diffs_smooth, s2_s3_fin_diffs_smooth))))
    #first_deriv_interleaved = torch.Tensor(list(itertools.chain(*zip(s1_s2_fin_diffs, s2_s3_fin_diffs))))
    #s2_fin_d2 = experiment.calc_finite_differences(first_deriv_interleaved, eval_step_size, skip=True, number_of_samples=number_of_samples)

    # Same as above
    #s2_fin_d2_smooth_c1 = scipy.signal.filtfilt(b, a, np.array(s2_fin_d2)[:,0])
    #s2_fin_d2_smooth_c2 = scipy.signal.filtfilt(b, a, np.array(s2_fin_d2)[:,1])
    #s2_fin_d2_smooth_c3 = scipy.signal.filtfilt(b, a, np.array(s2_fin_d2)[:,2])
    #s2_fin_d2_smooth = [[c1, c2, c3] for c1, c2, c3 in zip(s2_fin_d2_smooth_c1, s2_fin_d2_smooth_c2, s2_fin_d2_smooth_c3)]

    for index in range(number_of_samples):
        c1 = sample[index*divider+1][0]
        c2 = sample[index*divider+1][1]
        c3 = sample[index*divider+1][2]
        c4 = sample[index*divider+1][3]
        c5 = sample[index*divider+1][4]

        #A = matrix(R, 2, 3,  [x^2+9.81, 0, -1, 0, x^2+4.9, -1/2])

        # Check the error in solving the first differential equation
        dgl1_diff = abs(-s1_s2_fin_diffs[index][0]+c4)
        dgl1_difference.append(dgl1_diff)

        # Check the error in solving the second differential equation
        dgl2_diff = abs(-s1_s2_fin_diffs[index][1]+c4 +c5)
        dgl2_difference.append(dgl2_diff)

        dgl3_diff = abs(-s1_s2_fin_diffs[index][2]+c5)
        dgl3_difference.append(dgl3_diff)

    #experiment.print_log(f"model sample avg. error c1:{torch.mean(torch.Tensor(dgl1_difference))}")
    #experiment.print_log(f"model sample avg. error c2:{torch.mean(torch.Tensor(dgl2_difference))}")
    sample_result["avg_dgl_error"] = 0.333*(np.mean(dgl1_difference) + np.mean(dgl2_difference) + np.mean(dgl3_difference))
    sample_result["dgl1_difference"] = dgl1_difference
    sample_result["dgl2_difference"] = dgl2_difference
    sample_result["dgl3_difference"] = dgl3_difference
    all_samples.append([np.mean(dgl1_difference), np.mean(dgl2_difference), np.mean(dgl3_difference)])
    print(sample_result)
    print(all_samples)
