import pytest
import gpytorch
from kernels import *
from sage.all import *
import torch

def symmetry_check(model):
    model.matrix_multiplication - matrix([[cell.substitute(dx1=dx2, dx2=dx1) for cell in row] for row in model.matrix_multiplication.T])

class TestBipendulum:
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

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(3r, noise_constraint=gpytorch.constraints.Positive())
    model = LODEGP(train_x, train_y, likelihood, 3)

    def test_symmetry_check(self):
        # This matrix is suppossed to contain _only_ zeros
        assert symmetry_check(model)

    def test_satisfying_ode(self):
        l1, l2, s1, s2 = var("l1, l2, s1, s2")
        K_discussion = matrix([[cell.substitute(signal_variance_2=s2, lengthscale_2=l1, a=3, b=1) if type(cell) == sage.symbolic.expression.Expression else cell  for cell in row ] for row in model.diffed_kernel])
        for row in len(model.diffed_kernel):
            # Assert that they're 0 or at least very very very close to zero
            (K_discussion[row][0].diff(t2, 2) + 9.81*K_discussion[row][0] - K_discussion[row][2]).full_simplify()
            (K_discussion[row][1].diff(t2, 2) + 1/2*9.81*K_discussion[row][1] - 1/2*K_discussion[row][2]).simplify_full()

    def test_equality_maple_solution(self):
        # This covariance function is calculated using Maple, we _know_ that this is correct
        KK = Matrix(3, 3, [[1/4*exp(-1/2*(t1 - t2)^2/l^2)*(g^2*l^8 - 4*g*l^6 + (4*g*t1^2 - 8*g*t1*t2 + 4*g*t2^2 + 12)*l^4 - 24*(t1 - t2)^2*l^2 + 4*(t1 - t2)^4)/l^8, 1/4*(g^2*l^8 - 3*g*l^6 + (3*g*t1^2 - 6*g*t1*t2 + 3*g*t2^2 + 6)*l^4 - 12*(t1 - t2)^2*l^2 + 2*(t1 - t2)^4)*exp(-1/2*(t1 - t2)^2/l^2)/l^8, 1/4*(g^3*l^12 - 5*g^2*l^10 + (5*g^2*t1^2 - 10*g^2*t1*t2 + 5*g^2*t2^2 + 24*g)*l^8 + (-48*g*t1^2 + 96*g*t1*t2 - 48*g*t2^2 - 60)*l^6 + (8*g*t1^2 - 16*g*t1*t2 + 8*g*t2^2 + 180)*(t1 - t2)^2*l^4 - 60*(t1 - t2)^4*l^2 + 4*(t1 - t2)^6)*exp(-1/2*(t1 - t2)^2/l^2)/l^12], [1/4*(g^2*l^8 - 3*g*l^6 + (3*g*t1^2 - 6*g*t1*t2 + 3*g*t2^2 + 6)*l^4 - 12*(t1 - t2)^2*l^2 + 2*(t1 - t2)^4)*exp(-1/2*(t1 - t2)^2/l^2)/l^8, 1/4*(g^2*l^8 - 2*g*l^6 + (2*g*t1^2 - 4*g*t1*t2 + 2*g*t2^2 + 3)*l^4 - 6*(t1 - t2)^2*l^2 + (t1 - t2)^4)*exp(-1/2*(t1 - t2)^2/l^2)/l^8, 1/4*(g^3*l^12 - 4*g^2*l^10 + (4*g^2*t1^2 - 8*g^2*t1*t2 + 4*g^2*t2^2 + 15*g)*l^8 + (-30*g*t1^2 + 60*g*t1*t2 - 30*g*t2^2 - 30)*l^6 + 5*(t1 - t2)^2*(g*t1^2 - 2*g*t1*t2 + g*t2^2 + 18)*l^4 - 30*(t1 - t2)^4*l^2 + 2*(t1 - t2)^6)*exp(-1/2*(t1 - t2)^2/l^2)/l^12], [1/4*(g^3*l^12 - 5*g^2*l^10 + (5*g^2*t1^2 - 10*g^2*t1*t2 + 5*g^2*t2^2 + 24*g)*l^8 + (-48*g*t1^2 + 96*g*t1*t2 - 48*g*t2^2 - 60)*l^6 + (8*g*t1^2 - 16*g*t1*t2 + 8*g*t2^2 + 180)*(t1 - t2)^2*l^4 - 60*(t1 - t2)^4*l^2 + 4*(t1 - t2)^6)*exp(-1/2*(t1 - t2)^2/l^2)/l^12, 1/4*(g^3*l^12 - 4*g^2*l^10 + (4*g^2*t1^2 - 8*g^2*t1*t2 + 4*g^2*t2^2 + 15*g)*l^8 + (-30*g*t1^2 + 60*g*t1*t2 - 30*g*t2^2 - 30)*l^6 + 5*(t1 - t2)^2*(g*t1^2 - 2*g*t1*t2 + g*t2^2 + 18)*l^4 - 30*(t1 - t2)^4*l^2 + 2*(t1 - t2)^6)*exp(-1/2*(t1 - t2)^2/l^2)/l^12, 1/4*(g^4*l^16 - 6*g^3*l^14 + (6*g^3*t1^2 - 12*g^3*t1*t2 + 6*g^3*t2^2 + 39*g^2)*l^12 + (-78*g^2*t1^2 + 156*g^2*t1*t2 - 78*g^2*t2^2 - 180*g)*l^10 + (13*g^2*t1^4 - 52*g^2*t1^3*t2 + (78*g^2*t2^2 + 540*g)*t1^2 + (-52*g^2*t2^3 - 1080*g*t2)*t1 + 13*g^2*t2^4 + 540*g*t2^2 + 420)*l^8 - 180*(t1 - t2)^2*(g*t1^2 - 2*t2*t1*g + g*t2^2 + 28/3)*l^6 + 12*(t1 - t2)^4*(g*t1^2 - 2*g*t1*t2 + g*t2^2 + 70)*l^4 - 112*(t1 - t2)^6*l^2 + 4*(t1 - t2)^8)*exp(-1/2*(t1 - t2)^2/l^2)/l^16]])
        KK = matrix([[cell.substitute(g=9.81, l=1, t1=0, t2=0).simplify_full() for cell in row] for row in KK])
        #(K_discussion[row][0].diff(t2, 2) + 9.81*K_discussion[row][0] - K_discussion[row][2]).simplify_full()
        #(K_discussion[row][1].diff(t2, 2) + 4.905*K_discussion[row][1] - 0.5*K_discussion[row][2]).simplify_full() 

    def test_eigenvalues(self):


    def test_num_params(self):










class TestThreeTank:

    # This covariance functino is calculated using Maple, we _know_ that this is correct
    KK = Matrix(5, 5, [[exp(-1/2*(t1 - t2)^2/l1^2), 0, -exp(-1/2*(t1 - t2)^2/l1^2), exp(-1/2*(t1 - t2)^2/l1^2)*(t1 - t2)/l1^2, -exp(-1/2*(t1 - t2)^2/l1^2)*(t1 - t2)/l1^2], [0, exp(-1/2*(t1 - t2)^2/l2^2), exp(-1/2*(t1 - t2)^2/l2^2), 0, exp(-1/2*(t1 - t2)^2/l2^2)*(t1 - t2)/l2^2], [-exp(-1/2*(t1 - t2)^2/l1^2), exp(-1/2*(t1 - t2)^2/l2^2), 1 + exp(-1/2*(t1 - t2)^2/l1^2) + exp(-1/2*(t1 - t2)^2/l2^2), -exp(-1/2*(t1 - t2)^2/l1^2)*(t1 - t2)/l1^2, (l1^2*exp(-1/2*(t1 - t2)^2/l2^2) + l2^2*exp(-1/2*(t1 - t2)^2/l1^2))*(t1 - t2)/(l1^2*l2^2)], [-exp(-1/2*(t1 - t2)^2/l1^2)*(t1 - t2)/l1^2, 0, exp(-1/2*(t1 - t2)^2/l1^2)*(t1 - t2)/l1^2, exp(-1/2*(t1 - t2)^2/l1^2)*(l1 + t1 - t2)*(l1 - t1 + t2)/l1^4, -exp(-1/2*(t1 - t2)^2/l1^2)*(l1 + t1 - t2)*(l1 - t1 + t2)/l1^4], [exp(-1/2*(t1 - t2)^2/l1^2)*(t1 - t2)/l1^2, -exp(-1/2*(t1 - t2)^2/l2^2)*(t1 - t2)/l2^2, -(l1^2*exp(-1/2*(t1 - t2)^2/l2^2) + l2^2*exp(-1/2*(t1 - t2)^2/l1^2))*(t1 - t2)/(l1^2*l2^2), -exp(-1/2*(t1 - t2)^2/l1^2)*(l1 + t1 - t2)*(l1 - t1 + t2)/l1^4, (l2^4*(l1 + t1 - t2)*(l1 - t1 + t2)*exp(-1/2*(t1 - t2)^2/l1^2) + l1^4*exp(-1/2*(t1 - t2)^2/l2^2)*(l2 + t1 - t2)*(l2 - t1 + t2))/(l1^4*l2^4)]])
    # Numerically check if the covariance function we get is the same
    [[(cell1 - cell2).simplify_full().substitute(t2=2, t1=1, l1=1, l2=1).n() for cell1, cell2 in zip(row1, row2)] for row1, row2 in zip(K_discussion, KK)]