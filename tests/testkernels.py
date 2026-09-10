import pytest
import gpytorch
from lodegp.kernels import *
from sage.all import *
import torch

torch.set_default_tensor_type(torch.DoubleTensor)

def symmetry_check(model):
    symmetry_check_matrix = model.matrix_multiplication - matrix([[cell.substitute(dx1=dx2, dx2=dx1) for cell in row] for row in model.matrix_multiplication.T])
    # Symmetric up to numerical precision
    return all([all([cell < 1e-10 for cell in row]) for row in symmetry_check_matrix])

def eigval_check(model, train_x):
    covar_matrix = model(train_x).covariance_matrix
    eigvals = torch.linalg.eigh(covar_matrix)[0]
    compl_eigvals = torch.linalg.eig(covar_matrix)[0]
    return all([eig.real > -1e-10 for eig in eigvals]), all([eig.imag < 1e-10 for eig in compl_eigvals])
    



class LODEGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, ODE_type="Bipendulum"):
        super(LODEGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=num_tasks
        )
        # Put in front as they're either overwritten or used
        self.model_parameters = torch.nn.ParameterDict()
        R = QQ['x']; (x,) = R._first_ngens(1)
        if ODE_type == "Heating":
            #Heating system with parameters
            F = FunctionField(QQ, names=('a',)); (a,) = F._first_ngens(1)
            F = FunctionField(F, names=('b',)); (b,) = F._first_ngens(1)
            R = F['x']; (x,) = R._first_ngens(1)

            A = matrix(R, Integer(2), Integer(3), [x+a, -a, -1, -b, x+b, 0])
            self.model_parameters = torch.nn.ParameterDict({
                "a":torch.nn.Parameter(torch.tensor(0.0)),
                "b":torch.nn.Parameter(torch.tensor(0.0))
            })
        elif ODE_type == "Bipendulum":
            # Linearized bipendulum
            A = matrix(R, Integer(2), Integer(3), [x**2 + 9.81, 0, -1, 0, x**2+4.905, -1/2])
        elif ODE_type == "System 1":
            # System 1 (no idea)
            A = matrix(R, Integer(2), Integer(3), [x, -x**2+x-1, x-2, 2-x, x**2-x-1, -x])
        elif ODE_type == "Three Tank":
            # 3 Tank system (5 dimensional uncontrollable system)
            A = matrix(R, Integer(3), Integer(5), [-x, 0, 0, 1, 0, 0, -x, 0, 1, 1, 0, 0, -x, 0, 1])

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


class TestBipendulum:

    def test_symmetry_check(self):
        train_x = torch.linspace(0, 1, 10)
        train_y = torch.linspace(0, 1, 10)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(3, noise_constraint=gpytorch.constraints.Positive())
        model = LODEGP(train_x, train_y, likelihood, 3)
        # This matrix is suppossed to contain _only_ zeros
        assert symmetry_check(model)

    def test_satisfying_ode(self):
        train_x = torch.linspace(0, 1, 10)
        train_y = torch.linspace(0, 1, 10)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(3, noise_constraint=gpytorch.constraints.Positive())
        model = LODEGP(train_x, train_y, likelihood, 3)
        l1, l2, s1, s2 = var("l1, l2, s1, s2")
        #K_discussion = matrix([[cell.substitute(signal_variance_2=1, lengthscale_2=1, t1=1, t2=1) if type(cell) == sage.symbolic.expression.Expression else cell  for cell in row ] for row in model.diffed_kernel])
        K_discussion = matrix([[cell if type(cell) == sage.symbolic.expression.Expression else cell  for cell in row ] for row in model.diffed_kernel])
        for row in range(len(model.diffed_kernel)):
            # Assert that they're 0 or at least very very very close to zero
            assert (K_discussion[row][0].diff(t2, 2) + 9.81*K_discussion[row][0] - K_discussion[row][2]).full_simplify().substitute(t1=1, t2=1, lengthscale_2=1, signal_variance_2=1) < 1e-10
            assert (K_discussion[row][1].diff(t2, 2) + 1/2*9.81*K_discussion[row][1] - 1/2*K_discussion[row][2]).simplify_full().substitute(t1=1, t2=1, lengthscale_2=1, signal_variance_2=1) < 1e-10
            #assert (K_discussion[row][0].diff(t2, 2) + 9.81*K_discussion[row][0] - K_discussion[row][2]).full_simplify() < 1e-10
            #assert (K_discussion[row][1].diff(t2, 2) + 1/2*9.81*K_discussion[row][1] - 1/2*K_discussion[row][2]).simplify_full() < 1e-10

    def test_equality_maple_solution(self):
        train_x = torch.linspace(0, 1, 10)
        train_y = torch.linspace(0, 1, 10)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(3, noise_constraint=gpytorch.constraints.Positive())
        model = LODEGP(train_x, train_y, likelihood, 3)
        # This covariance function is calculated using Maple, we _know_ that this is correct
        t1, t2, l, g = var("t1, t2, l, g")
        K_discussion = matrix([[cell.substitute(signal_variance_2=1, lengthscale_2=l) if type(cell) == sage.symbolic.expression.Expression else cell  for cell in row ] for row in model.diffed_kernel])
        KK = matrix(3, 3, [[1/4*exp(-1/2*(t1 - t2)**2/l**2)*(g**2*l**8 - 4*g*l**6 + (4*g*t1**2 - 8*g*t1*t2 + 4*g*t2**2 + 12)*l**4 - 24*(t1 - t2)**2*l**2 + 4*(t1 - t2)**4)/l**8, 1/4*(g**2*l**8 - 3*g*l**6 + (3*g*t1**2 - 6*g*t1*t2 + 3*g*t2**2 + 6)*l**4 - 12*(t1 - t2)**2*l**2 + 2*(t1 - t2)**4)*exp(-1/2*(t1 - t2)**2/l**2)/l**8, 1/4*(g**3*l**12 - 5*g**2*l**10 + (5*g**2*t1**2 - 10*g**2*t1*t2 + 5*g**2*t2**2 + 24*g)*l**8 + (-48*g*t1**2 + 96*g*t1*t2 - 48*g*t2**2 - 60)*l**6 + (8*g*t1**2 - 16*g*t1*t2 + 8*g*t2**2 + 180)*(t1 - t2)**2*l**4 - 60*(t1 - t2)**4*l**2 + 4*(t1 - t2)**6)*exp(-1/2*(t1 - t2)**2/l**2)/l**12], [1/4*(g**2*l**8 - 3*g*l**6 + (3*g*t1**2 - 6*g*t1*t2 + 3*g*t2**2 + 6)*l**4 - 12*(t1 - t2)**2*l**2 + 2*(t1 - t2)**4)*exp(-1/2*(t1 - t2)**2/l**2)/l**8, 1/4*(g**2*l**8 - 2*g*l**6 + (2*g*t1**2 - 4*g*t1*t2 + 2*g*t2**2 + 3)*l**4 - 6*(t1 - t2)**2*l**2 + (t1 - t2)**4)*exp(-1/2*(t1 - t2)**2/l**2)/l**8, 1/4*(g**3*l**12 - 4*g**2*l**10 + (4*g**2*t1**2 - 8*g**2*t1*t2 + 4*g**2*t2**2 + 15*g)*l**8 + (-30*g*t1**2 + 60*g*t1*t2 - 30*g*t2**2 - 30)*l**6 + 5*(t1 - t2)**2*(g*t1**2 - 2*g*t1*t2 + g*t2**2 + 18)*l**4 - 30*(t1 - t2)**4*l**2 + 2*(t1 - t2)**6)*exp(-1/2*(t1 - t2)**2/l**2)/l**12], [1/4*(g**3*l**12 - 5*g**2*l**10 + (5*g**2*t1**2 - 10*g**2*t1*t2 + 5*g**2*t2**2 + 24*g)*l**8 + (-48*g*t1**2 + 96*g*t1*t2 - 48*g*t2**2 - 60)*l**6 + (8*g*t1**2 - 16*g*t1*t2 + 8*g*t2**2 + 180)*(t1 - t2)**2*l**4 - 60*(t1 - t2)**4*l**2 + 4*(t1 - t2)**6)*exp(-1/2*(t1 - t2)**2/l**2)/l**12, 1/4*(g**3*l**12 - 4*g**2*l**10 + (4*g**2*t1**2 - 8*g**2*t1*t2 + 4*g**2*t2**2 + 15*g)*l**8 + (-30*g*t1**2 + 60*g*t1*t2 - 30*g*t2**2 - 30)*l**6 + 5*(t1 - t2)**2*(g*t1**2 - 2*g*t1*t2 + g*t2**2 + 18)*l**4 - 30*(t1 - t2)**4*l**2 + 2*(t1 - t2)**6)*exp(-1/2*(t1 - t2)**2/l**2)/l**12, 1/4*(g**4*l**16 - 6*g**3*l**14 + (6*g**3*t1**2 - 12*g**3*t1*t2 + 6*g**3*t2**2 + 39*g**2)*l**12 + (-78*g**2*t1**2 + 156*g**2*t1*t2 - 78*g**2*t2**2 - 180*g)*l**10 + (13*g**2*t1**4 - 52*g**2*t1**3*t2 + (78*g**2*t2**2 + 540*g)*t1**2 + (-52*g**2*t2**3 - 1080*g*t2)*t1 + 13*g**2*t2**4 + 540*g*t2**2 + 420)*l**8 - 180*(t1 - t2)**2*(g*t1**2 - 2*t2*t1*g + g*t2**2 + 28/3)*l**6 + 12*(t1 - t2)**4*(g*t1**2 - 2*g*t1*t2 + g*t2**2 + 70)*l**4 - 112*(t1 - t2)**6*l**2 + 4*(t1 - t2)**8)*exp(-1/2*(t1 - t2)**2/l**2)/l**16]])
        KK = matrix([[cell.substitute(g=9.81, l=1).simplify_full() for cell in row] for row in KK])
        numerical_difference_sage_maple = [[(cell1 - cell2).simplify_full().substitute(t2=2, t1=1, l=1, s2=1).n() for cell1, cell2 in zip(row1, row2)] for row1, row2 in zip(K_discussion, KK)]
        assert all([all([cell < 1e-10 for cell in row]) for row in numerical_difference_sage_maple])

    def test_eigenvalues(self):
        train_x = torch.linspace(0, 1, 10)
        train_y = torch.linspace(0, 1, 10)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(3, noise_constraint=gpytorch.constraints.Positive())
        model = LODEGP(train_x, train_y, likelihood, 3)
        eig_positive, eig_real = eigval_check(model, train_x)
        # Check for positive eigvals (up to numerical precision)
        assert eig_positive 
        # Check for real values (up to numerical precision)
        assert eig_real 


    def test_num_params(self):
        train_x = torch.linspace(0, 1, 10)
        train_y = torch.linspace(0, 1, 10)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(3, noise_constraint=gpytorch.constraints.Positive())
        model = LODEGP(train_x, train_y, likelihood, 3)
        # This is hardcoded because I need to know how the smith form behaves
        # Bipendulum has 3 channels, two of which are 0 and one has SE
        # i.e. 2 parameters!
        # plus 3 channel noises (counted as 1) and 1 global noise => 4 parameters
        assert len(list(model.parameters())) == 4



class TestThreeTank:

    def test_symmetry_check(self):
        train_x = torch.linspace(0, 1, 10)
        train_y = torch.linspace(0, 1, 10)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(5, noise_constraint=gpytorch.constraints.Positive())
        model = LODEGP(train_x, train_y, likelihood, 5, ODE_type="Three Tank")
       # This matrix is suppossed to contain _only_ zeros
        assert symmetry_check(model)

    def test_satisfying_ode(self):
        train_x = torch.linspace(0, 1, 10)
        train_y = torch.linspace(0, 1, 10)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(5, noise_constraint=gpytorch.constraints.Positive())
        model = LODEGP(train_x, train_y, likelihood, 5, ODE_type="Three Tank")
        l1, l2, s1, s2 = var("l1, l2, s1, s2")
        #K_discussion = matrix([[cell.substitute(signal_variance_2=1, lengthscale_2=1, t1=1, t2=1) if type(cell) == sage.symbolic.expression.Expression else cell  for cell in row ] for row in model.diffed_kernel])
        K_discussion = matrix([[cell if type(cell) == sage.symbolic.expression.Expression else cell  for cell in row ] for row in model.diffed_kernel])
        for row in range(len(model.diffed_kernel)):
            # Assert that they're 0 or at least very very very close to zero
            assert (-K_discussion[row][0].diff(t2, 1) + K_discussion[row][3]).simplify_full().substitute(t1=1, t2=1, lengthscale_4=1, signal_variance_4=1, lengthscale_3=1, signal_variance_3=1, signal_variance_2 = 1) < 1e-10
            assert (-K_discussion[row][1].diff(t2, 1) + K_discussion[row][3] + K_discussion[row][4]).simplify_full().substitute(t1=1, t2=1, lengthscale_4=1, signal_variance_4=1, lengthscale_3=1, signal_variance_3=1, signal_variance_2 = 1) < 1e-10
            assert (-K_discussion[row][2].diff(t2, 1) + K_discussion[row][4]).simplify_full().substitute(t1=1, t2=1, lengthscale_4=1, signal_variance_4=1, lengthscale_3=1, signal_variance_3=1, signal_variance_2 = 1) < 1e-10

    def test_equality_maple_solution(self):
        train_x = torch.linspace(0, 1, 10)
        train_y = torch.linspace(0, 1, 10)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(5, noise_constraint=gpytorch.constraints.Positive())
        model = LODEGP(train_x, train_y, likelihood, 5, ODE_type="Three Tank")
        # This covariance function is calculated using Maple, we _know_ that this is correct
        t1, t2, l1, l2 = var("t1, t2, l1, l2")
        K_discussion = matrix([[cell.substitute(signal_variance_2_0=1, lengthscale_4=l2, signal_variance_4=1, lengthscale_3=l1, signal_variance_3=1) if type(cell) == sage.symbolic.expression.Expression else cell  for cell in row ] for row in model.diffed_kernel])
        KK = Matrix(5, 5, [[exp(-1/2*(t1 - t2)**2/l1**2), 0, -exp(-1/2*(t1 - t2)**2/l1**2), exp(-1/2*(t1 - t2)**2/l1**2)*(t1 - t2)/l1**2, -exp(-1/2*(t1 - t2)**2/l1**2)*(t1 - t2)/l1**2], [0, exp(-1/2*(t1 - t2)**2/l2**2), exp(-1/2*(t1 - t2)**2/l2**2), 0, exp(-1/2*(t1 - t2)**2/l2**2)*(t1 - t2)/l2**2], [-exp(-1/2*(t1 - t2)**2/l1**2), exp(-1/2*(t1 - t2)**2/l2**2), 1 + exp(-1/2*(t1 - t2)**2/l1**2) + exp(-1/2*(t1 - t2)**2/l2**2), -exp(-1/2*(t1 - t2)**2/l1**2)*(t1 - t2)/l1**2, (l1**2*exp(-1/2*(t1 - t2)**2/l2**2) + l2**2*exp(-1/2*(t1 - t2)**2/l1**2))*(t1 - t2)/(l1**2*l2**2)], [-exp(-1/2*(t1 - t2)**2/l1**2)*(t1 - t2)/l1**2, 0, exp(-1/2*(t1 - t2)**2/l1**2)*(t1 - t2)/l1**2, exp(-1/2*(t1 - t2)**2/l1**2)*(l1 + t1 - t2)*(l1 - t1 + t2)/l1**4, -exp(-1/2*(t1 - t2)**2/l1**2)*(l1 + t1 - t2)*(l1 - t1 + t2)/l1**4], [exp(-1/2*(t1 - t2)**2/l1**2)*(t1 - t2)/l1**2, -exp(-1/2*(t1 - t2)**2/l2**2)*(t1 - t2)/l2**2, -(l1**2*exp(-1/2*(t1 - t2)**2/l2**2) + l2**2*exp(-1/2*(t1 - t2)**2/l1**2))*(t1 - t2)/(l1**2*l2**2), -exp(-1/2*(t1 - t2)**2/l1**2)*(l1 + t1 - t2)*(l1 - t1 + t2)/l1**4, (l2**4*(l1 + t1 - t2)*(l1 - t1 + t2)*exp(-1/2*(t1 - t2)**2/l1**2) + l1**4*exp(-1/2*(t1 - t2)**2/l2**2)*(l2 + t1 - t2)*(l2 - t1 + t2))/(l1**4*l2**4)]])
        KK = matrix([[cell.substitute(l1=1, l2=1).simplify_full() for cell in row] for row in KK])
        # Numerically check if the covariance function we get is the same
        numerical_difference_sage_maple = [[(cell1 - cell2).simplify_full().substitute(t2=2, t1=1, l1=1, l2=1).n() for cell1, cell2 in zip(row1, row2)] for row1, row2 in zip(K_discussion, KK)]
        assert all([all([cell < 1e-10 for cell in row]) for row in numerical_difference_sage_maple])

    def test_eigenvalues(self):
        train_x = torch.linspace(0, 1, 10)
        train_y = torch.linspace(0, 1, 10)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(5, noise_constraint=gpytorch.constraints.Positive())
        model = LODEGP(train_x, train_y, likelihood, 5, ODE_type="Three Tank")
        eig_positive, eig_real = eigval_check(model, train_x)
        # Check for positive eigvals (up to numerical precision)
        assert eig_positive 
        # Check for real values (up to numerical precision)
        assert eig_real 


    def test_num_params(self):
        train_x = torch.linspace(0, 1, 10)
        train_y = torch.linspace(0, 1, 10)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(5, noise_constraint=gpytorch.constraints.Positive())
        model = LODEGP(train_x, train_y, likelihood, 5, ODE_type="Three Tank")
        assert len(list(model.parameters())) == 7



class TestHeating:

    def test_symmetry_check(self):
        train_x = torch.linspace(0, 1, 10)
        train_y = torch.linspace(0, 1, 10)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(3, noise_constraint=gpytorch.constraints.Positive())
        model = LODEGP(train_x, train_y, likelihood, 3, ODE_type="Heating")
       # This matrix is suppossed to contain _only_ zeros
        assert symmetry_check(model)

    def test_satisfying_ode(self):
        train_x = torch.linspace(0, 1, 10)
        train_y = torch.linspace(0, 1, 10)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(3, noise_constraint=gpytorch.constraints.Positive())
        model = LODEGP(train_x, train_y, likelihood, 3, ODE_type="Heating")
        l1, l2, s1, s2, a, b = var("l1, l2, s1, s2, a, b")
        #K_discussion = matrix([[cell.substitute(signal_variance_2=1, lengthscale_2=1, t1=1, t2=1) if type(cell) == sage.symbolic.expression.Expression else cell  for cell in row ] for row in model.diffed_kernel])
        K_discussion = matrix([[cell if type(cell) == sage.symbolic.expression.Expression else cell  for cell in row ] for row in model.diffed_kernel])
        for row in range(len(model.diffed_kernel)):
            # Assert that they're 0 or at least very very very close to zero
            a_val = model.model_parameters["a"].item()
            b_val = model.model_parameters["b"].item()
            assert (K_discussion[row][0].diff(t2, 1) + a_val*K_discussion[row][0] - a_val*K_discussion[row][1] - K_discussion[row][2]).simplify_full().substitute(t1=1, t2=1, lengthscale_2=1, signal_variance_2 = 1, a = a_val, b = b_val) < 1e-10
            assert (-b_val*K_discussion[row][0] + K_discussion[row][1].diff(t2, 1) + b_val*K_discussion[row][1]).simplify_full().substitute(t1=1, t2=1, lengthscale_2=1, signal_variance_2 = 1, a = a_val, b = b_val) < 1e-10


    

    def test_eigenvalues(self):
        train_x = torch.linspace(0, 1, 10)
        train_y = torch.linspace(0, 1, 10)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(3, noise_constraint=gpytorch.constraints.Positive())
        model = LODEGP(train_x, train_y, likelihood, 3, ODE_type="Heating")
        eig_positive, eig_real = eigval_check(model, train_x)
        # Check for positive eigvals (up to numerical precision)
        assert eig_positive 
        # Check for real values (up to numerical precision)
        assert eig_real 


    def test_num_params(self):
        train_x = torch.linspace(0, 1, 10)
        train_y = torch.linspace(0, 1, 10)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(3, noise_constraint=gpytorch.constraints.Positive())
        model = LODEGP(train_x, train_y, likelihood, 3, ODE_type="Heating")
        assert len(list(model.parameters())) == 6
