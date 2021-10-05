from sage.all import *

from kernels import *
import unittest

class TestKernel(unittest.TestCase):

    def test_deriv_term_extraction(self):
        context = Diff_SE_kernel()
        # var1*var2
        a, b = var('a, b')
        left_poly = a
        right_poly = b
        result = prepare_asym_deriv_dict(left_poly, right_poly, context)
        self.assertIsInstance(result[0][0][0], torch.nn.Parameter, "Hyperparameter 1 wasn't initialized properly")
        self.assertIsInstance(result[0][0][1], torch.nn.Parameter, "Hyperparameter 2 wasn't initialized properly")
        self.assertEqual(result[0][1], 0, "Exponent 1 not correctly calculated")
        self.assertEqual(result[0][2], 0, "Exponent 2 not correctly calculated")
        # var1*d*var2
        a, b, x1 = var('a, b, x1')
        left_poly = a*x1
        right_poly = b
        result = prepare_asym_deriv_dict(left_poly, right_poly, context)
        self.assertIsInstance(result[0][0][0], torch.nn.Parameter, "Hyperparameter 1 wasn't initialized properly")
        self.assertIsInstance(result[0][0][1], torch.nn.Parameter, "Hyperparameter 2 wasn't initialized properly")
        self.assertEqual(result[0][1], 1, "Exponent 1 not correctly calculated")
        self.assertEqual(result[0][2], 0, "Exponent 2 not correctly calculated")
        # var1*d1*var2*d2
        a, b, x1, x2 = var('a, b, x1, x2')
        left_poly = a*x1
        right_poly = b*x2
        result = prepare_asym_deriv_dict(left_poly, right_poly, context)
        self.assertIsInstance(result[0][0][0], torch.nn.Parameter, "Hyperparameter 1 wasn't initialized properly")
        self.assertIsInstance(result[0][0][1], torch.nn.Parameter, "Hyperparameter 2 wasn't initialized properly")
        self.assertEqual(result[0][1], 1, "Exponent 1 not correctly calculated")
        self.assertEqual(result[0][2], 1, "Exponent 2 not correctly calculated")
        # var1*var1
        left_poly = a
        right_poly = a
        result = prepare_asym_deriv_dict(left_poly, right_poly, context)
        self.assertIsInstance(result[0][0][0], torch.nn.Parameter, "Hyperparameter 1 wasn't initialized properly")
        self.assertIsInstance(result[0][0][1], torch.nn.Parameter, "Hyperparameter 2 wasn't initialized properly")
        self.assertEqual(result[0][0][0], result[0][0][1], "Hyperparamter entries aren't equal despite initial equality")
        self.assertEqual(result[0][1], 0, "Exponent 1 not correctly calculated")
        self.assertEqual(result[0][2], 0, "Exponent 2 not correctly calculated")
        # var1*d*var1
        left_poly = a*x1
        right_poly = a
        result = prepare_asym_deriv_dict(left_poly, right_poly, context)
        self.assertIsInstance(result[0][0][0], torch.nn.Parameter, "Hyperparameter 1 wasn't initialized properly")
        self.assertIsInstance(result[0][0][1], torch.nn.Parameter, "Hyperparameter 2 wasn't initialized properly")
        self.assertEqual(result[0][0][0], result[0][0][1], "Hyperparameter entries aren't equal despite initial equality")
        self.assertEqual(result[0][1], 1, "Exponent 1 not correctly calculated")
        self.assertEqual(result[0][2], 0, "Exponent 2 not correctly calculated")
        # var1*d1*var1*d2
        left_poly = a*x1
        right_poly = a*x2
        result = prepare_asym_deriv_dict(left_poly, right_poly, context)
        self.assertIsInstance(result[0][0][0], torch.nn.Parameter, "Hyperparameter 1 wasn't initialized properly")
        self.assertIsInstance(result[0][0][1], torch.nn.Parameter, "Hyperparameter 2 wasn't initialized properly")
        self.assertEqual(result[0][0][0], result[0][0][1], "Hyperparamter entries aren't equal despite intial equality")
        self.assertEqual(result[0][1], 1, "Exponent 1 not correctly calculated")
        self.assertEqual(result[0][2], 1, "Exponent 2 not correctly calculated")
        # num*num
        left_poly = 3
        right_poly = 7
        result = prepare_asym_deriv_dict(left_poly, right_poly, context)
        self.assertEqual(result[0][0][0], 3, "First coefficient is wrong")
        self.assertEqual(result[0][0][1], 7, "Second coefficient is wrong")
        self.assertEqual(result[0][1], 0, "Exponent 1 not correctly calculated")
        self.assertEqual(result[0][2], 0, "Exponent 2 not correctly calculated")
        # num*d*num
        left_poly = 15*x1
        right_poly = 92
        result = prepare_asym_deriv_dict(left_poly, right_poly, context)
        self.assertEqual(result[0][0][0], 15, "First coefficient is wrong")
        self.assertEqual(result[0][0][1], 92, "Second coefficient is wrong")
        self.assertEqual(result[0][1], 1, "Exponent 1 not correctly calculated")
        self.assertEqual(result[0][2], 0, "Exponent 2 not correctly calculated")
        # num*d1*num*d2
        left_poly = 42*x1
        right_poly = 148*x2
        result = prepare_asym_deriv_dict(left_poly, right_poly, context)
        self.assertEqual(result[0][0][0], 42, "First coefficient is wrong")
        self.assertEqual(result[0][0][1], 148, "Second coefficient is wrong")
        self.assertEqual(result[0][1], 1, "Exponent 1 not correctly calculated")
        self.assertEqual(result[0][2], 1, "Exponent 2 not correctly calculated")
        # var1*d^n*var2
        left_poly = a*x1^3
        right_poly = b
        result = prepare_asym_deriv_dict(left_poly, right_poly, context)
        self.assertIsInstance(result[0][0][0], torch.nn.Parameter, "Hyperparameter 1 wasn't initialized properly")
        self.assertIsInstance(result[0][0][1], torch.nn.Parameter, "Hyperparameter 2 wasn't initialized properly")
        self.assertEqual(result[0][1], 3, "Exponent 1 not correctly calculated")
        self.assertEqual(result[0][2], 0, "Exponent 2 not correctly calculated")
        # var1*d1^n*var2*d2^n
        left_poly = a*x1^7
        right_poly = b*x2^42
        result = prepare_asym_deriv_dict(left_poly, right_poly, context)
        self.assertIsInstance(result[0][0][0], torch.nn.Parameter, "Hyperparameter 1 wasn't initialized properly")
        self.assertIsInstance(result[0][0][1], torch.nn.Parameter, "Hyperparameter 2 wasn't initialized properly")
        self.assertEqual(result[0][1], 7, "Exponent 1 not correctly calculated")
        self.assertEqual(result[0][2], 42, "Exponent 2 not correctly calculated")
        # var1*d^n*var1
        left_poly = a*x1^13
        right_poly = a
        result = prepare_asym_deriv_dict(left_poly, right_poly, context)
        self.assertEqual(result[0][0][0], result[0][0][1], "Hyperparameter entries aren't equal despite initial equality")
        self.assertIsInstance(result[0][0][0], torch.nn.Parameter, "Hyperparameter 1 wasn't initialized properly")
        self.assertIsInstance(result[0][0][1], torch.nn.Parameter, "Hyperparameter 2 wasn't initialized properly")
        self.assertEqual(result[0][1], 13, "Exponent 1 not correctly calculated")
        self.assertEqual(result[0][2], 0, "Exponent 2 not correctly calculated")
        # var1*d1^n*var1*d2^n
        left_poly = a*x1^28
        right_poly = a*x2^83
        result = prepare_asym_deriv_dict(left_poly, right_poly, context)
        self.assertIsInstance(result[0][0][0], torch.nn.Parameter, "Hyperparameter 1 wasn't initialized properly")
        self.assertIsInstance(result[0][0][1], torch.nn.Parameter, "Hyperparameter 2 wasn't initialized properly")
        self.assertEqual(result[0][0][0], result[0][0][1], "Hyperparamter entries aren't equal despite initial equality")
        self.assertEqual(result[0][1], 28, "Exponent 1 not correctly calculated")
        self.assertEqual(result[0][2], 83, "Exponent 2 not correctly calculated")
        # num*d^n*num
        left_poly = 34*x1^72
        right_poly = 182
        result = prepare_asym_deriv_dict(left_poly, right_poly, context)
        self.assertEqual(result[0][0][0], 34, "First coefficient is wrong")
        self.assertEqual(result[0][0][1], 182, "Second coefficient is wrong")
        self.assertEqual(result[0][1], 72, "Exponent 1 not correctly calculated")
        self.assertEqual(result[0][2], 0, "Exponent 2 not correctly calculated")
        # num*d1^n*num*d2^n
        left_poly = 839840583*x1^75
        right_poly = 38748926495*x2^85739
        result = prepare_asym_deriv_dict(left_poly, right_poly, context)
        self.assertEqual(result[0][0][0], 839840583, "First coefficient is wrong")
        self.assertEqual(result[0][0][1], 38748926495, "Second coefficient is wrong")
        self.assertEqual(result[0][1], 75, "Exponent 1 not correctly calculated")
        self.assertEqual(result[0][2], 85739, "Exponent 2 not correctly calculated")
        # var*(x^n + x^0)*x2
        left_poly = a*x1^3 + a
        right_poly = x2
        result = prepare_asym_deriv_dict(left_poly, right_poly, context)
        # Result should be [[[a, 1], 3, 1], [[a, 1], 0, 1]]
        self.assertIsInstance(result[0][0][0], torch.nn.Parameter, "Hyperparameter 1 wasn't initialized properly")
        self.assertEqual(result[0][1], 3, "Exponent 1 not correctly calculated")
        self.assertEqual(result[0][2], 1, "Exponent 2 not correctly calculated")
        self.assertIsInstance(result[1][0][0], torch.nn.Parameter, "Hyperparameter 1 wasn't initialized properly")
        self.assertEqual(result[1][1], 0, "Exponent 1 not correctly calculated")
        self.assertEqual(result[1][2], 1, "Exponent 2 not correctly calculated")
        # var1*var2*num*x^n
        left_poly = a*b*x1^3
        right_poly = 42
        result = prepare_asym_deriv_dict(left_poly, right_poly, context)
        # Result should be [[[a, b, 42], 3, 0]]
        self.assertIsInstance(result[0][0][0], torch.nn.Parameter, "Hyperparameter 1 wasn't initialized properly")
        self.assertIsInstance(result[0][0][1], torch.nn.Parameter, "Hyperparameter 2 wasn't initialized properly")
        self.assertEqual(result[0][0][2], 42, "Coefficient wasn't initialized properly")
        self.assertEqual(result[0][1], 3, "Exponent 1 not correctly calculated")
        self.assertEqual(result[0][2], 0, "Exponent 2 not correctly calculated")
        # var1*num1*x^n
        left_poly = 42*a*x1^3
        right_poly = 1
        result = prepare_asym_deriv_dict(left_poly, right_poly, context)
        # Result should be [[[42, a, 1], 3, 0]]
        self.assertIsInstance(result[0][0][1], torch.nn.Parameter, "Hyperparameter 1 wasn't initialized properly")
        self.assertEqual(result[0][0][0], 42, "First coefficient is wrong")
        self.assertEqual(result[0][0][2], 1, "Second coefficient is wrong")
        self.assertEqual(result[0][1], 3, "Exponent 1 not correctly calculated")
        self.assertEqual(result[0][2], 0, "Exponent 2 not correctly calculated")



if __name__ == '__main__':
    unittest.main()

