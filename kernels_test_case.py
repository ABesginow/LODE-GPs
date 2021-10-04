from kernels import *

import unittest

class TestKernel(unittest.TestCase):

    def test_deriv_term_extraction(self):
        # var1*var2
        # var1*d*var2
        # var1*d1*var2*d2
        # var1*var1
        # var1*d*var1
        # var1*d1*var1*d2
        # num*num
        # num*d*num
        # num*d1*num*d2
        # var1*d^n*var2
        # var1*d1^n*var2*d2^n
        # var1*d^n*var1
        # var1*d1^n*var1*d2^n
        # num*d^n*num
        # num*d1^n*num*d2^n



if __name__ == '__main__':
    unittest.main()

