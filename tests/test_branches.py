#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

import os
import unittest
import pfnet as pf
import numpy as np
from . import test_cases
from numpy.linalg import norm

class TestBranches(unittest.TestCase):

    def setUp(self):

        # Random
        np.random.seed(0)

    def test_power_flow_utils(self):

        h = 1e-8
        tol = 1e-5
        eps = 1.0 # %

        for case in test_cases.CASES:
            
            net = pf.Parser(case).parse(case)

            net.set_flags('bus',
                          'variable',
                          'any',
                          ['voltage magnitude', 'voltage angle'])
            net.set_flags('branch',
                          'variable',
                          'any',
                          ['phase shift'])

            self.assertEqual(net.num_vars, 2*net.num_buses + net.num_branches)

            x0 = net.get_var_values() + np.random.randn(net.num_vars)*1e-4

            for branch in net.branches[:50]:

                # km
                f0 = np.array([branch.get_P_km(x0), branch.get_Q_km(x0)])
                J0 = branch.power_flow_Jacobian_km(x0)

                for i in range(10):
                    d = np.random.randn(x0.size)
                    x = x0 + h*d
                    f1 = np.array([branch.get_P_km(x), branch.get_Q_km(x)])
                    Jd_exact = J0*d
                    Jd_approx = (f1-f0)/h
                    error = 100.*norm(Jd_exact-Jd_approx)/(norm(Jd_exact)+tol)
                    self.assertLess(error, eps)

                # mk
                f0 = np.array([branch.get_P_mk(x0), branch.get_Q_mk(x0)])
                J0 = branch.power_flow_Jacobian_mk(x0)

                for i in range(10):
                    d = np.random.randn(x0.size)
                    x = x0 + h*d
                    f1 = np.array([branch.get_P_mk(x), branch.get_Q_mk(x)])
                    Jd_exact = J0*d
                    Jd_approx = (f1-f0)/h
                    error = 100.*norm(Jd_exact-Jd_approx)/(norm(Jd_exact)+tol)
                    self.assertLess(error, eps)
