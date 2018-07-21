#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

import os
import unittest
import numpy as np
import pfnet as pf
from . import test_cases

class TestLoads(unittest.TestCase):

    def test_GSO_5bus_vloads_case(self):

        T = 4
        
        case = os.path.join('data', 'GSO_5bus_vloads.raw')

        if not os.path.isfile(case):
            raise unittest.SkipTest('file not available')

        net = pf.ParserRAW().parse(case, T)

        self.assertEqual(net.num_loads, 3)
        self.assertEqual(net.get_num_vdep_loads(), 2)

        load1 = net.get_load_from_name_and_bus_number('1', 2)
        load2 = net.get_load_from_name_and_bus_number('1', 3)

        for t in range(T):

            self.assertEqual(load1.P[t], (500.+200.+100.)/net.base_power)
            self.assertEqual(load1.Q[t], (180.+50.-50.)/net.base_power)
            self.assertEqual(load1.comp_cp[t], 500./net.base_power)
            self.assertEqual(load1.comp_cq[t], 180./net.base_power)
            self.assertEqual(load1.comp_ci[t], 200./net.base_power)
            self.assertEqual(load1.comp_cj[t], 50./net.base_power)
            self.assertEqual(load1.comp_cg, 100./net.base_power)
            self.assertEqual(load1.comp_cb, 50./net.base_power)
            
            self.assertEqual(load2.P[t], (40.+20.+20.)/net.base_power)
            self.assertEqual(load2.Q[t], (20.+10.-10.)/net.base_power)
            self.assertEqual(load2.comp_cp[t], 40./net.base_power)
            self.assertEqual(load2.comp_cq[t], 20./net.base_power)
            self.assertEqual(load2.comp_ci[t], 20./net.base_power)
            self.assertEqual(load2.comp_cj[t], 10./net.base_power)
            self.assertEqual(load2.comp_cg, 20./net.base_power)
            self.assertEqual(load2.comp_cb, 10./net.base_power)
