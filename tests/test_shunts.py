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

class TestShunts(unittest.TestCase):

    def test_ieee25_raw_case(self):

        case = os.path.join('data', 'ieee25.raw')
        if not os.path.isfile(case):
            raise unittest.SkipTest('file not available')

        net = pf.Parser(case).parse(case)

        self.assertEqual(net.get_num_switched_v_shunts(), 1)

        s = net.get_switched_shunt_from_name_and_bus_number('', 106)

        self.assertTrue(s.is_switched())
        self.assertTrue(s.is_switched_v())

        self.assertTrue(s.is_continuous())
        self.assertFalse(s.is_discrete())

        self.assertTrue(np.all(s.b_values == np.array([-1., -0.8, -0.6, -0.4, -0.2, 0.])))

        n = s.round_b()
        self.assertEqual(n, 0)

        m = net.round_discrete_switched_shunts_b()
        self.assertEqual(m, 0)

    def test_psse_sample_raw_case(self):

        case = os.path.join('data', 'psse_sample_case.raw')
        if not os.path.isfile(case):
            raise unittest.SkipTest('file not available')

        p = pf.Parser(case)
        net = p.parse(case)

        self.assertEqual(net.get_num_switched_shunts(), 6)

        s1 = net.get_switched_shunt_from_name_and_bus_number('', 152)
        s2 = net.get_switched_shunt_from_name_and_bus_number('', 154)
        s3 = net.get_switched_shunt_from_name_and_bus_number('', 3005)
        s4 = net.get_switched_shunt_from_name_and_bus_number('', 3021)
        s5 = net.get_switched_shunt_from_name_and_bus_number('', 3022)
        s6 = net.get_switched_shunt_from_name_and_bus_number('', 93002)

        self.assertTrue(s1.is_discrete())
        self.assertFalse(s1.is_continuous())
        self.assertTrue(s1.is_switched_v())

        self.assertTrue(s2.is_discrete())
        self.assertFalse(s2.is_continuous())
        self.assertTrue(s2.is_switched_v())

        self.assertFalse(s3.is_discrete())
        self.assertTrue(s3.is_continuous())
        self.assertTrue(s3.is_switched_locked())

        self.assertFalse(s4.is_discrete())
        self.assertTrue(s4.is_continuous())
        self.assertTrue(s4.is_switched_v())

        self.assertFalse(s5.is_discrete())
        self.assertTrue(s5.is_continuous())
        self.assertTrue(s5.is_switched_v())

        self.assertTrue(s6.is_discrete())
        self.assertFalse(s6.is_continuous())
        self.assertTrue(s6.is_switched_v())

        eps = 1e-12

        # s1
        v = [(1, -15.00), (2, -5.00), (3, -10.00), (4, -8.00), (5, -7.00), (6, -5.00), (7, -7.00), (8, -4.00)]
        v.reverse()
        b = np.sum(list(map(lambda x: x[0]*x[1], v)))/100.
        values = [b]
        for steps, inc in v:
            for i in range(steps):
                b += -inc/100.
                values.append(b)

        self.assertLess(np.abs(values[-1]), eps)
        self.assertLess(np.linalg.norm(np.array(values)-s1.b_values), eps)
        self.assertLess(np.abs(s1.b+2.33), eps)
        self.assertLess(np.abs(s1.b_min-np.min(values)), eps)
        self.assertLess(np.abs(s1.b_max-np.max(values)), eps)
        s1.b = -2.30
        new_b = s1.b_values[np.argmin(np.abs(s1.b_values-s1.b))]
        self.assertEqual(s1.round_b(), 1)
        self.assertLess(np.abs(s1.b-new_b), eps)
        self.assertLess(np.abs(s1.b+2.29), eps)

        # s2
        v = [(1, 25.00), (2, 10.00), (2, 15.00), (1, 15.00), (2, 5.00), (3, 3.00), (2, 4.00), (1, 7.00)] 
        b = 0.
        values = [b]
        for steps, inc in v:
            for i in range(steps):
                b += inc/100.
                values.append(b)

        self.assertLess(np.abs(values[0]), eps)
        self.assertLess(np.linalg.norm(np.array(values)-s2.b_values), eps)
        self.assertLess(np.abs(s2.b-1.24), eps)
        self.assertLess(np.abs(s2.b_min-np.min(values)), eps)
        self.assertLess(np.abs(s2.b_max-np.max(values)), eps)
        s2.b = 0.56
        new_b = s2.b_values[np.argmin(np.abs(s2.b_values-s2.b))]
        self.assertEqual(s2.round_b(), 1)
        self.assertLess(np.abs(s2.b-new_b), eps)
        self.assertLess(np.abs(s2.b-0.6), eps)

        # s3
        values = [0., 0.3335]
        self.assertLess(np.linalg.norm(np.array(values)-s3.b_values), eps)
        self.assertLess(np.abs(s3.b-0.00), eps)
        self.assertLess(np.abs(s3.b_min-np.min(values)), eps)
        self.assertLess(np.abs(s3.b_max-np.max(values)), eps)
        s3.b = 0.
        new_b = 0.
        self.assertEqual(s3.round_b(), 0)
        self.assertLess(np.abs(s3.b-new_b), eps)
        self.assertLess(np.abs(s3.b-0.), eps)

        # s4
        v = [(2, 200.00), (1, 100.00), (2, 50.00), (4, 25.00)] 
        b = 0.
        values = [b]
        for steps, inc in v:
            for i in range(steps):
                b += inc/100.
                values.append(b)

        self.assertLess(np.abs(values[0]), eps)
        self.assertLess(np.linalg.norm(np.array(values)-s4.b_values), eps)
        self.assertLess(np.abs(s4.b-5.0154), eps)
        self.assertLess(np.abs(s4.b_min-np.min(values)), eps)
        self.assertLess(np.abs(s4.b_max-np.max(values)), eps)
        s4.b = 6.93
        new_b = s4.b_values[np.argmin(np.abs(s4.b_values-s4.b))]
        self.assertEqual(s4.round_b(), 1)
        self.assertLess(np.abs(s4.b-new_b), eps)
        self.assertLess(np.abs(s4.b-7.00), eps)

        # s5
        v = [(4, 100.00), (2, 50.00), (4, 25.00), (3, 20.00), (2, 20.00)] 
        b = 0.
        values = [b]
        for steps, inc in v:
            for i in range(steps):
                b += inc/100.
                values.append(b)

        self.assertLess(np.abs(values[0]), eps)
        self.assertLess(np.linalg.norm(np.array(values)-s5.b_values), eps)
        self.assertLess(np.abs(s5.b-6.0112), eps)
        self.assertLess(np.abs(s5.b_min-np.min(values)), eps)
        self.assertLess(np.abs(s5.b_max-np.max(values)), eps)
        s5.b = 2.3
        new_b = s5.b_values[np.argmin(np.abs(s5.b_values-s5.b))]
        self.assertEqual(s5.round_b(), 1)
        self.assertLess(np.abs(s5.b-new_b), eps)
        self.assertLess(np.abs(s5.b-2.), eps)
        
        # s6
        v1 = [(2, -30.00), (1, -5.00)]
        v2 = [(1, 1.44), (3, 10)]
        v1.reverse()
        b = np.sum(list(map(lambda x: x[0]*x[1], v1)))/100.
        values = [b]
        for steps, inc in v1:
            for i in range(steps):
                b += -inc/100.
                values.append(b)
        for steps, inc in v2:
            for i in range(steps):
                b += inc/100.
                values.append(b)

        self.assertLess(np.abs(values[3]), eps)
        self.assertLess(np.linalg.norm(np.array(values)-s6.b_values), eps)
        self.assertLess(np.abs(s6.b-0.0144), eps)
        self.assertLess(np.abs(s6.b_min-np.min(values)), eps)
        self.assertLess(np.abs(s6.b_max-np.max(values)), eps)
        s6.b = 0.15
        new_b = s6.b_values[np.argmin(np.abs(s6.b_values-s6.b))]
        self.assertEqual(s6.round_b(), 1)
        self.assertLess(np.abs(s6.b-new_b), eps)
        self.assertLess(np.abs(s6.b-0.1144), eps)

        # Fixed shunts
        self.assertGreaterEqual(net.get_num_fixed_shunts(), 13)
        tested = False
        for shunt in net.shunts:
            if shunt.is_fixed():
                b = shunt.b
                self.assertEqual(shunt.round_b(), 0)
                self.assertEqual(b, shunt.b)
                tested = True
        self.assertTrue(tested)

        m = 0
        for shunt in net.shunts:
            if shunt.is_switched() and shunt.is_discrete():
                shunt.b = shunt.b_max + 1.
                self.assertNotEqual(shunt.b, shunt.b_max)
                m += 1
        self.assertEqual(m, 3)
        
        self.assertEqual(net.round_discrete_switched_shunts_b(), 3)
        for shunt in net.shunts:
            if shunt.is_switched() and shunt.is_discrete():
                self.assertEqual(shunt.b, shunt.b_max)
        
