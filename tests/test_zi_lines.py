#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2019, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

import os
import unittest
import pfnet as pf
import numpy as np
from . import test_cases

class TestZILines(unittest.TestCase):

    def setUp(self):
        
        pass

    def test_aeso(self):

        case = os.path.join('data', 'aesoSL2014.raw')

        if not os.path.isfile(case):
            raise unittest.SkipTest('no .raw file')

        parser = pf.ParserRAW()
        parser.set('merge_buses', False)

        net1 = parser.parse(case)
        net1_copy = net1.get_copy(merge_buses=False)

        pf.tests.utils.compare_networks(self, net1, net1_copy)

        self.assertEqual(net1.num_buses, 2495)
        self.assertEqual(net1.num_branches, 2823)
        self.assertEqual(net1.get_num_zero_impedance_lines(), 42)

        net2 = net1.get_copy(merge_buses=True)

        net2.set_flags('bus', 'variable', 'any', 'voltage magnitude')

        self.assertEqual(net2.num_buses, 2454)
        self.assertEqual(net2.num_branches, 2823-42)
        self.assertEqual(net2.get_num_zero_impedance_lines(), 0)

        net1.copy_from_network(net2, merged=True)

        for bus in net2.buses:
            self.assertTrue(bus.has_flags('variable', 'voltage magnitude'))
        for bus in net1.buses:
            self.assertFalse(bus.has_flags('variable', 'voltage magnitude'))
        
        self.assertGreaterEqual(net2.num_vars, 1)
        self.assertEqual(net1.num_vars, 0)
        self.assertEqual(net1.num_fixed, 0)
        self.assertEqual(net1.num_bounded, 0)

        pf.tests.utils.compare_networks(self, net1, net1_copy)

        net1.copy_from_network(net2, merged=False)
        self.assertRaises(AssertionError, pf.tests.utils.compare_networks, self, net1, net1_copy)

        
        
        
