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

class TestZILines(unittest.TestCase):

    def setUp(self):
        
        pass

    def test_sample_case(self):

        case = os.path.join('data', 'psse_sample_case.raw')

        if os.path.isfile(case):
            
            # Parsed net
            net0 = pf.ParserRAW().parse(case)
            net0.get_bus_from_number(3006)

            # Merged
            net1 = net0.get_copy(merge_buses=True)
            net1.get_bus_from_number(3006)            

            # Copied
            net2 = net1.get_copy()
            
            # Extracted net (including)
            net3 = net1.extract_subnetwork([net1.get_bus_from_number(153),
                                            net1.get_bus_from_number(154),
                                            net1.get_bus_from_number(3005)])
            self.assertEqual(net3.num_buses, 3)
            
            # Extracted net (not including)
            net4 = net1.extract_subnetwork([net1.get_bus_from_number(101),
                                            net1.get_bus_from_number(102),
                                            net1.get_bus_from_number(151)])
            self.assertEqual(net4.num_buses, 3)
            self.assertRaises(pf.NetworkError, net4.get_bus_from_number, 153)
            self.assertRaises(pf.NetworkError, net4.get_bus_from_number, 3006)

            # Serialized net
            try:
                pf.ParserJSON().write(net1, 'foo.json')
                net5 = pf.ParserJSON().parse('foo.json')
            finally:
                if os.path.isfile('foo.json'):
                    os.remove('foo.json')

            pf.tests.utils.compare_networks(self, net1, net2)
            pf.tests.utils.compare_networks(self, net1, net5)

            # Test
            self.assertEqual(net0.get_num_redundant_buses(), 0)
            for net in [net1, net2, net3, net5]:
                self.assertEqual(net.get_num_redundant_buses(), 1)
                self.assertFalse(3006 in [bus.number for bus in net.buses])
                bus1 = net.get_bus_from_number(3006)
                bus2 = net.get_bus_from_number(153)
                self.assertTrue(bus1.is_equal(bus2))
                load1 = net.get_load_from_name_and_bus_number('1', 3006)
                load2 = net.get_load_from_name_and_bus_number('1', 153)
                self.assertTrue(load1.is_equal(load2))
                br1 = net.get_branch_from_name_and_bus_numbers('2', 3006, 154)
                br2 = net.get_branch_from_name_and_bus_numbers('2', 153, 154)
                self.assertTrue(br1.is_equal(br2))
                br3 = net.get_branch_from_name_and_bus_numbers('1', 3006, 3005)
                br4 = net.get_branch_from_name_and_bus_numbers('1', 153, 3005)
                self.assertTrue(br3.is_equal(br4))
        else:
            raise unittest.SkipTest('no .raw file')

    def test_aeso(self):

        case = os.path.join('data', 'aesoSL2014.raw')

        if not os.path.isfile(case):
            raise unittest.SkipTest('no .raw file')

        parser = pf.ParserRAW()

        net1 = parser.parse(case)

        for bus in net1.buses:
            bus.v_mag = bus.v_mag + 0.1
        
        net1_copy = net1.get_copy()
        net1_copy.update_properties()
        
        pf.tests.utils.compare_networks(self, net1, net1_copy)

        self.assertEqual(net1.num_buses, 2495)
        self.assertEqual(net1.num_branches, 2823)
        self.assertEqual(net1.get_num_zero_impedance_lines(), 42)
        self.assertEqual(net1.get_num_ZI_lines(), 42)

        net2 = net1.get_copy(merge_buses=True)
        net2.update_properties()
        
        net2.set_flags('bus', 'variable', 'any', 'voltage magnitude')
        
        self.assertEqual(net2.num_buses, 2454)
        self.assertEqual(net2.num_branches, 2823-42)
        self.assertEqual(net2.get_num_zero_impedance_lines(), 0)
        
        net1.copy_from_network(net2, merged=True)
        net1.update_properties()
        net1.update_properties()

        bus1 = net1.get_bus_from_number(25337)
        bus2 = net1.get_bus_from_number(25338)
        bus3 = net1.get_bus_from_number(25332)
        bus4 = net1.get_bus_from_number(25331)
        bus5 = net2.get_bus_from_number(25331)

        for bus in net2.buses:
            self.assertAlmostEqual(bus.P_mismatch, net1.get_bus_from_number(bus.number).P_mismatch, places=10)
            self.assertAlmostEqual(bus.Q_mismatch, net1.get_bus_from_number(bus.number).Q_mismatch, places=10)

        self.assertAlmostEqual(net1.bus_P_mis, net2.bus_P_mis, places=10)
        self.assertAlmostEqual(net1.bus_Q_mis, net2.bus_Q_mis, places=10)
        
        for bus in net2.buses:
            self.assertTrue(bus.has_flags('variable', 'voltage magnitude'))
        for bus in net1.buses:
            self.assertFalse(bus.has_flags('variable', 'voltage magnitude'))
        
        self.assertGreaterEqual(net2.num_vars, 1)
        self.assertEqual(net1.num_vars, 0)
        self.assertEqual(net1.num_fixed, 0)
        self.assertEqual(net1.num_bounded, 0)

        pf.tests.utils.compare_networks(self, net1, net1_copy)
            
        self.assertEqual(net1.bus_P_mis, net1_copy.bus_P_mis)
        self.assertEqual(net1.bus_Q_mis, net1_copy.bus_Q_mis)

        net1.copy_from_network(net2, merged=False)
        net1.update_properties()
        
        self.assertRaises(AssertionError, pf.tests.utils.compare_networks, self, net1, net1_copy)
        
        self.assertNotEqual(net1.bus_P_mis, net1_copy.bus_P_mis)
        self.assertNotEqual(net1.bus_Q_mis, net1_copy.bus_Q_mis)

        # Red buses in net2
        for bus in net1_copy.buses:
            self.assertEqual(bus.number, net1_copy.get_bus_from_number(bus.number).number)
            self.assertEqual(bus.name, net1_copy.get_bus_from_name(bus.name).name)
            self.assertEqual(bus.index, net1_copy.get_bus_from_number(bus.number).index)
            self.assertGreaterEqual(net2.get_bus_from_number(bus.number).index, 0)
            self.assertGreaterEqual(net2.get_bus_from_name(bus.name).index, 0)
        for gen in net1_copy.generators:
            self.assertGreaterEqual(net2.get_generator_from_name_and_bus_number(gen.name, gen.bus.number).index, 0)
        for load in net1_copy.loads:
            self.assertGreaterEqual(net2.get_load_from_name_and_bus_number(load.name, load.bus.number).index, 0)
        self.assertEqual(net2.get_num_redundant_buses(), 2495-2454)

        # Red buses in net3
        net3 = net2.get_copy(merge_buses=False)
        for bus in net1_copy.buses:
            self.assertGreaterEqual(net3.get_bus_from_number(bus.number).index, 0)
        for gen in net1_copy.generators:
            self.assertGreaterEqual(net3.get_generator_from_name_and_bus_number(gen.name, gen.bus.number).index, 0)
        for load in net1_copy.loads:
            self.assertGreaterEqual(net3.get_load_from_name_and_bus_number(load.name, load.bus.number).index, 0)
        self.assertEqual(net3.get_num_redundant_buses(), 2495-2454)
        
        
            
