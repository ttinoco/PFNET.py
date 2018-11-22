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

class TestParser(unittest.TestCase):

    def setUp(self):
        
        pass

    def test_parserraw_zi_lines(self):

        case = os.path.join('data', 'psse_sample_case.raw')

        if os.path.isfile(case):

            # Parsed net
            net1 = pf.ParserRAW().parse(case)

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

            # Test
            for net in [net1, net2, net3, net5]:
                self.assertEqual(net.get_num_redundant_buses(), 1)
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
        
    def test_parserraw_write(self):

        tested = False
        for case in test_cases.CASES:
            
            if os.path.splitext(case)[-1] != '.raw':
                continue
            
            parser = pf.ParserRAW()
            parser.set('output_level', 0)
            net1 = parser.parse(case, num_periods=2)
            
            try:
                parser.write(net1, 'foo.raw')

                net2 = parser.parse('foo.raw', num_periods=2)

                new_parser = pf.ParserRAW()
                net3 = parser.parse('foo.raw', num_periods=2)
                
            finally:
                if os.path.isfile('foo.raw'):
                    os.remove('foo.raw')
           
            pf.tests.utils.compare_networks(self, net1, net2, eps=1e-9)
            pf.tests.utils.compare_networks(self, net1, net3, eps=1e-9)
            tested = True
            
        if not tested:
            raise unittest.SkipTest("no .raw files")

    def test_case118_m(self):

        case_mat = os.path.join('data', 'case118.mat')
        case_m = os.path.join('data', 'case118.m')

        if os.path.isfile(case_mat) and os.path.isfile(case_m):
            net_mat = pf.Parser(case_mat).parse(case_mat, num_periods=2)
            net_m = pf.Parser(case_m).parse(case_m, num_periods=2)
            pf.tests.utils.compare_networks(self, net_mat, net_m)
        else:
            raise unittest.SkipTest('no .m files')

    def test_pyparsermat_write(self):

        tested = False
        for case in test_cases.CASES:
            
            if os.path.splitext(case)[-1] != '.m':
                continue

            parser = pf.PyParserMAT()
            parser.set('output_level', 0)
            net1 = parser.parse(case, num_periods=2)

            try:
                p = pf.PyParserMAT()
                p.write(net1, 'foo.m')

                parser = pf.PyParserMAT()
                net2 = parser.parse('foo.m', num_periods=2)
            finally:
                if os.path.isfile('foo.m'):
                    os.remove('foo.m')

            pf.tests.utils.compare_networks(self, net1, net2)
            tested = True
        if not tested:
            raise unittest.SkipTest("no .m files")

    def test_ieee25_raw(self):

        case = os.path.join('data', 'ieee25.raw')
        if not os.path.isfile(case):
            raise unittest.SkipTest('file not available')

        net = pf.ParserRAW().parse(case)
        
        self.assertEqual(net.num_buses,25)
        
        self.assertTrue(101 in [bus.number for bus in net.buses])
        self.assertTrue(104 in [bus.number for bus in net.buses])
        self.assertTrue(106 in [bus.number for bus in net.buses])
        self.assertTrue(222 in [bus.number for bus in net.buses])
        
        tested_101 = False
        tested_104 = False
        tested_106 = False
        tested_222 = False
        for bus in net.buses:
            
            # base kv
            if bus.number >= 101 and bus.number <= 110:
                self.assertEqual(bus.v_base,138.)
            else:
                self.assertLessEqual(bus.number,225)
                self.assertGreaterEqual(bus.number,211)
                self.assertEqual(bus.v_base,230.)
                
            # names
            if bus.number == 101:
                self.assertEqual(bus.name, 'COAL-A')
                self.assertEqual(len(bus.generators),3)
                self.assertEqual([g.name for g in bus.generators],
                                 ['5','4','3'])
                tested_101 = True
                
            if bus.number == 104:
                self.assertEqual(len(bus.loads),1)
                self.assertEqual(bus.loads[0].name, '1')
                tested_104 = True
                
            if bus.number == 106:
                self.assertEqual(len(bus.shunts),2)
                self.assertEqual([s.name for s in bus.shunts],
                                 ['','1'])
                tested_106 = True
                
            if bus.number == 222:
                self.assertEqual(bus.name,'HYDRO')
                self.assertEqual(len(bus.generators),10)
                self.assertEqual([g.name for g in bus.generators],
                                 ['A','9','8','7','6','5','4','3','2','1'])
                tested_222 = True
                
            brs = []
            for branch in net.branches:
                if (branch.bus_k.number, branch.bus_m.number) == (215,221):
                    brs.append(branch)
                if (branch.bus_k.number, branch.bus_m.number) == (103,224):
                    self.assertEqual(branch.name,'1')
            self.assertEqual(len(brs),2)
            self.assertEqual([br.name for br in brs],['2','1'])

        self.assertTrue(tested_101)
        self.assertTrue(tested_104)
        self.assertTrue(tested_106)
        self.assertTrue(tested_222)
                
    def test_sys_problem2(self):

        case = os.path.join('data', 'sys_problem2.mat')
        if not os.path.isfile(case):
            raise unittest.SkipTest('file not available')

        net = pf.ParserMAT().parse(case)

        self.assertEqual(net.base_power,100.)
        
        self.assertEqual(net.num_buses,3)
        self.assertEqual(net.num_generators,4)
        self.assertEqual(net.num_loads,3)
        self.assertEqual(net.num_branches,3)
        
        bus1 = net.get_bus_from_number(1)
        bus2 = net.get_bus_from_number(2)
        bus3 = net.get_bus_from_number(3)
        
        for bus in net.buses:
            self.assertEqual(bus.v_base,220.)
            
        self.assertEqual(bus1.number,1)
        self.assertEqual(bus2.number,2)
        self.assertEqual(bus3.number,3)
            
        branch13 = net.get_branch(0)
        branch23 = net.get_branch(1)
        branch12 = net.get_branch(2)
        
        self.assertEqual(len(bus1.generators),2)
        self.assertEqual(len(bus2.generators),1)
        self.assertEqual(len(bus3.generators),1)
        
        self.assertEqual(len(bus1.loads),1)
        self.assertEqual(len(bus2.loads),1)
        self.assertEqual(len(bus3.loads),1)
        
        gen0 = net.get_generator(3)
        gen1 = net.get_generator(2)
        gen2 = net.get_generator(1) 
        gen3 = net.get_generator(0)
        
        load0 = net.get_load(2)
        load1 = net.get_load(1)
        load2 = net.get_load(0)
    
        self.assertEqual(load0.bus,bus1)
        self.assertEqual(load1.bus,bus2)
        self.assertEqual(load2.bus,bus3)
        
        self.assertEqual(gen0.P,50/100.)
        self.assertEqual(gen1.P,40/100.)
        self.assertEqual(gen2.P,30/100.)
        self.assertEqual(gen3.P,20/100.)
        self.assertEqual(gen0.P_max,50)
        self.assertEqual(gen1.P_max,50)
        self.assertEqual(gen2.P_max,25)
        self.assertEqual(gen3.P_max,19)
        self.assertEqual(gen0.bus,bus1)
        self.assertEqual(gen1.bus,bus1)
        self.assertEqual(gen2.bus,bus2)
        self.assertEqual(gen3.bus,bus3)
        for gen in net.generators:
            self.assertEqual(gen.P_min,0.)

        self.assertEqual(branch13.bus_k.number,bus1.number)
        self.assertEqual(branch13.bus_m.number,bus3.number)
        
        self.assertEqual(branch23.bus_k.number,bus2.number)
        self.assertEqual(branch23.bus_m.number,bus3.number)
        
        self.assertEqual(branch12.bus_k.number,bus1.number)
        self.assertEqual(branch12.bus_m.number,bus2.number)
        
        self.assertEqual(branch13.g,0)
        self.assertLess(abs(branch13.b + 1./0.1),1e-10)
        self.assertEqual(branch13.ratingA,30.95)
        self.assertEqual(branch13.ratingB,30.95)
        self.assertEqual(branch13.ratingC,30.95)
        
        self.assertEqual(branch23.g,0)
        self.assertLess(abs(branch23.b + 1./0.2),1e-10)
        self.assertEqual(branch23.ratingA,13)
        self.assertEqual(branch23.ratingB,13)
        self.assertEqual(branch23.ratingC,13)
        
        self.assertEqual(branch12.g,0)
        self.assertLess(abs(branch12.b + 1./0.2),1e-10)
        self.assertEqual(branch12.ratingA,15)
        self.assertEqual(branch12.ratingB,15)
        self.assertEqual(branch12.ratingC,15)

        self.assertEqual(gen0.cost_coeff_Q0,0)
        self.assertEqual(gen0.cost_coeff_Q1,6.*net.base_power)
        self.assertEqual(gen0.cost_coeff_Q2,0.03*(net.base_power**2.))

        self.assertEqual(gen1.cost_coeff_Q0,0)
        self.assertEqual(gen1.cost_coeff_Q1,5.*net.base_power)
        self.assertEqual(gen1.cost_coeff_Q2,0.02*(net.base_power**2.))

        self.assertEqual(gen2.cost_coeff_Q0,0)
        self.assertEqual(gen2.cost_coeff_Q1,12.*net.base_power)
        self.assertEqual(gen2.cost_coeff_Q2,0.06*(net.base_power**2.))
        
        self.assertEqual(gen3.cost_coeff_Q0,0)
        self.assertEqual(gen3.cost_coeff_Q1,10.*net.base_power)
        self.assertEqual(gen3.cost_coeff_Q2,0.08*(net.base_power**2.))
        
        # Load utility
        self.assertEqual(load0.util_coeff_Q0,0)
        self.assertEqual(load0.util_coeff_Q1,400.*net.base_power)
        self.assertEqual(load0.util_coeff_Q2,-0.03*(net.base_power**2.))
        
        self.assertEqual(load1.util_coeff_Q0,0)
        self.assertEqual(load1.util_coeff_Q1,450.*net.base_power)
        self.assertEqual(load1.util_coeff_Q2,-0.02*(net.base_power**2.))
        
        self.assertEqual(load2.util_coeff_Q0,0)
        self.assertEqual(load2.util_coeff_Q1,300.*net.base_power)
        self.assertEqual(load2.util_coeff_Q2,-0.03*(net.base_power**2.))

    def test_sys_problem3(self):

        case = os.path.join('data', 'sys_problem3.mat')
        if not os.path.isfile(case):
            raise unittest.SkipTest('file not available')

        net = pf.ParserMAT().parse(case)
        
        self.assertEqual(net.base_power,100.)
        
        # numbers
        self.assertEqual(net.num_buses,10)
        self.assertEqual(net.num_generators,5)
        self.assertEqual(net.num_loads,10)
        self.assertEqual(net.num_shunts,0)
        self.assertEqual(net.num_var_generators,0)
        self.assertEqual(net.num_branches,13)
        
        # buses
        bus1 = net.get_bus_from_number(1)
        bus2 = net.get_bus_from_number(2)
        bus3 = net.get_bus_from_number(3)
        bus4 = net.get_bus_from_number(4)
        bus5 = net.get_bus_from_number(5)
        bus6 = net.get_bus_from_number(6)
        bus7 = net.get_bus_from_number(7)
        bus8 = net.get_bus_from_number(8)
        bus9 = net.get_bus_from_number(9)
        bus10 = net.get_bus_from_number(10)
        
        for bus in net.buses:
            self.assertEqual(bus.v_base,69.)
            
        # loads
        for bus in net.buses:
            self.assertEqual(len(bus.loads),1)
        load1 = bus1.loads[0]
        load2 = bus2.loads[0]
        load3 = bus3.loads[0]
        load4 = bus4.loads[0]
        load5 = bus5.loads[0]
        load6 = bus6.loads[0]
        load7 = bus7.loads[0]
        load8 = bus8.loads[0]
        load9 = bus9.loads[0]
        load10 = bus10.loads[0]
        
        self.assertEqual(load1.bus,bus1)
        self.assertEqual(load2.bus,bus2)
        self.assertEqual(load3.bus,bus3)
        self.assertEqual(load4.bus,bus4)
        self.assertEqual(load5.bus,bus5)
        self.assertEqual(load6.bus,bus6)
        self.assertEqual(load7.bus,bus7)
        self.assertEqual(load8.bus,bus8)
        self.assertEqual(load9.bus,bus9)
        self.assertEqual(load10.bus,bus10)
        
        self.assertEqual(load1.P,55./100.)
        self.assertEqual(load2.P,55/100.)
        self.assertEqual(load3.P,1300/100.)
        self.assertEqual(load4.P,650/100.)
        self.assertEqual(load5.P,650/100.)
        self.assertEqual(load6.P,200/100.)
        self.assertEqual(load7.P,2600/100.)
        self.assertEqual(load8.P,3600/100.)
        self.assertEqual(load9.P,1100/100.)
        self.assertEqual(load10.P,1900/100.)
        for load in net.loads:
            self.assertEqual(load.P_max,load.P)
            self.assertEqual(load.P_min,load.P)
            
        # generators
        self.assertEqual(len(bus1.generators),0)
        self.assertEqual(len(bus2.generators),1)
        self.assertEqual(len(bus3.generators),1)
        self.assertEqual(len(bus4.generators),0)
        self.assertEqual(len(bus5.generators),1)
        self.assertEqual(len(bus6.generators),0)
        self.assertEqual(len(bus7.generators),1)
        self.assertEqual(len(bus8.generators),1)
        self.assertEqual(len(bus9.generators),0)
        self.assertEqual(len(bus10.generators),0)
        gen1 = bus2.generators[0]
        gen2 = bus3.generators[0]
        gen3 = bus5.generators[0]
        gen4 = bus7.generators[0]
        gen5 = bus8.generators[0]
        
        self.assertEqual(gen1.bus,bus2)
        self.assertEqual(gen2.bus,bus3)
        self.assertEqual(gen3.bus,bus5)
        self.assertEqual(gen4.bus,bus7)
        self.assertEqual(gen5.bus,bus8)
        
        self.assertEqual(gen1.P,50./100.)
        self.assertEqual(gen1.P_min,0)
        self.assertEqual(gen1.P_max,1200./100.)
        self.assertEqual(gen1.cost_coeff_Q0,0)
        self.assertEqual(gen1.cost_coeff_Q1,6.9*100)
        self.assertEqual(gen1.cost_coeff_Q2,0.00067*(100**2.))
        
        self.assertEqual(gen2.P,40./100.)
        self.assertEqual(gen2.P_min,0)
        self.assertEqual(gen2.P_max,8000./100.)
        self.assertEqual(gen2.cost_coeff_Q0,0)
        self.assertEqual(gen2.cost_coeff_Q1,24.3*100)
        self.assertEqual(gen2.cost_coeff_Q2,0.00040*(100**2.))
        
        self.assertEqual(gen3.P,30./100.)
        self.assertEqual(gen3.P_min,0)
        self.assertEqual(gen3.P_max,3000./100.)
        self.assertEqual(gen3.cost_coeff_Q0,0)
        self.assertEqual(gen3.cost_coeff_Q1,29.1*100)
        self.assertEqual(gen3.cost_coeff_Q2,0.00006*(100**2.))
        
        self.assertEqual(gen4.P,20./100.)
        self.assertEqual(gen4.P_min,0)
        self.assertEqual(gen4.P_max,800./100.)
        self.assertEqual(gen4.cost_coeff_Q0,0)
        self.assertEqual(gen4.cost_coeff_Q1,6.9*100)
        self.assertEqual(gen4.cost_coeff_Q2,0.00026*(100**2.))
        
        self.assertEqual(gen5.P,10./100.)
        self.assertEqual(gen5.P_min,0)
        self.assertEqual(gen5.P_max,2000./100.)
        self.assertEqual(gen5.cost_coeff_Q0,0)
        self.assertEqual(gen5.cost_coeff_Q1,50.*100)
        self.assertEqual(gen5.cost_coeff_Q2,0.0015*(100**2.))
        
        # branches
        branch1 = net.get_branch(12)
        branch2 = net.get_branch(11)
        branch3 = net.get_branch(10)
        branch4 = net.get_branch(9)
        branch5 = net.get_branch(8)
        branch6 = net.get_branch(7)
        branch7 = net.get_branch(6)
        branch8 = net.get_branch(5)
        branch9 = net.get_branch(4)
        branch10 = net.get_branch(3)
        branch11 = net.get_branch(2)
        branch12 = net.get_branch(1)
        branch13 = net.get_branch(0)
        
        self.assertEqual(branch1.bus_k,bus1)
        self.assertEqual(branch1.bus_m,bus3)
        self.assertLess(abs(branch1.b + 1./0.1),1e-10)
        self.assertEqual(branch1.ratingA,3000./100.)
        self.assertEqual(branch1.ratingB,3000./100.)
        self.assertEqual(branch1.ratingC,3000./100.)
        
        self.assertEqual(branch2.bus_k,bus1)
        self.assertEqual(branch2.bus_m,bus10)
        self.assertLess(abs(branch2.b + 1./0.27),1e-10)
        self.assertEqual(branch2.ratingA,2000./100.)
        self.assertEqual(branch2.ratingB,2000./100.)
        self.assertEqual(branch2.ratingC,2000./100.)
        
        self.assertEqual(branch3.bus_k,bus2)
        self.assertEqual(branch3.bus_m,bus3)
        self.assertLess(abs(branch3.b + 1./0.12),1e-10)
        self.assertEqual(branch3.ratingA,6500./100.)
        self.assertEqual(branch3.ratingB,6500./100.)
        self.assertEqual(branch3.ratingC,6500./100.)
        
        self.assertEqual(branch4.bus_k,bus2)
        self.assertEqual(branch4.bus_m,bus9)
        self.assertLess(abs(branch4.b + 1./0.07),1e-10)
        self.assertEqual(branch4.ratingA,5500./100.)
        self.assertEqual(branch4.ratingB,5500./100.)
        self.assertEqual(branch4.ratingC,5500./100.)
        
        self.assertEqual(branch5.bus_k,bus2)
        self.assertEqual(branch5.bus_m,bus10)
        self.assertLess(abs(branch5.b + 1./0.14),1e-10)
        self.assertEqual(branch5.ratingA,5500./100.)
        self.assertEqual(branch5.ratingB,5500./100.)
        self.assertEqual(branch5.ratingC,5500./100.)
        
        self.assertEqual(branch6.bus_k,bus3)
        self.assertEqual(branch6.bus_m,bus4)
        self.assertLess(abs(branch6.b + 1./0.1),1e-10)
        self.assertEqual(branch6.ratingA,3000./100.)
        self.assertEqual(branch6.ratingB,3000./100.)
        self.assertEqual(branch6.ratingC,3000./100.)
        
        self.assertEqual(branch7.bus_k,bus3)
        self.assertEqual(branch7.bus_m,bus5)
        self.assertLess(abs(branch7.b + 1./0.17),1e-10)
        self.assertEqual(branch7.ratingA,4000./100.)
        self.assertEqual(branch7.ratingB,4000./100.)
        self.assertEqual(branch7.ratingC,4000./100.)
        
        self.assertEqual(branch8.bus_k,bus4)
        self.assertEqual(branch8.bus_m,bus5)
        self.assertLess(abs(branch8.b + 1./0.17),1e-10)
        self.assertEqual(branch8.ratingA,4000./100.)
        self.assertEqual(branch8.ratingB,4000./100.)
        self.assertEqual(branch8.ratingC,4000./100.)
        
        self.assertEqual(branch9.bus_k,bus5)
        self.assertEqual(branch9.bus_m,bus6)
        self.assertLess(abs(branch9.b + 1./0.17),1e-10)
        self.assertEqual(branch9.ratingA,5000./100.)
        self.assertEqual(branch9.ratingB,5000./100.)
        self.assertEqual(branch9.ratingC,5000./100.)
        
        self.assertEqual(branch10.bus_k,bus6)
        self.assertEqual(branch10.bus_m,bus7)
        self.assertLess(abs(branch10.b + 1./0.16),1e-10)
        self.assertEqual(branch10.ratingA,2000./100.)
        self.assertEqual(branch10.ratingB,2000./100.)
        self.assertEqual(branch10.ratingC,2000./100.)
        
        self.assertEqual(branch11.bus_k,bus7)
        self.assertEqual(branch11.bus_m,bus8)
        self.assertLess(abs(branch11.b + 1./0.25),1e-10)
        self.assertEqual(branch11.ratingA,3000./100.)
        self.assertEqual(branch11.ratingB,3000./100.)
        self.assertEqual(branch11.ratingC,3000./100.)
        
        self.assertEqual(branch12.bus_k,bus8)
        self.assertEqual(branch12.bus_m,bus9)
        self.assertLess(abs(branch12.b + 1./0.25),1e-10)
        self.assertEqual(branch12.ratingA,2500./100.)
        self.assertEqual(branch12.ratingB,2500./100.)
        self.assertEqual(branch12.ratingC,2500./100.)
        
        self.assertEqual(branch13.bus_k,bus8)
        self.assertEqual(branch13.bus_m,bus10)
        self.assertLess(abs(branch13.b + 1./0.07),1e-10)
        self.assertEqual(branch13.ratingA,4000./100.)
        self.assertEqual(branch13.ratingB,4000./100.)
        self.assertEqual(branch13.ratingC,4000./100.)

    def test_ieee14_gen_cost(self):

        case = os.path.join('data', 'ieee14.mat')
        if not os.path.isfile(case):
            raise unittest.SkipTest('file not available')

        net = pf.ParserMAT().parse(case)

        self.assertEqual(net.base_power,100.)
        self.assertEqual(net.num_buses,14)
        self.assertEqual(net.num_generators,5)
        
        gen0 = net.get_generator(net.num_generators-1)
        gen1 = net.get_generator(net.num_generators-2)
        
        self.assertEqual(gen0.P,232.4/100.)
        self.assertEqual(gen0.cost_coeff_Q2,(4.3029259900e-02)*(net.base_power**2.))
        self.assertEqual(gen0.cost_coeff_Q1,20.*net.base_power)
        
        self.assertEqual(gen1.P,40./100.)
        self.assertEqual(gen1.cost_coeff_Q2,0.25*(net.base_power**2.))
        self.assertEqual(gen1.cost_coeff_Q1,20.*net.base_power)
            
    def test_type_parsers(self):

        for case in test_cases.CASES:

            if case.split('.')[-1] == 'mat':
                self.assertRaises(pf.ParserError,pf.ParserJSON().parse,case)
                self.assertRaises(pf.ParserError,pf.PyParserMAT().parse,case)
                if pf.has_raw_parser():
                    self.assertRaises(pf.ParserError,pf.ParserRAW().parse,case)
                net = pf.ParserMAT().parse(case)
                self.assertGreater(net.num_buses,0)
            if case.split('.')[-1] == 'm':
                self.assertRaises(pf.ParserError,pf.ParserJSON().parse,case)
                self.assertRaises(pf.ParserError,pf.ParserMAT().parse,case)
                if pf.has_raw_parser():
                    self.assertRaises(pf.ParserError,pf.ParserRAW().parse,case)
                net = pf.PyParserMAT().parse(case)
                self.assertGreater(net.num_buses,0)
            elif case.split('.')[-1] == 'raw':
                self.assertRaises(pf.ParserError,pf.ParserMAT().parse,case)
                self.assertRaises(pf.ParserError,pf.ParserJSON().parse,case)
                self.assertRaises(pf.ParserError,pf.PyParserMAT().parse,case)
                if pf.has_raw_parser():
                    net = pf.ParserRAW().parse(case)
                    self.assertGreater(net.num_buses,0)

    def test_json_parser(self):

        import os
        from numpy.linalg import norm
        eps = 1e-10

        norminf = lambda x: norm(x,np.inf) if not np.isscalar(x) else np.abs(x)

        for case in test_cases.CASES:

            T = 4
                
            net = pf.Parser(case).parse(case,T)
            self.assertEqual(net.num_periods,T)

            # Bus sens
            for bus in net.buses:
                bus.sens_P_balance = np.random.randn(net.num_periods)
                bus.sens_Q_balance = np.random.randn(net.num_periods)
                bus.sens_v_mag_u_bound = np.random.randn(net.num_periods)
                bus.sens_v_mag_l_bound = np.random.randn(net.num_periods)
                bus.sens_v_ang_u_bound = np.random.randn(net.num_periods)
                bus.sens_v_ang_l_bound = np.random.randn(net.num_periods)
                bus.sens_v_set_reg = np.random.randn(net.num_periods)
                bus.sens_v_reg_by_tran = np.random.randn(net.num_periods)
                bus.sens_v_reg_by_shunt = np.random.randn(net.num_periods)
            
            # Set flags
            net.set_flags('bus','variable','any','voltage magnitude')
            self.assertEqual(net.num_vars,net.num_buses*T)
            
            # Add vargens and betteries
            net.add_var_generators_from_parameters(net.get_load_buses(),100.,50.,30.,5,0.05)
            net.add_batteries_from_parameters(net.get_generator_buses(),20.,50.)
            
            # Some perturbations to reduce luck
            for bus in net.buses:
                bus.price = np.random.randn(T)

            try:
                
                json_parser = pf.ParserJSON()
                
                json_parser.write(net,"temp_json.json")
                
                new_net = json_parser.parse("temp_json.json")
                self.assertEqual(new_net.num_periods,T)

                # Compare
                pf.tests.utils.compare_networks(self, net, new_net)
                                    
            finally:
                
                os.remove("temp_json.json")
