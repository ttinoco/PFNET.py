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

    def test_parserepc(self):
        
        case = os.path.join('data', 'sample.epc')
        if not os.path.isfile(case):
            raise unittest.SkipTest('epc file not available')

        parser = pf.ParserEPC()
        parser.set('output_level', 0)

        net = parser.parse(case)

        self.assertEqual(net.num_buses, 56)

    def test_parserraw_star_bus_area_zone(self):
        
        case = os.path.join('data', 'psse_sample_case.raw')
        if not os.path.isfile(case):
            raise unittest.SkipTest('raw file not available')

        parser = pf.ParserRAW()
        parser.set('keep_all_out_of_service', True)
        net = parser.parse(case)

        counter = 0
        for bus in net.buses:
            if bus.is_star():
                neighbors = []
                for br in bus.branches:
                    if bus.index == br.bus_k.index:
                        neighbors.append(br.bus_m)
                    else:
                        neighbors.append(br.bus_k)
                self.assertEqual(len(neighbors), 3)
                for b in neighbors:
                    self.assertEqual(b.area, bus.area)
                    self.assertEqual(b.zone, bus.zone)
                counter += 1
        self.assertGreaterEqual(counter, 1)

    def test_parserraw_keep_all_lossless(self):

        try:
            
            for case in test_cases.CASES:
                
                if os.path.splitext(case)[-1] != '.raw':
                    continue                
                
                parser = pf.ParserRAW()
                parser.set('keep_all_out_of_service', 1)

                net1 = parser.parse(case)
                
                parser = pf.ParserRAW()
                parser.write(net1, 'foo.raw')

                parser = pf.ParserRAW()
                parser.set('keep_all_out_of_service', 1)

                net2 = parser.parse('foo.raw')

                pf.tests.utils.compare_networks(self, net1, net2, eps=1e-9)
        finally:

            if os.path.isfile('foo.raw'):
                os.remove('foo.raw')            

    def test_parserraw_in_out(self):
        
        case = os.path.join('data', 'psse_sample_case_oos.raw')
        if not os.path.isfile(case):
            raise unittest.SkipTest('raw file not available')

        parser = pf.ParserRAW()

        net = parser.parse(case)

        parser = pf.ParserRAW()
        parser.set('keep_all_out_of_service', True)
        net_oos = parser.parse(case)

        # Buses
        self.assertEqual(net.num_buses+3, net_oos.num_buses)
        self.assertRaises(pf.NetworkError, net.get_bus_from_number, 208)
        self.assertRaises(pf.NetworkError, net.get_bus_from_number, 209)
        self.assertRaises(pf.NetworkError, net.get_bus_from_number, 3012)
        bus = net_oos.get_bus_from_number(208)
        self.assertFalse(bus.in_service)
        bus = net_oos.get_bus_from_number(209)
        self.assertFalse(bus.in_service)
        bus = net_oos.get_bus_from_number(3012)
        self.assertFalse(bus.in_service)

        # star buses
        for bus in net_oos.buses:
            if bus.is_star():
                b = net.get_bus_from_number(bus.number)
                self.assertTrue(b.is_star())

        # Loads
        self.assertEqual(net.num_loads+2, net_oos.num_loads)
        self.assertRaises(pf.NetworkError, net.get_load_from_name_and_bus_number, '2', 154)
        self.assertRaises(pf.NetworkError, net.get_load_from_name_and_bus_number, '3', 154)
        load = net_oos.get_load_from_name_and_bus_number('2', 154)
        self.assertFalse(load.in_service)
        load = net_oos.get_load_from_name_and_bus_number('3', 154)
        self.assertFalse(load.in_service)

        # Generators
        self.assertEqual(net.num_generators+2, net_oos.num_generators)
        self.assertRaises(pf.NetworkError, net.get_generator_from_name_and_bus_number, '2', 301)
        self.assertRaises(pf.NetworkError, net.get_generator_from_name_and_bus_number, '3', 301)
        gen = net_oos.get_generator_from_name_and_bus_number('2', 301)
        self.assertFalse(gen.in_service)
        gen = net_oos.get_generator_from_name_and_bus_number('3', 301)
        self.assertFalse(gen.in_service)

        # Facts
        self.assertEqual(net.num_facts+2, net_oos.num_facts)
        self.assertRaises(pf.NetworkError, net.get_facts_from_name_and_bus_numbers, 'FACTS_DVCE_1', 153, 0)
        self.assertRaises(pf.NetworkError, net.get_facts_from_name_and_bus_numbers, 'FACTS_DVCE_2', 153, 155)
        facts = net_oos.get_facts_from_name_and_bus_numbers('FACTS_DVCE_1', 153, 0)
        self.assertFalse(facts.in_service)
        facts = net_oos.get_facts_from_name_and_bus_numbers('FACTS_DVCE_2', 153, 155)
        self.assertFalse(facts.in_service)

        # CSC
        self.assertEqual(net.num_csc_converters+2, net_oos.num_csc_converters)
        conv = net_oos.get_csc_converter_from_name_and_ac_bus_number('TWO_TERM_DC2', 301)
        self.assertFalse(conv.in_service)
        conv = net_oos.get_csc_converter_from_name_and_ac_bus_number('TWO_TERM_DC2', 3022)
        self.assertFalse(conv.in_service)

        # VSC
        self.assertEqual(net.num_vsc_converters+2, net_oos.num_vsc_converters)
        conv = net_oos.get_vsc_converter_from_name_and_ac_bus_number('VDCLINE2', 203)
        self.assertFalse(conv.in_service)
        conv = net_oos.get_vsc_converter_from_name_and_ac_bus_number('VDCLINE2', 205)
        self.assertFalse(conv.in_service)

        # DC buses
        self.assertEqual(net.num_dc_buses+4, net_oos.num_dc_buses)
        self.assertEqual(net.num_dc_buses, net_oos.get_num_dc_buses(only_in_service=True))
        
        # DC branches
        self.assertEqual(net.num_dc_branches+2, net_oos.num_dc_branches)
        self.assertEqual(net.num_dc_branches, net_oos.get_num_dc_branches(only_in_service=True))

        # fixed shunts
        self.assertEqual(net.get_num_fixed_shunts()+2+len([s for s in net_oos.shunts if s.is_part_of_transformer() and not s.in_service]),
                         net_oos.get_num_fixed_shunts())
        self.assertRaises(pf.NetworkError, net.get_fixed_shunt_from_name_and_bus_number, '1', 152)
        self.assertRaises(pf.NetworkError, net.get_fixed_shunt_from_name_and_bus_number, '1', 154)
        shunt = net_oos.get_fixed_shunt_from_name_and_bus_number('1', 152)
        self.assertFalse(shunt.in_service)
        shunt = net_oos.get_fixed_shunt_from_name_and_bus_number('1', 154)
        self.assertFalse(shunt.in_service)

        # switched shunts
        self.assertEqual(net.get_num_switched_shunts()+1, net_oos.get_num_switched_shunts())
        self.assertRaises(pf.NetworkError, net.get_switched_shunt_from_name_and_bus_number, '', 3021)
        shunt = net_oos.get_switched_shunt_from_name_and_bus_number('', 3021)
        self.assertFalse(shunt.in_service)

        # Branches       
        self.assertEqual(net.get_num_branches()+4, net_oos.get_num_branches())
        self.assertRaises(pf.NetworkError, net.get_branch_from_name_and_bus_numbers, '3', 208, 93003)
        self.assertRaises(pf.NetworkError, net.get_branch_from_name_and_bus_numbers, '4', 209, 93004)
        self.assertRaises(pf.NetworkError, net.get_branch_from_name_and_bus_numbers, '2', 3012, 93006)
        br = net_oos.get_branch_from_name_and_bus_numbers('3', 208, 93003)
        self.assertFalse(br.in_service)
        br = net_oos.get_branch_from_name_and_bus_numbers('4', 209, 93004)
        self.assertFalse(br.in_service)
        br = net_oos.get_branch_from_name_and_bus_numbers('2', 3012, 93006)
        self.assertFalse(br.in_service)
        
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

    def test_rts96_epc(self):

        case_epc = os.path.join('data', 'IEEE_RTS_96_bus.epc')
        case_raw = os.path.join('data', 'IEEE_RTS_96_bus.raw')

        if not os.path.isfile(case_epc):
            raise unittest.SkipTest('epc file not available')
        if not os.path.isfile(case_raw):
            raise unittest.SkipTest('raw file not available')

        parser_epc = pf.ParserEPC()
        parser_raw = pf.ParserRAW()

        net_epc = parser_epc.parse(case_epc)
        net_raw = parser_raw.parse(case_raw)

        # pf.tests.utils.compare_networks(self, net_epc, net_raw, check_internals=False, eps=1e-4)
        self.assertEqual(net_epc.num_buses, net_raw.num_buses)
        self.assertEqual(net_epc.num_branches, net_raw.num_branches)
        self.assertEqual(net_epc.num_generators, net_raw.num_generators)
        self.assertEqual(net_epc.num_loads, net_raw.num_loads)
        self.assertEqual(net_epc.num_shunts, net_raw.num_shunts)

        for i in range(net_epc.num_buses):
            bus1 = net_epc.get_bus(i)
            bus2 = net_raw.get_bus(i)
            pf.tests.utils.compare_buses(self, bus1, bus2, check_internals=False, check_indices=False, eps=1e-4)

        for i in range(10):
            branch_epc = net_epc.get_branch(i)
            branch_raw = net_raw.get_branch(109-i)

            self.assertEqual(branch_epc.b_k, branch_raw.b_k)
            self.assertEqual(branch_epc.b_m, branch_raw.b_m)
            self.assertEqual(branch_epc.bus_k.index, branch_raw.bus_k.index)
            self.assertEqual(branch_epc.is_fixed_tran(), branch_raw.is_fixed_tran())
            self.assertEqual(branch_epc.is_line(), branch_raw.is_line())
            self.assertEqual(branch_epc.ratingA, branch_raw.ratingA)

        for i in range(net_epc.num_generators):
            gen1 = net_epc.get_generator(i)
            gen2 = net_raw.get_generator(i)
            pf.tests.utils.compare_generators(self, gen1, gen2, check_internals=False, eps=1e-4)

        for i in range(net_epc.num_shunts):
            shunt1 = net_epc.get_shunt(i)
            shunt2 = net_raw.get_shunt(i)
            self.assertEqual(shunt1.in_service, shunt2.in_service)
            self.assertEqual(shunt1.num_periods, shunt2.num_periods)
            self.assertEqual(shunt1.bus.index, shunt2.bus.index)
            self.assertEqual(shunt1.is_fixed(), shunt2.is_fixed())
            self.assertEqual(shunt1.b, shunt2.b)
            self.assertEqual(shunt1.g, shunt2.g)
            self.assertEqual(shunt1.b_max, shunt2.b_max)
            self.assertEqual(shunt1.b_min, shunt2.b_min)

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

    def test_ACTIVSg10k_raw(self):

        case = os.path.join('data', 'ACTIVSg10k.raw')
        if not os.path.isfile(case):
            raise unittest.SkipTest('file not available')

        parser = pf.ParserRAW()

        net = parser.parse(case)

        results = []
        for shunt in net.shunts:
            if shunt.is_switched() and shunt.is_discrete():
                results.append(any(shunt.b == shunt.b_values))

        self.assertGreater(len(results), 0)
        self.assertTrue(all(results))

        parser.set('round_switched_shunts', False)

        net = parser.parse(case)

        results = []
        for shunt in net.shunts:
            if shunt.is_switched() and shunt.is_discrete():
                results.append(any(shunt.b == shunt.b_values))

        self.assertGreater(len(results), 0)
        self.assertFalse(all(results))

    def test_ieee25_raw(self):

        case = os.path.join('data', 'ieee25.raw')
        if not os.path.isfile(case):
            raise unittest.SkipTest('file not available')

        parser = pf.ParserRAW()
        
        self.assertRaises(pf.ParserError, parser.set, 'foo', 5)
        
        parser.set('round_tap_ratios', False)

        net = parser.parse(case)
        
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

        case = os.path.join('data', 'ieee14.m')
        if not os.path.isfile(case):
            raise unittest.SkipTest('file not available')

        net = pf.PyParserMAT().parse(case)

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
