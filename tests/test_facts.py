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

class TestFACTS(unittest.TestCase):

    def test_psse_sample_raw_case(self):

        T = 4

        case = os.path.join('data','psse_sample_case.raw')
        if not os.path.isfile(case):
            raise unittest.SkipTest('file not available')
        
        p = pf.ParserRAW()

        net = p.parse(case, T)

        # Network
        self.assertEqual(net.num_buses-net.get_num_star_buses(), 41)

        self.assertEqual(net.num_facts,3)
        
        f1 = net.get_facts(0) # 153/3006 - 155 (upfc)
        f2 = net.get_facts(1) # 153/3006 - 0 (statcom)
        f3 = net.get_facts(2) # 153/3006 - 155 (sssc)

        self.assertTrue(isinstance(f1, pf.Facts))
        self.assertTrue(isinstance(f2, pf.Facts))
        self.assertTrue(isinstance(f3, pf.Facts))

        self.assertEqual(len(net.facts),3)
        self.assertTrue(net.facts[0].is_equal(f1))
        self.assertTrue(net.facts[1].is_equal(f2))
        self.assertTrue(net.facts[2].is_equal(f3))

        self.assertEqual(f2.bus_k.number, 3006) # zi with 153

        f = net.get_facts_from_name_and_bus_numbers('FACTS_DVCE_1',
                                                    3006,
                                                    0)

        self.assertTrue(f.is_equal(f2))
        self.assertFalse(f.is_equal(f1))
        self.assertEqual(f.name, 'FACTS_DVCE_1')

        f = net.get_facts_from_name_and_bus_numbers('FACTS_DVCE_2',
                                                    3006,
                                                    155)

        self.assertTrue(f.is_equal(f1))
        self.assertFalse(f.is_equal(f2))
        self.assertEqual(f.name, 'FACTS_DVCE_2')

        f = net.get_facts_from_name_and_bus_numbers('FACTS_DVCE_3',
                                                    3006,
                                                    155)

        self.assertTrue(f.is_equal(f3))
        self.assertFalse(f.is_equal(f2))
        self.assertEqual(f.name, 'FACTS_DVCE_3')

        self.assertEqual(net.get_num_buses_reg_by_facts(), 1)

        # Types
        self.assertTrue(f2.is_STATCOM())
        self.assertFalse(f2.is_SSSC())
        self.assertFalse(f2.is_UPFC())
        self.assertFalse(f1.is_STATCOM())
        self.assertFalse(f1.is_SSSC())
        self.assertTrue(f1.is_UPFC())
        self.assertFalse(f3.is_STATCOM())
        self.assertTrue(f3.is_SSSC())
        self.assertFalse(f3.is_UPFC())
        self.assertTrue(f1.is_regulator())
        self.assertTrue(f2.is_regulator())
        self.assertFalse(f3.is_regulator())

        # Bus
        bus1 = net.get_bus_from_number(3006)
        bus2 = net.get_bus_from_number(155)

        self.assertTrue(bus1.is_regulated_by_facts())
        self.assertFalse(bus2.is_regulated_by_facts())

        self.assertEqual(len(bus1.facts_k),3)
        self.assertEqual(len(bus2.facts_k),0)

        self.assertEqual(len(bus1.facts_m),0)
        self.assertEqual(len(bus2.facts_m),2)

        self.assertEqual(len(bus1.facts),3)
        self.assertEqual(len(bus2.facts),2)

        self.assertTrue(bus1.facts[0].is_equal(f1))
        self.assertTrue(bus1.facts[1].is_equal(f2))
        self.assertTrue(bus1.facts[2].is_equal(f3))
        self.assertTrue(bus2.facts[0].is_equal(f1))
        self.assertTrue(bus2.facts[1].is_equal(f3))

        self.assertEqual(len(bus1.reg_facts),2)
        self.assertEqual(len(bus2.reg_facts),0)
        self.assertTrue(bus1.reg_facts[0].is_equal(f1))
        self.assertTrue(bus1.reg_facts[1].is_equal(f2))

        for t in range(T):
            self.assertEqual(bus1.v_set[t], 1.015)
            self.assertEqual(bus2.v_set[t], 1.000)

        # Facts 153/3006 - 155 (UPFC)
        self.assertEqual(f1.name, 'FACTS_DVCE_2')
        self.assertFalse(f1.is_series_link_disabled())
        self.assertFalse(f1.is_series_link_bypassed())
        self.assertTrue(f1.is_in_normal_series_mode())
        self.assertFalse(f1.is_in_constant_series_z_mode())
        self.assertFalse(f1.is_in_constant_series_v_mode())
        self.assertEqual(f1.num_periods, T)
        self.assertEqual(f1.obj_type, 'facts')
        self.assertEqual(f1.index, 0)
        for t in range(T):
            self.assertEqual(f1.v_mag_s[t], 0.01)
            self.assertEqual(f1.v_ang_s[t], 0.01)
            self.assertEqual(f1.P_k[t], -3.5)
            self.assertEqual(f1.P_m[t], 3.5)
            self.assertEqual(f1.Q_k[t], -0.4)
            self.assertEqual(f1.Q_m[t], 0.4)
            self.assertEqual(f1.Q_sh[t], 0.)
            self.assertEqual(f1.Q_s[t], 0.)
            self.assertEqual(f1.P_dc[t], 0.)
            self.assertEqual(f1.P_set[t], 3.5)
            self.assertEqual(f1.Q_set[t], 0.4)
        self.assertEqual(f1.Q_par, 1.)
        self.assertEqual(f1.v_max_s, 1.)
        self.assertEqual(f1.g, 0.)
        self.assertEqual(f1.b, 0.)
        self.assertEqual(f1.i_max_s, 0.)
        self.assertEqual(f1.Q_max_s, pf.FACTS_INF_Q)
        self.assertEqual(f1.Q_min_s, -pf.FACTS_INF_Q)
        self.assertEqual(f1.i_max_sh, 0.25)
        self.assertEqual(f1.Q_max_sh, 0.25)
        self.assertEqual(f1.Q_min_sh, -0.25)
        self.assertEqual(f1.P_max_dc, 99.99)
        self.assertEqual(f1.v_min_m, 0.9)
        self.assertEqual(f1.v_max_m, 1.1)
        self.assertEqual(f1.bus_k.number, 3006)
        self.assertEqual(f1.bus_m.number, 155)

        # Facts 153/3006 - 0 (STATCOM)
        self.assertEqual(f2.name, 'FACTS_DVCE_1')
        self.assertTrue(f2.is_series_link_disabled())
        self.assertFalse(f2.is_series_link_bypassed())
        self.assertFalse(f2.is_in_normal_series_mode())
        self.assertFalse(f2.is_in_constant_series_z_mode())
        self.assertFalse(f2.is_in_constant_series_v_mode())
        self.assertEqual(f2.num_periods, T)
        self.assertEqual(f2.obj_type, 'facts')
        self.assertEqual(f2.index, 1)
        for t in range(T):
            self.assertEqual(f2.v_mag_s[t], 0.01)
            self.assertEqual(f2.v_ang_s[t], 0.01)
            self.assertEqual(f2.P_k[t], 0.)
            self.assertEqual(f2.P_m[t], 0.)
            self.assertEqual(f2.Q_k[t], 0.)
            self.assertEqual(f2.Q_m[t], 0.)
            self.assertEqual(f2.Q_sh[t], 0.)
            self.assertEqual(f2.Q_s[t], 0.)
            self.assertEqual(f2.P_dc[t], 0.)
            self.assertEqual(f2.P_set[t], 0.)
            self.assertEqual(f2.Q_set[t], 0.)
        self.assertEqual(f2.Q_par, 1.)
        self.assertEqual(f2.v_max_s, 1.)
        self.assertEqual(f2.g, 0.)
        self.assertEqual(f2.b, 0.)
        self.assertEqual(f2.i_max_s, 0.)
        self.assertEqual(f2.i_max_sh, 0.5)
        self.assertEqual(f2.P_max_dc, 1.)
        self.assertEqual(f2.v_min_m, 0.9263)
        self.assertEqual(f2.v_max_m, 1.134)
        self.assertEqual(f2.bus_k.number, 3006)
        self.assertTrue(f2.bus_m is None)

        # Facts 153/3006 - 155 (SSSC)
        self.assertEqual(f3.name, 'FACTS_DVCE_3')
        self.assertFalse(f3.is_series_link_disabled())
        self.assertFalse(f3.is_series_link_bypassed())
        self.assertTrue(f3.is_in_normal_series_mode())
        self.assertFalse(f3.is_in_constant_series_z_mode())
        self.assertFalse(f3.is_in_constant_series_v_mode())
        self.assertEqual(f3.num_periods, T)
        self.assertEqual(f3.obj_type, 'facts')
        self.assertEqual(f3.index, 2)
        self.assertEqual(f3.i_max_sh, 0.)
        self.assertEqual(f3.P_max_dc, 0.)
