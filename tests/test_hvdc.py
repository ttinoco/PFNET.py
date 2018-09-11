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
from scipy.sparse import bmat

class TestHVDC(unittest.TestCase):

    def test_ieee300_raw_case(self):

        T = 4

        case = os.path.join('data', 'ieee300.raw')
        if not os.path.isfile(case):
            raise unittest.SkipTest('file not available')

        net = pf.ParserRAW().parse(case, T)

        # Buses

        # Branches

        # Converters
        self.assertEqual(net.num_csc_converters, 2)
        
        convR = net.csc_converters[0]
        convI = net.csc_converters[1]

        self.assertTrue(convR.is_rectifier())
        self.assertFalse(convR.is_inverter())
        self.assertFalse(convI.is_rectifier())
        self.assertTrue(convI.is_inverter())

        self.assertEqual(convR.name, "1")
        self.assertEqual(convI.name, "1")
        
        ac_busR = convR.ac_bus
        dc_busR = convR.dc_bus
        self.assertEqual(ac_busR.number, 119)
        self.assertEqual(dc_busR.name, "TTDC 1 bus 0")
        self.assertEqual(len(ac_busR.csc_converters), 1)
        self.assertEqual(len(dc_busR.csc_converters), 1)
        self.assertTrue(ac_busR.csc_converters[0].is_equal(convR))
        self.assertTrue(dc_busR.csc_converters[0].is_equal(convR))
        self.assertFalse(ac_busR.csc_converters[0].is_equal(convI))
        self.assertFalse(dc_busR.csc_converters[0].is_equal(convI))

        ac_busI = convI.ac_bus
        dc_busI = convI.dc_bus
        self.assertEqual(ac_busI.number, 120)
        self.assertEqual(dc_busI.name, "TTDC 1 bus 1")
        self.assertEqual(len(ac_busI.csc_converters), 1)
        self.assertEqual(len(dc_busI.csc_converters), 1)
        self.assertTrue(ac_busI.csc_converters[0].is_equal(convI))
        self.assertTrue(dc_busI.csc_converters[0].is_equal(convI))
        self.assertFalse(ac_busI.csc_converters[0].is_equal(convR))
        self.assertFalse(dc_busI.csc_converters[0].is_equal(convR))
        
        self.assertTrue(convR.is_equal(net.get_csc_converter_from_name_and_ac_bus_number(convR.name,
                                                                                         ac_busR.number)))
        self.assertTrue(convR.is_equal(net.get_csc_converter_from_name_and_dc_bus_name(convR.name,
                                                                                       dc_busR.name)))
        self.assertTrue(convI.is_equal(net.get_csc_converter_from_name_and_ac_bus_number(convI.name,
                                                                                         ac_busI.number)))
        self.assertTrue(convI.is_equal(net.get_csc_converter_from_name_and_dc_bus_name(convI.name,
                                                                                       dc_busI.name)))

        self.assertTrue(convR.is_in_P_dc_mode())
        self.assertFalse(convR.is_in_i_dc_mode())
        self.assertFalse(convR.is_in_v_dc_mode())
        
        self.assertFalse(convI.is_in_P_dc_mode())
        self.assertFalse(convI.is_in_i_dc_mode())
        self.assertTrue(convI.is_in_v_dc_mode())
        
        for t in range(T):
            v_base = dc_busR.v_base
            idc = (dc_busR.v[t]-dc_busI.v[t])*v_base/6.2 # kA
            self.assertLess(np.abs(idc*dc_busI.v[t]*v_base-100.),1e-10)
            self.assertLess(np.abs(convR.P_dc_set[t]*net.base_power-idc*dc_busR.v[t]*v_base),1e-10)
            self.assertEqual(convR.i_dc_set[t], 0.)
            self.assertEqual(convR.v_dc_set[t], 0.)
            self.assertEqual(convI.P_dc_set[t], 0.)
            idc = -convI.i_dc_set[t]/(1.-0.1) # p.u.
            rcomp = 0.0/(dc_busI.v_base**2./net.base_power)
            vschd = (convI.v_dc_set[t]+idc*rcomp)*dc_busI.v_base
            self.assertEqual(vschd, 460.)
            self.assertLess(np.abs(convR.P[t]+dc_busR.v[t]*idc),1e-10)
            self.assertLess(np.abs(convI.P[t]-dc_busI.v[t]*idc),1e-10)
            self.assertLess(np.abs(np.tan(np.arccos(np.minimum(dc_busR.v[t],1.)))*convR.P[t]-convR.Q[t]),1e-10)
            self.assertLess(np.abs(np.tan(np.arccos(np.minimum(dc_busI.v[t],1.)))*convI.P[t]+convI.Q[t]),1e-10)

    def test_GSO_5bus_ttdc_case(self):
        
        T = 4
        
        case = os.path.join('data', 'GSO_5bus_ttdc.raw')
        if not os.path.isfile(case):
            raise unittest.SkipTest('file not available')

        net = pf.ParserRAW().parse(case, T)
        
        # Buses
        self.assertEqual(net.num_dc_buses, 2)
        self.assertEqual(net.get_num_dc_buses(), 2)
        self.assertEqual(len(net.dc_buses), 2)

        bus1 = net.dc_buses[0]
        bus2 = net.dc_buses[1]

        self.assertEqual(bus1.index, 0)
        self.assertEqual(bus2.index, 1)
        
        self.assertEqual(bus1.number, 0)
        self.assertEqual(bus2.number, 1)

        self.assertEqual(bus1.name, "TTDC LINE 1 bus 0")
        self.assertEqual(bus2.name, "TTDC LINE 1 bus 1")

        self.assertEqual(bus1.num_periods, T)
        self.assertEqual(bus2.num_periods, T)

        self.assertTrue(bus1.is_equal(net.get_dc_bus_from_number(bus1.number)))
        self.assertFalse(bus1.is_equal(net.get_dc_bus_from_number(bus2.number)))
        self.assertTrue(bus2.is_equal(net.get_dc_bus_from_number(bus2.number)))
        self.assertFalse(bus2.is_equal(net.get_dc_bus_from_number(bus1.number)))

        self.assertTrue(bus1.is_equal(net.get_dc_bus_from_name(bus1.name)))
        self.assertFalse(bus1.is_equal(net.get_dc_bus_from_name(bus2.name)))
        self.assertTrue(bus2.is_equal(net.get_dc_bus_from_name(bus2.name)))
        self.assertFalse(bus2.is_equal(net.get_dc_bus_from_name(bus1.name)))

        self.assertEqual(bus1.v_base, bus2.v_base)

        # Initial bus voltages that are consistent
        # with power order and scheduled voltage at given
        # point along the line
        rdc = 8.2 # ohms
        rcomp = 4.1 # ohms
        for t in range(T):
            idc = 1000.*(bus1.v[t]*bus1.v_base-bus2.v[t]*bus2.v_base)/rdc # amps
            vs = bus2.v[t]*bus2.v_base + idc*rcomp*1e-3 # kv
            P = bus1.v[t]*bus1.v_base*idc*1e-3
            self.assertLess(np.abs(vs-525.), 1e-10)
            self.assertLess(np.abs(P-400.), 1e-10)

        for t in range(T):
            self.assertEqual(bus1.index_v[t], 0)
            self.assertEqual(bus2.index_v[t], 0)

        # Branches
        self.assertEqual(net.num_dc_branches, 1)
        self.assertEqual(net.get_num_dc_branches(), 1)
        self.assertEqual(len(net.dc_branches), 1)

        branch = net.dc_branches[0]

        self.assertEqual(branch.name, "TTDC LINE 1")
        self.assertEqual(branch.num_periods, T)

        busk = branch.bus_k
        busm = branch.bus_m

        self.assertTrue(busk.is_equal(bus1))
        self.assertTrue(busm.is_equal(bus2))
        self.assertEqual(busk.name, "TTDC LINE 1 bus 0")
        self.assertEqual(busm.name, "TTDC LINE 1 bus 1")
        
        self.assertTrue(branch.is_equal(net.get_dc_branch_from_name_and_dc_bus_names(branch.name,
                                                                                     busk.name,
                                                                                     busm.name)))
        

        self.assertEqual(len(busk.branches), 1)
        self.assertEqual(len(busk.branches_k), 1)
        self.assertEqual(len(busk.branches_m), 0)

        self.assertEqual(len(busm.branches), 1)
        self.assertEqual(len(busm.branches_k), 0)
        self.assertEqual(len(busm.branches_m), 1)
        
        self.assertTrue(busk.branches[0].is_equal(branch))
        self.assertTrue(busk.branches_k[0].is_equal(branch))
        self.assertTrue(busm.branches[0].is_equal(branch))
        self.assertTrue(busm.branches_m[0].is_equal(branch))

        r_base = (busk.v_base**2.)/net.base_power
        self.assertEqual(branch.r*r_base,8.2)
        
        # CSC converters
        self.assertEqual(net.num_csc_converters, 2)
        self.assertEqual(net.get_num_csc_converters(), 2)
        self.assertEqual(len(net.csc_converters), 2)

        convR = net.csc_converters[0]
        convI = net.csc_converters[1]

        self.assertTrue(convR.is_rectifier())
        self.assertFalse(convR.is_inverter())
        self.assertFalse(convI.is_rectifier())
        self.assertTrue(convI.is_inverter())

        self.assertEqual(convR.name, "LINE 1")
        self.assertEqual(convI.name, "LINE 1")

        self.assertEqual(convR.num_periods, T)
        self.assertEqual(convI.num_periods, T)

        ac_busR = convR.ac_bus
        dc_busR = convR.dc_bus
        self.assertEqual(ac_busR.number, 5)
        self.assertEqual(dc_busR.name, "TTDC LINE 1 bus 0")
        self.assertEqual(len(ac_busR.csc_converters), 1)
        self.assertEqual(len(dc_busR.csc_converters), 1)
        self.assertTrue(ac_busR.csc_converters[0].is_equal(convR))
        self.assertTrue(dc_busR.csc_converters[0].is_equal(convR))
        self.assertFalse(ac_busR.csc_converters[0].is_equal(convI))
        self.assertFalse(dc_busR.csc_converters[0].is_equal(convI))

        ac_busI = convI.ac_bus
        dc_busI = convI.dc_bus
        self.assertEqual(ac_busI.number, 2)
        self.assertEqual(dc_busI.name, "TTDC LINE 1 bus 1")
        self.assertEqual(len(ac_busI.csc_converters), 1)
        self.assertEqual(len(dc_busI.csc_converters), 1)
        self.assertTrue(ac_busI.csc_converters[0].is_equal(convI))
        self.assertTrue(dc_busI.csc_converters[0].is_equal(convI))
        self.assertFalse(ac_busI.csc_converters[0].is_equal(convR))
        self.assertFalse(dc_busI.csc_converters[0].is_equal(convR))
        
        self.assertTrue(convR.is_equal(net.get_csc_converter_from_name_and_ac_bus_number(convR.name,
                                                                                         ac_busR.number)))
        self.assertTrue(convR.is_equal(net.get_csc_converter_from_name_and_dc_bus_name(convR.name,
                                                                                       dc_busR.name)))
        self.assertTrue(convI.is_equal(net.get_csc_converter_from_name_and_ac_bus_number(convI.name,
                                                                                         ac_busI.number)))
        self.assertTrue(convI.is_equal(net.get_csc_converter_from_name_and_dc_bus_name(convI.name,
                                                                                       dc_busI.name)))

        self.assertTrue(convR.is_in_P_dc_mode())
        self.assertFalse(convR.is_in_i_dc_mode())
        self.assertFalse(convR.is_in_v_dc_mode())
        
        self.assertFalse(convI.is_in_P_dc_mode())
        self.assertFalse(convI.is_in_i_dc_mode())
        self.assertTrue(convI.is_in_v_dc_mode())

        for t in range(T):
            self.assertEqual(convR.P_dc_set[t]*net.base_power, 400.)
            self.assertEqual(convR.i_dc_set[t], 0.)
            self.assertEqual(convR.v_dc_set[t], 0.)
            self.assertEqual(convI.P_dc_set[t], 0.)
            idc = -convI.i_dc_set[t]/(1.-0.15) # p.u.
            rcomp = 4.1/(dc_busI.v_base**2./net.base_power)
            vschd = (convI.v_dc_set[t]+idc*rcomp)*dc_busI.v_base
            self.assertEqual(vschd,525)
            self.assertLess(np.abs(convR.P[t]+dc_busR.v[t]*idc),1e-10)
            self.assertLess(np.abs(convI.P[t]-dc_busI.v[t]*idc),1e-10)
            self.assertLess(np.abs(np.tan(np.arccos(np.minimum(dc_busR.v[t],1.)))*convR.P[t]-convR.Q[t]),1e-10)
            self.assertLess(np.abs(np.tan(np.arccos(np.minimum(dc_busI.v[t],1.)))*convI.P[t]+convI.Q[t]),1e-10)

            self.assertLess(np.abs(convR.angle[t]*180./np.pi-8.),1e-10)
            self.assertLess(np.abs(convI.angle[t]*180./np.pi-16.),1e-10)

            self.assertLess(np.abs(convR.ratio[t]-(1./1.06480)*0.44000*ac_busR.v_base/dc_busR.v_base),1e-10)
            self.assertLess(np.abs(convI.ratio[t]-(1./0.89375)*0.95650*ac_busI.v_base/dc_busI.v_base),1e-10)

            self.assertEqual(convR.P_dc[t],-convR.P[t])
            self.assertEqual(convI.P_dc[t],-convI.P[t])

        self.assertEqual(convR.num_bridges, 2)
        self.assertEqual(convI.num_bridges, 2)

        self.assertLess(np.abs(convR.x*(dc_busR.v_base**2.)/net.base_power - 3.8800),1e-10)
        self.assertLess(np.abs(convR.r*(dc_busR.v_base**2.)/net.base_power - 0.0110),1e-10)
        self.assertLess(np.abs(convI.x*(dc_busI.v_base**2.)/net.base_power - 3.0470),1e-10)
        self.assertLess(np.abs(convI.r*(dc_busI.v_base**2.)/net.base_power - 0.0120),1e-10)

        self.assertLess(np.abs(convR.x_cap*(dc_busR.v_base**2.)/net.base_power - 0.0098),1e-10)
        self.assertLess(np.abs(convI.x_cap*(dc_busI.v_base**2.)/net.base_power - 0.0074),1e-10)
        
        self.assertLess(np.abs(convR.angle_min*180./np.pi-8.),1e-10)
        self.assertLess(np.abs(convI.angle_min*180./np.pi-16.),1e-10)

        self.assertLess(np.abs(convR.angle_max*180./np.pi-12.),1e-10)
        self.assertLess(np.abs(convI.angle_max*180./np.pi-20.),1e-10)

        self.assertEqual(convR.v_base_p, 500.)
        self.assertEqual(convI.v_base_p, 230)
        self.assertLess(np.abs(convR.v_base_s-220.), 1e-2)
        self.assertLess(np.abs(convI.v_base_s-220.), 1e-2)
        
    def test_GSO_5bus_vscdc_case(self):
        
        T = 4

        case = os.path.join('data', 'GSO_5bus_vscdc.raw')
        if not os.path.isfile(case):
            raise unittest.SkipTest('file not available')

        net = pf.ParserRAW().parse(case, T)

        # Buses
        self.assertEqual(net.num_dc_buses, 2)
        self.assertEqual(net.get_num_dc_buses(), 2)
        self.assertEqual(len(net.dc_buses), 2)

        bus1 = net.dc_buses[0]
        bus2 = net.dc_buses[1]

        self.assertEqual(bus1.index, 0)
        self.assertEqual(bus2.index, 1)
        
        self.assertEqual(bus1.number, 0)
        self.assertEqual(bus2.number, 1)

        self.assertEqual(bus1.name, "VSCDC LINE 1 bus 0")
        self.assertEqual(bus2.name, "VSCDC LINE 1 bus 1")

        self.assertEqual(bus1.num_periods, T)
        self.assertEqual(bus2.num_periods, T)

        self.assertTrue(bus1.is_equal(net.get_dc_bus_from_number(bus1.number)))
        self.assertFalse(bus1.is_equal(net.get_dc_bus_from_number(bus2.number)))
        self.assertTrue(bus2.is_equal(net.get_dc_bus_from_number(bus2.number)))
        self.assertFalse(bus2.is_equal(net.get_dc_bus_from_number(bus1.number)))

        self.assertTrue(bus1.is_equal(net.get_dc_bus_from_name(bus1.name)))
        self.assertFalse(bus1.is_equal(net.get_dc_bus_from_name(bus2.name)))
        self.assertTrue(bus2.is_equal(net.get_dc_bus_from_name(bus2.name)))
        self.assertFalse(bus2.is_equal(net.get_dc_bus_from_name(bus1.name)))

        self.assertEqual(bus1.v_base, bus2.v_base)

        rdc = 0.7100 # ohms
        for t in range(T):
            self.assertEqual(bus2.v_base, 100.)
            self.assertEqual(bus1.v_base, 100.)
            self.assertEqual(bus2.v[t], 1.)
            i10 = (bus2.v[t]-bus1.v[t])*bus1.v_base/rdc # kamps
            P0 = bus1.v[t]*bus1.v_base*i10 # MW
            self.assertLess(np.abs(P0-(-200.)), 1e-10)
            
        for t in range(T):
            self.assertEqual(bus1.index_v[t], 0)
            self.assertEqual(bus2.index_v[t], 0)

        # Branches
        self.assertEqual(net.num_dc_branches, 1)
        self.assertEqual(net.get_num_dc_branches(), 1)
        self.assertEqual(len(net.dc_branches), 1)

        branch = net.dc_branches[0]

        self.assertEqual(branch.name, "VSCDC LINE 1")
        self.assertEqual(branch.num_periods, T)

        busk = branch.bus_k
        busm = branch.bus_m

        self.assertTrue(busk.is_equal(bus1))
        self.assertTrue(busm.is_equal(bus2))
        self.assertEqual(busk.name, "VSCDC LINE 1 bus 0")
        self.assertEqual(busm.name, "VSCDC LINE 1 bus 1")

        self.assertTrue(branch.is_equal(net.get_dc_branch_from_name_and_dc_bus_names(branch.name,
                                                                                     busk.name,
                                                                                     busm.name)))

        self.assertEqual(len(busk.branches), 1)
        self.assertEqual(len(busk.branches_k), 1)
        self.assertEqual(len(busk.branches_m), 0)

        self.assertEqual(len(busm.branches), 1)
        self.assertEqual(len(busm.branches_k), 0)
        self.assertEqual(len(busm.branches_m), 1)
        
        self.assertTrue(busk.branches[0].is_equal(branch))
        self.assertTrue(busk.branches_k[0].is_equal(branch))
        self.assertTrue(busm.branches[0].is_equal(branch))
        self.assertTrue(busm.branches_m[0].is_equal(branch))

        r_base = (busk.v_base**2.)/net.base_power
        self.assertEqual(branch.r*r_base,0.7100)

        # VSC converters
        self.assertEqual(net.num_vsc_converters, 2)
        self.assertEqual(net.get_num_vsc_converters(), 2)
        self.assertEqual(len(net.vsc_converters), 2)

        conv0 = net.vsc_converters[0]
        conv1 = net.vsc_converters[1]

        self.assertEqual(conv0.name, "LINE 1")
        self.assertEqual(conv1.name, "LINE 1")

        self.assertEqual(conv0.num_periods, T)
        self.assertEqual(conv1.num_periods, T)
        
        ac_bus0 = conv0.ac_bus
        dc_bus0 = conv0.dc_bus
        self.assertEqual(ac_bus0.number, 4)
        self.assertEqual(dc_bus0.name, "VSCDC LINE 1 bus 0")
        self.assertEqual(len(ac_bus0.vsc_converters), 1)
        self.assertEqual(len(dc_bus0.vsc_converters), 1)
        self.assertTrue(ac_bus0.vsc_converters[0].is_equal(conv0))
        self.assertTrue(dc_bus0.vsc_converters[0].is_equal(conv0))
        self.assertFalse(ac_bus0.vsc_converters[0].is_equal(conv1))
        self.assertFalse(dc_bus0.vsc_converters[0].is_equal(conv1))

        ac_bus1 = conv1.ac_bus
        dc_bus1 = conv1.dc_bus
        self.assertEqual(ac_bus1.number, 5)
        self.assertEqual(dc_bus1.name, "VSCDC LINE 1 bus 1")
        self.assertEqual(len(ac_bus1.vsc_converters), 1)
        self.assertEqual(len(dc_bus1.vsc_converters), 1)
        self.assertTrue(ac_bus1.vsc_converters[0].is_equal(conv1))
        self.assertTrue(dc_bus1.vsc_converters[0].is_equal(conv1))
        self.assertFalse(ac_bus1.vsc_converters[0].is_equal(conv0))
        self.assertFalse(dc_bus1.vsc_converters[0].is_equal(conv0))


        self.assertTrue(conv0.is_equal(net.get_vsc_converter_from_name_and_ac_bus_number(conv0.name,
                                                                                         ac_bus0.number)))
        self.assertTrue(conv0.is_equal(net.get_vsc_converter_from_name_and_dc_bus_name(conv0.name,
                                                                                       dc_bus0.name)))
        self.assertTrue(conv1.is_equal(net.get_vsc_converter_from_name_and_ac_bus_number(conv1.name,
                                                                                         ac_bus1.number)))
        self.assertTrue(conv1.is_equal(net.get_vsc_converter_from_name_and_dc_bus_name(conv1.name,
                                                                                       dc_bus1.name)))

        self.assertTrue(conv0.is_in_P_dc_mode())
        self.assertFalse(conv0.is_in_v_dc_mode())
        self.assertFalse(conv0.is_in_f_ac_mode())
        self.assertTrue(conv0.is_in_v_ac_mode())

        self.assertEqual(conv0.target_power_factor, 1.)
        self.assertEqual(conv0.Q_par, 1.)
        self.assertEqual(conv0.reg_bus.number, ac_bus0.number)
        
        self.assertFalse(conv1.is_in_P_dc_mode())
        self.assertTrue(conv1.is_in_v_dc_mode())
        self.assertFalse(conv1.is_in_f_ac_mode())
        self.assertTrue(conv1.is_in_v_ac_mode())

        self.assertEqual(conv1.target_power_factor, 1.)
        self.assertEqual(conv1.Q_par, 1.)
        self.assertEqual(conv1.reg_bus.number, ac_bus1.number)
        self.assertTrue(conv1.reg_bus.is_equal(ac_bus1))
        self.assertTrue(ac_bus1.is_regulated_by_vsc_converter())
        self.assertEqual(len(ac_bus1.reg_vsc_converters),1)
        self.assertTrue(conv1.is_equal(ac_bus1.reg_vsc_converters[0]))
        self.assertEqual(net.get_num_buses_reg_by_vsc_converter(), 2)
        self.assertFalse(ac_bus0.is_regulated_by_gen())
        self.assertFalse(ac_bus1.is_regulated_by_gen())
        
        self.assertEqual(net.num_vars, 0)
        net.set_flags('bus',
                      'variable',
                      'v set regulated',
                      'voltage magnitude')        
        self.assertEqual(net.num_vars, (net.get_num_buses_reg_by_gen()+2)*T)
        self.assertTrue(ac_bus1.has_flags('variable', 'voltage magnitude'))

        for t in range(T):

            self.assertEqual(ac_bus0.v_set[0], 1.00000)
            self.assertEqual(ac_bus1.v_set[t], 1.00001)
            
            self.assertEqual(conv0.P_dc_set[t]*net.base_power, 200.) # injection into DC grid
            self.assertEqual(conv0.v_dc_set[t], 0.)
            self.assertEqual(conv1.P_dc_set[t], 0.)
            self.assertEqual(conv1.v_dc_set[t]*conv1.dc_bus.v_base, 100.)

            self.assertEqual(conv0.P_dc[t], -conv0.P[t])
            self.assertEqual(conv1.P_dc[t], -conv1.P[t])

            idc = conv0.P_dc_set[t]/conv0.dc_bus.v[t]
            
            self.assertLess(np.abs(conv0.P[t]-(-idc*conv0.dc_bus.v[t])),1e-10)
            self.assertEqual(conv0.Q[t], 0.)
            self.assertLess(np.abs(conv1.P[t]-(idc*conv1.dc_bus.v[t])),1e-10)
            self.assertEqual(conv1.Q[t], 0.)

        self.assertEqual(conv0.Q_max*net.base_power, 100.)
        self.assertEqual(conv0.Q_min*net.base_power, -100.)
        self.assertEqual(conv1.Q_max*net.base_power, 350.)
        self.assertEqual(conv1.Q_min*net.base_power, -140.)
        self.assertEqual(conv0.P_max*net.base_power, 120.)
        self.assertEqual(conv0.P_min*net.base_power, -120.)
        self.assertEqual(conv1.P_max*net.base_power, 450.)
        self.assertEqual(conv1.P_min*net.base_power, -450.)
        
        
        ibase = net.base_power/dc_bus0.v_base
        
        self.assertEqual(conv0.loss_coeff_A*net.base_power, 100.*1e-3)
        self.assertEqual(conv1.loss_coeff_A*net.base_power, 90.*1e-3)
        self.assertEqual(conv0.loss_coeff_B*net.base_power/ibase, 0.100)
        self.assertEqual(conv1.loss_coeff_B*net.base_power/ibase, 0.150)

    def test_ieee25_vsc_case(self):
        
        T = 4
        
        case = os.path.join('data', 'ieee25_vsc.raw')
        if not os.path.isfile(case):
            raise unittest.SkipTest('file not available')
        
        net = pf.ParserRAW().parse(case, T)
        
        self.assertEqual(net.num_vsc_converters, 2)
        self.assertEqual(net.get_num_vsc_converters_in_P_dc_mode(), 1)
        self.assertEqual(net.get_num_vsc_converters_in_v_dc_mode(), 1)
        self.assertEqual(net.get_num_vsc_converters_in_v_ac_mode(), 2)
        self.assertEqual(net.get_num_vsc_converters_in_f_ac_mode(), 0)
        
        # Total number of converters should be equal to total of DC modes
        self.assertEqual(net.num_vsc_converters, (net.get_num_vsc_converters_in_P_dc_mode() +
                                                  net.get_num_vsc_converters_in_v_dc_mode()))
        
        # Total number of converters should be equal to total of AC modes
        self.assertEqual(net.num_vsc_converters, (net.get_num_vsc_converters_in_v_ac_mode()+
                                                  net.get_num_vsc_converters_in_f_ac_mode()))

    def test_ieee25_vsc_nr(self):
                
        case = os.path.join('data', 'ieee25_vsc.raw')
        if not os.path.isfile(case):
            raise unittest.SkipTest('file not available')
        
        net = pf.ParserRAW().parse(case)
        
        # Voltages
        net.set_flags('bus',
                      'variable',
                      'not slack',
                      ['voltage magnitude','voltage angle'])
        
        # Gen active powers
        net.set_flags('generator',
                      'variable',
                      'slack',
                      'active power')
        
        # Gen reactive powers
        net.set_flags('generator',
                      'variable',
                      'slack',
                      'reactive power')

        # Set up problem
        problem = pf.Problem(net)
        problem.add_constraint(pf.Constraint('AC power balance', net))
        problem.add_constraint(pf.Constraint('generator active power participation', net))
        problem.add_constraint(pf.Constraint('PVPQ switching', net))
        problem.analyze()
        problem.eval(problem.get_init_point())

        # NR matrix - no HVDC
        M = bmat([[problem.A], [problem.J]])
        M = M.todense()
        self.assertEqual(np.linalg.matrix_rank(M), M.shape[0])

        # DC buses
        net.set_flags('dc bus',
                      'variable',
                      'any',
                      'voltage')

        problem.add_constraint(pf.Constraint('HVDC power balance', net))
        problem.analyze()
        problem.eval(problem.get_init_point())

        # NR matrix - DC buses and HVDC power balance
        M = bmat([[problem.A], [problem.J]])
        M = M.todense()
        self.assertEqual(np.linalg.matrix_rank(M), M.shape[0]-1)

        # VSC
        net.set_flags('vsc converter',
                      'variable',
                      'any',
                      ['active power', 'reactive power', 'dc power'])

        problem.add_constraint(pf.Constraint('VSC DC power control', net))
        problem.add_constraint(pf.Constraint('VSC DC voltage control', net))
        problem.add_constraint(pf.Constraint('VSC converter equations', net))
        problem.analyze()
        problem.eval(problem.get_init_point())
        
        # NR matrix - VSC
        M = bmat([[problem.A], [problem.J]])
        M = M.todense()
        self.assertEqual(np.linalg.matrix_rank(M), M.shape[0])

        c = problem.find_constraint('VSC DC voltage control')
        Ac = c.A.copy()
        bc = c.b.copy()
        A = problem.A.copy()
        J = problem.J.copy()

        # PVPQ Switching heuristic
        problem.add_heuristic(pf.Heuristic('PVPQ switching', net))
        problem.apply_heuristics(problem.get_init_point())
        problem.analyze()
        problem.eval(problem.get_init_point())

        # Check before/after
        self.assertEqual((Ac-c.A).nnz, 0)
        self.assertEqual((J-problem.J).nnz, 0)
        self.assertEqual((A-problem.A).nnz, 0)
        
        # NR matrix - VSC after PVPQ
        M = bmat([[problem.A], [problem.J]])
        M = M.todense()
        self.assertEqual(np.linalg.matrix_rank(M), M.shape[0])
