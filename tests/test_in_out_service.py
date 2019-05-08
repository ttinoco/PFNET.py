#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

import json
import unittest
import numpy as np
import pfnet as pf
from . import test_cases

class TestInOutService(unittest.TestCase):

    def test_state_tag(self):

        for case in test_cases.CASES:
            
            net = pf.Parser(case).parse(case)

            self.assertEqual(net.state_tag, 0)

            for gen in net.generators:
                gen.in_service = False
            for branch in net.branches:
                branch.in_service = False

            self.assertEqual(net.state_tag, net.num_generators+net.num_branches)

    def test_generators(self):

        for case in test_cases.CASES:
            
            net = pf.Parser(case).parse(case)

            net.make_all_in_service()
            
            for gen in net.generators:

                self.assertTrue(gen.is_in_service())
                self.assertTrue(gen.in_service)
                
                reg = gen.is_regulator()
                slack = gen.is_slack()

                # Out of service
                gen.in_service = False

                self.assertFalse(gen.is_in_service())
                self.assertFalse(gen.in_service)

                # bus
                self.assertTrue(gen.bus is not None)
                self.assertTrue(gen.index in [g.index for g in gen.bus.generators])

                # regulation
                if reg:
                    self.assertTrue(gen.is_regulator())
                    self.assertTrue(gen.reg_bus is not None)
                    self.assertTrue(gen.index in [g.index for g in gen.reg_bus.reg_generators])

                    if all([not g.is_in_service() for g in gen.reg_bus.reg_generators]):
                        self.assertFalse(gen.reg_bus.is_regulated_by_gen())
                    else:
                        self.assertTrue(gen.reg_bus.is_regulated_by_gen())

                # slack
                if slack:
                    self.assertTrue(gen.is_slack())
                    self.assertTrue(gen.bus.is_slack())

                # clear
                gen.in_service = True
                self.assertTrue(gen.is_in_service())

                # out of service
                gen.in_service = False
            
                # json
                json_string = gen.json_string
                d = json.loads(json_string)
                self.assertTrue('in_service' in d)
                self.assertFalse(d['in_service'])

            # copy
            new_net = net.get_copy()
            for new_gen in new_net.generators:
                self.assertFalse(new_gen.is_in_service())
                
    def test_branches(self):

        for case in test_cases.CASES:
            
            net = pf.Parser(case).parse(case)

            net.make_all_in_service()

            for branch in net.branches:

                self.assertTrue(branch.is_in_service())
                self.assertTrue(branch.in_service)

                reg = branch.is_tap_changer_v()
                
                # out of service
                branch.in_service = False
                self.assertFalse(branch.is_in_service())
                self.assertFalse(branch.in_service)

                # buses
                self.assertTrue(branch.bus_k is not None)
                self.assertTrue(branch.bus_m is not None)
                self.assertTrue(branch.index in [br.index for br in branch.bus_k.branches_k])
                self.assertTrue(branch.index in [br.index for br in branch.bus_m.branches_m])

                # regulation
                if reg:
                    self.assertTrue(branch.is_tap_changer_v())
                    self.assertTrue(branch.reg_bus is not None)
                    self.assertTrue(branch.index in [br.index for br in branch.reg_bus.reg_trans])

                    if all([not br.is_in_service() for br in branch.reg_bus.reg_trans]):
                        self.assertFalse(branch.reg_bus.is_regulated_by_tran())
                    else:
                        self.assertTrue(branch.reg_bus.is_regulated_by_tran())

                # clear
                branch.in_service = True
                self.assertTrue(branch.is_in_service())
                self.assertTrue(branch.in_service)
                
                # out of service
                branch.in_service = False
                
                # json
                json_string = branch.json_string
                d = json.loads(json_string)
                self.assertTrue('in_service' in d)
                self.assertFalse(d['in_service'])
                
            # copy  
            new_net = net.get_copy()
            for new_branch in new_net.branches:
                self.assertFalse(new_branch.is_in_service())
                
    def test_buses(self):

        for case in test_cases.CASES:
            
            net = pf.Parser(case).parse(case)

            for bus in net.buses:

                net.make_all_in_service()

                # total gen P
                total = bus.get_total_gen_P()
                for gen in bus.generators:
                    gen.in_service = False
                    new_total = bus.get_total_gen_P()
                    self.assertLess(np.abs(new_total-(total-gen.P)),1e-8)
                    total = new_total
                    
                net.make_all_in_service()
                
                # total gen Q
                total = bus.get_total_gen_Q()
                for gen in bus.generators:
                    gen.in_service = False
                    new_total = bus.get_total_gen_Q()
                    self.assertLess(np.abs(new_total-(total-gen.Q)),1e-8)
                    total = new_total

                net.make_all_in_service()
                
                # total gen Qmin
                total = bus.get_total_gen_Q_min()
                for gen in bus.generators:
                    gen.in_service = False
                    new_total = bus.get_total_gen_Q_min()
                    self.assertLess(np.abs(new_total-(total-gen.Q_min)),1e-8)
                    total = new_total

                net.make_all_in_service()
                
                # total gen Qmax
                total = bus.get_total_gen_Q_max()
                for gen in bus.generators:
                    gen.in_service = False
                    new_total = bus.get_total_gen_Q_max()
                    self.assertLess(np.abs(new_total-(total-gen.Q_max)),1e-8)
                    total = new_total

                net.make_all_in_service()

                # toatl reg gen Q
                total = bus.get_total_reg_gen_Q()
                self.assertLess(np.abs(total-sum([g.Q for g in bus.reg_generators])), 1e-8)
                for gen in bus.reg_generators:
                    gen.in_service = False
                    new_total = bus.get_total_reg_gen_Q()
                    self.assertLess(np.abs(new_total-(total-gen.Q)),1e-8)
                    total = new_total

                net.make_all_in_service()
                
                # total reg gen Qmin
                total = bus.get_total_reg_gen_Q_min()
                self.assertLess(np.abs(total-sum([g.Q_min for g in bus.reg_generators])), 1e-8)
                for gen in bus.reg_generators:
                    gen.in_service = False
                    new_total = bus.get_total_reg_gen_Q_min()
                    self.assertLess(np.abs(new_total-(total-gen.Q_min)),1e-8)
                    total = new_total

                net.make_all_in_service()
                
                # total reg gen Qmax
                total = bus.get_total_reg_gen_Q_max()
                self.assertLess(np.abs(total-sum([g.Q_max for g in bus.reg_generators])), 1e-8)
                for gen in bus.reg_generators:
                    gen.in_service = False
                    new_total = bus.get_total_reg_gen_Q_max()
                    self.assertLess(np.abs(new_total-(total-gen.Q_max)),1e-8)
                    total = new_total

                net.make_all_in_service()
                
                # reg by gen
                if bus.is_regulated_by_gen():
                    for i in range(len(bus.reg_generators)):
                        gen = bus.reg_generators[i]
                        gen.in_service = False
                        if i < len(bus.reg_generators)-1:
                            self.assertTrue(bus.is_regulated_by_gen())
                        else:
                            self.assertFalse(bus.is_regulated_by_gen())

                net.make_all_in_service()
                            
                # reg by tran
                if bus.is_regulated_by_tran():
                    for i in range(len(bus.reg_trans)):
                        br = bus.reg_trans[i]
                        br.in_service = False
                        if i < len(bus.reg_trans)-1:
                            self.assertTrue(bus.is_regulated_by_tran())
                        else:
                            self.assertFalse(bus.is_regulated_by_tran())

                net.make_all_in_service()
                            
                # slack
                if bus.is_slack():
                    for gen in bus.generators:
                        gen.in_service = False
                        self.assertFalse(gen.is_in_service())
                        self.assertTrue(gen.is_slack())
                        self.assertTrue(bus.is_slack())
    
    def test_network(self):

        for case in test_cases.CASES:
            
            net = pf.Parser(case).parse(case)

            # clear outages
            net.make_all_in_service()

            self.assertEqual(net.get_num_branches(only_in_service=True), net.num_branches)
            self.assertEqual(net.get_num_branches_out_of_service(), 0)
            self.assertEqual(net.get_num_generators(True), net.num_generators)
            self.assertEqual(net.get_num_generators_out_of_service(), 0)
            
            # num branches on outage
            for branch in net.branches:
                branch.in_service = False
            self.assertEqual(net.get_num_branches(True), 0)
            self.assertEqual(net.get_num_branches_out_of_service(), net.num_branches)
            
            # num gens on outage
            for gen in net.generators:
                gen.in_service = False
            self.assertEqual(net.get_num_generators(True), 0)
            self.assertEqual(net.get_num_generators_out_of_service(), net.num_generators)

    def test_network_properties(self):

        for case in test_cases.CASES:
            
            net = pf.Parser(case).parse(case)
            
            # tap ratio vio
            net.make_all_in_service()
            for branch in net.branches:
                if branch.is_tap_changer():
                    branch.ratio = branch.ratio_max + 10.
                    net.update_properties()
                    self.assertEqual(net.tran_r_vio, 10.)
                    branch.in_service = False
                    net.update_properties()
                    self.assertNotEqual(net.tran_r_vio, 10.)
                    break
            
            # phase shift vio
            net.make_all_in_service()
            for branch in net.branches:
                if branch.is_phase_shifter():
                    branch.phase = branch.phase_max + 20.
                    net.update_properties()
                    self.assertEqual(net.tran_p_vio, 20.)
                    branch.in_service = False
                    net.update_properties()
                    self.assertNotEqual(net.tran_p_vio, 20.)
                    break
            
            # mismatches (branch and gen outages)
            net.make_all_in_service()
            net.update_properties()
            bus = net.get_bus(0)
            p_mis = bus.P_mismatch
            q_mis = bus.Q_mismatch
            for branch in bus.branches:
                branch.in_service = False
            for gen in bus.generators:
                gen.in_service = False
            net.update_properties()
            self.assertLess(np.abs(p_mis-(bus.P_mismatch + 
                                          sum([g.P for g in bus.generators]) -
                                          sum([br.P_km for br in bus.branches_k]) -
                                          sum([br.P_mk for br in bus.branches_m]))), 1e-8)
            self.assertLess(np.abs(q_mis-(bus.Q_mismatch +
                                          sum([g.Q for g in bus.generators]) -
                                          sum([br.Q_km for br in bus.branches_k]) -
                                          sum([br.Q_mk for br in bus.branches_m]))), 1e-8)
            
            # v reg limit violations
            net.make_all_in_service()
            for bus in net.buses:
                if bus.is_regulated_by_tran():
                    net.update_properties()
                    bus.v_mag = bus.v_max_reg + 15.
                    net.update_properties()
                    self.assertLess(np.abs(net.tran_v_vio-15.), 1e-8)
                    for branch in bus.reg_trans:
                        branch.in_service = False
                    self.assertFalse(bus.is_regulated_by_tran())
                    net.update_properties()
                    self.assertGreater(np.abs(net.tran_v_vio-15.), 1e-8)
                    break
                            
            # v set deviations
            net.make_all_in_service()
            for bus in net.buses:
                if bus.is_regulated_by_gen():
                    bus.v_mag = bus.v_set + 33.
                    net.update_properties()
                    self.assertLess(np.abs(net.gen_v_dev-33.), 1e-8)
                    for gen in bus.reg_generators:
                        gen.in_service = False
                    self.assertFalse(bus.is_regulated_by_gen())
                    net.update_properties()
                    self.assertGreater(np.abs(net.gen_v_dev-33.), 1e-8)
                    break
            
            # gen active power cost
            net.make_all_in_service()
            net.update_properties()
            cost = net.gen_P_cost
            for gen in net.generators:
                gen.in_service = False
                net.update_properties()
                cost -= gen.cost_coeff_Q0 + gen.cost_coeff_Q1*gen.P + gen.cost_coeff_Q2*(gen.P**2.)
                self.assertLess(np.abs(cost-net.gen_P_cost), 1e-8*(1.+np.abs(cost)))
            
            # gen Q vio
            net.make_all_in_service()
            for gen in net.generators:
                if gen.is_regulator():
                    gen.Q = gen.Q_max + 340.
                    net.update_properties()
                    self.assertLess(np.abs(net.gen_Q_vio-340.*net.base_power), 1e-8)
                    gen.in_service = False
                    net.update_properties()
                    self.assertGreater(np.abs(net.gen_Q_vio-340.*net.base_power), 1e-8)
                    break
            
            # gen P vio
            net.make_all_in_service()
            for gen in net.generators:
                gen.P = gen.P_min - 540.
                net.update_properties()
                self.assertLess(np.abs(net.gen_P_vio-540.*net.base_power), 1e-8)
                gen.in_service = False
                net.update_properties()
                self.assertGreater(np.abs(net.gen_P_vio-540.*net.base_power), 1e-8)
                break
