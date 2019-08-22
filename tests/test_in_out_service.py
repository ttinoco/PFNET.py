#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

import os
import json
import unittest
import numpy as np
import pfnet as pf
from . import test_cases

class TestInOutService(unittest.TestCase):

    def test_bus_effects(self):

        for case in test_cases.CASES:
            
            net = pf.Parser(case).parse(case)
            
            net.add_var_generators_from_parameters(net.get_load_buses(),100.,50.,30.,5,0.05)
            net.add_batteries_from_parameters(net.get_generator_buses(),20.,50.)
            
            net.make_all_in_service()

            for bus in net.buses:
                self.assertTrue(bus.in_service)
            for bus in net.dc_buses:
                self.assertTrue(bus.in_service)
            for gen in net.generators:
                self.assertTrue(gen.in_service)
            for branch in net.branches:
                self.assertTrue(branch.in_service)
            for facts in net.facts:
                self.assertTrue(facts.in_service)
            for conv in net.csc_converters:
                self.assertTrue(conv.in_service)
            for conv in net.vsc_converters:
                self.assertTrue(conv.in_service)
            for load in net.loads:
                self.assertTrue(load.in_service)
            for branch in net.dc_branches:
                self.assertTrue(branch.in_service)
            for shunt in net.shunts:
                self.assertTrue(shunt.in_service)
            for bat in net.batteries:
                self.assertTrue(bat.in_service)
            for gen in net.var_generators:
                self.assertTrue(gen.in_service)

            for bus in net.buses:
                bus.in_service = False

            for bus in net.buses:
                self.assertFalse(bus.in_service)
            for bus in net.dc_buses:
                self.assertTrue(bus.in_service)
            for gen in net.generators:
                self.assertFalse(gen.in_service)
            for branch in net.branches:
                self.assertFalse(branch.in_service)
            for facts in net.facts:
                self.assertFalse(facts.in_service)
            for conv in net.csc_converters:
                self.assertFalse(conv.in_service)
            for conv in net.vsc_converters:
                self.assertFalse(conv.in_service)
            for load in net.loads:
                self.assertFalse(load.in_service)
            for branch in net.dc_branches:
                self.assertTrue(branch.in_service)
            for shunt in net.shunts:
                self.assertFalse(shunt.in_service)
            for bat in net.batteries:
                self.assertFalse(bat.in_service)
            for gen in net.var_generators:
                self.assertFalse(gen.in_service)

            # Changes have no effect on comps
            # connected to out of service buses
            for gen in net.generators:
                gen.in_service = False
            for branch in net.branches:
                branch.in_service = False
            for load in net.loads:
                load.in_service = False
            for conv in net.csc_converters:
                conv.in_service = False
            for conv in net.vsc_converters:
                conv.in_service = False
            for facts in net.facts:
                facts.in_service = False
            for bat in net.batteries:
                bat.in_service = False
            for gen in net.var_generators:
                gen.in_service = False
            for shunt in net.shunts:
                shunt.in_service = False

            for bus in net.buses:
                bus.in_service = True

            # Previous state remembered
            for bus in net.buses:
                self.assertTrue(bus.in_service)
            for bus in net.dc_buses:
                self.assertTrue(bus.in_service)
            for gen in net.generators:
                self.assertTrue(gen.in_service)
            for branch in net.branches:
                self.assertTrue(branch.in_service)
            for facts in net.facts:
                self.assertTrue(facts.in_service)
            for conv in net.csc_converters:
                self.assertTrue(conv.in_service)
            for conv in net.vsc_converters:
                self.assertTrue(conv.in_service)
            for load in net.loads:
                self.assertTrue(load.in_service)
            for branch in net.dc_branches:
                self.assertTrue(branch.in_service)
            for shunt in net.shunts:
                self.assertTrue(shunt.in_service)
            for bat in net.batteries:
                self.assertTrue(bat.in_service)
            for gen in net.var_generators:
                self.assertTrue(gen.in_service)

    def test_dc_bus_effects(self):

        for case in test_cases.CASES:
            
            net = pf.Parser(case).parse(case)

            net.make_all_in_service()

            for bus in net.buses:
                self.assertTrue(bus.in_service)
            for bus in net.dc_buses:
                self.assertTrue(bus.in_service)
            for gen in net.generators:
                self.assertTrue(gen.in_service)
            for branch in net.branches:
                self.assertTrue(branch.in_service)
            for facts in net.facts:
                self.assertTrue(facts.in_service)
            for conv in net.csc_converters:
                self.assertTrue(conv.in_service)
            for conv in net.vsc_converters:
                self.assertTrue(conv.in_service)
            for load in net.loads:
                self.assertTrue(load.in_service)
            for branch in net.dc_branches:
                self.assertTrue(branch.in_service)
            for shunt in net.shunts:
                self.assertTrue(shunt.in_service)
            for bat in net.batteries:
                self.assertTrue(bat.in_service)
            for gen in net.var_generators:
                self.assertTrue(gen.in_service)

            for bus in net.dc_buses:
                bus.in_service = False

            for bus in net.buses:
                self.assertTrue(bus.in_service)
            for bus in net.dc_buses:
                self.assertFalse(bus.in_service)
            for gen in net.generators:
                self.assertTrue(gen.in_service)
            for branch in net.branches:
                self.assertTrue(branch.in_service)
            for facts in net.facts:
                self.assertTrue(facts.in_service)
            for conv in net.csc_converters:
                self.assertFalse(conv.in_service)
            for conv in net.vsc_converters:
                self.assertFalse(conv.in_service)
            for load in net.loads:
                self.assertTrue(load.in_service)
            for branch in net.dc_branches:
                self.assertFalse(branch.in_service)
            for shunt in net.shunts:
                self.assertTrue(shunt.in_service)
            for bat in net.batteries:
                self.assertTrue(bat.in_service)
            for gen in net.var_generators:
                self.assertTrue(gen.in_service)

            # Changes have no effect
            for branch in net.dc_branches:
                branch.in_service = False
            for conv in net.csc_converters:
                conv.in_service = False
            for conv in net.vsc_converters:
                conv.in_service = False
                
            for bus in net.dc_buses:
                bus.in_service = True

            # Previous state remembered
            for bus in net.buses:
                self.assertTrue(bus.in_service)
            for bus in net.dc_buses:
                self.assertTrue(bus.in_service)
            for gen in net.generators:
                self.assertTrue(gen.in_service)
            for branch in net.branches:
                self.assertTrue(branch.in_service)
            for facts in net.facts:
                self.assertTrue(facts.in_service)
            for conv in net.csc_converters:
                self.assertTrue(conv.in_service)
            for conv in net.vsc_converters:
                self.assertTrue(conv.in_service)
            for load in net.loads:
                self.assertTrue(load.in_service)
            for branch in net.dc_branches:
                self.assertTrue(branch.in_service)
            for shunt in net.shunts:
                self.assertTrue(shunt.in_service)
            for bat in net.batteries:
                self.assertTrue(bat.in_service)
            for gen in net.var_generators:
                self.assertTrue(gen.in_service)

    def test_json(self):

        for case in test_cases.CASES:
            
            net = pf.Parser(case).parse(case)

            for gen in net.generators:
                gen.in_service = False
            for branch in net.branches:
                branch.in_service = False
            for bus in net.buses:
                bus.in_service = False
            for load in net.loads:
                load.in_service = False
            for bus in net.dc_buses:
                bus.in_service = False
            for branch in net.dc_branches:
                branch.in_service = False
            for conv in net.csc_converters:
                conv.in_service = False
            for conv in net.vsc_converters:
                conv.in_service = False
            for facts in net.facts:
                facts.in_service = False
            for bat in net.batteries:
                bat.in_service = False
            for gen in net.var_generators:
                gen.in_service = False
            for shunt in net.shunts:
                shunt.in_service = False

            new_net = json.loads(json.dumps(net, cls=pf.NetworkJSONEncoder),
                                 cls=pf.NetworkJSONDecoder)

            for bus in new_net.buses:
                self.assertFalse(bus.in_service)
            for bus in new_net.dc_buses:
                self.assertFalse(bus.in_service)
            for gen in new_net.generators:
                self.assertFalse(gen.in_service)
            for branch in new_net.branches:
                self.assertFalse(branch.in_service)
            for facts in new_net.facts:
                self.assertFalse(facts.in_service)
            for conv in new_net.csc_converters:
                self.assertFalse(conv.in_service)
            for conv in new_net.vsc_converters:
                self.assertFalse(conv.in_service)
            for load in new_net.loads:
                self.assertFalse(load.in_service)
            for branch in new_net.dc_branches:
                self.assertFalse(branch.in_service)
            for shunt in new_net.shunts:
                self.assertFalse(shunt.in_service)
            for bat in new_net.batteries:
                self.assertFalse(bat.in_service)
            for gen in new_net.var_generators:
                self.assertTrue(gen.in_service)

    def test_copy(self):

        for case in test_cases.CASES:
            
            net = pf.Parser(case).parse(case)

            for gen in net.generators:
                gen.in_service = False
            for branch in net.branches:
                branch.in_service = False
            for bus in net.buses:
                bus.in_service = False
            for load in net.loads:
                load.in_service = False
            for bus in net.dc_buses:
                bus.in_service = False
            for branch in net.dc_branches:
                branch.in_service = False
            for conv in net.csc_converters:
                conv.in_service = False
            for conv in net.vsc_converters:
                conv.in_service = False
            for facts in net.facts:
                facts.in_service = False
            for bat in net.batteries:
                bat.in_service = False
            for gen in net.var_generators:
                gen.in_service = False
            for shunt in net.shunts:
                shunt.in_service = False

            new_net = net.get_copy()

            for bus in new_net.buses:
                self.assertFalse(bus.in_service)
            for bus in new_net.dc_buses:
                self.assertFalse(bus.in_service)
            for gen in new_net.generators:
                self.assertFalse(gen.in_service)
            for branch in new_net.branches:
                self.assertFalse(branch.in_service)
            for facts in new_net.facts:
                self.assertFalse(facts.in_service)
            for conv in new_net.csc_converters:
                self.assertFalse(conv.in_service)
            for conv in new_net.vsc_converters:
                self.assertFalse(conv.in_service)
            for load in new_net.loads:
                self.assertFalse(load.in_service)
            for branch in new_net.dc_branches:
                self.assertFalse(branch.in_service)
            for shunt in new_net.shunts:
                self.assertFalse(shunt.in_service)
            for bat in new_net.batteries:
                self.assertFalse(bat.in_service)
            for gen in new_net.var_generators:
                self.assertTrue(gen.in_service)

    def test_state_tag(self):

        for case in test_cases.CASES:
            
            net = pf.Parser(case).parse(case)
            
            net.add_var_generators_from_parameters(net.get_load_buses(),100.,50.,30.,5,0.05)
            net.add_batteries_from_parameters(net.get_generator_buses(),20.,50.)

            self.assertEqual(net.state_tag, 0)

            for gen in net.generators:
                gen.in_service = False
            for branch in net.branches:
                branch.in_service = False
            for load in net.loads:
                load.in_service = False
            for branch in net.dc_branches:
                branch.in_service = False
            for conv in net.csc_converters:
                conv.in_service = False
            for conv in net.vsc_converters:
                conv.in_service = False
            for facts in net.facts:
                facts.in_service = False
            for bat in net.batteries:
                bat.in_service = False
            for gen in net.var_generators:
                gen.in_service = False
            for shunt in net.shunts:
                shunt.in_service = False
            for bus in net.dc_buses:
                bus.in_service = False
            for bus in net.buses:
                bus.in_service = False

            self.assertEqual(net.state_tag,
                             (net.num_generators+
                              net.num_branches+
                              net.num_buses+
                              net.num_loads+
                              net.num_facts+
                              net.num_dc_buses+
                              net.num_dc_branches+
                              net.num_vsc_converters+
                              net.num_csc_converters+
                              net.num_shunts+
                              net.num_batteries+
                              net.num_var_generators))

    def test_other_components(self):

        for case in test_cases.CASES:
            
            net = pf.Parser(case).parse(case)

            net.add_var_generators_from_parameters(net.get_load_buses(),100.,50.,30.,5,0.05)
            net.add_batteries_from_parameters(net.get_generator_buses(),20.,50.)

            # loads
            for load in net.loads:
                pass

            net_copy = net.get_copy()

            # facts
            for facts in net.facts:
                reg = facts.is_regulator()
                facts.in_service = False
                old_facts = net_copy.get_facts(facts.index)
                self.assertTrue(old_facts.in_service)
                self.assertEqual(facts.is_STATCOM(), old_facts.is_STATCOM())
                self.assertEqual(facts.is_SSSC(), old_facts.is_SSSC())
                self.assertEqual(facts.is_UPFC(), old_facts.is_UPFC())
                self.assertEqual(facts.is_series_link_disabled(), old_facts.is_series_link_disabled())
                self.assertEqual(facts.is_series_link_bypassed(), old_facts.is_series_link_bypassed())
                self.assertEqual(facts.is_in_normal_series_mode(), old_facts.is_in_normal_series_mode())
                self.assertEqual(facts.is_in_constant_series_z_mode(), old_facts.is_in_constant_series_z_mode())
                self.assertEqual(facts.is_in_constant_series_v_mode(), old_facts.is_in_constant_series_v_mode())
                if reg:
                    self.assertTrue(facts.is_regulator())
                    self.assertTrue(facts.reg_bus is not None)
                    self.assertTrue(facts.index in [f.index for f in facts.reg_bus.reg_facts])
                    if all([not f.is_in_service() for f in facts.reg_bus.reg_facts]):
                        self.assertFalse(facts.reg_bus.is_regulated_by_facts(only_in_service=True))
                    else:
                        self.assertTrue(facts.reg_bus.is_regulated_by_facts(only_in_service=True))
                    self.assertTrue(facts.reg_bus.is_regulated_by_facts())

            # csc
            for conv in net.csc_converters:
                conv.in_service = False
                old_conv = net_copy.get_csc_converter(conv.index)
                self.assertFalse(conv.is_in_service())
                self.assertTrue(old_conv.in_service)
                self.assertEqual(conv.is_in_v_dc_mode(), old_conv.is_in_v_dc_mode())

            # vsc
            for conv in net.vsc_converters:
                reg = conv.is_in_v_ac_mode()
                conv.in_service = False
                old_conv = net_copy.get_vsc_converter(conv.index)
                self.assertTrue(old_conv.in_service)
                self.assertEqual(conv.is_in_v_ac_mode(), old_conv.is_in_v_ac_mode())
                if reg:
                    self.assertTrue(conv.is_in_v_ac_mode())
                    self.assertTrue(conv.reg_bus is not None)
                    self.assertTrue(conv.index in [c.index for c in conv.reg_bus.reg_vsc_converters])
                    if all([not c.is_in_service() for c in conv.reg_bus.reg_vsc_converters]):
                        self.assertFalse(conv.reg_bus.is_regulated_by_vsc_converter(only_in_service=True))
                    else:
                        self.assertTrue(conv.reg_bus.is_regulated_by_vsc_converter(only_in_service=True))
                    self.assertTrue(conv.reg_bus.is_regulated_by_vsc_converter())

            # shunts
            for shunt in net.shunts:
                reg = shunt.is_switched_v()
                shunt.in_service = False
                old_shunt = net_copy.get_shunt(shunt.index)
                self.assertFalse(shunt.is_in_service())
                self.assertTrue(old_shunt.in_service)
                self.assertEqual(shunt.is_fixed(), old_shunt.is_fixed())
                self.assertEqual(shunt.is_switched(), old_shunt.is_switched())
                self.assertEqual(shunt.is_switched_locked(), old_shunt.is_switched_locked())
                self.assertEqual(shunt.is_switched_v(), old_shunt.is_switched_v())
                self.assertEqual(shunt.is_continuous(), old_shunt.is_continuous())
                self.assertEqual(shunt.is_discrete(), old_shunt.is_discrete())
                if reg:
                    self.assertTrue(shunt.is_switched_v())
                    self.assertTrue(shunt.reg_bus is not None)
                    self.assertTrue(shunt.index in [s.index for s in shunt.reg_bus.reg_shunts])
                    if all([not s.is_in_service() for s in shunt.reg_bus.reg_shunts]):
                        self.assertFalse(shunt.reg_bus.is_regulated_by_shunt(only_in_service=True))
                    else:
                        self.assertTrue(shunt.reg_bus.is_regulated_by_shunt(only_in_service=True))
                    self.assertTrue(shunt.reg_bus.is_regulated_by_shunt())

            # vargens
            for gen in net.var_generators:
                gen.in_service = False
                self.assertFalse(gen.is_in_service())

            # bats
            for bat in net.batteries:
                bat.in_service = False
                self.assertFalse(bat.is_in_service())

            # dc branches
            for br in net.dc_branches:
                br.in_service = False
                self.assertFalse(br.is_in_service())

            net.make_all_in_service()
            
            # dc buses
            for bus in net.dc_buses:
                bus.in_service = False
                self.assertFalse(bus.is_in_service())
                for conv in bus.csc_converters:
                    self.assertFalse(conv.is_in_service())
                for conv in bus.vsc_converters:
                    self.assertFalse(conv.is_in_service())
                for br in bus.branches_k:
                    self.assertFalse(br.is_in_service())
                for br in bus.branches_m:
                    self.assertFalse(br.is_in_service())
        
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
                        self.assertFalse(gen.reg_bus.is_regulated_by_gen(only_in_service=True))
                    else:
                        self.assertTrue(gen.reg_bus.is_regulated_by_gen(only_in_service=True))
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
                        self.assertFalse(branch.reg_bus.is_regulated_by_tran(only_in_service=True))
                    else:
                        self.assertTrue(branch.reg_bus.is_regulated_by_tran(only_in_service=True))
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

            net.add_var_generators_from_parameters(net.get_load_buses(),100.,50.,30.,5,0.05)
            net.add_batteries_from_parameters(net.get_generator_buses(),20.,50.)

            for bus in net.buses:

                slack = bus.is_slack()
                star = bus.is_star()
                
                bus.in_service = False
                
                self.assertEqual(bus.get_total_gen_P(), 0.)
                self.assertEqual(bus.get_total_gen_Q(), 0.)
                self.assertEqual(bus.get_total_gen_Q_min(), 0.)
                self.assertEqual(bus.get_total_gen_Q_max(), 0.)
                self.assertEqual(bus.get_total_load_P(), 0.)
                self.assertEqual(bus.get_total_load_Q(), 0.)
                self.assertEqual(bus.get_total_shunt_g(), 0.)
                self.assertEqual(bus.get_total_shunt_b(), 0.)

                self.assertEqual(bus.is_slack(), slack)
                self.assertEqual(bus.is_star(), star)
                
                # reg 
                self.assertEqual(bus.is_regulated_by_gen(only_in_service=True),
                                 any([x for x in bus.reg_generators if x.in_service]))
                self.assertEqual(bus.is_regulated_by_tran(only_in_service=True),
                                 any([x for x in bus.reg_trans if x.in_service]))
                self.assertEqual(bus.is_regulated_by_shunt(only_in_service=True),
                                 any([x for x in bus.reg_shunts if x.in_service]))
                self.assertEqual(bus.is_v_set_regulated(only_in_service=True),
                                 (bus.is_regulated_by_gen(only_in_service=True) or
                                  bus.is_regulated_by_shunt(only_in_service=True) or
                                  bus.is_regulated_by_facts(only_in_service=True)))
                self.assertEqual(bus.is_regulated_by_vsc_converter(only_in_service=True),
                                 any([x for x in bus.reg_vsc_converters if x.in_service]))
                self.assertEqual(bus.is_regulated_by_facts(only_in_service=True),
                                 any([x for x in bus.reg_facts if x.in_service]))

                for gen in bus.generators:
                    self.assertFalse(gen.in_service)
                for br in bus.branches_k:
                    self.assertFalse(br.in_service)
                for br in bus.branches_m:
                    self.assertFalse(br.in_service)
                for s in bus.shunts:
                    self.assertFalse(s.in_service)
                for load in bus.loads:
                    self.assertFalse(load.in_service)
                for conv in bus.csc_converters:
                    self.assertFalse(conv.in_service)
                for conv in bus.vsc_converters:
                    self.assertFalse(conv.in_service)
                for bat in bus.batteries:
                    self.assertFalse(bat.in_service)
                for gen in bus.var_generators:
                    self.assertFalse(gen.in_service)

                # json
                json_string = bus.json_string
                d = json.loads(json_string)
                self.assertTrue('in_service' in d)
                self.assertFalse(d['in_service'])
    
    def test_network(self):

        for case in test_cases.CASES:
            
            net = pf.Parser(case).parse(case)

            net.make_all_in_service()
            
            self.assertEqual(net.get_num_branches(only_in_service=True), net.num_branches)
            self.assertEqual(net.get_num_branches(only_in_service=False), net.num_branches)
            self.assertEqual(net.get_num_branches_out_of_service(), 0)

            self.assertEqual(net.get_num_generators(only_in_service=True), net.num_generators)
            self.assertEqual(net.get_num_generators(only_in_service=False), net.num_generators)
            self.assertEqual(net.get_num_generators_out_of_service(), 0)

            self.assertEqual(net.get_num_buses(only_in_service=True), net.num_buses)
            self.assertEqual(net.get_num_buses(only_in_service=False), net.num_buses)
            self.assertEqual(net.get_num_buses_out_of_service(), 0)

            self.assertEqual(net.get_num_loads(only_in_service=True), net.num_loads)
            self.assertEqual(net.get_num_loads(only_in_service=False), net.num_loads)
            self.assertEqual(net.get_num_loads_out_of_service(), 0)

            self.assertEqual(net.get_num_shunts(only_in_service=True), net.num_shunts)
            self.assertEqual(net.get_num_shunts(only_in_service=False), net.num_shunts)
            self.assertEqual(net.get_num_shunts_out_of_service(), 0)

            self.assertEqual(net.get_num_batteries(only_in_service=True), net.num_batteries)
            self.assertEqual(net.get_num_batteries(only_in_service=False), net.num_batteries)
            self.assertEqual(net.get_num_batteries_out_of_service(), 0)

            self.assertEqual(net.get_num_var_generators(only_in_service=True), net.num_var_generators)
            self.assertEqual(net.get_num_var_generators(only_in_service=False), net.num_var_generators)
            self.assertEqual(net.get_num_var_generators_out_of_service(), 0)

            self.assertEqual(net.get_num_facts(only_in_service=True), net.num_facts)
            self.assertEqual(net.get_num_facts(only_in_service=False), net.num_facts)
            self.assertEqual(net.get_num_facts_out_of_service(), 0)

            self.assertEqual(net.get_num_dc_buses(only_in_service=True), net.num_dc_buses)
            self.assertEqual(net.get_num_dc_buses(only_in_service=False), net.num_dc_buses)
            self.assertEqual(net.get_num_dc_buses_out_of_service(), 0)

            self.assertEqual(net.get_num_dc_branches(only_in_service=True), net.num_dc_branches)
            self.assertEqual(net.get_num_dc_branches(only_in_service=False), net.num_dc_branches)
            self.assertEqual(net.get_num_dc_branches_out_of_service(), 0)

            self.assertEqual(net.get_num_csc_converters(only_in_service=True), net.num_csc_converters)
            self.assertEqual(net.get_num_csc_converters(only_in_service=False), net.num_csc_converters)
            self.assertEqual(net.get_num_csc_converters_out_of_service(), 0)

            self.assertEqual(net.get_num_vsc_converters(only_in_service=True), net.num_vsc_converters)
            self.assertEqual(net.get_num_vsc_converters(only_in_service=False), net.num_vsc_converters)
            self.assertEqual(net.get_num_vsc_converters_out_of_service(), 0)
            
            # out of service
            for bus in net.buses:
                bus.in_service = False
            for bus in net.dc_buses:
                bus.in_service = False
            for gen in net.generators:
                gen.in_service = False
            for branch in net.branches:
                branch.in_service = False
            for facts in net.facts:
                facts.in_service = False
            for conv in net.csc_converters:
                conv.in_service = False
            for conv in net.vsc_converters:
                conv.in_service = False
            for load in net.loads:
                load.in_service = False
            for branch in net.dc_branches:
                branch.in_service = False
            for shunt in net.shunts:
                shunt.in_service = False
            for bat in net.batteries:
                bat.in_service = False
            for gen in net.var_generators:
                gen.in_service = False
                
            self.assertEqual(net.get_num_branches(True), 0)
            self.assertEqual(net.get_num_branches(False), net.num_branches)
            self.assertEqual(net.get_num_branches_out_of_service(), net.num_branches)
            self.assertEqual(net.get_num_fixed_trans(only_in_service=True), 0)
            self.assertEqual(net.get_num_fixed_trans(only_in_service=False),
                             len([br for br in net.branches if br.is_fixed_tran()]))
            self.assertEqual(net.get_num_lines(only_in_service=True), 0)
            self.assertEqual(net.get_num_lines(only_in_service=False),
                             len([br for br in net.branches if br.is_line()]))
            self.assertEqual(net.get_num_zero_impedance_lines(only_in_service=True), 0)
            self.assertEqual(net.get_num_zero_impedance_lines(only_in_service=False),
                             len([br for br in net.branches if br.is_zero_impedance_line()]))
            self.assertEqual(net.get_num_phase_shifters(only_in_service=True), 0)
            self.assertEqual(net.get_num_phase_shifters(only_in_service=False),
                             len([br for br in net.branches if br.is_phase_shifter()]))
            self.assertEqual(net.get_num_tap_changers(only_in_service=True), 0)
            self.assertEqual(net.get_num_tap_changers(only_in_service=False),
                             len([br for br in net.branches if br.is_tap_changer()]))
            self.assertEqual(net.get_num_tap_changers_v(only_in_service=True), 0)
            self.assertEqual(net.get_num_tap_changers_v(only_in_service=False),
                             len([br for br in net.branches if br.is_tap_changer_v()]))
            self.assertEqual(net.get_num_tap_changers_Q(only_in_service=True), 0)
            self.assertEqual(net.get_num_tap_changers_Q(only_in_service=False),
                             len([br for br in net.branches if br.is_tap_changer_Q()]))

            self.assertEqual(net.get_num_generators(only_in_service=True), 0)
            self.assertEqual(net.get_num_generators(only_in_service=False), net.num_generators)
            self.assertEqual(net.get_num_generators_out_of_service(), net.num_generators)
            self.assertEqual(net.get_num_reg_gens(only_in_service=True), 0)
            self.assertEqual(net.get_num_reg_gens(only_in_service=False), len([g for g in net.generators if g.is_regulator()]))
            self.assertEqual(net.get_num_slack_gens(only_in_service=True), 0)
            self.assertEqual(net.get_num_slack_gens(only_in_service=False), len([g for g in net.generators if g.is_slack()]))
            
            self.assertEqual(net.get_num_buses(only_in_service=True), 0)
            self.assertEqual(net.get_num_buses(only_in_service=False), net.num_buses)
            self.assertEqual(net.get_num_buses_out_of_service(), net.num_buses)
            self.assertEqual(net.get_num_slack_buses(only_in_service=True), 0)
            self.assertEqual(net.get_num_star_buses(only_in_service=True), 0)
            self.assertEqual(net.get_num_buses_reg_by_gen(only_in_service=True), 0)
            self.assertEqual(net.get_num_buses_reg_by_tran(only_in_service=True), 0)
            self.assertEqual(net.get_num_buses_reg_by_shunt(only_in_service=True), 0)
            self.assertEqual(net.get_num_buses_reg_by_facts(only_in_service=True), 0)
            self.assertEqual(net.get_num_buses_reg_by_vsc_converter(only_in_service=True), 0)
            self.assertEqual(net.get_num_slack_buses(only_in_service=False),
                             len([b for b in net.buses if b.is_slack()]))
            self.assertEqual(net.get_num_star_buses(only_in_service=False),
                             len([b for b in net.buses if b.is_star()]))
            self.assertEqual(net.get_num_buses_reg_by_gen(only_in_service=False),
                             len([b for b in net.buses if b.is_regulated_by_gen()]))
            self.assertEqual(net.get_num_buses_reg_by_tran(only_in_service=False),
                             len([b for b in net.buses if b.is_regulated_by_tran()]))
            self.assertEqual(net.get_num_buses_reg_by_shunt(only_in_service=False),
                             len([b for b in net.buses if b.is_regulated_by_shunt()]))
            self.assertEqual(net.get_num_buses_reg_by_facts(only_in_service=False),
                             len([b for b in net.buses if b.is_regulated_by_facts()]))
            self.assertEqual(net.get_num_buses_reg_by_vsc_converter(only_in_service=False),
                             len([b for b in net.buses if b.is_regulated_by_vsc_converter()]))
            
            self.assertEqual(net.get_num_loads(only_in_service=True), 0)
            self.assertEqual(net.get_num_loads(only_in_service=False), net.num_loads)
            self.assertEqual(net.get_num_loads_out_of_service(), net.num_loads)
            self.assertEqual(net.get_num_vdep_loads(only_in_service=True), 0)
            self.assertEqual(net.get_num_vdep_loads(only_in_service=False), len([l for l in net.loads if l.is_voltage_dependent()]))
            
            self.assertEqual(net.get_num_shunts(only_in_service=True), 0)
            self.assertEqual(net.get_num_shunts(only_in_service=False), net.num_shunts)
            self.assertEqual(net.get_num_shunts_out_of_service(), net.num_shunts)
            self.assertEqual(net.get_num_fixed_shunts(only_in_service=True), 0)
            self.assertEqual(net.get_num_fixed_shunts(only_in_service=False),
                             len([s for s in net.shunts if s.is_fixed()]))
            self.assertEqual(net.get_num_switched_shunts(only_in_service=True), 0)
            self.assertEqual(net.get_num_switched_shunts(only_in_service=False),
                             len([s for s in net.shunts if s.is_switched()]))
            self.assertEqual(net.get_num_switched_v_shunts(only_in_service=True), 0)
            self.assertEqual(net.get_num_switched_v_shunts(only_in_service=False),
                             len([s for s in net.shunts if s.is_switched_v()]))            

            self.assertEqual(net.get_num_batteries(only_in_service=True), 0)
            self.assertEqual(net.get_num_batteries(only_in_service=False), net.num_batteries)
            self.assertEqual(net.get_num_batteries_out_of_service(), net.num_batteries)

            self.assertEqual(net.get_num_var_generators(only_in_service=True), 0)
            self.assertEqual(net.get_num_var_generators(only_in_service=False), net.num_var_generators)
            self.assertEqual(net.get_num_var_generators_out_of_service(), net.num_var_generators)

            self.assertEqual(net.get_num_facts(only_in_service=True), 0)
            self.assertEqual(net.get_num_facts(only_in_service=False), net.num_facts)
            self.assertEqual(net.get_num_facts_out_of_service(), net.num_facts)
            self.assertEqual(net.get_num_facts_in_normal_series_mode(only_in_service=True), 0)
            self.assertEqual(net.get_num_facts_in_normal_series_mode(only_in_service=False),
                             len([f for f in net.facts if f.is_in_normal_series_mode()]))
            self.assertEqual(net.get_num_reg_facts(only_in_service=True), 0)
            self.assertEqual(net.get_num_reg_facts(only_in_service=False), len([f for f in net.facts if f.is_regulator()]))

            self.assertEqual(net.get_num_dc_buses(only_in_service=True), 0)
            self.assertEqual(net.get_num_dc_buses(only_in_service=False), net.num_dc_buses)
            self.assertEqual(net.get_num_dc_buses_out_of_service(), net.num_dc_buses)

            self.assertEqual(net.get_num_dc_branches(only_in_service=True), 0)
            self.assertEqual(net.get_num_dc_branches(only_in_service=False), net.num_dc_branches)
            self.assertEqual(net.get_num_dc_branches_out_of_service(), net.num_dc_branches)

            self.assertEqual(net.get_num_csc_converters(only_in_service=True), 0)
            self.assertEqual(net.get_num_csc_converters(only_in_service=False), net.num_csc_converters)
            self.assertEqual(net.get_num_csc_converters_out_of_service(), net.num_csc_converters)
            
            self.assertEqual(net.get_num_vsc_converters(only_in_service=True), 0)
            self.assertEqual(net.get_num_vsc_converters(only_in_service=False), net.num_vsc_converters)
            self.assertEqual(net.get_num_vsc_converters_out_of_service(), net.num_vsc_converters)
            self.assertEqual(net.get_num_vsc_converters_in_P_dc_mode(only_in_service=True), 0)
            self.assertEqual(net.get_num_vsc_converters_in_P_dc_mode(only_in_service=False),
                             len([c for c in net.vsc_converters if c.is_in_P_dc_mode()]))
            self.assertEqual(net.get_num_vsc_converters_in_v_dc_mode(only_in_service=False),
                             len([c for c in net.vsc_converters if c.is_in_v_dc_mode()]))
            self.assertEqual(net.get_num_vsc_converters_in_v_ac_mode(only_in_service=False),
                             len([c for c in net.vsc_converters if c.is_in_v_ac_mode()]))
            self.assertEqual(net.get_num_vsc_converters_in_f_ac_mode(only_in_service=False),
                             len([c for c in net.vsc_converters if c.is_in_f_ac_mode()]))

            net.make_all_in_service()

            # in of service
            for bus in net.buses:
                self.assertTrue(bus.in_service)
            for bus in net.dc_buses:
                self.assertTrue(bus.in_service)
            for gen in net.generators:
                self.assertTrue(gen.in_service)
            for branch in net.branches:
                self.assertTrue(branch.in_service)
            for facts in net.facts:
                self.assertTrue(facts.in_service)
            for conv in net.csc_converters:
                self.assertTrue(conv.in_service)
            for conv in net.vsc_converters:
                self.assertTrue(conv.in_service)
            for load in net.loads:
                self.assertTrue(load.in_service)
            for branch in net.dc_branches:
                self.assertTrue(branch.in_service)
            for shunt in net.shunts:
                self.assertTrue(shunt.in_service)
            for bat in net.batteries:
                self.assertTrue(bat.in_service)
            for gen in net.var_generators:
                self.assertTrue(gen.in_service)

    def test_network_properties(self):

        for case in test_cases.CASES:

            if os.path.basename(case) in ['sys_problem3.mat',
                                          'sys_problem2.mat']:
                continue
            
            net = pf.Parser(case).parse(case)

            if net.num_buses > 2000:
                continue

            # bus vmax vmin
            net.make_all_in_service()
            for bus in net.buses:
                vmag = bus.v_mag
                bus.v_mag = 100.
                net.update_properties()
                self.assertEqual(net.bus_v_max, 100.)
                bus.in_service = False
                net.update_properties()
                self.assertNotEqual(net.bus_v_max, 100.)
                bus.in_service = True
                bus.v_mag = -100.
                net.update_properties()
                self.assertEqual(net.bus_v_min, -100.)
                bus.in_service = False
                net.update_properties()
                self.assertNotEqual(net.bus_v_min, -100.)
                bus.v_mag = vmag

            # mismatches bus outages
            net.make_all_in_service()
            for bus in net.buses:
                bus.in_service = False
            net.update_properties()
            for bus in net.buses:
                self.assertEqual(bus.P_mismatch, 0.)
                self.assertEqual(bus.Q_mismatch, 0.)
            self.assertEqual(net.bus_P_mis, 0.)
            self.assertEqual(net.bus_Q_mis, 0.)

            # mismatches all comp outages except buses
            net.make_all_in_service()
            net.update_properties()
            self.assertNotEqual(net.bus_P_mis, 0.)
            self.assertNotEqual(net.bus_Q_mis, 0.)
            for gen in net.generators:
                gen.in_service = False
            for branch in net.branches:
                branch.in_service = False
            for bus in net.buses:
                bus.in_service = False
            for load in net.loads:
                load.in_service = False
            for bus in net.dc_buses:
                bus.in_service = False
            for branch in net.dc_branches:
                branch.in_service = False
            for conv in net.csc_converters:
                conv.in_service = False
            for conv in net.vsc_converters:
                conv.in_service = False
            for facts in net.facts:
                facts.in_service = False
            for bat in net.batteries:
                bat.in_service = False
            for gen in net.var_generators:
                gen.in_service = False
            for shunt in net.shunts:
                shunt.in_service = False
            net.update_properties()
            for bus in net.buses:
                self.assertEqual(bus.P_mismatch, 0.)
                self.assertEqual(bus.Q_mismatch, 0.)
            self.assertEqual(net.bus_P_mis, 0.)
            self.assertEqual(net.bus_Q_mis, 0.)
            
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
                                          sum([br.P_mk for br in bus.branches_m]))), 1e-4)
            self.assertLess(np.abs(q_mis-(bus.Q_mismatch +
                                          sum([g.Q for g in bus.generators]) -
                                          sum([br.Q_km for br in bus.branches_k]) -
                                          sum([br.Q_mk for br in bus.branches_m]))), 1e-4)
            
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
                    self.assertTrue(bus.is_regulated_by_tran())
                    self.assertFalse(bus.is_regulated_by_tran(only_in_service=True))
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
                    self.assertTrue(bus.is_regulated_by_gen())
                    self.assertFalse(bus.is_regulated_by_gen(only_in_service=True))
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
                if gen.is_regulator() and not gen.is_slack():
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
