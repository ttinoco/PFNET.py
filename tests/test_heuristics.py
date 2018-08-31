#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

import unittest
import numpy as np
import pfnet as pf
from . import test_cases

class TestHeuristics(unittest.TestCase):
    
    def test_PVPQ_switching(self):

        T = 2

        for case in test_cases.CASES:
            
            net = pf.Parser(case).parse(case, T)
            self.assertEqual(net.num_periods, T)

            # Variables
            net.set_flags('bus',
                          'variable',
                          'not slack',
                          ['voltage magnitude','voltage angle'])
            net.set_flags('generator',
                          'variable',
                          'slack',
                          'active power')
            net.set_flags('generator',
                          'variable',
                          'regulator',
                          'reactive power')
            net.set_flags('branch',
                          'variable',
                          'tap changer - v',
                          'tap ratio')
            net.set_flags('branch',
                          'variable',
                          'phase shifter',
                          'phase shift')
            net.set_flags('shunt',
                          'variable',
                          'switching - v',
                          'susceptance')
            
            self.assertEqual(net.num_vars,
                             (2*(net.num_buses-net.get_num_slack_buses()) +
                              net.get_num_slack_gens() +
                              net.get_num_reg_gens() +
                              net.get_num_tap_changers_v() + 
                              net.get_num_phase_shifters() +
                              net.get_num_switched_v_shunts())*T)
                             
            # Fixed
            net.set_flags('branch',
                          'fixed',
                          'tap changer - v',
                          'tap ratio')
            net.set_flags('branch',
                          'fixed',
                          'phase shifter',
                          'phase shift')
            net.set_flags('shunt',
                          'fixed',
                          'switching - v',
                          'susceptance')
            self.assertEqual(net.num_fixed,
                             (net.get_num_tap_changers_v() +
                              net.get_num_phase_shifters() +
                              net.get_num_switched_v_shunts())*T)
                             

            self.assertRaises(pf.HeuristicError, pf.Heuristic, 'foo', net)

            heur = pf.Heuristic('PVPQ switching', net)

            self.assertEqual(heur.name, 'PVPQ switching')

            self.assertTrue(heur.network.has_same_ptr(net))

            x = net.get_var_values()

            acpf = pf.Constraint('AC power balance', net)
            pvpq = pf.Constraint('PVPQ switching', net)
            fix = pf.Constraint('variable fixing', net)

            self.assertRaises(pf.HeuristicError, heur.apply, [], x)
            self.assertRaises(pf.HeuristicError, heur.apply, [fix], x)
            self.assertRaises(pf.HeuristicError, heur.apply, [fix, acpf], x)
            self.assertRaises(pf.HeuristicError, heur.apply, [fix, pvpq], x)
            self.assertRaises(pf.HeuristicError, heur.apply, [acpf, pvpq], x)
            self.assertRaises(pf.HeuristicError, heur.apply, [acpf, fix, pvpq], x)

            for c in [acpf, pvpq, fix]:
                c.analyze()

            self.assertRaises(pf.HeuristicError, heur.apply, [], x)
            self.assertRaises(pf.HeuristicError, heur.apply, [fix], x)
            self.assertRaises(pf.HeuristicError, heur.apply, [fix, acpf], x)
            self.assertRaises(pf.HeuristicError, heur.apply, [fix, pvpq], x)
            heur.apply([acpf, pvpq], x)
            self.assertEqual(acpf.f.size, 2*net.num_buses*T)
            self.assertEqual(pvpq.A.shape[1], net.num_vars)
            heur.apply([acpf, fix, pvpq], x)
            self.assertEqual(acpf.f.size, 2*net.num_buses*T)
            self.assertEqual(pvpq.A.shape[1], net.num_vars)
            
