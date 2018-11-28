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

class TestBRYCOR(unittest.TestCase):

    def test_aeso_raw_case(self):

        T = 4

        case = os.path.join('data', 'aesoSL2014.raw')

        if os.path.isfile(case):

            net = pf.ParserRAW().parse(case, T)
            
            self.assertGreater(net.num_buses, 0)
            
            self.assertEqual(len([br for br in net.branches if br.has_y_correction()]), 3)
            
            br1 = net.get_branch_from_name_and_bus_numbers('PS',
                                                           421,
                                                           420)
            br2 = net.get_branch_from_name_and_bus_numbers('PS',
                                                           1602,
                                                           1288)
            br3 = net.get_branch_from_name_and_bus_numbers('PS',
                                                           656,
                                                           229)
            for br in [br1, br2, br3]:
                self.assertTrue(br.has_y_correction())

            ycorr1 = br1.y_correction
            ycorr2 = br2.y_correction
            ycorr3 = br3.y_correction

            for ycorr in [ycorr1, ycorr2, ycorr3]:
                self.assertTrue(isinstance(ycorr, pf.BranchYCorrection))
                self.assertTrue(ycorr.is_based_on_phase_shift())
                self.assertEqual(ycorr.max_num_values, 20)
                self.assertTrue(isinstance(ycorr.values, np.ndarray))
                self.assertTrue(isinstance(ycorr.corrections, np.ndarray))

            self.assertEqual(ycorr1.name, 'Y Correction 12')
            self.assertEqual(ycorr2.name, 'Y Correction 12')
            self.assertEqual(ycorr3.name, 'Y Correction 13')

            raw12_val = [-30.00, -22.90, -18.90, -8.40, -4.20, 0.00, 6.30, 10.50, 18.90, 22.90, 30.00]
            raw12_cor = [1.28000, 1.17000, 1.11000, 1.04000, 1.00900, 1.00000, 1.02600, 1.06000, 1.11000, 1.17000, 1.28000]

            raw13_val = [-60.00, -39.70, -24.40, 0.00, 24.40, 39.70, 60.00]
            raw13_cor = [1.57174, 1.26912, 1.10920, 1.00000, 1.11282, 1.27373, 1.57165]
            
            self.assertEqual(ycorr1.num_values, 11)
            self.assertEqual(ycorr2.num_values, 11)
            self.assertEqual(ycorr3.num_values, 7)
            
            self.assertEqual(ycorr1.values.size, 11)
            self.assertEqual(ycorr1.corrections.size, 11)
            self.assertEqual(ycorr2.values.size, 11)
            self.assertEqual(ycorr2.corrections.size, 11)
            self.assertEqual(ycorr3.values.size, 7)
            self.assertEqual(ycorr3.corrections.size, 7)
            
            for i in range(11):
                self.assertLess(np.abs(ycorr1.values[i]*180./np.pi-raw12_val[i]), 1e-8)
                self.assertLess(np.abs(ycorr1.corrections[i]-1./raw12_cor[i]), 1e-8)
                self.assertLess(np.abs(ycorr2.values[i]*180./np.pi-raw12_val[i]), 1e-8)
                self.assertLess(np.abs(ycorr2.corrections[i]-1./raw12_cor[i]), 1e-8)

            for i in range(7):
                self.assertLess(np.abs(ycorr3.values[i]*180./np.pi-raw13_val[i]), 1e-8)
                self.assertLess(np.abs(ycorr3.corrections[i]-1./raw13_cor[i]), 1e-8)

        else:
            raise unittest.SkipTest('no .raw file')

    def test_psse_sample_raw_case(self):

        T = 4

        case = os.path.join('data', 'psse_sample_case.raw')

        if os.path.isfile(case):
        
            p = pf.ParserRAW()
            
            net = p.parse(case, T)
            
            self.assertGreater(net.num_buses, 0)
            
            self.assertEqual(len([br for br in net.branches if br.has_y_correction()]), 4)
            
            star_bus = None
            for bus in net.buses:
                if not bus.is_star():
                    continue
                for br in bus.branches:
                    if br.bus_k.number == 3010 or br.bus_m.number == 3010:
                        star_bus = bus
            self.assertTrue(star_bus is not None)

            br1 = net.get_branch_from_name_and_bus_numbers('T4', # ratio-based
                                                           152,
                                                           3021)

            br2 = net.get_branch_from_name_and_bus_numbers('T7', # phase-based
                                                           203,
                                                           202)
            
            br3 = net.get_branch_from_name_and_bus_numbers('11', # ratio-based
                                                           3008,
                                                           3018)
            
            br4 = net.get_branch_from_name_and_bus_numbers('2',  # phase-based
                                                           3010, 
                                                           star_bus.number)
       
            
            for br in [br1, br2, br3, br4]:
                self.assertTrue(br.has_y_correction())

            ycorr1 = br1.y_correction
            ycorr2 = br2.y_correction
            ycorr3 = br3.y_correction
            ycorr4 = br4.y_correction
            
            for ycorr in [ycorr1, ycorr2, ycorr3, ycorr4]:
                self.assertTrue(isinstance(ycorr, pf.BranchYCorrection))
                self.assertEqual(ycorr.max_num_values, 20)
                self.assertTrue(isinstance(ycorr.values, np.ndarray))
                self.assertTrue(isinstance(ycorr.corrections, np.ndarray))
                self.assertEqual(ycorr.num_values, 11)
                self.assertEqual(ycorr.values.size, 11)
                self.assertEqual(ycorr.corrections.size, 11)

            for ycorr in [ycorr2, ycorr4]:
                self.assertTrue(ycorr.is_based_on_phase_shift())
                self.assertFalse(ycorr.is_based_on_tap_ratio())

            for ycorr in [ycorr1, ycorr3]:
                self.assertFalse(ycorr.is_based_on_phase_shift())
                self.assertTrue(ycorr.is_based_on_tap_ratio())

            self.assertEqual(ycorr1.name, 'Y Correction 2')
            self.assertEqual(ycorr2.name, 'Y Correction 1')
            self.assertEqual(ycorr3.name, 'Y Correction 2')
            self.assertEqual(ycorr4.name, 'Y Correction 1')

            raw1_val = [-30.00, -24.00, -18.00, -12.00, -6.00, 0.00, 6.00, 12.00, 18.00, 24.00, 30.00]
            raw1_cor = [1.10000, 1.09100, 1.08400, 1.06300, 1.03200, 1.00000, 1.03000, 1.06000, 1.08000, 1.09000, 1.11000]
            
            raw2_val = [0.60000, 0.70000, 0.80000, 0.90000, 0.95000, 1.00000, 1.05000, 1.10000, 1.20000, 1.30000, 1.40000]
            raw2_cor = [1.06000, 1.05000, 1.04000, 1.03000, 1.02000, 1.01000, 0.99000, 0.98000, 0.97000, 0.96000, 0.95000]
            
            for ycorr in [ycorr2, ycorr4]:
                for i in range(10):
                    self.assertTrue(ycorr.values[i] < ycorr.values[i+1])
                for i in range(11):
                    self.assertLess(np.abs(ycorr.values[i]*180./np.pi-raw1_val[i]), 1e-8)
                    self.assertLess(np.abs(ycorr.corrections[i]-1./raw1_cor[i]), 1e-8)

            for ycorr in [ycorr1, ycorr3]:
                t2 = 1.
                for i in range(10):
                    self.assertTrue(ycorr.values[i] < ycorr.values[i+1])
                for i in range(11):
                    self.assertLess(np.abs(ycorr.corrections[i]-1./raw2_cor[11-i-1]), 1e-8)
                    self.assertLess(np.abs(ycorr.values[i]-t2/raw2_val[11-i-1]), 1e-8)

        else:
            raise unittest.SkipTest('no .raw file')
