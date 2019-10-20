#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

import os
import csv
import pfnet
import unittest
import numpy as np
from . import test_cases

class TestUtils(unittest.TestCase):

    def setUp(self):
        
        pass

    def test_ptdf_case14(self):
        
        case = os.path.join('data', 'case14.m')
        case_ptdf = os.path.join('tests', 'resources', 'case14_ptdf.csv')
        if not os.path.isfile(case) or not os.path.isfile(case_ptdf):
            raise unittest.SkipTest('file not available')
    
        net = pfnet.Parser(case).parse(case)
        
        ptdf = pfnet.utils.make_PTDF(net)
        
        self.assertTrue(isinstance(ptdf, np.ndarray))
        self.assertEqual(ptdf.shape[0], net.get_num_branches(False))
        self.assertEqual(ptdf.shape[1], net.get_num_buses(False))
        
        for br in net.branches:
            br.name = ''

        f = open(case_ptdf, 'r')
        errors = []
        rows = csv.reader(f)
        row = next(rows)
        for row in rows:
            nk = int(row[0])
            nm = int(row[1])
            n = int(row[2])
            val1 = float(row[3])
            bus = net.get_bus_from_number(n)
            branch = net.get_branch_from_name_and_bus_numbers('', nk, nm)
            val2 = ptdf[branch.index, bus.index]
            if val1 == 0:
                self.assertEqual(val2, 0.)
            else:
                errors.append(abs(val1-val2)/abs(val1))
        self.assertGreater(len(errors), 0)
        self.assertLess(np.average(errors), 0.06)
        self.assertLess(np.std(errors), 0.084)

        f.close()
