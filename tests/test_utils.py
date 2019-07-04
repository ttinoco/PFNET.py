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

class TestUtils(unittest.TestCase):

    def setUp(self):
        
        pass

    def test_ptdf(self):

        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case)
            if net.num_buses > 300:
                continue

            print(case)

            ptdf = pf.utils.make_PTDF(net)

            break
