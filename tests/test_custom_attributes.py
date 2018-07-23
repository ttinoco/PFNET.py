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

class TestCustomAttributes(unittest.TestCase):

    def setUp(self):

        pass

    def test_attribute_int(self):

        x = pf.AttributeInt(5)

        self.assertEqual(x, 5)
        self.assertEqual(x[0], 5)

        a = np.array([x])
        self.assertTrue(np.all(a == [5]))

    def test_attribute_float(self):

        x = pf.AttributeFloat(5.231)

        self.assertEqual(x, 5.231)
        self.assertEqual(x[0], 5.231)

        a = np.array([x])
        self.assertTrue(np.all(a == [5.231]))
        
        
