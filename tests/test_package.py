#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

import pfnet as pf
import unittest
from . import test_cases
import numpy as np

class TestPackage(unittest.TestCase):
    
    def setUp(self):
        
        pass

    def test_has_parsers(self):

        self.assertTrue(hasattr(pf, 'has_raw_parser'))
        self.assertTrue(hasattr(pf, 'has_epc_parser'))

    def test_version(self):

        self.assertTrue(hasattr(pf, '__version__'))
