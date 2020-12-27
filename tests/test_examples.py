import os
import sys
import unittest
from subprocess import call, STDOUT

class TestExamples(unittest.TestCase):

    PYTHON = "python2" if sys.version_info[2] < 3 else "python3"

    def test_constraints(self):
        
        FNULL = open(os.devnull, 'w')
        retcode = call([self.PYTHON, "./examples/constraints.py", "./data/ieee14.m"],
                       stdout=FNULL,
                       stderr=STDOUT)
        
        self.assertEqual(retcode, 0)

    def test_contingencies(self):
        
        FNULL = open(os.devnull, 'w')
        retcode = call([self.PYTHON, "./examples/contingencies.py", "./data/ieee14.m"],
                       stdout=FNULL,
                       stderr=STDOUT)
        
        self.assertEqual(retcode, 0)

    def test_functions(self):
        
        FNULL = open(os.devnull, 'w')
        retcode = call([self.PYTHON, "./examples/functions.py", "./data/ieee14.m"],
                       stdout=FNULL,
                       stderr=STDOUT)
        
        self.assertEqual(retcode, 0)

    def test_muti_period(self):
        
        FNULL = open(os.devnull, 'w')
        retcode = call([self.PYTHON, "./examples/multi_period.py", "./data/ieee14.m"],
                       stdout=FNULL,
                       stderr=STDOUT)
        
        self.assertEqual(retcode, 0)

    def test_networks(self):
        
        FNULL = open(os.devnull, 'w')
        retcode = call([self.PYTHON, "./examples/networks.py", "./data/ieee14.m"],
                       stdout=FNULL,
                       stderr=STDOUT)
        
        self.assertEqual(retcode, 0)

    def test_parsers(self):
        
        FNULL = open(os.devnull, 'w')
        retcode = call([self.PYTHON, "./examples/parsers.py", "./data/ieee14.m"],
                       stdout=FNULL,
                       stderr=STDOUT)
        
        self.assertEqual(retcode, 0)

    def test_problems(self):
        
        FNULL = open(os.devnull, 'w')
        retcode = call([self.PYTHON, "./examples/problems.py", "./data/ieee14.m"],
                       stdout=FNULL,
                       stderr=STDOUT)
        
        self.assertEqual(retcode, 0)

    def test_projections(self):
        
        FNULL = open(os.devnull, 'w')
        retcode = call([self.PYTHON, "./examples/projections.py", "./data/ieee14.m"],
                       stdout=FNULL,
                       stderr=STDOUT)
        
        self.assertEqual(retcode, 0)

    def test_start(self):
        
        FNULL = open(os.devnull, 'w')
        retcode = call([self.PYTHON, "./examples/start.py", "./data/ieee14.m"],
                       stdout=FNULL,
                       stderr=STDOUT)
        
        self.assertEqual(retcode, 0)

    def test_variables(self):
        
        FNULL = open(os.devnull, 'w')
        retcode = call([self.PYTHON, "./examples/variables.py", "./data/ieee14.m"],
                       stdout=FNULL,
                       stderr=STDOUT)
        
        self.assertEqual(retcode, 0)
