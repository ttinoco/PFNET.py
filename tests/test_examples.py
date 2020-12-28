import os
import sys
import unittest
import contextlib

class TestExamples(unittest.TestCase):

    def test_constraints(self):

        import examples.constraints
        with contextlib.redirect_stdout(None):
            examples.constraints.main(["./data/ieee14.m"])
            
    def test_contingencies(self):
        
        import examples.contingencies
        with contextlib.redirect_stdout(None):
            examples.contingencies.main(["./data/ieee14.m"])

    def test_functions(self):
        
        import examples.functions
        with contextlib.redirect_stdout(None):
            examples.functions.main(["./data/ieee14.m"])

    def test_muti_period(self):
        
        import examples.multi_period
        with contextlib.redirect_stdout(None):
            examples.multi_period.main(["./data/ieee14.m"])

    def test_networks(self):
        
        import examples.networks
        with contextlib.redirect_stdout(None):
            examples.networks.main(["./data/ieee14.m"])
    
    def test_parsers(self):
        
        import examples.parsers
        with contextlib.redirect_stdout(None):
            examples.parsers.main(["./data/ieee14.m"])

    def test_problems(self):
        
        import examples.problems
        with contextlib.redirect_stdout(None):
            examples.problems.main(["./data/ieee14.m"])

    def test_projections(self):
        
        import examples.projections
        with contextlib.redirect_stdout(None):
            examples.projections.main(["./data/ieee14.m"])

    def test_start(self):
        
        import examples.start
        with contextlib.redirect_stdout(None):
            examples.start.main(["./data/ieee14.m"])

    def test_variables(self):
        
        import examples.variables
        with contextlib.redirect_stdout(None):
            examples.variables.main(["./data/ieee14.m"])
