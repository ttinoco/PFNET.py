# PSO to solve the PF problem

import pfnet
import numpy as np

from numpy import hstack
from numpy.linalg import norm
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve

#Import Case
case = pfnet.PyParserRAW().parse('C:\\Users\\Barberia Juan Luis\\Desktop\\PFNET_Archives\\IEEE 39 bus.raw')

case.clear_flags()

# bus voltage angles
case.set_flags('bus',
               'variable',
               'not slack',
               'voltage angle')
    
# bus voltage magnitudes
case.set_flags('bus',
               'variable',
               'not regulated by generator',
               'voltage magnitude')
    
# slack gens active powers
case.set_flags('generator',
               'variable',
               'slack',
               'active power')
    
# regulator gens reactive powers
case.set_flags('generator',
               'variable',
               'regulator',
               'reactive power')

#Problem Formulation

p = pfnet.Problem(case)
p.add_constraint(pfnet.Constraint('AC power balance', case))  
p.add_constraint(pfnet.Constraint('generator active power participation', case))
p.add_constraint(pfnet.Constraint('PVPQ switching', case))
p.add_heuristic(pfnet.Heuristic('PVPQ switching', case))
p.analyze()

x = p.get_init_point()
p.eval(x)


residual = lambda x: hstack((p.A*x-p.b, p.f))

    
p.apply_heuristics(x)
x = x + spsolve(bmat([[p.A],[p.J]],format='csr'), -residual(x))
p.eval(x)

case.set_var_values(x)
case.update_properties()







class Particle():
    
    def __init__(self,case,problem):
        
        self.case    = case
        self.p       = problem
        self.x       = p.get_init_point()*(0.2*np.random.rand(1)-0.1)
        self.x_best  = 0
        
    def crossover(self, other):
        
        x1 = np.array(self.x)
        x2 = np.array(other.x)
        
        x = (x1+x2)/2.
        
        self.x = x
        
        
        
        
        
        
        
        
        
        

