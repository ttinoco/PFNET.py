import pfnet
import numpy as np

parser=pfnet.PyParserRAW()


#net=parser.parse('C:\\Users\\Barberia Juan Luis\\Desktop\\PFNET_Archives\\two_area_case.raw')
#net=parser.parse('C:\\Users\\Barberia Juan Luis\\Desktop\\PFNET_Archives\\tap_changer_trafo.raw')
#net=parser.parse('C:\\Users\\Barberia Juan Luis\\Desktop\\3w_trafo.raw')
#net=parser.parse('C:\\Users\\Barberia Juan Luis\\Desktop\\PFNET_Archives\\case_trafo.raw')
#net=parser.parse('C:\\Users\\Barberia Juan Luis\\Desktop\\PFNET_Archives\\Brazilian_7_bus_Equiv_Model.raw')
#net=parser.parse('C:\\Users\\Barberia Juan Luis\\Desktop\\PFNET_Archives\\WSCC 9 bus.raw')
#net=parser.parse('C:\\Users\\Barberia Juan Luis\\Desktop\\PFNET_Archives\\IEEE_14_bus.raw')
#net=parser.parse('C:\\Users\\Barberia Juan Luis\\Desktop\\PFNET_Archives\\IEEE 30 bus.raw')
#net=parser.parse('C:\\Users\\Barberia Juan Luis\\Desktop\\PFNET_Archives\\IEEE 39 bus.raw')
#net=parser.parse('C:\\Users\\Barberia Juan Luis\\Desktop\\PFNET_Archives\\IEEE 118 bus.raw')
#net=parser.parse('C:\\Users\\Barberia Juan Luis\\Desktop\\PFNET_Archives\\ver1718pid.raw')
#net=parser.parse('C:\\Users\\Barberia Juan Luis\\Desktop\\PFNET_Archives\\Texas2000_June2016.raw')
net=parser.parse('C:\\Users\\Barberia Juan Luis\\Desktop\\PFNET_Archives\\SouthCarolina500.raw')

from numpy import hstack
from numpy.linalg import norm
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve

def NRsolve(net):

    net.clear_flags()

    # bus voltage angles
    net.set_flags('bus',
                  'variable',
                  'not slack',
                  'voltage angle')
    
    # bus voltage magnitudes
    net.set_flags('bus',
                  'variable',
                  'not regulated by generator',
                  'voltage magnitude')
    
    # slack gens active powers
    net.set_flags('generator',
                  'variable',
                  'slack',
                  'active power')
    
    # regulator gens reactive powers
    net.set_flags('generator',
                  'variable',
                  'regulator',
                  'reactive power')

    p = pfnet.Problem(net)
    p.add_constraint(pfnet.Constraint('AC power balance', net))  
    p.add_constraint(pfnet.Constraint('generator active power participation', net))
    p.add_constraint(pfnet.Constraint('PVPQ switching', net))
    p.add_heuristic(pfnet.Heuristic('PVPQ switching', net))
    p.analyze()
    
    x = p.get_init_point()
    p.eval(x)

    residual = lambda x: hstack((p.A*x-p.b, p.f))

    while norm(residual(x)) > 1e-4:
        p.apply_heuristics(x)
        x = x + spsolve(bmat([[p.A],[p.J]],format='csr'), -residual(x))
        p.eval(x)

    net.set_var_values(x)
    net.update_properties()



NRsolve(net)
result_p = [GEN.P*net.base_power for GEN in net.generators]
P_max = [GEN.P_max*net.base_power for GEN in net.generators]
P_min = [GEN.P_min*net.base_power for GEN in net.generators]
result_q = [GEN.Q*net.base_power for GEN in net.generators]
result_v_mag = [BUS.v_mag for BUS in net.buses]
result_v_ang = [BUS.v_ang*180/np.pi for BUS in net.buses]
B_shunt=[SHUNT.b*net.base_power for SHUNT in net.shunts]
P_km   =[BRANCH.P_km for BRANCH in net.branches]
P_mk   =[BRANCH.P_mk for BRANCH in net.branches]

#print(result_p)
#print(result_v_mag)
#print(result_v_ang)