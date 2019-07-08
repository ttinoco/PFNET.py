import pfnet
import numpy as np
from scipy.sparse import bmat, eye, coo_matrix
from scipy.sparse.linalg import spsolve

def make_PTDF(net):
    """
    Constructs matrix of DC participation factors.
    Handles in- and out-of-service buses and branches.
    
    Parameters
    ----------
    net : |Network|

    Returns
    -------
    PTDF : |Array| (to be accessed by PTDF[branch.index, bus.index])
    """

    net = net.get_copy()

    net.clear_flags()

    net.set_flags('bus', 'variable', 'any', 'voltage angle')
    net.set_flags('bus', 'fixed', 'slack', 'voltage angle')
    net.set_flags('generator', 'variable', 'slack', 'active power')

    c1 = pfnet.Constraint('DC power balance', net)
    c2 = pfnet.Constraint('generator active power participation', net)
    c3 = pfnet.Constraint('variable fixing', net)

    c1.analyze()
    c2.analyze()
    c3.analyze()
    
    A = bmat([[c1.A], [c2.A], [c3.A]], format='csc')
    I = eye(c1.A.shape[0], net.get_num_buses(True))
    O = coo_matrix((c2.A.shape[0]+c3.A.shape[0], net.get_num_buses(True)))
    P = coo_matrix((np.ones(net.get_num_buses(True)),
                    (zip(*[(bus.dP_index, bus.index) for bus in net.buses if bus.is_in_service()]))),
                   shape=(net.get_num_buses(True), net.get_num_buses(False)))
    Ibar = bmat([[I], [O]], format='csc')

    Erows = []
    Ecols = []
    Edata = []
    for branch in net.branches:
        if branch.is_in_service():
            Erows.extend([branch.index, branch.index])
            Ecols.extend([branch.bus_k.index_v_ang, branch.bus_m.index_v_ang])
            Edata.extend([branch.b, -branch.b])
    E = coo_matrix((Edata, (Erows, Ecols)), shape=(net.get_num_branches(False), net.num_vars))
    
    PTDF = (E*spsolve(A, Ibar)*P).toarray()

    return PTDF
    
