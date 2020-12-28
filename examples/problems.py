#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

# Optimization Problems - Problems

import sys
import pfnet
from examples.power_flow import NRsolve

def main(args=None):
    
    if args is None:
        args = sys.argv[1:]
        
    net = pfnet.Parser(args[0]).parse(args[0])

    print('%.2e %.2e' %(net.bus_P_mis, net.bus_Q_mis))

    NRsolve(net)
    
    print('%.2e %.2e' %(net.bus_P_mis, net.bus_Q_mis))
    
if __name__ == "__main__":
    main()
