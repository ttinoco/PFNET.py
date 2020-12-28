#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

import sys

# Getting Started - Example

import pfnet
import numpy as np

def main(args=None):
    
    if args is None:
        args = sys.argv[1:]

    net = pfnet.PyParserMAT().parse(args[0])

    print(np.average([bus.degree for bus in net.buses]))

if __name__ == "__main__":
    main()
