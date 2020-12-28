#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

# Power Networks - Contingencies

import sys
import pfnet

def main(args=None):
    
    if args is None:
        args = sys.argv[1:]

    net = pfnet.Parser(args[0]).parse(args[0])

    gen = net.get_generator(3)
    branch = net.get_branch(2)
    
    gen.in_service = False
    branch.in_service = False
    
    print(net.get_num_generators_out_of_service(), net.get_num_branches_out_of_service())
    
    net.make_all_in_service()
    
    gen = net.get_generator(3)
    branch = net.get_branch(2)
    
    c1 = pfnet.Contingency(generators=[gen],branches=[branch])
    
    print(c1.num_generator_outages, c1.num_branch_outages)
    
    print(c1.outages)
    
    print(gen.is_in_service(), branch.is_in_service())
    
    c1.apply(net)
    
    print(gen.is_in_service(), branch.is_in_service())
    
    c1.clear(net)
    
    print(gen.is_in_service(), branch.is_in_service())
    
if __name__ == "__main__":
    main()
