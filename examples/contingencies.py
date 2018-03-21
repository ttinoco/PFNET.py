#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

# Power Networks - Contingencies

import sys
sys.path.append('.')
import pfnet

net = pfnet.Parser(sys.argv[1]).parse(sys.argv[1])

net.clear_outages()

gen = net.get_generator(3)
branch = net.get_branch(2)

gen.outage = True
branch.outage = True

print(net.get_num_generators_on_outage(), net.get_num_branches_on_outage())

net.clear_outages()

gen = net.get_generator(3)
branch = net.get_branch(2)

c1 = pfnet.Contingency(generators=[gen],branches=[branch])

print(c1.num_generator_outages, c1.num_branch_outages)

print(c1.has_generator_outage(gen), c1.has_branch_outage(branch))

print(gen.is_on_outage(), branch.is_on_outage())

print(c1.outages)

c1.apply(net)

print(gen.is_on_outage(), branch.is_on_outage())

c1.clear(net)

print(gen.is_on_outage(), branch.is_on_outage())
