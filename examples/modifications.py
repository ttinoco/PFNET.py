#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

import sys
import pfnet

net = pfnet.Parser(sys.argv[1]).parse(sys.argv[1])

bus = net.buses[8]

print(len(bus.generators), len(bus.loads))

gen = bus.generators[0]
load = bus.loads[0]

print(gen.bus == bus, load.bus == bus)

gen.bus = None
bus.remove_load(load)

print(len(bus.generators), len(bus.loads))

bus.add_generator(gen)
load.bus = bus

print(len(bus.generators), len(bus.loads))
print(gen.bus == bus, load.bus == bus)

new_gen = pfnet.Generator()
new_gen.bus = bus

net.add_generators([new_gen])

print(new_gen == net.generators[-1])

print(len(bus.generators), len(bus.loads))

net.remove_generators([new_gen])

print(len(bus.generators), len(bus.loads))
