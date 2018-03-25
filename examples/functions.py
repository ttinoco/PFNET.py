#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

# Optimization Problems - Functions

import sys
sys.path.append('.')
import pfnet

net = pfnet.Parser(sys.argv[1]).parse(sys.argv[1])

net.set_flags('bus',
              'variable',
              'any',
              'voltage magnitude')

print(net.num_vars == net.num_buses)

func = pfnet.Function('voltage magnitude regularization',0.3,net)

print(func.name == 'voltage magnitude regularization')

print(func.weight)

x = net.get_var_values()

func.analyze()
func.eval(x)

print(x.shape)

print(func.phi)

print(type(func.gphi), func.gphi.shape)

print(type(func.Hphi), func.Hphi.shape)

print(func.Hphi)

# Set parameters
import numpy as np

func = pfnet.Function('variable regularization', 1., net)
w = np.random.randn(net.num_vars)
x0 = np.random.randn(net.num_vars)
func.set_parameter('w', w)
func.set_parameter('x0', x0)
x = np.random.randn(net.num_vars)
func.analyze()
func.eval(x)
print(func.phi)



