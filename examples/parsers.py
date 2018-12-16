#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

import os
import sys
sys.path.append('.')

# Parsers - Overview

import pfnet

parser = pfnet.Parser(sys.argv[1])
network = parser.parse(sys.argv[1])

parser_json = pfnet.ParserJSON()
parser_json.write(network, 'new_network.json')
network = parser_json.parse('new_network.json')
os.remove('new_network.json')

parser_mat = pfnet.PyParserMAT()
parser_mat.write(network, 'new_network.m')
network = parser_mat.parse('new_network.m')
os.remove('new_network.m')
