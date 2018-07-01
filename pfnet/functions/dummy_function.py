#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

import numpy as np
from scipy.sparse import coo_matrix
from pfnet import CustomFunction

class DummyGenCost(CustomFunction):

    def init(self):
        
        self.name = "dummy generation cost"
    
    def count_step(self, branch, t):

        buses = [branch.bus_k, branch.bus_m]
        
        if branch.is_on_outage():
            return

        for bus in buses:
            index = bus.index_t[t]
            if not self.bus_counted[index]:
                for gen in bus.generators:
                    if gen.has_flags('variable','active power'):
                        self.Hphi_nnz = self.Hphi_nnz+1
            self.bus_counted[index] = True
                
    def analyze_step(self,branch,t):
        
        buses = [branch.bus_k, branch.bus_m]
        
        if branch.is_on_outage():
            return

        for bus in buses:
            index = bus.index_t[t]
            if not self.bus_counted[index]:
                for gen in bus.generators:
                    if gen.has_flags('variable','active power'):
                        self.Hphi.row[self.Hphi_nnz] = gen.index_P[t]
                        self.Hphi.col[self.Hphi_nnz] = gen.index_P[t]
                        self.Hphi_nnz = self.Hphi_nnz+1
            self.bus_counted[index] = True

    def eval_step(self,branch,t,x):

        buses = [branch.bus_k, branch.bus_m]
        
        if branch.is_on_outage():
            return

        for bus in buses:
            index = bus.index_t[t]
            if not self.bus_counted[index]:
                for gen in bus.generators:
                    Q0 = gen.cost_coeff_Q0
                    Q1 = gen.cost_coeff_Q1
                    Q2 = gen.cost_coeff_Q2
                    if gen.has_flags('variable','active power'):
                        P = x[gen.index_P[t]]
                        self.gphi[gen.index_P[t]] = Q1+2.*Q2*P
                        self.Hphi.data[self.Hphi_nnz] = 2.*Q2
                        self.Hphi_nnz = self.Hphi_nnz + 1
                    else:
                        P = gen.P[t]
                    self.phi = self.phi + Q0+Q1*P+Q2*(P**2.)
            self.bus_counted[index] = True
