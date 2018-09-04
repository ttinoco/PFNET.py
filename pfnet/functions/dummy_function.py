#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

import numpy as np
from pfnet import CustomFunction
from scipy.sparse import coo_matrix

class DummyGenCost(CustomFunction):

    def init(self):
        
        self.name = "dummy generation cost"
        
    def count_step(self, bus, busdc, t):

        if bus is None:
            return

        for gen in bus.generators:
            if gen.has_flags('variable','active power') and not gen.is_on_outage():
                self.Hphi_nnz = self.Hphi_nnz+1
                
    def analyze_step(self, bus, busdc, t):

        if bus is None:
            return
        
        for gen in bus.generators:
            if gen.has_flags('variable','active power') and not gen.is_on_outage():
                self.Hphi.row[self.Hphi_nnz] = gen.index_P[t]
                self.Hphi.col[self.Hphi_nnz] = gen.index_P[t]
                self.Hphi_nnz = self.Hphi_nnz+1

    def eval_step(self, bus, busdc, t, x):

        if bus is None:
            return

        for gen in bus.generators:
            Q0 = gen.cost_coeff_Q0
            Q1 = gen.cost_coeff_Q1
            Q2 = gen.cost_coeff_Q2
            if gen.has_flags('variable','active power') and not gen.is_on_outage():
                P = x[gen.index_P[t]]
                self.gphi[gen.index_P[t]] = Q1+2.*Q2*P
                self.Hphi.data[self.Hphi_nnz] = 2.*Q2
                self.Hphi_nnz = self.Hphi_nnz + 1
            else:
                P = gen.P[t]
            self.phi = self.phi + Q0+Q1*P+Q2*(P**2.)
