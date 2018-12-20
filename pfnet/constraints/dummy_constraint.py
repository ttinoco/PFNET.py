#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

import numpy as np
from scipy.sparse import coo_matrix
from pfnet import CustomConstraint

class DummyDCPF(CustomConstraint):

    def init(self):
        
        self.name = "dummy DC power balance"

    def count_step(self, bus, busdc, t):

        if bus is None:
            return
                
        for gen in bus.generators:
            if gen.is_on_outage():
                continue
            if gen.has_flags('variable','active power'):
                self.A_nnz = self.A_nnz+1
        for load in bus.loads:
            if load.has_flags('variable','active power'):
                self.A_nnz = self.A_nnz+1
        for vargen in bus.var_generators:
            if vargen.has_flags('variable','active power'):
                self.A_nnz = self.A_nnz+1
        for bat in bus.batteries:
            if bat.has_flags('variable','charging power'):
                self.A_nnz = self.A_nnz+2
                
        for branch in bus.branches_k:
            if branch.is_on_outage():
                continue
            buses = [branch.bus_k, branch.bus_m]
            for k in range(2):
                m = 1 if k == 0 else 0
                if buses[k].has_flags('variable','voltage angle'):
                    self.A_nnz = self.A_nnz+1
                if buses[m].has_flags('variable','voltage angle'):
                    self.A_nnz = self.A_nnz+1
                if branch.has_flags('variable','phase shift'):
                    self.A_nnz = self.A_nnz+1

        self.A_row = self.A_row+1

    def analyze_step(self, bus, busdc, t):

        if bus is None:
            return
            
        for gen in bus.generators:
            if gen.is_on_outage():
                continue
            if gen.has_flags('variable','active power'):
                self.A.row[self.A_nnz] = bus.index_P[t]
                self.A.col[self.A_nnz] = gen.index_P[t]
                self.A.data[self.A_nnz] = 1.
                self.A_nnz = self.A_nnz+1
            else:
                self.b[bus.index_P[t]] += -gen.P[t]
        for load in bus.loads:
            if load.has_flags('variable','active power'):
                self.A.row[self.A_nnz] = bus.index_P[t]
                self.A.col[self.A_nnz] = load.index_P[t]
                self.A.data[self.A_nnz] = -1.
                self.A_nnz = self.A_nnz+1
            else:
                self.b[bus.index_P[t]] += load.P[t]
        for vargen in bus.var_generators:
            if vargen.has_flags('variable','active power'):
                self.A.row[self.A_nnz] = bus.index_P[t]
                self.A.col[self.A_nnz] = vargen.index_P[t]
                self.A.data[self.A_nnz] = 1.
                self.A_nnz = self.A_nnz+1
            else:
                self.b[bus.index_P[t]] += -vargen.P[t]
        for bat in bus.batteries:
            if bat.has_flags('variable','charging power'):
                self.A.row[self.A_nnz] = bus.index_P[t]
                self.A.col[self.A_nnz] = bat.index_Pc[t]
                self.A.data[self.A_nnz] = -1.
                self.A_nnz = self.A_nnz+1
                self.A.row[self.A_nnz] = bus.index_P[t]
                self.A.col[self.A_nnz] = bat.index_Pd[t]
                self.A.data[self.A_nnz] = 1.
                self.A_nnz = self.A_nnz+1
            else:
                self.b[bus.index_P[t]] += bat.P[t]

        for branch in bus.branches_k:
            if branch.is_on_outage():
                continue
            buses = [branch.bus_k, branch.bus_m]
            for k in range(2):
                m = 1 if k == 0 else 0
                sign_phi = 1. if k == 0 else -1.
                if buses[k].has_flags('variable','voltage angle'):
                    self.A.row[self.A_nnz] = buses[k].index_P[t]
                    self.A.col[self.A_nnz] = buses[k].index_v_ang[t]
                    self.A.data[self.A_nnz] = branch.b
                    self.A_nnz = self.A_nnz+1
                else:
                    self.b[buses[k].index_P[t]] += -branch.b*buses[k].v_ang[t]
                if buses[m].has_flags('variable','voltage angle'):
                    self.A.row[self.A_nnz] = buses[k].index_P[t]
                    self.A.col[self.A_nnz] = buses[m].index_v_ang[t]
                    self.A.data[self.A_nnz] = -branch.b
                    self.A_nnz = self.A_nnz+1
                else:
                    self.b[buses[k].index_P[t]] += branch.b*buses[m].v_ang[t]
                if branch.has_flags('variable','phase shift'):
                    self.A.row[self.A_nnz] = buses[k].index_P[t]
                    self.A.col[self.A_nnz] = branch.index_phase[t]
                    self.A.data[self.A_nnz] = -branch.b*sign_phi
                    self.A_nnz = self.A_nnz+1
                else:
                    self.b[buses[k].index_P[t]] += branch.b*branch.phase[t]*sign_phi
                    
        self.A_row = self.A_row+1

    def eval_step(self, bus, busdc, t, x, y):
 
        pass
        
    def store_sens_step(self, bus, busdc, t, sA, sf, sGu, sGl):
        
        pass
        
