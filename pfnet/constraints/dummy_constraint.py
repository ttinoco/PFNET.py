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

        if not bus.is_in_service():
            return

        bus.set_dP_index(self.A_row, t)
                
        for gen in bus.generators:
            if not gen.is_in_service():
                continue
            if gen.has_flags('variable','active power'):
                self.A_nnz = self.A_nnz+1
        for load in bus.loads:
            if not load.is_in_service():
                continue
            if load.has_flags('variable','active power'):
                self.A_nnz = self.A_nnz+1
        for vargen in bus.var_generators:
            if not vargen.is_in_service():
                continue
            if vargen.has_flags('variable','active power'):
                self.A_nnz = self.A_nnz+1
        for bat in bus.batteries:
            if not bat.is_in_service():
                continue
            if bat.has_flags('variable','charging power'):
                self.A_nnz = self.A_nnz+2
                
        for branch in bus.branches_k:
            if not branch.is_in_service():
                continue
            buses = [branch.bus_k, branch.bus_m]
            if buses[0].has_flags('variable','voltage angle'):
                self.A_nnz = self.A_nnz+1
            if buses[1].has_flags('variable','voltage angle'):
                self.A_nnz = self.A_nnz+1
            if branch.has_flags('variable','phase shift'):
                self.A_nnz = self.A_nnz+1

        for branch in bus.branches_m:
            if not branch.is_in_service():
                continue
            buses = [branch.bus_m, branch.bus_k]
            if buses[0].has_flags('variable','voltage angle'):
                self.A_nnz = self.A_nnz+1
            if buses[1].has_flags('variable','voltage angle'):
                self.A_nnz = self.A_nnz+1
            if branch.has_flags('variable','phase shift'):
                self.A_nnz = self.A_nnz+1

        self.A_row = self.A_row+1

    def analyze_step(self, bus, busdc, t):

        if bus is None:
            return

        if not bus.is_in_service():
            return
            
        for gen in bus.generators:
            if not gen.is_in_service():
                continue
            if gen.has_flags('variable','active power'):
                self.A.row[self.A_nnz] = self.A_row
                self.A.col[self.A_nnz] = gen.index_P[t]
                self.A.data[self.A_nnz] = 1.
                self.A_nnz = self.A_nnz+1
            else:
                self.b[self.A_row] += -gen.P[t]
        for load in bus.loads:
            if not load.is_in_service():
                continue
            if load.has_flags('variable','active power'):
                self.A.row[self.A_nnz] = self.A_row
                self.A.col[self.A_nnz] = load.index_P[t]
                self.A.data[self.A_nnz] = -1.
                self.A_nnz = self.A_nnz+1
            else:
                self.b[self.A_row] += load.P[t]
        for vargen in bus.var_generators:
            if not vargen.is_in_service():
                continue
            if vargen.has_flags('variable','active power'):
                self.A.row[self.A_nnz] = self.A_row
                self.A.col[self.A_nnz] = vargen.index_P[t]
                self.A.data[self.A_nnz] = 1.
                self.A_nnz = self.A_nnz+1
            else:
                self.b[self.A_row] += -vargen.P[t]
        for bat in bus.batteries:
            if not bat.is_in_service():
                continue
            if bat.has_flags('variable','charging power'):
                self.A.row[self.A_nnz] = self.A_row
                self.A.col[self.A_nnz] = bat.index_Pc[t]
                self.A.data[self.A_nnz] = -1.
                self.A_nnz = self.A_nnz+1
                self.A.row[self.A_nnz] = self.A_row
                self.A.col[self.A_nnz] = bat.index_Pd[t]
                self.A.data[self.A_nnz] = 1.
                self.A_nnz = self.A_nnz+1
            else:
                self.b[self.A_row] += bat.P[t]

        for branch in bus.branches_k:
            if not branch.is_in_service():
                continue
            buses = [branch.bus_k, branch.bus_m]
            if buses[0].has_flags('variable','voltage angle'):
                self.A.row[self.A_nnz] = self.A_row
                self.A.col[self.A_nnz] = buses[0].index_v_ang[t]
                self.A.data[self.A_nnz] = branch.b
                self.A_nnz = self.A_nnz+1
            else:
                self.b[self.A_row] += -branch.b*buses[0].v_ang[t]
            if buses[1].has_flags('variable','voltage angle'):
                self.A.row[self.A_nnz] = self.A_row
                self.A.col[self.A_nnz] = buses[1].index_v_ang[t]
                self.A.data[self.A_nnz] = -branch.b
                self.A_nnz = self.A_nnz+1
            else:
                self.b[self.A_row] += branch.b*buses[1].v_ang[t]
            if branch.has_flags('variable','phase shift'):
                self.A.row[self.A_nnz] = self.A_row
                self.A.col[self.A_nnz] = branch.index_phase[t]
                self.A.data[self.A_nnz] = -branch.b
                self.A_nnz = self.A_nnz+1
            else:
                self.b[self.A_row] += branch.b*branch.phase[t]

        for branch in bus.branches_m:
            if not branch.is_in_service():
                continue
            buses = [branch.bus_m, branch.bus_k]
            if buses[0].has_flags('variable','voltage angle'):
                self.A.row[self.A_nnz] = self.A_row
                self.A.col[self.A_nnz] = buses[0].index_v_ang[t]
                self.A.data[self.A_nnz] = branch.b
                self.A_nnz = self.A_nnz+1
            else:
                self.b[self.A_row] += -branch.b*buses[0].v_ang[t]
            if buses[1].has_flags('variable','voltage angle'):
                self.A.row[self.A_nnz] = self.A_row
                self.A.col[self.A_nnz] = buses[1].index_v_ang[t]
                self.A.data[self.A_nnz] = -branch.b
                self.A_nnz = self.A_nnz+1
            else:
                self.b[self.A_row] += branch.b*buses[1].v_ang[t]
            if branch.has_flags('variable','phase shift'):
                self.A.row[self.A_nnz] = self.A_row
                self.A.col[self.A_nnz] = branch.index_phase[t]
                self.A.data[self.A_nnz] = branch.b
                self.A_nnz = self.A_nnz+1
            else:
                self.b[self.A_row] += -branch.b*branch.phase[t]
                    
        self.A_row = self.A_row+1

    def eval_step(self, bus, busdc, t, x, y):
 
        pass
        
    def store_sens_step(self, bus, busdc, t, sA, sf, sGu, sGl):
        
        pass
        
