#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

import os
import unittest
import pfnet as pf
import numpy as np
from . import test_cases
from numpy.linalg import norm
from scipy.sparse import coo_matrix,triu,tril,eye

NUM_TRIALS = 25
EPS = 5.0 # %
TOL = 1e-4

class TestConstraints(unittest.TestCase):

    def setUp(self):

        # Network
        self.T = 2

        # Random
        np.random.seed(0)

    def test_constr_FACTS_EQ(self):

        # Constants
        h = 1e-8

        # Multiperiods
        for case in test_cases.CASES:
            
            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)

            # Vars
            net.set_flags('bus',
                          'variable',
                          'any',
                          ['voltage magnitude', 'voltage angle'])
            net.set_flags('facts',
                          'variable',
                          'any',
                          'all')
            self.assertEqual(net.num_vars, (2*net.num_buses+9*net.num_facts)*self.T)

            x0 = net.get_var_values()+1e-4*np.random.randn(net.num_vars)
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            # Constraint
            constr = pf.Constraint('FACTS equations',net)
            self.assertEqual(constr.name,'FACTS equations')

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b

            # Before
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,0))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,0))
            self.assertEqual(A.nnz,0)
            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.J_row,0)
            self.assertEqual(constr.A_row,0)
            self.assertEqual(constr.num_extra_vars,0)

            num_statcom = len([f for f in net.facts if f.is_STATCOM()])
            num_SSSC = len([f for f in net.facts if f.is_SSSC()])
            num_UPFC = len([f for f in net.facts if f.is_UPFC()])
            num_seriesenabled = len([f for f in net.facts if f.is_in_normal_series_mode()])
            num_seriesdisabled = len([f for f in net.facts if f.is_series_link_disabled()])

            # Verify analyze
            Jnnz = 28*num_seriesenabled;
            rowsJ = 4*num_seriesenabled
            rowsA = 2*net.num_facts
            Annz = 7*net.num_facts
            for facts in net.facts:
                if not facts.is_regulator():
                    rowsA = rowsA+1
                    Annz = Annz+1
                if facts.P_max_dc == 0 or facts.is_series_link_disabled():
                    rowsA = rowsA+1
                    Annz = Annz+1
                if facts.is_series_link_disabled():
                    rowsA = rowsA+5
                    Annz = Annz+5

            constr.analyze()
            self.assertEqual(constr.J_nnz, Jnnz*self.T)
            self.assertEqual(constr.A_nnz, Annz*self.T)
            self.assertEqual(constr.J_row, rowsJ*self.T)
            self.assertEqual(constr.A_row, rowsA*self.T)

            y_init = constr.init_extra_vars
            self.assertEqual(y_init.size,constr.num_extra_vars)
            self.assertTrue(np.all(y_init == 0.))
            
            y0 = np.random.rand(constr.num_extra_vars)
            constr.eval(x0,y0)
            self.assertEqual(constr.J_nnz,Jnnz*self.T)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.J_row,rowsJ*self.T)
            self.assertEqual(constr.A_row,0)

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b

            # After
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(rowsJ*self.T,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(rowsA*self.T,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(rowsJ*self.T,net.num_vars+constr.num_extra_vars))
            self.assertEqual(J.nnz,Jnnz*self.T)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(rowsA*self.T,net.num_vars+constr.num_extra_vars))
            self.assertEqual(A.nnz,Annz*self.T)

            self.assertTrue(not np.any(np.isinf(b)))
            self.assertTrue(not np.any(np.isnan(b)))
            self.assertTrue(not np.any(np.isinf(f)))
            self.assertTrue(not np.any(np.isnan(f)))
            self.assertTrue(not np.any(np.isinf(J.data)))
            self.assertTrue(not np.any(np.isnan(J.data)))
            self.assertTrue(not np.any(np.isinf(A.data)))
            self.assertTrue(not np.any(np.isnan(A.data)))

            # Ax=b check
            for k in range(self.T):
                for facts in net.facts:
                    self.assertTrue(facts.has_flags('variable',
                                                    ['active power',
                                                     'reactive power',
                                                     'series voltage magnitude',
                                                     'series voltage angle']))
                    index_Pk = np.where(A.col == facts.index_P_k[k])[0]
                    index_Pm = np.where(A.col == facts.index_P_m[k])[0]
                    index_Pdc = np.where(A.col == facts.index_P_dc[k])[0]
                    index_Qk = np.where(A.col == facts.index_Q_k[k])[0]
                    index_Qm = np.where(A.col == facts.index_Q_m[k])[0]
                    index_Qsh = np.where(A.col == facts.index_Q_sh[k])[0]
                    index_Qs = np.where(A.col == facts.index_Q_s[k])[0]
                    index_vmags = np.where(A.col == facts.index_v_mag_s[k])[0]
                    index_vangs = np.where(A.col == facts.index_v_ang_s[k])[0]
                    self.assertEqual(index_Pk.size,1)
                    self.assertEqual(index_Qk.size,1)
                    self.assertEqual(A.data[index_Pk],1.)
                    self.assertEqual(A.data[index_Qk],1.)
                    self.assertEqual(b[A.row[index_Pk]],0.)
                    self.assertEqual(b[A.row[index_Qk]],0.)
                    if not facts.is_regulator():
                        self.assertEqual(index_Qsh.size,2)
                        self.assertEqual(A.data[index_Qsh[0]],1.)
                        self.assertEqual(A.data[index_Qsh[1]],-1.)
                        self.assertEqual(b[A.row[index_Qsh[0]]],0.)
                        self.assertEqual(b[A.row[index_Qsh[1]]],0.)
                    if facts.P_max_dc ==0 or facts.is_series_link_disabled():
                        self.assertEqual(index_Pdc.size,2)
                        self.assertEqual(A.data[index_Pdc[0]],-1.)
                        self.assertEqual(A.data[index_Pdc[1]],1.)
                        self.assertEqual(b[A.row[index_Pdc[0]]],0.)
                        self.assertEqual(b[A.row[index_Pdc[1]]],0.)
                    else:
                        self.assertEqual(index_Pdc.size,1)
                        self.assertEqual(A.data[index_Pdc],-1.)
                        self.assertEqual(b[A.row[index_Pdc]],0.)
                    if facts.is_series_link_disabled():
                        self.assertEqual(index_Pm.size,2)
                        for index in index_Pm:
                            self.assertEqual(A.data[index],1.)
                            self.assertEqual(b[A.row[index]],0.)
                        self.assertEqual(index_Qm.size,2)
                        for index in index_Qm:
                            self.assertEqual(A.data[index],1.)
                            self.assertEqual(b[A.row[index]],0.)
                        self.assertEqual(index_Qs.size,2)
                        self.assertEqual(index_vmags.size,1)
                        self.assertEqual(index_vangs.size,1)
                        self.assertEqual(A.data[index_Qs[0]],-1.)
                        self.assertEqual(A.data[index_Qs[1]],1.)
                        self.assertEqual(A.data[index_vmags],1.)
                        self.assertEqual(A.data[index_vangs],1.)
                        self.assertEqual(b[A.row[index_Qs[0]]],0.)
                        self.assertEqual(b[A.row[index_Qs[1]]],0.)
                        self.assertEqual(b[A.row[index_vmags]],0.)
                        self.assertEqual(b[A.row[index_vangs]],0.)

            # f check
            flags = {}
            for t in range(self.T):
                for bus in net.buses:
                    flags[(t,bus.index)] = False
            J_row = 0
            for t in range(self.T):
                for branch in net.branches:
                    for bus in [branch.bus_k, branch.bus_m]:
                        if not flags[(t, bus.index)]:
                            facts_onthisbus = [facts for facts in net.facts if ((facts.bus_k == bus) and (facts.is_in_normal_series_mode()))]
                            for facts in facts_onthisbus:
                                busk = facts.bus_k
                                busm = facts.bus_m
                                vmag_k = x0[busk.index_v_mag[t]]
                                vang_k = x0[busk.index_v_ang[t]]
                                vmag_m = x0[busm.index_v_mag[t]]
                                vang_m = x0[busm.index_v_ang[t]]
                                vmag_s = x0[facts.index_v_mag_s[t]]
                                vang_s = x0[facts.index_v_ang_s[t]]
                                P_m = x0[facts.index_P_m[t]]
                                P_dc = x0[facts.index_P_dc[t]]
                                Q_m = x0[facts.index_Q_m[t]]
                                Q_s = x0[facts.index_Q_s[t]]
                                f1 = -vmag_k*np.cos(vang_k)+vmag_m*np.cos(vang_m)-vmag_s*np.cos(vang_s)
                                f2 = -vmag_k*np.sin(vang_k)+vmag_m*np.sin(vang_m)-vmag_s*np.sin(vang_s)
                                f3 = vmag_s*P_m*np.cos(vang_s)-vmag_s*Q_m*np.sin(vang_s)-vmag_m*P_dc*np.cos(vang_m)+vmag_m*Q_s*np.sin(vang_m)
                                f4 = vmag_s*P_m*np.sin(vang_s)+vmag_s*Q_m*np.cos(vang_s)-vmag_m*P_dc*np.sin(vang_m)-vmag_m*Q_s*np.cos(vang_m)
                                self.assertAlmostEqual(f1,f[J_row])
                                self.assertAlmostEqual(f2,f[J_row+1])
                                self.assertAlmostEqual(f3,f[J_row+2])
                                self.assertAlmostEqual(f4,f[J_row+3])
                                J_row += 4
                        flags[(t,bus.index)] = True

            # Jacobian check
            pf.tests.utils.check_constraint_Jacobian(self,
                                                     constr,
                                                     x0,
                                                     np.zeros(0),
                                                     NUM_TRIALS,
                                                     TOL,
                                                     EPS,
                                                     h)

            # Sigle Hessian check
            pf.tests.utils.check_constraint_single_Hessian(self,
                                                           constr,
                                                           x0,
                                                           np.zeros(0),
                                                           NUM_TRIALS,
                                                           TOL,
                                                           EPS,
                                                           h)

            # Combined Hessian check
            pf.tests.utils.check_constraint_combined_Hessian(self,
                                                             constr,
                                                             x0,
                                                             np.zeros(0),
                                                             NUM_TRIALS,
                                                             TOL,
                                                             EPS,
                                                             h)

    def test_constr_FACTS_PSET_SWITCH(self):

        # Multiperiods
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)

            # Vars
            net.set_flags('facts',
                          'variable',
                          'any',
                          'active power')

            self.assertEqual(net.num_vars, 3*net.num_facts*self.T)

            # Constraint
            constr = pf.Constraint('switching FACTS active power control',net)
            self.assertEqual(constr.name,'switching FACTS active power control')

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b

            # Before
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,0))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,0))
            self.assertEqual(A.nnz,0)
            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.J_row,0)
            self.assertEqual(constr.A_row,0)
            self.assertEqual(constr.num_extra_vars,0)

            x0 = net.get_var_values()
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            # Verify analyze
            constr.analyze()
            num = len([f for f in net.facts if f.is_in_normal_series_mode() and f.P_max_dc > 0.])
            Annz = num*self.T
            Arow = Annz
            self.assertEqual(constr.A_nnz,Annz)
            self.assertEqual(constr.A_row,Arow)

            # Verify evaluation
            constr.eval(x0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.A_row,0)

            f = constr.f
            J = constr.J
            b = constr.b
            A = constr.A

            # After
            self.assertTrue(not np.any(np.isinf(b)))
            self.assertTrue(not np.any(np.isnan(b)))
            self.assertTrue(not np.any(np.isinf(f)))
            self.assertTrue(not np.any(np.isnan(f)))
            self.assertTrue(not np.any(np.isinf(J.data)))
            self.assertTrue(not np.any(np.isnan(J.data)))
            self.assertTrue(not np.any(np.isinf(A.data)))
            self.assertTrue(not np.any(np.isnan(A.data)))

            # Ax = b Check
            for t in range(self.T):
                for f in net.facts:
                    self.assertTrue(f.has_flags('variable', 'active power'))
                    if f.is_in_normal_series_mode() and f.P_max_dc > 0.:
                        indexP = np.where(A.col == f.index_P_m[t])[0]
                        self.assertEqual(indexP.size,1)
                        self.assertEqual(A.data[indexP],1)
                        self.assertEqual(b[A.row[indexP]],f.P_set[t])

    def test_constr_FACTS_QSET_SWITCH(self):

        # Multiperiods
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)
            num_vsc = net.num_vsc_converters
            if num_vsc == 0:
                continue

            # Vars
            net.set_flags('facts',
                          'variable',
                          'any',
                          'reactive power')

            self.assertEqual(net.num_vars, 4*net.num_facts*self.T)

            # Constraint
            constr = pf.Constraint('switching FACTS reactive power control',net)
            self.assertEqual(constr.name,'switching FACTS reactive power control')

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b

            # Before
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,0))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,0))
            self.assertEqual(A.nnz,0)
            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.J_row,0)
            self.assertEqual(constr.A_row,0)
            self.assertEqual(constr.num_extra_vars,0)

            x0 = net.get_var_values()
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            # Verify analyze
            constr.analyze()
            num = len([f for f in net.facts if f.is_in_normal_series_mode()])
            Annz = num*self.T
            Arow = Annz
            self.assertEqual(constr.A_nnz,Annz)
            self.assertEqual(constr.A_row,Arow)

            # Verify evaluation
            constr.eval(x0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.A_row,0)

            f = constr.f
            J = constr.J
            b = constr.b
            A = constr.A

            # After
            self.assertTrue(not np.any(np.isinf(b)))
            self.assertTrue(not np.any(np.isnan(b)))
            self.assertTrue(not np.any(np.isinf(f)))
            self.assertTrue(not np.any(np.isnan(f)))
            self.assertTrue(not np.any(np.isinf(J.data)))
            self.assertTrue(not np.any(np.isnan(J.data)))
            self.assertTrue(not np.any(np.isinf(A.data)))
            self.assertTrue(not np.any(np.isnan(A.data)))

            # Ax = b Check
            for t in range(self.T):
                for f in net.facts:
                    self.assertTrue(f.has_flags('variable', 'reactive power'))
                    if f.is_in_normal_series_mode():
                        indexP = np.where(A.col == f.index_Q_m[t])[0]
                        self.assertEqual(indexP.size,1)
                        self.assertEqual(A.data[indexP],1)
                        self.assertEqual(b[A.row[indexP]],f.Q_set[t])

    def test_constr_REG_PF(self):

        # Constants
        h = 1e-8

        # Multiperiod
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)

            # Vars
            net.set_flags('vsc converter',
                          'variable',
                          'any',
                          ['active power', 'reactive power'])
            self.assertEqual(net.num_vars, 2*net.get_num_vsc_converters()*self.T)

            x0 = net.get_var_values()
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            # PF
            for vsc in net.vsc_converters:
                if vsc.is_in_f_ac_mode():
                    vsc.target_power_factor = np.sign(np.random.randn())*np.minimum(np.random.rand(), 0.2)
                    
            # Constraint
            constr = pf.Constraint('power factor regulation',net)
            self.assertEqual(constr.name,'power factor regulation')

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b

            # Before
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,0))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,0))
            self.assertEqual(A.nnz,0)

            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.J_row,0)
            self.assertEqual(constr.A_row,0)
            self.assertEqual(constr.num_extra_vars,0)

            Jnnz = 0
            for i in range(net.num_buses):
                bus = net.get_bus(i)
                for vsc in bus.vsc_converters:
                    if vsc.is_in_f_ac_mode():
                        Jnnz += 4

            Annz = 4*net.get_num_vsc_converters_in_f_ac_mode()

            rowsJ = 2*net.get_num_vsc_converters_in_f_ac_mode()
            rowsA = net.get_num_vsc_converters_in_f_ac_mode()

            constr.analyze()
            self.assertEqual(constr.J_nnz, Jnnz*self.T)
            self.assertEqual(constr.A_nnz, Annz*self.T)
            self.assertEqual(constr.J_row, rowsJ*self.T)
            self.assertEqual(constr.A_row, rowsA*self.T)
            self.assertEqual(constr.num_extra_vars, rowsJ*self.T)

            y_init = constr.init_extra_vars
            self.assertEqual(y_init.size,constr.num_extra_vars)
            self.assertTrue(np.all(y_init == 0.))
            
            y0 = np.random.rand(constr.num_extra_vars)
            constr.eval(x0,y0)
            self.assertEqual(constr.J_nnz,Jnnz*self.T)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.J_row,rowsJ*self.T)
            self.assertEqual(constr.A_row,0)

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b

            # After
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(rowsJ*self.T,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(rowsA*self.T,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(rowsJ*self.T,net.num_vars+constr.num_extra_vars))
            self.assertEqual(J.nnz,Jnnz*self.T)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(rowsA*self.T,net.num_vars+constr.num_extra_vars))
            self.assertEqual(A.nnz,Annz*self.T)

            self.assertTrue(not np.any(np.isinf(b)))
            self.assertTrue(not np.any(np.isnan(b)))
            self.assertTrue(not np.any(np.isinf(f)))
            self.assertTrue(not np.any(np.isnan(f)))
            self.assertTrue(not np.any(np.isinf(J.data)))
            self.assertTrue(not np.any(np.isnan(J.data)))
            self.assertTrue(not np.any(np.isinf(A.data)))
            self.assertTrue(not np.any(np.isnan(A.data)))

            # Ax=b check
            for k in range(J.shape[0]//2):
                index1 = np.where(A.col == net.num_vars+2*k)[0]
                index2 = np.where(A.col == net.num_vars+2*k+1)[0]
                self.assertEqual(index1.size,1)
                self.assertEqual(index2.size,1)
                self.assertEqual(A.row[index1[0]],A.row[index2[0]])
                index3 = np.where(A.row == A.row[index1[0]])[0]
                self.assertEqual(index3.size,4)
            for vsc in net.vsc_converters:
                if vsc.is_in_f_ac_mode():
                    gamma = vsc.target_power_factor
                    factor = np.sqrt((1-gamma**2.)/(gamma**2.))
                    for t in range(self.T):
                        iQ = vsc.index_Q[t]
                        iP = vsc.index_P[t]
                        k = np.where(A.col == iQ)[0]
                        self.assertEqual(k.size, 1)
                        k = np.where(A.row == A.row[k])[0]
                        self.assertEqual(k.size, 4)
                        for kk in k:
                            if A.col[kk] == iQ:
                                self.assertEqual(A.data[kk], 1.)
                            elif A.col[kk] == iP:
                                if vsc.target_power_factor >= 0:
                                    self.assertAlmostEqual(A.data[kk], -factor)
                                else:
                                    self.assertAlmostEqual(A.data[kk], factor)
                            else:
                                if (A.col[kk]-net.num_vars) % 2 == 0: 
                                    self.assertAlmostEqual(A.data[kk], -factor) # y
                                else:                 
                                    self.assertAlmostEqual(A.data[kk], factor) # z
                                    
            # f check
            eps = 1e-8
            J_row = 0
            for t in range(self.T):
                for bus in net.buses:
                    for vsc in bus.vsc_converters:
                        if vsc.is_in_f_ac_mode():
                            self.assertTrue(vsc.has_flags('variable', ['active power', 'reactive power']))
                            y = y0[J_row]
                            z = y0[J_row+1]
                            Q = vsc.Q[t]
                            Qmax = vsc.Q_max
                            Qmin = vsc.Q_min
                            CompY = (Q-Qmin)+y-np.sqrt((Q-Qmin)**2.+y**2.+2*eps)
                            CompZ = (Qmax-Q)+z-np.sqrt((Qmax-Q)**2.+z**2.+2*eps)
                            self.assertAlmostEqual(CompY,f[J_row])
                            self.assertAlmostEqual(CompZ,f[J_row+1])
                            J_row += 2

            # Jacobian check
            pf.tests.utils.check_constraint_Jacobian(self,
                                                     constr,
                                                     x0,
                                                     y0,
                                                     NUM_TRIALS,
                                                     TOL,
                                                     EPS,
                                                     h)

            # Sigle Hessian check
            pf.tests.utils.check_constraint_single_Hessian(self,
                                                           constr,
                                                           x0,
                                                           y0,
                                                           NUM_TRIALS,
                                                           TOL,
                                                           EPS,
                                                           h)
            
            # Combined Hessian check
            pf.tests.utils.check_constraint_combined_Hessian(self,
                                                             constr,
                                                             x0,
                                                             y0,
                                                             NUM_TRIALS,
                                                             TOL,
                                                             EPS,
                                                             h)
            
    def test_constr_REG_PF_SWITCH(self):

        # Multiperiod
        for case in test_cases.CASES:
            
            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)
            self.assertEqual(net.num_vars,0)

            # Vars
            net.set_flags('vsc converter',
                          'variable',
                          'any',
                          ['active power', 'reactive power'])
            self.assertEqual(net.num_vars, 2*net.get_num_vsc_converters()*self.T)
            
            x0 = net.get_var_values()
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            # PF
            for vsc in net.vsc_converters:
                if vsc.is_in_f_ac_mode():
                    vsc.target_power_factor = np.sign(np.random.randn())*np.minimum(np.random.rand(), 0.2)
            
            # Constraint
            constr = pf.Constraint('switching power factor regulation',net)
            self.assertEqual(constr.name,'switching power factor regulation')
            
            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            
            # Before
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,0))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,0))
            self.assertEqual(A.nnz,0)
            
            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            
            # Manual count
            nnz = 0
            num_constr = 0
            for vsc in net.vsc_converters:
                if vsc.is_in_f_ac_mode() and vsc.has_flags('variable', ['active power', 'reactive power']):
                    num_constr += 1
                    nnz += 2

            constr.analyze()
            self.assertEqual(constr.A.shape[0],num_constr*self.T)
            self.assertEqual(nnz*self.T,constr.A_nnz)
            constr.eval(x0)
            self.assertEqual(0,constr.A_nnz)
                
            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            
            # After
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(num_constr*self.T,))
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(num_constr*self.T,net.num_vars))
            self.assertEqual(A.nnz,nnz*self.T)
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,net.num_vars))
            self.assertEqual(J.nnz,0)
            
            self.assertTrue(not np.any(np.isinf(b)))
            self.assertTrue(not np.any(np.isnan(b)))
                
            # Detailed check
            Ai = A.row
            Aj = A.col
            Ad = A.data
            self.assertEqual(Ai.size,nnz*self.T)
            self.assertEqual(Aj.size,nnz*self.T)
            self.assertEqual(Ad.size,nnz*self.T)
            nnz = 0
            row = 0

            for t in range(self.T):
                for bus in net.buses:
                    for vsc in bus.vsc_converters:
                        if vsc.is_in_f_ac_mode():
                            gamma = vsc.target_power_factor
                            factor = np.sqrt(1-gamma**2.)/np.abs(gamma)
                            self.assertEqual(b[row], 0.)
                            self.assertEqual(Ai[nnz], row)
                            self.assertEqual(Aj[nnz], vsc.index_P[t])
                            if gamma >= 0.:
                                self.assertAlmostEqual(Ad[nnz], -factor)
                            else:
                                self.assertAlmostEqual(Ad[nnz], factor)
                            nnz += 1
                            self.assertEqual(Ai[nnz], row)
                            self.assertEqual(Aj[nnz], vsc.index_Q[t])
                            self.assertEqual(Ad[nnz], 1.)
                            nnz += 1
                            row += 1

            self.assertEqual(row,A.shape[0])
            self.assertEqual(nnz,A.nnz)

    def test_constr_HVDCPF(self):

        # Multiperiods
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case, self.T)
            self.assertEqual(net.num_periods, self.T)

            # Vars
            net.set_flags('dc bus',
                          'variable',
                          'any',
                          'voltage')

            # Vars
            net.set_flags('vsc converter',
                          'variable',
                          'any',
                          'dc power')

            net.set_flags('csc converter',
                          'variable',
                          'any',
                          'dc power')

            self.assertEqual(net.num_vars, (net.num_dc_buses +
                                            2*net.num_vsc_converters +
                                            2*net.num_csc_converters)*self.T)

            # Constraint
            constr = pf.Constraint('HVDC power balance',net)
            self.assertEqual(constr.name,'HVDC power balance')

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            G = constr.G
            l = constr.l
            u = constr.u

            # Before
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,0))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,0))
            self.assertEqual(A.nnz,0)
            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.J_row,0)
            self.assertEqual(constr.A_row,0)
            self.assertEqual(constr.H_combined.nnz, 0)
            self.assertTupleEqual(constr.H_combined.shape, (0,0))
            self.assertTrue(type(G) is coo_matrix)
            self.assertTupleEqual(G.shape,(0,0))
            self.assertEqual(G.nnz,0)
            self.assertEqual(constr.num_extra_vars,0)
            
            x0 = net.get_var_values()+1e-1*np.random.randn(net.num_vars)
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            flags = np.zeros(net.num_dc_buses*self.T, dtype=int)
            for t in range(self.T):
                for bus in net.dc_buses:
                    flags[bus.index_t[t]] = 1
            self.assertEqual(np.sum(flags), flags.size)

            constr.analyze()

            A = constr.A
            b = constr.b

            self.assertTupleEqual(constr.J.shape, (0, net.num_vars))
            self.assertTupleEqual(constr.G.shape, (0, net.num_vars))
            self.assertEqual(constr.J.nnz, 0)
            self.assertEqual(constr.G.nnz, 0)
            self.assertEqual(constr.l.size, 0)
            self.assertEqual(constr.u.size, 0)
            self.assertEqual(constr.f.size, 0)

            self.assertTupleEqual(constr.A.shape, (net.num_dc_buses*self.T, net.num_vars))
            self.assertEqual(constr.A.nnz, (net.num_vsc_converters +
                                            net.num_csc_converters +
                                            4*net.num_dc_branches)*self.T)
            self.assertEqual(constr.b.size, net.num_dc_buses*self.T)

            i_mis_manual = np.zeros(net.num_dc_buses*self.T)
            i_mis = A*x0-b

            for t in range(self.T):
                for bus in net.dc_buses:
                    for branch in bus.branches:
                        self.assertTrue(branch.bus_k.has_flags('variable', 'voltage'))
                        self.assertTrue(branch.bus_m.has_flags('variable', 'voltage'))
                        ikm = (x0[branch.bus_k.index_v[t]]-x0[branch.bus_m.index_v[t]])/branch.r
                        self.assertEqual(ikm, branch.get_i_km(x0)[t])
                        if bus.is_equal(branch.bus_k):
                            i_out = ikm
                        else:
                            i_out = -ikm
                        i_mis_manual[bus.index_t[t]] -= i_out
                for conv in net.vsc_converters:
                    self.assertTrue(conv.has_flags('variable', 'dc power'))
                    i_in = x0[conv.index_i_dc[t]]
                    i_mis_manual[conv.dc_bus.index_t[t]] += i_in
                for conv in net.csc_converters:
                    self.assertTrue(conv.has_flags('variable', 'dc power'))
                    i_in = x0[conv.index_i_dc[t]]
                    i_mis_manual[conv.dc_bus.index_t[t]] += i_in
            if not i_mis.size:
                self.assertTrue(np.all(i_mis_manual == i_mis))
            else:
                self.assertLessEqual(np.max(np.abs(i_mis_manual-i_mis)), 1e-10)

            net.set_var_values(x0)

            for t in range(self.T):
                for bus in net.dc_buses:
                    self.assertNotEqual(bus.v[t], 0.)
                    self.assertNotEqual(bus.v[t], 1.)
                for conv in net.vsc_converters:
                    self.assertNotEqual(conv.P_dc[t], 0.)
                    self.assertNotEqual(conv.i_dc[t], 0.)

            # Test with no variables
            net.clear_flags()
            self.assertEqual(net.num_vars, 0)

            constr.analyze()
            
            A = constr.A
            b = constr.b

            self.assertTupleEqual(constr.J.shape, (0, net.num_vars))
            self.assertTupleEqual(constr.G.shape, (0, net.num_vars))
            self.assertEqual(constr.J.nnz, 0)
            self.assertEqual(constr.G.nnz, 0)
            self.assertEqual(constr.l.size, 0)
            self.assertEqual(constr.u.size, 0)
            self.assertEqual(constr.f.size, 0)

            self.assertTupleEqual(constr.A.shape, (net.num_dc_buses*self.T, 0))
            self.assertEqual(constr.A.nnz, 0)
            self.assertEqual(constr.b.size, net.num_dc_buses*self.T)

            x0 = net.get_var_values()
            self.assertEqual(x0.size, 0)
            
            i_mis_manual = np.zeros(net.num_dc_buses*self.T)
            i_mis = A*x0-b

            for t in range(self.T):
                for bus in net.dc_buses:
                    for branch in bus.branches:
                        self.assertFalse(branch.bus_k.has_flags('variable', 'voltage'))
                        self.assertFalse(branch.bus_m.has_flags('variable', 'voltage'))
                        ikm = branch.i_km[t]
                        if bus.is_equal(branch.bus_k):
                            i_out = ikm
                        else:
                            i_out = -ikm
                        i_mis_manual[bus.index_t[t]] -= i_out
                for conv in net.vsc_converters:
                    self.assertFalse(conv.has_flags('variable', 'dc power'))
                    i_in = conv.i_dc[t]
                    i_mis_manual[conv.dc_bus.index_t[t]] += i_in
                for conv in net.csc_converters:
                    self.assertFalse(conv.has_flags('variable', 'dc power'))
                    i_in = conv.i_dc[t]
                    i_mis_manual[conv.dc_bus.index_t[t]] += i_in
            if not i_mis.size:
                self.assertTrue(np.all(i_mis_manual == i_mis))
            else:
                self.assertLessEqual(np.max(np.abs(i_mis_manual-i_mis)), 1e-10)

    def test_constr_VSC_EQ(self):

        # Constants
        h = 1e-10

        # Multiperiods
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)

            # Vars
            net.set_flags('vsc converter',
                          'variable',
                          'any',
                          ['dc power', 'active power'])
            net.set_flags('dc bus',
                          'variable',
                          'any',
                          'voltage')

            # Check if dc bus indexes are setting to unique values
            busindicest = [bus.index_t for bus in net.dc_buses]
            self.assertEqual(len(np.unique(busindicest)), net.num_dc_buses*self.T)
            busindicesv = [bus.index_v for bus in net.dc_buses]
            self.assertEqual(len(np.unique(busindicesv)), net.num_dc_buses*self.T)

            # Check if vsc different variables index are setting to unique values
            vscindicesPac = [vsc.index_P for vsc in net.vsc_converters]
            self.assertEqual(len(np.unique(vscindicesPac)), net.num_vsc_converters*self.T)
            vscindicesPdc = [vsc.index_P_dc for vsc in net.vsc_converters]
            self.assertEqual(len(np.unique(vscindicesPdc)), net.num_vsc_converters*self.T)
            vscindicesidc = [vsc.index_i_dc for vsc in net.vsc_converters]
            self.assertEqual(len(np.unique(vscindicesidc)), net.num_vsc_converters*self.T)

            self.assertEqual(net.num_vars, (3*net.num_vsc_converters+net.num_dc_buses)*self.T)

            # Constraint
            constr = pf.Constraint('VSC converter equations',net)
            self.assertEqual(constr.name,'VSC converter equations')

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b

            # Before
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,0))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,0))
            self.assertEqual(A.nnz,0)
            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.J_row,0)
            self.assertEqual(constr.A_row,0)
            self.assertEqual(constr.num_extra_vars,0)

            x0 = net.get_var_values()+1e-4*np.random.randn(net.num_vars)
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            # Verify analyze
            constr.analyze()

            Annz = 3*net.num_vsc_converters*self.T
            Jnnz =3*net.num_vsc_converters*self.T
            Arow = net.num_vsc_converters*self.T
            Jrow = net.num_vsc_converters*self.T

            self.assertEqual(constr.A_nnz,Annz)
            self.assertEqual(constr.A_row,Arow)
            self.assertEqual(constr.J_nnz,Jnnz)
            self.assertEqual(constr.J_row,Jrow)

            # Verify evaluation
            constr.eval(x0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.A_row,0)
            self.assertEqual(constr.J_nnz,Jnnz)
            self.assertEqual(constr.J_row,Jrow)

            f = constr.f
            J = constr.J
            b = constr.b
            A = constr.A

            # After
            self.assertTrue(not np.any(np.isinf(b)))
            self.assertTrue(not np.any(np.isnan(b)))
            self.assertTupleEqual(b.shape,(Arow,))
            self.assertTrue(not np.any(np.isinf(f)))
            self.assertTrue(not np.any(np.isnan(f)))
            self.assertTupleEqual(f.shape,(Jrow,))
            self.assertTrue(not np.any(np.isinf(J.data)))
            self.assertTrue(not np.any(np.isnan(J.data)))
            self.assertTupleEqual(J.shape,(Jrow,net.num_vars))
            self.assertTrue(not np.any(np.isinf(A.data)))
            self.assertTrue(not np.any(np.isnan(A.data)))
            self.assertTupleEqual(A.shape,(Arow,net.num_vars))

            # Ax = b check
            coeffB = [vsc.loss_coeff_B for vsc in net.vsc_converters]
            sumcoefB = np.sum(np.abs(coeffB))
            self.assertAlmostEqual(norm(A.data,1),2*Arow+sumcoefB*self.T) # Almost, because of float type, to avoid precision errors
            for k in range(self.T):
                for vsc in net.vsc_converters:
                    self.assertTrue(vsc.has_flags('variable',['dc power','active power']))
                    indexP = np.where(A.col == vsc.index_P[k])[0]
                    indexPdc = np.where(A.col == vsc.index_P_dc[k])[0]
                    indexidc = np.where(A.col == vsc.index_i_dc[k])[0]
                    self.assertEqual(indexP.size,1)
                    self.assertEqual(indexPdc.size,1)
                    self.assertEqual(indexidc.size,1)
                    self.assertEqual(A.data[indexP],1.)
                    self.assertEqual(A.data[indexPdc],1.)
                    if vsc.P_dc_set[k] <= 0:
                        self.assertEqual(A.data[indexidc],-vsc.loss_coeff_B)
                    else:
                        self.assertEqual(A.data[indexidc],vsc.loss_coeff_B)
                    self.assertEqual(b[A.row[indexP]],-1.*vsc.loss_coeff_A)

            # f check
            J_row = 0
            for t in range(self.T):
                for bus in net.dc_buses:
                    vsc_onthisbus = [vsc for vsc in net.vsc_converters if vsc.dc_bus == bus]
                    for vsc in vsc_onthisbus:
                        indexPdc = np.where(J.col == vsc.index_P_dc[t])[0]
                        indexidc = np.where(J.col == vsc.index_i_dc[t])[0]
                        indexv = np.where(J.col == bus.index_v[t])[0]
                        dP = x0[J.col[indexPdc]] - x0[J.col[indexidc]]*x0[J.col[indexv]]
                        self.assertAlmostEqual(f[J_row],dP)
                        J_row += 1

            # Jacobian check
            pf.tests.utils.check_constraint_Jacobian(self,
                                                     constr,
                                                     x0,
                                                     np.zeros(0),
                                                     NUM_TRIALS,
                                                     TOL,
                                                     EPS,
                                                     h)

            # Sigle Hessian check
            pf.tests.utils.check_constraint_single_Hessian(self,
                                                           constr,
                                                           x0,
                                                           np.zeros(0),
                                                           NUM_TRIALS,
                                                           TOL,
                                                           EPS,
                                                           h)

            # Combined Hessian check
            pf.tests.utils.check_constraint_combined_Hessian(self,
                                                             constr,
                                                             x0,
                                                             np.zeros(0),
                                                             NUM_TRIALS,
                                                             TOL,
                                                             EPS,
                                                             h)

    def test_constr_VSC_DC_PSET(self):

        # Multiperiods
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)

            # Check if bus indexes are setting to unique values
            busindices = [bus.index_t for bus in net.dc_buses]
            self.assertEqual(len(np.unique(busindices)), net.num_dc_buses*self.T)

            # Vars
            net.set_flags('vsc converter',
                          'variable',
                          'any',
                          'active power')

            self.assertEqual(net.num_vars, net.num_vsc_converters*self.T)

            # Constraint
            constr = pf.Constraint('VSC DC power control',net)
            self.assertEqual(constr.name,'VSC DC power control')

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b

            # Before
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,0))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,0))
            self.assertEqual(A.nnz,0)
            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.J_row,0)
            self.assertEqual(constr.A_row,0)
            self.assertEqual(constr.num_extra_vars,0)

            x0 = net.get_var_values()
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            # Verify analyze
            constr.analyze()

            # Check if vsc index are setting to unique values
            vscindices = [vsc.index_P for vsc in net.vsc_converters]
            self.assertEqual(len(np.unique(vscindices)), net.num_vsc_converters*self.T)
            dcmodevsc = [vsc for vsc in net.vsc_converters if vsc.is_in_P_dc_mode()]
            Annz = len(dcmodevsc)*self.T
            Arow = Annz
            self.assertEqual(constr.A_nnz,Annz)
            self.assertEqual(constr.A_row,Arow)

            # Verify evaluation
            constr.eval(x0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.A_row,0)

            f = constr.f
            J = constr.J
            b = constr.b
            A = constr.A

            # After
            self.assertTrue(not np.any(np.isinf(b)))
            self.assertTrue(not np.any(np.isnan(b)))
            self.assertTrue(not np.any(np.isinf(f)))
            self.assertTrue(not np.any(np.isnan(f)))
            self.assertTrue(not np.any(np.isinf(J.data)))
            self.assertTrue(not np.any(np.isnan(J.data)))
            self.assertTrue(not np.any(np.isinf(A.data)))
            self.assertTrue(not np.any(np.isnan(A.data)))

            # Ax = b Check
            for k in range(self.T):
                for vsc in dcmodevsc:
                    self.assertTrue(vsc.has_flags('variable', ['active power']))
                    indexP = np.where(A.col == vsc.index_P[k])[0]
                    self.assertEqual(indexP.size,1)
                    self.assertEqual(A.data[indexP],-1)
                    self.assertEqual(b[A.row[indexP]],vsc.P_dc_set[k])

    def test_constr_VSC_DC_VSET(self):

        # Multiperiods
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)
            
            # Check if bus indexes are setting to unique values
            busindices = [bus.index_t for bus in net.dc_buses]
            self.assertEqual(len(np.unique(busindices)), net.num_dc_buses*self.T)

            # Vars
            net.set_flags('dc bus',
                          'variable',
                          'any',
                          'voltage')

            self.assertEqual(net.num_vars, net.num_dc_buses*self.T)

            # Constraint
            constr = pf.Constraint('VSC DC voltage control',net)
            self.assertEqual(constr.name,'VSC DC voltage control')

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b

            # Before
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,0))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,0))
            self.assertEqual(A.nnz,0)
            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.J_row,0)
            self.assertEqual(constr.A_row,0)
            self.assertEqual(constr.num_extra_vars,0)

            x0 = net.get_var_values()
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            # Verify analyze
            constr.analyze()
            dcmodevsc = [vsc for vsc in net.vsc_converters if vsc.is_in_v_dc_mode()]
            Annz = len(dcmodevsc)*self.T
            Arow = Annz
            self.assertEqual(constr.A_nnz,Annz)
            self.assertEqual(constr.A_row,Arow)

            # Verify evaluation
            constr.eval(x0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.A_row,0)

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b

            # After
            self.assertTrue(not np.any(np.isinf(b)))
            self.assertTrue(not np.any(np.isnan(b)))
            self.assertTrue(not np.any(np.isinf(f)))
            self.assertTrue(not np.any(np.isnan(f)))
            self.assertTrue(not np.any(np.isinf(J.data)))
            self.assertTrue(not np.any(np.isnan(J.data)))
            self.assertTrue(not np.any(np.isinf(A.data)))
            self.assertTrue(not np.any(np.isnan(A.data)))

            # Verify A matrix
            self.assertTrue(np.all(A.data == 1))
            for t in range(0,self.T):
                indices = [vsc.dc_bus.index_v[t] for vsc in dcmodevsc]
                self.assertTrue(np.all(A.col[t*len(dcmodevsc):(t*len(dcmodevsc)+len(dcmodevsc))] == indices))

            # Verify b vector
            for t in range(0,self.T):
                setpoints = [vsc.v_dc_set[t] for vsc in dcmodevsc]
                self.assertTrue(np.all(b[t*len(dcmodevsc):(t*len(dcmodevsc)+len(dcmodevsc))] == setpoints))
                    
    def test_constr_LOAD_VDEP(self):

        # Constants
        h = 1e-10

        # Multiperiods
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)

            # Vars
            net.set_flags('bus',
                          'variable',
                          'any',
                          'voltage magnitude')
            net.set_flags('load',
                          'variable',
                          'any',
                          ['active power', 'reactive power'])
            self.assertEqual(net.num_vars, (2*net.num_loads+net.num_buses)*self.T)

            x0 = net.get_var_values()+1e-5*np.random.randn(net.num_vars)
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            # Loads comps
            for load in net.loads:
                load.comp_ci = np.random.randn(self.T)
                load.comp_cj = np.random.randn(self.T)
                load.comp_cg = np.random.randn()
                load.comp_cb = np.random.randn()
                load.comp_cp = load.P
                load.comp_cq = load.Q

            # Constraint
            constr = pf.Constraint('load voltage dependence',net)
            self.assertEqual(constr.name,'load voltage dependence')

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b

            # Before
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,0))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,0))
            self.assertEqual(A.nnz,0)

            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.J_row,0)
            self.assertEqual(constr.A_row,0)
            self.assertEqual(constr.num_extra_vars,0)

            Jnnz = 4*net.num_loads*self.T
            rowsJ = 2*net.num_loads*self.T

            constr.analyze()
            self.assertEqual(constr.J_nnz,Jnnz)
            self.assertEqual(constr.J_row,rowsJ)
            self.assertEqual(constr.num_extra_vars,0)
            self.assertLessEqual(constr.J_row, constr.H_nnz.size)
            self.assertLessEqual(2*net.num_loads*net.num_periods, constr.H_nnz.size)
            self.assertTrue(np.all(constr.H_nnz[:2*net.num_loads*net.num_periods] == 1))

            for i in range(rowsJ):
                H = constr.get_H_single(i)
                self.assertEqual(H.shape[0], net.num_vars)
                self.assertEqual(H.shape[1], net.num_vars)
                self.assertEqual(H.nnz, 1)
            H = constr.H_combined
            self.assertEqual(H.shape[0], net.num_vars)
            self.assertEqual(H.shape[1], net.num_vars)
            self.assertEqual(H.nnz, rowsJ)

            constr.eval(x0)
            self.assertEqual(constr.J_nnz,Jnnz)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.J_row,rowsJ)
            self.assertEqual(constr.A_row,0)

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b

            # After
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(rowsJ,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(rowsJ,net.num_vars))
            self.assertEqual(J.nnz,Jnnz)

            self.assertTrue(not np.any(np.isinf(b)))
            self.assertTrue(not np.any(np.isnan(b)))
            self.assertTrue(not np.any(np.isinf(f)))
            self.assertTrue(not np.any(np.isnan(f)))
            self.assertTrue(not np.any(np.isinf(J.data)))
            self.assertTrue(not np.any(np.isnan(J.data)))
            self.assertTrue(not np.any(np.isinf(A.data)))
            self.assertTrue(not np.any(np.isnan(A.data)))

            # f check
            J_row = 0
            for t in range(self.T):
                for bus in net.buses:
                    for load in bus.loads:
                        Sp = (x0[load.index_P[t]] -
                              load.comp_cp[t] -
                              load.comp_ci[t]*x0[bus.index_v_mag[t]] -
                              load.comp_cg*(x0[bus.index_v_mag[t]])**2.)
                        Sq = (x0[load.index_Q[t]] -
                              load.comp_cq[t] -
                              load.comp_cj[t]*x0[bus.index_v_mag[t]] +
                              load.comp_cb*(x0[bus.index_v_mag[t]])**2.)
                        self.assertAlmostEqual(Sp,f[J_row])
                        self.assertAlmostEqual(Sq,f[J_row+1])
                        J_row += 2
                        
            # Jacobian check
            pf.tests.utils.check_constraint_Jacobian(self,
                                                     constr,
                                                     x0,
                                                     np.zeros(0),
                                                     NUM_TRIALS,
                                                     TOL,
                                                     EPS,
                                                     h)
            # Single Hessian check
            pf.tests.utils.check_constraint_single_Hessian(self,
                                                           constr,
                                                           x0,
                                                           np.zeros(0),
                                                           NUM_TRIALS,
                                                           TOL,
                                                           EPS,
                                                           h)
            # Combined Hessian check
            pf.tests.utils.check_constraint_combined_Hessian(self,
                                                             constr,
                                                             x0,
                                                             np.zeros(0),
                                                             NUM_TRIALS,
                                                             TOL,
                                                             EPS,
                                                             h)

    def test_constr_CFUNC(self):

        h = 1e-9
        
        # Multiperiod
        for case in test_cases.CASES:
            
            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)

            net.set_flags('bus',
                          'variable',
                          'any',
                          'voltage magnitude')
            net.set_flags('generator',
                          'variable',
                          'any',
                          'active power')

            self.assertEqual(net.num_vars, (net.num_buses+net.num_generators)*net.num_periods)

            x = net.get_var_values() + 1e-2*np.random.rand(net.num_vars)
            
            func = pf.Function('generation cost', 1., net)

            constr = pf.Constraint('constrained function', net)
            
            rhs = 100.
            
            constr.set_parameter("rhs", rhs)
            constr.set_parameter("func", func)

            # Equality
            constr.set_parameter("op", "=")
            constr.analyze()
            self.assertEqual(constr.num_extra_vars, 0)
            self.assertEqual(constr.G.nnz, 0)
            self.assertTupleEqual(constr.G.shape, (0, net.num_vars))
            self.assertEqual(constr.l.size, 0)
            self.assertEqual(constr.u.size, 0)
            self.assertEqual(constr.l_extra_vars.size, 0)
            self.assertEqual(constr.u_extra_vars.size, 0)
            self.assertEqual(constr.init_extra_vars.size, 0)
            self.assertEqual(constr.f.size, 1)
            self.assertEqual(constr.J.nnz, net.num_vars)
            self.assertTupleEqual(constr.J.shape, (1, net.num_vars))
            H = constr.get_H_single(0)
            self.assertEqual(H.nnz, func.Hphi.nnz)
            self.assertTupleEqual(H.shape, (net.num_vars, net.num_vars))
            self.assertEqual(func.phi, 0.)
            constr.eval(x)
            net.update_properties(x)
            self.assertNotEqual(func.phi, 0.)
            self.assertLess(np.abs(func.phi-np.sum(net.gen_P_cost)), 1e-12*(np.abs(func.phi)+1.))
            self.assertEqual(constr.f[0], func.phi - rhs - 0.)
            pf.tests.utils.check_constraint_Jacobian(self, constr, x, np.zeros(0), NUM_TRIALS, TOL, EPS, h, quiet=True)
            pf.tests.utils.check_constraint_single_Hessian(self, constr, x, np.zeros(0), NUM_TRIALS, TOL, EPS, h, quiet=True)
            pf.tests.utils.check_constraint_combined_Hessian(self, constr, x, np.zeros(0), NUM_TRIALS, TOL, EPS, h, quiet=True)
            
            # Inequality >=
            constr.set_parameter("op", ">=")
            constr.analyze()
            self.assertEqual(constr.num_extra_vars, 1)
            self.assertEqual(constr.G.nnz, 1)
            self.assertEqual(constr.G.row[0], 0)
            self.assertEqual(constr.G.col[0], net.num_vars)
            self.assertEqual(constr.G.data[0], 1.)
            self.assertTupleEqual(constr.G.shape, (1, net.num_vars+1))
            self.assertEqual(constr.l.size, 1)
            self.assertEqual(constr.u.size, 1)
            self.assertEqual(constr.l_extra_vars.size, 1)
            self.assertEqual(constr.u_extra_vars.size, 1)
            self.assertEqual(constr.init_extra_vars.size, 1)
            self.assertEqual(constr.l[0],0)
            self.assertEqual(constr.l_extra_vars[0],0)
            self.assertEqual(constr.u[0],1e8)
            self.assertEqual(constr.u_extra_vars[0],1e8)
            self.assertEqual(constr.f.size, 1)
            self.assertEqual(constr.J.nnz, net.num_vars+1)
            self.assertTupleEqual(constr.J.shape, (1, net.num_vars+1))
            H = constr.get_H_single(0)
            self.assertEqual(H.nnz, func.Hphi.nnz)
            self.assertTupleEqual(H.shape, (net.num_vars+1, net.num_vars+1))
            self.assertEqual(func.phi, 0.)
            y = np.random.randn(1)
            constr.eval(x,y)
            net.update_properties(x)
            self.assertNotEqual(func.phi, 0.)
            self.assertLess(np.abs(func.phi-np.sum(net.gen_P_cost)), 1e-12*(np.abs(func.phi)+1.))
            self.assertEqual(constr.f[0], func.phi - rhs - y[0])
            pf.tests.utils.check_constraint_Jacobian(self, constr, x, y, NUM_TRIALS, TOL, EPS, h, quiet=True)
            pf.tests.utils.check_constraint_single_Hessian(self, constr, x, y, NUM_TRIALS, TOL, EPS, h, quiet=True)
            pf.tests.utils.check_constraint_combined_Hessian(self, constr, x, y, NUM_TRIALS, TOL, EPS, h, quiet=True)
            self.assertEqual(constr.G*np.hstack((x,y)),y[0])

            # Inequality <=
            constr.set_parameter("op", "<=")
            constr.analyze()
            self.assertEqual(constr.num_extra_vars, 1)
            self.assertEqual(constr.G.nnz, 1)
            self.assertEqual(constr.G.row[0], 0)
            self.assertEqual(constr.G.col[0], net.num_vars)
            self.assertEqual(constr.G.data[0], 1.)
            self.assertTupleEqual(constr.G.shape, (1, net.num_vars+1))
            self.assertEqual(constr.l.size, 1)
            self.assertEqual(constr.u.size, 1)
            self.assertEqual(constr.l_extra_vars.size, 1)
            self.assertEqual(constr.u_extra_vars.size, 1)
            self.assertEqual(constr.init_extra_vars.size, 1)
            self.assertEqual(constr.l[0],-1e8)
            self.assertEqual(constr.l_extra_vars[0],-1e8)
            self.assertEqual(constr.u[0],0)
            self.assertEqual(constr.u_extra_vars[0],0)
            self.assertEqual(constr.f.size, 1)
            self.assertEqual(constr.J.nnz, net.num_vars+1)
            self.assertTupleEqual(constr.J.shape, (1, net.num_vars+1))
            H = constr.get_H_single(0)
            self.assertEqual(H.nnz, func.Hphi.nnz)
            self.assertTupleEqual(H.shape, (net.num_vars+1, net.num_vars+1))
            self.assertEqual(func.phi, 0.)
            y = np.random.randn(1)
            constr.eval(x,y)
            net.update_properties(x)
            self.assertNotEqual(func.phi, 0.)
            self.assertLess(np.abs(func.phi-np.sum(net.gen_P_cost)), 1e-12*(np.abs(func.phi)+1.))
            self.assertEqual(constr.f[0], func.phi - rhs - y[0])
            pf.tests.utils.check_constraint_Jacobian(self, constr, x, y, NUM_TRIALS, TOL, EPS, h, quiet=True)
            pf.tests.utils.check_constraint_single_Hessian(self, constr, x, y, NUM_TRIALS, TOL, EPS, h, quiet=True)
            pf.tests.utils.check_constraint_combined_Hessian(self, constr, x, y, NUM_TRIALS, TOL, EPS, h, quiet=True)
            self.assertEqual(constr.G*np.hstack((x,y)),y[0])

    def test_constr_FIX(self):
        
        # Single period
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case)
            self.assertEqual(net.num_periods,1)

            # add vargens
            net.add_var_generators_from_parameters(net.get_load_buses(),80.,50.,30.,5,0.05)
            for vargen in net.var_generators:
                vargen.P = vargen.index*1.5
                vargen.Q = vargen.index*2.5
            self.assertGreater(net.num_var_generators,0)

            self.assertEqual(net.num_vars,0)
            self.assertEqual(net.num_fixed,0)

            # Vars
            net.set_flags('bus',
                          'variable',
                          'any',
                          ['voltage magnitude','voltage angle'])
            net.set_flags('generator',
                          'variable',
                          'slack',
                          'active power')
            net.set_flags('generator',
                          'variable',
                          'regulator',
                          'reactive power')
            net.set_flags('branch',
                          'variable',
                          'tap changer',
                          'tap ratio')
            net.set_flags('branch',
                          'variable',
                          'phase shifter',
                          'phase shift')
            net.set_flags('shunt',
                          'variable',
                          'switching - v',
                          'susceptance')
            net.set_flags('variable generator',
                          'variable',
                          'any',
                          ['active power','reactive power'])
            net.set_flags('battery',
                          'variable',
                          'any',
                          ['charging power','energy level'])
            net.set_flags('load',
                          'variable',
                          'any',
                          'active power')
            self.assertGreater(net.num_vars,0)
            self.assertEqual(net.num_fixed,0)
            self.assertEqual(net.num_vars,
                             2*net.num_buses +
                             net.get_num_slack_gens() +
                             net.get_num_reg_gens() +
                             net.get_num_tap_changers() +
                             net.get_num_phase_shifters() +
                             net.get_num_switched_v_shunts() +
                             net.num_var_generators*2+
                             3*net.num_batteries+
                             net.num_loads)

            # Fixed
            net.set_flags('bus',
                          'fixed',
                          'slack',
                          ['voltage magnitude','voltage angle'])
            net.set_flags('bus',
                          'fixed',
                          'regulated by generator',
                          'voltage magnitude')
            net.set_flags('generator',
                          'fixed',
                          'regulator',
                          'reactive power')
            net.set_flags('branch',
                          'fixed',
                          'tap changer',
                          'tap ratio')
            net.set_flags('branch',
                          'fixed',
                          'phase shifter',
                          'phase shift')
            net.set_flags('shunt',
                          'fixed',
                          'switching - v',
                          'susceptance')
            net.set_flags('variable generator',
                          'fixed',
                          'any',
                          ['active power','reactive power'])
            net.set_flags('battery',
                          'fixed',
                          'any',
                          ['charging power','energy level'])
            net.set_flags('load',
                          'fixed',
                          'any',
                          'active power')
            self.assertGreater(net.num_fixed,0)
            self.assertEqual(net.num_fixed,
                             2*(net.get_num_slack_buses()) +
                             (net.get_num_buses_reg_by_gen()-net.get_num_slack_buses()) +
                             net.get_num_reg_gens() +
                             net.get_num_tap_changers() +
                             net.get_num_phase_shifters() +
                             net.get_num_switched_v_shunts() +
                             net.num_var_generators*2+
                             3*net.num_batteries+
                             net.num_loads)

            x0 = net.get_var_values()
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            constr = pf.Constraint('variable fixing',net)
            self.assertEqual(constr.name,'variable fixing')

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            G = constr.G
            l = constr.l
            u = constr.u

            # Before
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,0))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,0))
            self.assertEqual(A.nnz,0)
            self.assertTrue(type(G) is coo_matrix)
            self.assertTupleEqual(G.shape,(0,0))
            self.assertEqual(G.nnz,0)
            self.assertTrue(type(l) is np.ndarray)
            self.assertTupleEqual(l.shape,(0,))
            self.assertTrue(type(u) is np.ndarray)
            self.assertTupleEqual(u.shape,(0,))

            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.G_nnz,0)

            A_nnz = net.num_fixed
            constr.analyze()
            self.assertEqual(A_nnz,constr.A_nnz)
            constr.eval(x0)
            self.assertEqual(0,constr.A_nnz)

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            G = constr.G
            l = constr.l
            u = constr.u

            # After
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(net.num_fixed,))
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(net.num_fixed,net.num_vars))
            self.assertEqual(A.nnz,net.num_fixed)
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,net.num_vars))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(G) is coo_matrix)
            self.assertTupleEqual(G.shape,(0,net.num_vars))
            self.assertEqual(G.nnz,0)
            self.assertTrue(type(l) is np.ndarray)
            self.assertTupleEqual(l.shape,(0,))
            self.assertTrue(type(u) is np.ndarray)
            self.assertTupleEqual(u.shape,(0,))

            self.assertTrue(not np.any(np.isinf(b)))
            self.assertTrue(not np.any(np.isnan(b)))

            # Vargen
            for vargen in net.var_generators:
                ar = np.where(A.col == vargen.index_P)[0]
                self.assertEqual(ar.size,1)
                self.assertEqual(A.col[ar[0]],vargen.index_P)
                self.assertEqual(b[A.row[ar[0]]],vargen.P)
                self.assertEqual(b[A.row[ar[0]]],vargen.index*1.5)
            for vargen in net.var_generators:
                ar = np.where(A.col == vargen.index_Q)[0]
                self.assertEqual(ar.size,1)
                self.assertEqual(A.col[ar[0]],vargen.index_Q)
                self.assertEqual(b[A.row[ar[0]]],vargen.Q)
                self.assertEqual(b[A.row[ar[0]]],vargen.index*2.5)

            # Batteries
            for bat in net.batteries:
                ar = np.where(A.col == bat.index_Pc)[0]
                self.assertEqual(ar.size,1)
                self.assertEqual(A.col[ar[0]],bat.index_Pc)
                self.assertEqual(b[A.row[ar[0]]],max([bat.P,0]))
            for bat in net.batteries:
                ar = np.where(A.col == bat.index_Pd)[0]
                self.assertEqual(ar.size,1)
                self.assertEqual(A.col[ar[0]],bat.index_Pd)
                self.assertEqual(b[A.row[ar[0]]],max([-bat.P,0]))
            for bat in net.batteries:
                ar = np.where(A.col == bat.index_E)[0]
                self.assertEqual(ar.size,1)
                self.assertEqual(A.col[ar[0]],bat.index_E)
                self.assertEqual(b[A.row[ar[0]]],bat.E)

            # Load
            for load in net.loads:
                self.assertTrue(load.has_flags('variable','active power'))
                self.assertTrue(load.has_flags('fixed','active power'))
                ar = np.where(A.col == load.index_P)[0]
                self.assertEqual(ar.size,1)
                self.assertEqual(A.col[ar[0]],load.index_P)
                self.assertEqual(b[A.row[ar[0]]],load.P)

            # Projections
            P1 = constr.get_var_projection()
            P2 = constr.get_extra_var_projection()
            self.assertTrue(isinstance(P1,coo_matrix))
            self.assertTrue(isinstance(P2,coo_matrix))
            self.assertEqual(P1.shape[0],net.num_vars)
            self.assertEqual(P2.shape[0],0)
            self.assertEqual(P1.shape[1],net.num_vars)
            self.assertEqual(P2.shape[1],net.num_vars)
            self.assertEqual(P1.nnz,net.num_vars)
            self.assertEqual(P2.nnz,0)
            self.assertLess(np.linalg.norm(x0-P1*x0),1e-12)
        
        # Multiperiods
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)

            # add vargens
            net.add_var_generators_from_parameters(net.get_load_buses(),80.,50.,30.,5,0.05)
            for vargen in net.var_generators:
                vargen.P = np.random.rand(self.T)*10
                vargen.Q = np.random.rand(self.T)*10
                self.assertEqual(vargen.num_periods,self.T)
            self.assertGreater(net.num_var_generators,0)
            self.assertEqual(net.num_vars,0)
            self.assertEqual(net.num_fixed,0)

            # Vars
            net.set_flags('bus',
                          ['variable','fixed'],
                          'any',
                          ['voltage magnitude','voltage angle'])
            net.set_flags('generator',
                          ['variable','fixed'],
                          'slack',
                          'active power')
            net.set_flags('generator',
                          ['variable','fixed'],
                          'regulator',
                          'reactive power')
            net.set_flags('branch',
                          ['variable','fixed'],
                          'tap changer',
                          'tap ratio')
            net.set_flags('branch',
                          ['variable','fixed'],
                          'phase shifter',
                          'phase shift')
            net.set_flags('shunt',
                          ['variable','fixed'],
                          'switching - v',
                          'susceptance')
            net.set_flags('variable generator',
                          ['variable','fixed'],
                          'any',
                          ['active power','reactive power'])
            net.set_flags('battery',
                          ['variable','fixed'],
                          'any',
                          ['charging power','energy level'])
            net.set_flags('load',
                          ['variable','fixed'],
                          'any',
                          'active power')
            self.assertGreater(net.num_vars,0)
            self.assertEqual(net.num_fixed,net.num_vars)
            self.assertEqual(net.num_vars,
                             (2*net.num_buses +
                              net.get_num_slack_gens() +
                              net.get_num_reg_gens() +
                              net.get_num_tap_changers() +
                              net.get_num_phase_shifters() +
                              net.get_num_switched_v_shunts() +
                              net.num_var_generators*2+
                              3*net.num_batteries+
                              net.num_loads)*self.T)

            x0 = net.get_var_values()
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            constr = pf.Constraint('variable fixing',net)
            self.assertEqual(constr.name,'variable fixing')

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            G = constr.G
            l = constr.l
            u = constr.u

            # Before
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,0))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,0))
            self.assertEqual(A.nnz,0)
            self.assertTrue(type(G) is coo_matrix)
            self.assertTupleEqual(G.shape,(0,0))
            self.assertEqual(G.nnz,0)
            self.assertTrue(type(l) is np.ndarray)
            self.assertTupleEqual(l.shape,(0,))
            self.assertTrue(type(u) is np.ndarray)
            self.assertTupleEqual(u.shape,(0,))

            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.G_nnz,0)

            A_nnz = net.num_fixed
            constr.analyze()
            self.assertEqual(A_nnz,constr.A_nnz)
            constr.eval(x0)
            self.assertEqual(0,constr.A_nnz)

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            G = constr.G
            l = constr.l
            u = constr.u

            # After
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(net.num_fixed,))
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(net.num_fixed,net.num_vars))
            self.assertEqual(A.nnz,net.num_fixed)
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,net.num_vars))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(G) is coo_matrix)
            self.assertTupleEqual(G.shape,(0,net.num_vars))
            self.assertEqual(G.nnz,0)
            self.assertTrue(type(l) is np.ndarray)
            self.assertTupleEqual(l.shape,(0,))
            self.assertTrue(type(u) is np.ndarray)
            self.assertTupleEqual(u.shape,(0,))

            self.assertTrue(not np.any(np.isinf(b)))
            self.assertTrue(not np.any(np.isnan(b)))

            # Time loop
            for t in range(self.T):

                # bus
                for bus in net.buses:
                    ar = np.where(A.col == bus.index_v_mag[t])[0]
                    self.assertEqual(ar.size,1)
                    self.assertEqual(A.col[ar[0]],bus.index_v_mag[t])
                    if bus.is_regulated_by_gen():
                        self.assertEqual(b[A.row[ar[0]]],bus.v_set[t])
                    else:
                        self.assertEqual(b[A.row[ar[0]]],bus.v_mag[t])
                    ar = np.where(A.col == bus.index_v_ang[t])[0]
                    self.assertEqual(ar.size,1)
                    self.assertEqual(A.col[ar[0]],bus.index_v_ang[t])
                    self.assertEqual(b[A.row[ar[0]]],bus.v_ang[t])

                # Gens
                for gen in net.generators:
                    if gen.is_slack():
                        ar = np.where(A.col == gen.index_P[t])[0]
                        self.assertEqual(ar.size,1)
                        self.assertEqual(A.col[ar[0]],gen.index_P[t])
                        self.assertEqual(b[A.row[ar[0]]],gen.P[t])
                    if gen.is_regulator():
                        ar = np.where(A.col == gen.index_Q[t])[0]
                        self.assertEqual(ar.size,1)
                        self.assertEqual(A.col[ar[0]],gen.index_Q[t])
                        self.assertEqual(A.data[ar[0]],1.)
                        self.assertEqual(b[A.row[ar[0]]],gen.Q[t])

                # Shunts
                for shunt in net.shunts:
                    if shunt.is_switched_v():
                        ar = np.where(A.col == shunt.index_b[t])[0]
                        self.assertEqual(ar.size,1)
                        self.assertEqual(A.col[ar[0]],shunt.index_b[t])
                        self.assertEqual(b[A.row[ar[0]]],shunt.b[t])

                # Branch
                for branch in net.branches:
                    if branch.is_tap_changer():
                        ar = np.where(A.col == branch.index_ratio[t])[0]
                        self.assertEqual(ar.size,1)
                        self.assertEqual(A.col[ar[0]],branch.index_ratio[t])
                        self.assertEqual(b[A.row[ar[0]]],branch.ratio[t])
                    if branch.is_phase_shifter():
                        ar = np.where(A.col == branch.index_phase[t])[0]
                        self.assertEqual(ar.size,1)
                        self.assertEqual(A.col[ar[0]],branch.index_phase[t])
                        self.assertEqual(b[A.row[ar[0]]],branch.phase[t])

                # Vargen
                for vargen in net.var_generators:
                    ar = np.where(A.col == vargen.index_P[t])[0]
                    self.assertEqual(ar.size,1)
                    self.assertEqual(A.col[ar[0]],vargen.index_P[t])
                    self.assertEqual(b[A.row[ar[0]]],vargen.P[t])
                    ar = np.where(A.col == vargen.index_Q[t])[0]
                    self.assertEqual(ar.size,1)
                    self.assertEqual(A.col[ar[0]],vargen.index_Q[t])
                    self.assertEqual(b[A.row[ar[0]]],vargen.Q[t])

                # Batteries
                for bat in net.batteries:
                    ar = np.where(A.col == bat.index_Pc[t])[0]
                    self.assertEqual(ar.size,1)
                    self.assertEqual(A.col[ar[0]],bat.index_Pc[t])
                    self.assertEqual(b[A.row[ar[0]]],max([bat.P[t],0]))
                    ar = np.where(A.col == bat.index_Pd[t])[0]
                    self.assertEqual(ar.size,1)
                    self.assertEqual(A.col[ar[0]],bat.index_Pd[t])
                    self.assertEqual(b[A.row[ar[0]]],max([-bat.P[t],0]))
                    ar = np.where(A.col == bat.index_E[t])[0]
                    self.assertEqual(ar.size,1)
                    self.assertEqual(A.col[ar[0]],bat.index_E[t])
                    self.assertEqual(b[A.row[ar[0]]],bat.E[t])

                # Load
                for load in net.loads:
                    self.assertTrue(load.has_flags('variable','active power'))
                    self.assertTrue(load.has_flags('fixed','active power'))
                    ar = np.where(A.col == load.index_P[t])[0]
                    self.assertEqual(ar.size,1)
                    self.assertEqual(A.col[ar[0]],load.index_P[t])
                    self.assertEqual(b[A.row[ar[0]]],load.P[t])

    def test_constr_FIX_with_outages(self):
        
        # Multiperiod
        for case in test_cases.CASES:
            
            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)

            net.clear_outages()

            gen = net.get_generator(0)
            branch = net.get_branch(0)

            gen.outage = True
            branch.outage = True

            self.assertTrue(gen.is_on_outage())
            self.assertTrue(branch.is_on_outage())

            gen.P = np.random.rand(self.T)
            gen.Q = np.random.rand(self.T)
            branch.ratio = np.random.randn(self.T)
            branch.phase = np.random.randn(self.T)

            net.set_flags('generator',
                          ['variable','fixed'],
                          'any',
                          ['active power', 'reactive power'])
            net.set_flags('branch',
                          ['variable','fixed'],
                          'any',
                          ['tap ratio', 'phase shift'])
            self.assertEqual(net.num_vars,
                             self.T*(2*net.num_generators + 2*net.num_branches))
            self.assertEqual(net.num_vars, net.num_fixed)

            constr = pf.Constraint('variable fixing', net)
            constr.analyze()

            A = constr.A
            b = constr.b

            for t in range(self.T):

                # gen P
                k = np.where(A.col == gen.index_P[t])[0]
                self.assertEqual(k.size, 1)
                k = k[0]
                i = A.row[k]
                self.assertEqual(A.data[k], 1.)
                self.assertEqual(b[i], gen.P[t])

                # gen Q
                k = np.where(A.col == gen.index_Q[t])[0]
                self.assertEqual(k.size, 1)
                k = k[0]
                i = A.row[k]
                self.assertEqual(A.data[k], 1.)
                self.assertEqual(b[i], gen.Q[t])

                # branch ratio
                k = np.where(A.col == branch.index_ratio[t])[0]
                self.assertEqual(k.size, 1)
                k = k[0]
                i = A.row[k]
                self.assertEqual(A.data[k], 1.)
                self.assertEqual(b[i], branch.ratio[t])

                # branch phase
                k = np.where(A.col == branch.index_phase[t])[0]
                self.assertEqual(k.size, 1)
                k = k[0]
                i = A.row[k]
                self.assertEqual(A.data[k], 1.)
                self.assertEqual(b[i], branch.phase[t])

            # Disconnect
            net.clear_outages()
            net.clear_flags()
            self.assertEqual(net.num_vars, 0)
            for bus in net.buses:
                if bus.degree == 1:
                    self.assertEqual(len(bus.branches), 1)
                    bus.branches[0].outage = True
                    self.assertTrue(bus.branches[0].is_on_outage())
                    net.set_flags_of_component(bus,
                                               ['variable', 'fixed'],
                                               ['voltage magnitude', 'voltage angle'])
                    self.assertEqual(net.num_vars, 2*self.T)
                    self.assertEqual(net.num_vars, net.num_fixed)
                    self.assertTrue(bus.has_flags('variable', ['voltage magnitude',
                                                               'voltage angle']))
                    self.assertTrue(bus.has_flags('fixed', ['voltage magnitude',
                                                            'voltage angle']))
                    constr = pf.Constraint('variable fixing', net)
                    constr.analyze()
                    A = constr.A
                    b = constr.b
                    self.assertEqual(A.shape[0], 2*self.T)
                    for t in range(self.T):

                        # bus v mag
                        k = np.where(A.col == bus.index_v_mag[t])[0]
                        self.assertEqual(k.size, 1)
                        k = k[0]
                        self.assertEqual(A.data[k], 1.)
                        self.assertEqual(b[A.row[k]], bus.v_mag[t])

                        # bus v ang
                        k = np.where(A.col == bus.index_v_ang[t])[0]
                        self.assertEqual(k.size, 1)
                        k = k[0]
                        self.assertEqual(A.data[k], 1.)
                        self.assertEqual(b[A.row[k]], bus.v_ang[t])
                    break                                    
                    
    def test_constr_BOUND(self):

        # Single period
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case)
            self.assertEqual(net.num_periods,1)

            # add vargens
            net.add_var_generators_from_parameters(net.get_load_buses(),80.,50.,30.,5,0.05)
            for vargen in net.var_generators:
                vargen.P = vargen.index*1.5
                vargen.Q = vargen.index*2.5
                vargen.P_ava = vargen.index*3.
                vargen.P_max = 100.
                vargen.P_min = 0.
                vargen.Q_max = 50.
                vargen.Q_min = -50.
            self.assertGreater(net.num_var_generators,0)

            self.assertEqual(net.num_bounded,0)
            self.assertEqual(net.num_vars,0)
            self.assertEqual(net.num_fixed,0)

            # loads
            for load in net.loads:
                load.P_min = -2.4*(load.index+1)
                load.P_max = 3.3*(load.index+1)
                load.Q_min = 1.2*(load.index+2.)
                load.Q_max = 5.8*(load.index+3.)
            
            # Vars
            net.set_flags('bus',
                          'variable',
                          'regulated by generator',
                          ['voltage magnitude','voltage angle'])
            net.set_flags('generator',
                          'variable',
                          'regulator',
                          ['active power','reactive power'])
            net.set_flags('load',
                          'variable',
                          'adjustable active power',
                          ['active power','reactive power'])
            net.set_flags('branch',
                          'variable',
                          'tap changer',
                          'tap ratio')
            net.set_flags('branch',
                          'variable',
                          'phase shifter',
                          'phase shift')
            net.set_flags('shunt',
                          'variable',
                          'switching - v',
                          'susceptance')
            net.set_flags('variable generator',
                          'variable',
                          'any',
                          ['active power','reactive power'])
            net.set_flags('battery',
                          'variable',
                          'any',
                          ['charging power','energy level'])
            net.set_flags('vsc converter',
                          'variable',
                          'any',
                          ['dc power', 'active power', 'reactive power'])
            net.set_flags('facts',
                          'variable',
                          'any',
                          ['series voltage magnitude','series voltage angle',
                           'active power', 'reactive power'])
            net.set_flags('dc bus',
                          'variable',
                          'any',
                          'voltage')
            net.set_flags('csc converter',
                          'variable',
                          'any',
                          'all')

            num_vars_saved = net.num_vars
            self.assertGreater(net.num_vars,0)
            self.assertEqual(net.num_fixed,0)
            self.assertEqual(net.num_vars,
                             (net.get_num_buses_reg_by_gen()*2 +
                              net.get_num_reg_gens()*2 +
                              2*net.get_num_P_adjust_loads() +
                              net.get_num_tap_changers() +
                              net.get_num_phase_shifters()*1 +
                              net.get_num_switched_v_shunts() +
                              net.num_var_generators*2+
                              3*net.num_batteries+
                              4*net.num_vsc_converters+
                              9*net.num_facts +
                              net.num_dc_buses +
                              6*net.num_csc_converters))
            
            x0 = net.get_var_values()
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            constr = pf.Constraint('variable bounds',net)
            self.assertEqual(constr.name,'variable bounds')

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            G = constr.G
            l = constr.l
            u = constr.u

            # Before
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,0))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,0))
            self.assertEqual(A.nnz,0)
            self.assertTrue(type(G) is coo_matrix)
            self.assertTupleEqual(G.shape,(0,0))
            self.assertEqual(G.nnz,0)
            self.assertTrue(type(l) is np.ndarray)
            self.assertTupleEqual(l.shape,(0,))
            self.assertTrue(type(u) is np.ndarray)
            self.assertTupleEqual(u.shape,(0,))

            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.G_nnz,0)

            constr.analyze()
            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.G_nnz,0)
            constr.eval(x0)
            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.G_nnz,0)

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            G = constr.G
            l = constr.l
            u = constr.u

            # After
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,net.num_vars))
            self.assertEqual(A.nnz,0)
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,net.num_vars))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(G) is coo_matrix)
            self.assertTupleEqual(G.shape,(net.num_vars,net.num_vars))
            self.assertEqual(G.nnz,net.num_vars)
            self.assertTrue(np.all(G.row == np.array(range(net.num_vars))))
            self.assertTrue(np.all(G.col == np.array(range(net.num_vars))))
            self.assertTrue(np.all(G.data == np.ones(net.num_vars)))
            self.assertTrue(type(l) is np.ndarray)
            self.assertTupleEqual(l.shape,(net.num_vars,))
            self.assertTrue(type(u) is np.ndarray)
            self.assertTupleEqual(u.shape,(net.num_vars,))

            E = G-eye(net.num_vars)
            self.assertGreater(G.nnz,0)
            self.assertGreater(norm(G.data,np.inf),0.5)
            self.assertEqual(E.nnz,0)

            self.assertTrue(not np.any(np.isinf(l)))
            self.assertTrue(not np.any(np.isnan(l)))
            self.assertTrue(not np.any(np.isinf(u)))
            self.assertTrue(not np.any(np.isnan(u)))
            self.assertTrue(not np.any(np.isinf(b)))
            self.assertTrue(not np.any(np.isnan(b)))            

            # Bounds
            for bus in net.buses:
                if bus.is_regulated_by_gen():
                    self.assertTrue(bus.has_flags('variable',
                                                  ['voltage magnitude',
                                                   'voltage angle']))
                    self.assertEqual(u[bus.index_v_mag],pf.BUS_INF_V_MAG)
                    self.assertEqual(u[bus.index_v_ang],pf.BUS_INF_V_ANG)
                    self.assertEqual(l[bus.index_v_mag],0.)
                    self.assertEqual(l[bus.index_v_ang],-pf.BUS_INF_V_ANG)
                else:
                    self.assertFalse(bus.has_flags('variable',
                                                   ['voltage magnitude',
                                                    'voltage angle']))

            for branch in net.branches:
                if branch.is_tap_changer():
                    self.assertTrue(branch.has_flags('variable','tap ratio')) 
                    self.assertEqual(u[branch.index_ratio],pf.BRANCH_INF_RATIO)
                    self.assertEqual(l[branch.index_ratio],0.)
                else:
                    self.assertFalse(branch.has_flags('variable','tap ratio'))
                if branch.is_phase_shifter():
                    self.assertTrue(branch.has_flags('variable','phase shift'))
                    self.assertLess(np.abs(u[branch.index_phase]-np.pi*2.),1e-10)
                    self.assertLess(np.abs(l[branch.index_phase]+np.pi*2.),1e-10)
                else:
                    self.assertFalse(branch.has_flags('variable','phase shift'))

            for gen in net.generators:
                if gen.is_regulator():
                    self.assertTrue(gen.has_flags('variable',['active power','reactive power']))
                    self.assertEqual(u[gen.index_P],pf.GEN_INF_P)
                    self.assertEqual(u[gen.index_Q],pf.GEN_INF_Q)
                    self.assertEqual(l[gen.index_P],-pf.GEN_INF_P)
                    self.assertEqual(l[gen.index_Q],-pf.GEN_INF_Q)
                else:
                    self.assertFalse(gen.has_flags('variable',
                                                   ['active power','reactive power']))

            for load in net.loads:
                self.assertTrue(load.has_flags('variable','active power'))
                self.assertTrue(load.has_flags('variable','reactive power'))
                self.assertTrue(load.has_flags('variable',['active power','reactive power']))
                self.assertEqual(u[load.index_P],pf.LOAD_INF_P)
                self.assertEqual(l[load.index_P],-pf.LOAD_INF_P)
                self.assertEqual(u[load.index_Q],pf.LOAD_INF_Q)
                self.assertEqual(l[load.index_Q],-pf.LOAD_INF_Q)

            for vargen in net.var_generators:
                self.assertTrue(vargen.has_flags('variable',
                                                 ['active power','reactive power']))
                self.assertEqual(u[vargen.index_P],pf.VARGEN_INF_P)
                self.assertEqual(u[vargen.index_Q],pf.VARGEN_INF_Q)
                self.assertEqual(l[vargen.index_P],-pf.VARGEN_INF_P)
                self.assertEqual(l[vargen.index_Q],-pf.VARGEN_INF_Q)

            for shunt in net.shunts:
                if shunt.is_switched_v():
                    self.assertTrue(shunt.has_flags('variable','susceptance'))
                    self.assertEqual(u[shunt.index_b],pf.SHUNT_INF_SUSC)
                    self.assertEqual(l[shunt.index_b],-pf.SHUNT_INF_SUSC)
                else:
                    self.assertFalse(shunt.has_flags('variable','susceptance'))

            for bat in net.batteries:
                self.assertTrue(bat.has_flags('variable','charging power'))
                self.assertTrue(bat.has_flags('variable','energy level'))
                self.assertEqual(u[bat.index_Pc],pf.BAT_INF_P)
                self.assertEqual(l[bat.index_Pc],0.)
                self.assertEqual(u[bat.index_Pd],pf.BAT_INF_P)
                self.assertEqual(l[bat.index_Pd],0.)
                self.assertEqual(u[bat.index_E],pf.BAT_INF_E)
                self.assertEqual(l[bat.index_E],0.)
                
            for vsc_conv in net.vsc_converters:
                self.assertTrue(vsc_conv.has_flags('variable','active power'))
                self.assertTrue(vsc_conv.has_flags('variable','reactive power'))
                self.assertTrue(vsc_conv.has_flags('variable','dc power'))
                self.assertEqual(u[vsc_conv.index_P],pf.CONVVSC_INF_P)
                self.assertEqual(l[vsc_conv.index_P],-pf.CONVVSC_INF_P)
                self.assertEqual(u[vsc_conv.index_Q],pf.CONVVSC_INF_Q)
                self.assertEqual(l[vsc_conv.index_Q],-pf.CONVVSC_INF_Q)
                self.assertEqual(u[vsc_conv.index_P_dc],pf.CONVVSC_INF_PDC)
                self.assertEqual(l[vsc_conv.index_P_dc],-pf.CONVVSC_INF_PDC)
                self.assertEqual(u[vsc_conv.index_i_dc],pf.CONVVSC_INF_PDC)
                self.assertEqual(l[vsc_conv.index_i_dc],-pf.CONVVSC_INF_PDC)
            
            for facts in net.facts:
                self.assertTrue(facts.has_flags('variable','series voltage magnitude'))
                self.assertTrue(facts.has_flags('variable','series voltage angle'))
                self.assertTrue(facts.has_flags('variable','active power'))
                self.assertTrue(facts.has_flags('variable','reactive power'))
                self.assertEqual(u[facts.index_v_mag_s],pf.FACTS_INF_VMAG_S)
                self.assertEqual(l[facts.index_v_mag_s],0.)
                self.assertEqual(u[facts.index_v_ang_s],pf.FACTS_INF_VANG_S)
                self.assertEqual(l[facts.index_v_ang_s],-pf.FACTS_INF_VANG_S)
                self.assertEqual(u[facts.index_P_k],pf.FACTS_INF_P)
                self.assertEqual(l[facts.index_P_k],-pf.FACTS_INF_P)
                self.assertEqual(u[facts.index_P_m],pf.FACTS_INF_P)
                self.assertEqual(l[facts.index_P_m],-pf.FACTS_INF_P)
                self.assertEqual(u[facts.index_P_dc],pf.FACTS_INF_P)
                self.assertEqual(l[facts.index_P_dc],-pf.FACTS_INF_P)
                self.assertEqual(u[facts.index_Q_k],pf.FACTS_INF_Q)
                self.assertEqual(l[facts.index_Q_k],-pf.FACTS_INF_Q)
                self.assertEqual(u[facts.index_Q_m],pf.FACTS_INF_Q)
                self.assertEqual(l[facts.index_Q_m],-pf.FACTS_INF_Q)
                self.assertEqual(u[facts.index_Q_s],pf.FACTS_INF_Q)
                self.assertEqual(l[facts.index_Q_s],-pf.FACTS_INF_Q)
                self.assertEqual(u[facts.index_Q_sh],pf.FACTS_INF_Q)
                self.assertEqual(l[facts.index_Q_sh],-pf.FACTS_INF_Q)

            for bus in net.dc_buses:
                self.assertEqual(u[bus.index_v], pf.BUSDC_INF_V)
                self.assertEqual(l[bus.index_v], -pf.BUSDC_INF_V)

            for csc in net.csc_converters:
                self.assertEqual(u[csc.index_P], pf.CONVCSC_INF_P)
                self.assertEqual(l[csc.index_P], -pf.CONVCSC_INF_P)
                self.assertEqual(u[csc.index_Q], pf.CONVCSC_INF_Q)
                self.assertEqual(l[csc.index_Q], -pf.CONVCSC_INF_Q)
                self.assertEqual(u[csc.index_P_dc], pf.CONVCSC_INF_PDC)
                self.assertEqual(l[csc.index_P_dc], -pf.CONVCSC_INF_PDC)
                self.assertEqual(u[csc.index_i_dc], pf.CONVCSC_INF_PDC)
                self.assertEqual(l[csc.index_i_dc], -pf.CONVCSC_INF_PDC)
                self.assertEqual(u[csc.index_angle], pf.CONVCSC_INF_ANGLE)
                self.assertEqual(l[csc.index_angle], -pf.CONVCSC_INF_ANGLE)
                self.assertEqual(u[csc.index_ratio], pf.CONVCSC_INF_RATIO)
                self.assertEqual(l[csc.index_ratio], -pf.CONVCSC_INF_RATIO)

            # Add bounded flags
            net.set_flags('bus',
                          'bounded',
                          'regulated by generator',
                          ['voltage magnitude','voltage angle'])
            net.set_flags('generator',
                          'bounded',
                          'regulator',
                          ['active power','reactive power'])
            net.set_flags('load',
                          'bounded',
                          'adjustable active power',
                          ['active power','reactive power'])
            net.set_flags('branch',
                          'bounded',
                          'tap changer',
                          'tap ratio')
            net.set_flags('branch',
                          'bounded',
                          'phase shifter',
                          'phase shift')
            net.set_flags('shunt',
                          'bounded',
                          'switching - v',
                          'susceptance')
            net.set_flags('variable generator',
                          'bounded',
                          'any',
                          ['active power','reactive power'])
            net.set_flags('battery',
                          'bounded',
                          'any',
                          ['charging power','energy level'])
            net.set_flags('vsc converter',
                          'bounded',
                          'any',
                          ['dc power', 'active power','reactive power'])
            net.set_flags('facts',
                          'bounded',
                          'any',
                          ['series voltage magnitude','series voltage angle',
                           'active power','reactive power'])
            net.set_flags('dc bus',
                          'bounded',
                          'any',
                          'voltage')
            net.set_flags('csc converter',
                          'bounded',
                          'any',
                          'all')
            self.assertEqual(net.num_vars,num_vars_saved)
            self.assertEqual(net.num_fixed,0)
            self.assertEqual(net.num_bounded,net.num_vars)

            x0 = net.get_var_values()
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            constr = pf.Constraint('variable bounds',net)
            self.assertEqual(constr.name,'variable bounds')

            constr.analyze()

            G = constr.G
            l = constr.l
            u = constr.u

            self.assertTrue(type(G) is coo_matrix)
            self.assertTupleEqual(G.shape,(net.num_vars,net.num_vars))
            self.assertEqual(G.nnz,net.num_vars)
            self.assertTrue(np.all(G.row == np.array(range(net.num_vars))))
            self.assertTrue(np.all(G.col == np.array(range(net.num_vars))))
            self.assertTrue(np.all(G.data == np.ones(net.num_vars)))
            self.assertTrue(type(l) is np.ndarray)
            self.assertTupleEqual(l.shape,(net.num_vars,))
            self.assertTrue(type(u) is np.ndarray)
            self.assertTupleEqual(u.shape,(net.num_vars,))

            E = G-eye(net.num_vars)
            self.assertGreater(G.nnz,0)
            self.assertGreater(norm(G.data,np.inf),0.5)
            self.assertEqual(E.nnz,0)

            # Bounds
            for bus in net.buses:
                if bus.is_regulated_by_gen():
                    self.assertTrue(bus.has_flags('bounded',
                                                  ['voltage magnitude',
                                                   'voltage angle']))
                    self.assertTrue(bus.has_flags('variable',
                                                  ['voltage magnitude',
                                                   'voltage angle']))
                    self.assertEqual(u[bus.index_v_mag],bus.v_max)
                    self.assertEqual(u[bus.index_v_ang],pf.BUS_INF_V_ANG)
                    self.assertEqual(l[bus.index_v_mag],bus.v_min)
                    self.assertEqual(l[bus.index_v_ang],-pf.BUS_INF_V_ANG)
                else:
                    self.assertFalse(bus.has_flags('bounded',
                                                   ['voltage magnitude',
                                                    'voltage angle']))

            for branch in net.branches:
                if branch.is_tap_changer():
                    self.assertTrue(branch.has_flags('bounded','tap ratio'))
                    self.assertEqual(u[branch.index_ratio],branch.ratio_max)
                    self.assertEqual(l[branch.index_ratio],branch.ratio_min)
                else:
                    self.assertFalse(branch.has_flags('bounded','tap ratio'))
                if branch.is_phase_shifter():
                    self.assertTrue(branch.has_flags('bounded','phase shift'))
                    self.assertEqual(u[branch.index_phase],branch.phase_max)
                    self.assertEqual(l[branch.index_phase],branch.phase_min)
                else:
                    self.assertFalse(branch.has_flags('bounded','phase shift'))

            for gen in net.generators:
                if gen.is_regulator():
                    self.assertTrue(gen.has_flags('bounded',['active power','reactive power']))
                    self.assertEqual(u[gen.index_P],gen.P_max)
                    self.assertEqual(u[gen.index_Q],gen.Q_max)
                    self.assertEqual(l[gen.index_P],gen.P_min)
                    self.assertEqual(l[gen.index_Q],gen.Q_min)
                else:
                    self.assertFalse(gen.has_flags('bounded',['active power','reactive power']))

            for load in net.loads:
                self.assertTrue(load.has_flags('bounded','active power'))
                self.assertTrue(load.has_flags('bounded','reactive power'))
                self.assertTrue(load.has_flags('bounded',['active power','reactive power']))
                self.assertEqual(u[load.index_P],load.P_max)
                self.assertEqual(l[load.index_P],load.P_min)
                self.assertEqual(u[load.index_Q],load.Q_max)
                self.assertEqual(l[load.index_Q],load.Q_min)

            for vargen in net.var_generators:
                self.assertTrue(vargen.has_flags('bounded',['active power','reactive power']))
                self.assertEqual(u[vargen.index_P],vargen.P_ava)
                self.assertEqual(u[vargen.index_Q],vargen.Q_max)
                self.assertEqual(l[vargen.index_P],vargen.P_min)
                self.assertEqual(l[vargen.index_Q],vargen.Q_min)

            for shunt in net.shunts:
                if shunt.is_switched_v():
                    self.assertTrue(shunt.has_flags('bounded','susceptance'))
                    self.assertEqual(u[shunt.index_b],shunt.b_max)
                    self.assertEqual(l[shunt.index_b],shunt.b_min)
                else:
                    self.assertFalse(shunt.has_flags('bounded','susceptance'))

            for bat in net.batteries:
                self.assertTrue(bat.has_flags('bounded','charging power'))
                self.assertTrue(bat.has_flags('bounded','energy level'))
                self.assertEqual(u[bat.index_Pc],bat.P_max)
                self.assertEqual(l[bat.index_Pc],0.)
                self.assertEqual(u[bat.index_Pd],-bat.P_min)
                self.assertEqual(l[bat.index_Pd],0.)
                self.assertEqual(u[bat.index_E],bat.E_max)
                self.assertEqual(l[bat.index_E],0.)
                
            for vsc_conv in net.vsc_converters:
                self.assertTrue(vsc_conv.has_flags('bounded','active power'))
                self.assertTrue(vsc_conv.has_flags('bounded','reactive power'))
                self.assertTrue(vsc_conv.has_flags('bounded','dc power'))
                self.assertEqual(u[vsc_conv.index_P],vsc_conv.P_max)
                self.assertEqual(l[vsc_conv.index_P],vsc_conv.P_min)
                self.assertEqual(u[vsc_conv.index_Q],vsc_conv.Q_max)
                self.assertEqual(l[vsc_conv.index_Q],vsc_conv.Q_min)
                self.assertEqual(u[vsc_conv.index_P_dc],pf.CONVVSC_INF_PDC)
                self.assertEqual(l[vsc_conv.index_P_dc],-pf.CONVVSC_INF_PDC)
                self.assertEqual(u[vsc_conv.index_i_dc],pf.CONVVSC_INF_PDC)
                self.assertEqual(l[vsc_conv.index_i_dc],-pf.CONVVSC_INF_PDC)
                
            for facts in net.facts:
                self.assertTrue(facts.has_flags('bounded','series voltage magnitude'))
                self.assertTrue(facts.has_flags('bounded','series voltage angle'))
                self.assertTrue(facts.has_flags('bounded','active power'))
                self.assertTrue(facts.has_flags('bounded','reactive power'))
                self.assertEqual(u[facts.index_v_mag_s],facts.v_max_s)
                self.assertEqual(l[facts.index_v_mag_s],0.)
                self.assertEqual(u[facts.index_v_ang_s],pf.FACTS_INF_VANG_S)
                self.assertEqual(l[facts.index_v_ang_s],-pf.FACTS_INF_VANG_S)
                self.assertEqual(u[facts.index_P_k],pf.FACTS_INF_P)
                self.assertEqual(l[facts.index_P_k],-pf.FACTS_INF_P)
                self.assertEqual(u[facts.index_P_m],pf.FACTS_INF_P)
                self.assertEqual(l[facts.index_P_m],-pf.FACTS_INF_P)
                self.assertEqual(u[facts.index_P_dc],facts.P_max_dc)
                self.assertEqual(l[facts.index_P_dc],-facts.P_max_dc)
                self.assertEqual(u[facts.index_Q_k],pf.FACTS_INF_Q)
                self.assertEqual(l[facts.index_Q_k],-pf.FACTS_INF_Q)
                self.assertEqual(u[facts.index_Q_m],pf.FACTS_INF_Q)
                self.assertEqual(l[facts.index_Q_m],-pf.FACTS_INF_Q)
                self.assertEqual(u[facts.index_Q_s],facts.Q_max_s)
                self.assertEqual(l[facts.index_Q_s],facts.Q_min_s)
                self.assertEqual(u[facts.index_Q_sh],facts.Q_max_sh)
                self.assertEqual(l[facts.index_Q_sh],facts.Q_min_sh)

            for bus in net.dc_buses:
                self.assertEqual(u[bus.index_v], pf.BUSDC_INF_V)
                self.assertEqual(l[bus.index_v], -pf.BUSDC_INF_V)

            for csc in net.csc_converters:
                self.assertEqual(u[csc.index_P], pf.CONVCSC_INF_P)
                self.assertEqual(l[csc.index_P], -pf.CONVCSC_INF_P)
                self.assertEqual(u[csc.index_Q], pf.CONVCSC_INF_Q)
                self.assertEqual(l[csc.index_Q], -pf.CONVCSC_INF_Q)
                self.assertEqual(u[csc.index_P_dc], pf.CONVCSC_INF_PDC)
                self.assertEqual(l[csc.index_P_dc], -pf.CONVCSC_INF_PDC)
                self.assertEqual(u[csc.index_i_dc], pf.CONVCSC_INF_PDC)
                self.assertEqual(l[csc.index_i_dc], -pf.CONVCSC_INF_PDC)
                self.assertEqual(u[csc.index_angle], pf.CONVCSC_INF_ANGLE)
                self.assertEqual(l[csc.index_angle], -pf.CONVCSC_INF_ANGLE)
                self.assertEqual(u[csc.index_ratio], pf.CONVCSC_INF_RATIO)
                self.assertEqual(l[csc.index_ratio], -pf.CONVCSC_INF_RATIO)
                
            # Sensitivities
            net.clear_sensitivities()
            for branch in net.branches:
                self.assertEqual(branch.sens_ratio_u_bound, 0.)
                self.assertEqual(branch.sens_ratio_l_bound, 0.)
                self.assertEqual(branch.sens_phase_u_bound, 0.)
                self.assertEqual(branch.sens_phase_l_bound, 0.)
            for bus in net.buses:
                self.assertEqual(bus.sens_P_balance,0.)
                self.assertEqual(bus.sens_Q_balance,0.)
                self.assertEqual(bus.sens_v_mag_u_bound,0.)
                self.assertEqual(bus.sens_v_mag_l_bound,0.)
                self.assertEqual(bus.sens_v_ang_u_bound,0.)
                self.assertEqual(bus.sens_v_ang_l_bound,0.)
            for gen in net.generators:
                self.assertEqual(gen.sens_P_u_bound,0.)
                self.assertEqual(gen.sens_P_l_bound,0.)
                self.assertEqual(gen.sens_Q_u_bound,0.)
                self.assertEqual(gen.sens_Q_l_bound,0.)
            for load in net.loads:
                self.assertEqual(load.sens_P_u_bound,0.)
                self.assertEqual(load.sens_P_l_bound,0.)
            for shunt in net.shunts:
                self.assertEqual(shunt.sens_b_u_bound, 0.)
                self.assertEqual(shunt.sens_b_l_bound, 0.)
                
            mu = np.random.randn(net.num_vars)
            pi = np.random.randn(net.num_vars)

            constr.store_sensitivities(None,None,mu,pi)

            # Branch sens
            for branch in net.branches:
                if branch.is_tap_changer():
                    self.assertEqual(branch.sens_ratio_u_bound, mu[branch.index_ratio])
                    self.assertEqual(branch.sens_ratio_l_bound, pi[branch.index_ratio])
                else:
                    self.assertEqual(branch.sens_ratio_u_bound, 0.)
                    self.assertEqual(branch.sens_ratio_l_bound, 0.)
                if branch.is_phase_shifter():
                    self.assertEqual(branch.sens_phase_u_bound, mu[branch.index_phase])
                    self.assertEqual(branch.sens_phase_l_bound, pi[branch.index_phase])
                else:
                    self.assertEqual(branch.sens_phase_u_bound, 0.)
                    self.assertEqual(branch.sens_phase_l_bound, 0.)

            # Bus sens
            for bus in net.buses:
                self.assertEqual(bus.sens_P_balance,0.)
                self.assertEqual(bus.sens_Q_balance,0.)
                if bus.is_regulated_by_gen():
                    self.assertTrue(bus.has_flags('variable','voltage angle'))
                    self.assertNotEqual(bus.sens_v_ang_u_bound,0.)
                    self.assertNotEqual(bus.sens_v_ang_l_bound,0.)
                    self.assertEqual(bus.sens_v_mag_u_bound,mu[bus.index_v_mag])
                    self.assertEqual(bus.sens_v_mag_l_bound,pi[bus.index_v_mag])
                    self.assertEqual(bus.sens_v_ang_u_bound,mu[bus.index_v_ang])
                    self.assertEqual(bus.sens_v_ang_l_bound,pi[bus.index_v_ang])
                else:
                    self.assertEqual(bus.sens_v_mag_u_bound,0.)
                    self.assertEqual(bus.sens_v_mag_l_bound,0.)
                    self.assertEqual(bus.sens_v_ang_u_bound,0.)
                    self.assertEqual(bus.sens_v_ang_l_bound,0.)

            # Gen sens
            for gen in net.generators:
                if gen.is_regulator():
                    self.assertTrue(gen.has_flags('variable','active power'))
                    self.assertNotEqual(gen.sens_P_u_bound,0.)
                    self.assertNotEqual(gen.sens_P_l_bound,0.)
                    self.assertEqual(gen.sens_P_u_bound, mu[gen.index_P])
                    self.assertEqual(gen.sens_P_l_bound, pi[gen.index_P])
                    self.assertEqual(gen.sens_Q_u_bound, mu[gen.index_Q])
                    self.assertEqual(gen.sens_Q_l_bound, pi[gen.index_Q])
                else:
                    self.assertEqual(gen.sens_P_u_bound, 0.)
                    self.assertEqual(gen.sens_P_l_bound, 0.)
                    self.assertEqual(gen.sens_Q_u_bound, 0.)
                    self.assertEqual(gen.sens_Q_l_bound, 0.)

            # Load sens
            for load in net.loads:
                self.assertTrue(load.has_flags('variable','active power'))
                self.assertNotEqual(load.sens_P_u_bound,0.)
                self.assertNotEqual(load.sens_P_l_bound,0.)
                self.assertEqual(load.sens_P_u_bound,mu[load.index_P])
                self.assertEqual(load.sens_P_l_bound,pi[load.index_P])

            # Shunts
            for shunt in net.shunts:
                if shunt.is_switched_v():
                    self.assertEqual(shunt.sens_b_u_bound,mu[shunt.index_b])
                    self.assertEqual(shunt.sens_b_l_bound,pi[shunt.index_b])
                else:
                    self.assertEqual(shunt.sens_b_u_bound, 0.)
                    self.assertEqual(shunt.sens_b_l_bound, 0.)

        # Multi period
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)

            # add vargens
            net.add_var_generators_from_parameters(net.get_load_buses(),80.,50.,30.,5,0.05)
            for vargen in net.var_generators:
                vargen.P = np.random.rand(self.T)
                vargen.Q = np.random.rand(self.T)
                vargen.P_ava = vargen.P*3.4
                vargen.P_max = 100.
                vargen.P_min = 0.
                vargen.Q_max = 50.
                vargen.Q_min = -50.
                self.assertEqual(vargen.num_periods,self.T)
                for t in range(self.T):
                    self.assertEqual(vargen.P_ava[t],vargen.P[t]*3.4)
            self.assertGreater(net.num_var_generators,0)
            self.assertEqual(net.num_bounded,0)
            self.assertEqual(net.num_vars,0)
            self.assertEqual(net.num_fixed,0)

            # add batteries
            gen_buses = net.get_generator_buses()
            net.add_batteries_from_parameters(gen_buses,20.,40.,0.8,0.9)

            # loads
            for load in net.loads:
                load.P_min = -2.4*(load.index+1)*np.array(range(net.num_periods))
                load.P_max = 3.3*(load.index+1)*np.array(range(net.num_periods))
                load.Q = 3.5*load.index*np.array(range(net.num_periods))
                load.Q_min = 1.2*(load.index+1)*np.array(range(net.num_periods))
                load.Q_max = 7.5*(load.index+1)*np.array(range(net.num_periods))

            # Vars
            net.set_flags('bus',
                          'variable',
                          'any',
                          ['voltage magnitude','voltage angle'])
            net.set_flags('generator',
                          'variable',
                          'any',
                          ['active power','reactive power'])
            net.set_flags('load',
                          'variable',
                          'any',
                          ['active power','reactive power'])
            net.set_flags('branch',
                          'variable',
                          'tap changer',
                          ['tap ratio'])
            net.set_flags('branch',
                          'variable',
                          'phase shifter',
                          'phase shift')
            net.set_flags('shunt',
                          'variable',
                          'switching - v',
                          ['susceptance'])
            net.set_flags('variable generator',
                          'variable',
                          'any',
                          ['active power','reactive power'])
            net.set_flags('battery',
                          'variable',
                          'any',
                          ['charging power','energy level'])
            net.set_flags('vsc converter',
                          'variable',
                          'any',
                          ['dc power', 'active power', 'reactive power'])
            net.set_flags('facts',
                          'variable',
                          'any',
                          ['series voltage magnitude','series voltage angle',
                           'active power', 'reactive power'])
            net.set_flags('dc bus',
                          'variable',
                          'any',
                          'voltage')
            net.set_flags('csc converter',
                          'variable',
                          'any',
                          'all')

            self.assertGreater(net.num_vars,0)
            self.assertEqual(net.num_fixed,0)
            self.assertEqual(net.num_vars,
                             (net.num_buses*2 +
                              net.num_generators*2 +
                              2*net.num_loads +
                              net.get_num_tap_changers() +
                              net.get_num_phase_shifters() +
                              net.get_num_switched_v_shunts() +
                              net.num_var_generators*2 +
                              3*net.num_batteries +
                              4*net.num_vsc_converters +
                              9*net.num_facts +
                              net.num_dc_buses +
                              6*net.num_csc_converters)*self.T)

            x0 = net.get_var_values()
            constr = pf.Constraint('variable bounds',net)
            self.assertEqual(constr.name,'variable bounds')
            constr.analyze()
            constr.eval(x0)

            G = constr.G
            l = constr.l
            u = constr.u

            self.assertTrue(type(G) is coo_matrix)
            self.assertTupleEqual(G.shape,(net.num_vars,net.num_vars))
            self.assertEqual(G.nnz,net.num_vars)
            self.assertTrue(np.all(G.row == np.array(range(net.num_vars))))
            self.assertTrue(np.all(G.col == np.array(range(net.num_vars))))
            self.assertTrue(np.all(G.data == np.ones(net.num_vars)))
            self.assertTrue(type(l) is np.ndarray)
            self.assertTupleEqual(l.shape,(net.num_vars,))
            self.assertTrue(type(u) is np.ndarray)
            self.assertTupleEqual(u.shape,(net.num_vars,))

            for t in range(self.T):
                for bus in net.buses:
                    self.assertEqual(u[bus.index_v_mag[t]],pf.BUS_INF_V_MAG)
                    self.assertEqual(u[bus.index_v_ang[t]],pf.BUS_INF_V_ANG)
                    self.assertEqual(l[bus.index_v_mag[t]],0)
                    self.assertEqual(l[bus.index_v_ang[t]],-pf.BUS_INF_V_ANG)
                for gen in net.generators:
                    self.assertEqual(u[gen.index_P[t]],pf.GEN_INF_P)
                    self.assertEqual(u[gen.index_Q[t]],pf.GEN_INF_Q)
                    self.assertEqual(l[gen.index_P[t]],-pf.GEN_INF_P)
                    self.assertEqual(l[gen.index_Q[t]],-pf.GEN_INF_Q)
                for branch in net.branches:
                    if branch.is_tap_changer():
                        self.assertEqual(u[branch.index_ratio[t]],pf.BRANCH_INF_RATIO)
                        self.assertEqual(l[branch.index_ratio[t]],0.)
                    if branch.is_phase_shifter():
                        self.assertLess(np.abs(u[branch.index_phase[t]]-np.pi*2.),1e-10)
                        self.assertLess(np.abs(l[branch.index_phase[t]]+np.pi*2.),1e-10)
                for vargen in net.var_generators:
                    self.assertEqual(u[vargen.index_P[t]],pf.VARGEN_INF_P)
                    self.assertEqual(u[vargen.index_Q[t]],pf.VARGEN_INF_Q)
                    self.assertEqual(l[vargen.index_P[t]],-pf.VARGEN_INF_P)
                    self.assertEqual(l[vargen.index_Q[t]],-pf.VARGEN_INF_Q)
                for load in net.loads:
                    self.assertEqual(u[load.index_P[t]],pf.LOAD_INF_P)
                    self.assertEqual(l[load.index_P[t]],-pf.LOAD_INF_P)
                    self.assertEqual(u[load.index_Q[t]],pf.LOAD_INF_Q)
                    self.assertEqual(l[load.index_Q[t]],-pf.LOAD_INF_Q)
                for shunt in net.shunts:
                    if shunt.is_switched_v():
                        self.assertEqual(u[shunt.index_b[t]],pf.SHUNT_INF_SUSC)
                        self.assertEqual(l[shunt.index_b[t]],-pf.SHUNT_INF_SUSC)
                for vsc_conv in net.vsc_converters:
                    self.assertTrue(vsc_conv.has_flags('variable','active power'))
                    self.assertTrue(vsc_conv.has_flags('variable','reactive power'))
                    self.assertTrue(vsc_conv.has_flags('variable','dc power'))
                    self.assertEqual(u[vsc_conv.index_P[t]],pf.CONVVSC_INF_P)
                    self.assertEqual(l[vsc_conv.index_P[t]],-pf.CONVVSC_INF_P)
                    self.assertEqual(u[vsc_conv.index_Q[t]],pf.CONVVSC_INF_Q)
                    self.assertEqual(l[vsc_conv.index_Q[t]],-pf.CONVVSC_INF_Q)
                    self.assertEqual(u[vsc_conv.index_P_dc[t]],pf.CONVVSC_INF_PDC)
                    self.assertEqual(l[vsc_conv.index_P_dc[t]],-pf.CONVVSC_INF_PDC)
                    self.assertEqual(u[vsc_conv.index_i_dc[t]],pf.CONVVSC_INF_PDC)
                    self.assertEqual(l[vsc_conv.index_i_dc[t]],-pf.CONVVSC_INF_PDC)
                for facts in net.facts:
                    self.assertTrue(facts.has_flags('variable','series voltage magnitude'))
                    self.assertTrue(facts.has_flags('variable','series voltage angle'))
                    self.assertTrue(facts.has_flags('variable','active power'))
                    self.assertTrue(facts.has_flags('variable','reactive power'))
                    self.assertEqual(u[facts.index_v_mag_s[t]],pf.FACTS_INF_VMAG_S)
                    self.assertEqual(l[facts.index_v_mag_s[t]],0.)
                    self.assertEqual(u[facts.index_v_ang_s[t]],pf.FACTS_INF_VANG_S)
                    self.assertEqual(l[facts.index_v_ang_s[t]],-pf.FACTS_INF_VANG_S)
                    self.assertEqual(u[facts.index_P_k[t]],pf.FACTS_INF_P)
                    self.assertEqual(l[facts.index_P_k[t]],-pf.FACTS_INF_P)
                    self.assertEqual(u[facts.index_P_m[t]],pf.FACTS_INF_P)
                    self.assertEqual(l[facts.index_P_m[t]],-pf.FACTS_INF_P)
                    self.assertEqual(u[facts.index_P_dc[t]],pf.FACTS_INF_P)
                    self.assertEqual(l[facts.index_P_dc[t]],-pf.FACTS_INF_P)
                    self.assertEqual(u[facts.index_Q_k[t]],pf.FACTS_INF_Q)
                    self.assertEqual(l[facts.index_Q_k[t]],-pf.FACTS_INF_Q)
                    self.assertEqual(u[facts.index_Q_m[t]],pf.FACTS_INF_Q)
                    self.assertEqual(l[facts.index_Q_m[t]],-pf.FACTS_INF_Q)
                    self.assertEqual(u[facts.index_Q_s[t]],pf.FACTS_INF_Q)
                    self.assertEqual(l[facts.index_Q_s[t]],-pf.FACTS_INF_Q)
                    self.assertEqual(u[facts.index_Q_sh[t]],pf.FACTS_INF_Q)
                    self.assertEqual(l[facts.index_Q_sh[t]],-pf.FACTS_INF_Q)
                    
                for bus in net.dc_buses:
                    self.assertEqual(u[bus.index_v[t]], pf.BUSDC_INF_V)
                    self.assertEqual(l[bus.index_v[t]], -pf.BUSDC_INF_V)

                for csc in net.csc_converters:
                    self.assertEqual(u[csc.index_P[t]], pf.CONVCSC_INF_P)
                    self.assertEqual(l[csc.index_P[t]], -pf.CONVCSC_INF_P)
                    self.assertEqual(u[csc.index_Q[t]], pf.CONVCSC_INF_Q)
                    self.assertEqual(l[csc.index_Q[t]], -pf.CONVCSC_INF_Q)
                    self.assertEqual(u[csc.index_P_dc[t]], pf.CONVCSC_INF_PDC)
                    self.assertEqual(l[csc.index_P_dc[t]], -pf.CONVCSC_INF_PDC)
                    self.assertEqual(u[csc.index_i_dc[t]], pf.CONVCSC_INF_PDC)
                    self.assertEqual(l[csc.index_i_dc[t]], -pf.CONVCSC_INF_PDC)
                    self.assertEqual(u[csc.index_angle[t]], pf.CONVCSC_INF_ANGLE)
                    self.assertEqual(l[csc.index_angle[t]], -pf.CONVCSC_INF_ANGLE)
                    self.assertEqual(u[csc.index_ratio[t]], pf.CONVCSC_INF_RATIO)
                    self.assertEqual(l[csc.index_ratio[t]], -pf.CONVCSC_INF_RATIO)
                    
            # Row info
            for t in range(self.T):
                for bus in net.buses:
                    i = bus.index_v_mag[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:bus:%d:voltage magnitude:%d' %(bus.index,t))
                    i = bus.index_v_ang[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:bus:%d:voltage angle:%d' %(bus.index,t))
                for gen in net.generators:
                    i = gen.index_P[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:generator:%d:active power:%d' %(gen.index,t))
                    i = gen.index_Q[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:generator:%d:reactive power:%d' %(gen.index,t))
                for load in net.loads:
                    i = load.index_P[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:load:%d:active power:%d' %(load.index,t))
                    i = load.index_Q[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:load:%d:reactive power:%d' %(load.index,t))
                for vargen in net.var_generators:
                    i = vargen.index_P[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:variable generator:%d:active power:%d' %(vargen.index,t))
                    i = vargen.index_Q[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:variable generator:%d:reactive power:%d' %(vargen.index,t))
                for branch in net.branches:
                    if branch.is_tap_changer():
                        i = branch.index_ratio[t]
                        s = constr.get_G_row_info_string(i)
                        self.assertEqual(constr.get_A_row_info_string(i),"")
                        self.assertEqual(constr.get_J_row_info_string(i),"")
                        self.assertEqual(s,'variable bounds:branch:%d:tap ratio:%d' %(branch.index,t))
                    if branch.is_phase_shifter():
                        i = branch.index_phase[t]
                        s = constr.get_G_row_info_string(i)
                        self.assertEqual(constr.get_A_row_info_string(i),"")
                        self.assertEqual(constr.get_J_row_info_string(i),"")
                        self.assertEqual(s,'variable bounds:branch:%d:phase shift:%d' %(branch.index,t))
                for shunt in net.shunts:
                    if shunt.is_switched_v():
                        i = shunt.index_b[t]
                        s = constr.get_G_row_info_string(i)
                        self.assertEqual(constr.get_A_row_info_string(i),"")
                        self.assertEqual(constr.get_J_row_info_string(i),"")
                        self.assertEqual(s,'variable bounds:shunt:%d:susceptance:%d' %(shunt.index,t))
                for bat in net.batteries:
                    i = bat.index_Pc[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:battery:%d:charging power:%d' %(bat.index,t))
                    i = bat.index_Pd[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:battery:%d:discharging power:%d' %(bat.index,t))
                    i = bat.index_E[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:battery:%d:energy level:%d' %(bat.index,t))

                for vsc_conv in net.vsc_converters:
                    i = vsc_conv.index_P[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:vsc converter:%d:active power:%d' %(vsc_conv.index,t))
                    i = vsc_conv.index_Q[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:vsc converter:%d:reactive power:%d' %(vsc_conv.index,t))
                    i = vsc_conv.index_P_dc[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:vsc converter:%d:dc power:%d' %(vsc_conv.index,t))
                    i = vsc_conv.index_i_dc[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:vsc converter:%d:dc current:%d' %(vsc_conv.index,t))
                    
                for facts in net.facts:
                    i = facts.index_v_mag_s[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:facts:%d:series voltage magnitude:%d' %(facts.index,t))
                    i = facts.index_v_ang_s[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:facts:%d:series voltage angle:%d' %(facts.index,t))
                    i = facts.index_P_k[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:facts:%d:active power k:%d' %(facts.index,t))
                    i = facts.index_P_m[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:facts:%d:active power m:%d' %(facts.index,t))
                    i = facts.index_P_dc[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:facts:%d:dc power:%d' %(facts.index,t))
                    i = facts.index_Q_k[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:facts:%d:reactive power k:%d' %(facts.index,t))
                    i = facts.index_Q_m[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:facts:%d:reactive power m:%d' %(facts.index,t))
                    i = facts.index_Q_s[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:facts:%d:series reactive power:%d' %(facts.index,t))
                    i = facts.index_Q_sh[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:facts:%d:shunt reactive power:%d' %(facts.index,t))

                for bus in net.dc_buses:
                    i = bus.index_v[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:dc bus:%d:voltage:%d' %(bus.index,t))
                    
                for csc in net.csc_converters:
                    i = csc.index_P[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:csc converter:%d:active power:%d' %(csc.index,t))
                    i = csc.index_Q[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:csc converter:%d:reactive power:%d' %(csc.index,t))
                    i = csc.index_P_dc[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:csc converter:%d:dc power:%d' %(csc.index,t))
                    i = csc.index_i_dc[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:csc converter:%d:dc current:%d' %(csc.index,t))
                    i = csc.index_angle[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:csc converter:%d:angle:%d' %(csc.index,t))
                    i = csc.index_ratio[t]
                    s = constr.get_G_row_info_string(i)
                    self.assertEqual(constr.get_A_row_info_string(i),"")
                    self.assertEqual(constr.get_J_row_info_string(i),"")
                    self.assertEqual(s,'variable bounds:csc converter:%d:tap ratio:%d' %(csc.index,t))
                    
            # Bounded
            net.set_flags('bus',
                          'bounded',
                          'any',
                          ['voltage magnitude','voltage angle'])
            net.set_flags('generator',
                          'bounded',
                          'any',
                          ['active power','reactive power'])
            net.set_flags('load',
                          'bounded',
                          'any',
                          ['active power','reactive power'])
            net.set_flags('branch',
                          'bounded',
                          'tap changer',
                          ['tap ratio'])
            net.set_flags('branch',
                          'bounded',
                          'phase shifter',
                          'phase shift')
            net.set_flags('shunt',
                          'bounded',
                          'switching - v',
                          ['susceptance'])
            net.set_flags('variable generator',
                          'bounded',
                          'any',
                          ['active power','reactive power'])
            net.set_flags('battery',
                          'bounded',
                          'any',
                          ['charging power','energy level'])
            net.set_flags('vsc converter',
                          'bounded',
                          'any',
                          ['dc power', 'active power','reactive power'])
            net.set_flags('facts',
                          'bounded',
                          'any',
                          ['series voltage magnitude','series voltage angle',
                           'active power','reactive power'])
            net.set_flags('dc bus',
                          'bounded',
                          'any',
                          'voltage')
            net.set_flags('csc converter',
                          'bounded',
                          'any',
                          'all')
            self.assertGreater(net.num_vars,0)
            self.assertEqual(net.num_bounded,net.num_vars)

            x0 = net.get_var_values()
            constr = pf.Constraint('variable bounds',net)
            self.assertEqual(constr.name,'variable bounds')
            constr.analyze()
            constr.eval(x0)

            G = constr.G
            l = constr.l
            u = constr.u

            for t in range(self.T):
                for bus in net.buses:
                    self.assertEqual(u[bus.index_v_mag[t]],bus.v_max)
                    self.assertEqual(u[bus.index_v_ang[t]],pf.BUS_INF_V_ANG)
                    self.assertEqual(l[bus.index_v_mag[t]],bus.v_min)
                    self.assertEqual(l[bus.index_v_ang[t]],-pf.BUS_INF_V_ANG)
                for gen in net.generators:
                    self.assertEqual(u[gen.index_P[t]],gen.P_max)
                    self.assertEqual(u[gen.index_Q[t]],gen.Q_max)
                    self.assertEqual(l[gen.index_P[t]],gen.P_min)
                    self.assertEqual(l[gen.index_Q[t]],gen.Q_min)
                for branch in net.branches:
                    if branch.is_tap_changer():
                        self.assertEqual(u[branch.index_ratio[t]],branch.ratio_max)
                        self.assertEqual(l[branch.index_ratio[t]],branch.ratio_min)
                    if branch.is_phase_shifter():
                        self.assertEqual(u[branch.index_phase[t]],branch.phase_max)
                        self.assertEqual(l[branch.index_phase[t]],branch.phase_min)
                for vargen in net.var_generators:
                    self.assertEqual(u[vargen.index_P[t]],vargen.P_ava[t])
                    self.assertEqual(u[vargen.index_Q[t]],vargen.Q_max)
                    self.assertEqual(l[vargen.index_P[t]],vargen.P_min)
                    self.assertEqual(l[vargen.index_Q[t]],vargen.Q_min)
                for load in net.loads:
                    self.assertEqual(u[load.index_P[t]],load.P_max[t])
                    self.assertEqual(l[load.index_P[t]],load.P_min[t])
                    self.assertEqual(u[load.index_P[t]],3.3*(load.index+1)*t)
                    self.assertEqual(l[load.index_P[t]],-2.4*(load.index+1)*t)
                    self.assertEqual(u[load.index_Q[t]],7.5*(load.index+1)*t)
                    self.assertEqual(l[load.index_Q[t]],1.2*(load.index+1)*t)
                for shunt in net.shunts:
                    if shunt.is_switched_v():
                        self.assertEqual(u[shunt.index_b[t]],shunt.b_max)
                        self.assertEqual(l[shunt.index_b[t]],shunt.b_min)
                for vsc_conv in net.vsc_converters:
                    self.assertTrue(vsc_conv.has_flags('bounded','active power'))
                    self.assertTrue(vsc_conv.has_flags('bounded','reactive power'))
                    self.assertTrue(vsc_conv.has_flags('bounded','dc power'))
                    self.assertEqual(u[vsc_conv.index_P[t]],vsc_conv.P_max)
                    self.assertEqual(l[vsc_conv.index_P[t]],vsc_conv.P_min)
                    self.assertEqual(u[vsc_conv.index_Q[t]],vsc_conv.Q_max)
                    self.assertEqual(l[vsc_conv.index_Q[t]],vsc_conv.Q_min)
                    self.assertEqual(u[vsc_conv.index_P_dc[t]],pf.CONVVSC_INF_PDC)
                    self.assertEqual(l[vsc_conv.index_P_dc[t]],-pf.CONVVSC_INF_PDC)
                    self.assertEqual(u[vsc_conv.index_i_dc[t]],pf.CONVVSC_INF_PDC)
                    self.assertEqual(l[vsc_conv.index_i_dc[t]],-pf.CONVVSC_INF_PDC)
                    
                for facts in net.facts:
                    self.assertTrue(facts.has_flags('bounded','series voltage magnitude'))
                    self.assertTrue(facts.has_flags('bounded','series voltage angle'))
                    self.assertTrue(facts.has_flags('bounded','active power'))
                    self.assertTrue(facts.has_flags('bounded','reactive power'))
                    self.assertEqual(u[facts.index_v_mag_s[t]],facts.v_max_s)
                    self.assertEqual(l[facts.index_v_mag_s[t]],0.)
                    self.assertEqual(u[facts.index_v_ang_s[t]],pf.FACTS_INF_VANG_S)
                    self.assertEqual(l[facts.index_v_ang_s[t]],-pf.FACTS_INF_VANG_S)
                    self.assertEqual(u[facts.index_P_k[t]],pf.FACTS_INF_P)
                    self.assertEqual(l[facts.index_P_k[t]],-pf.FACTS_INF_P)
                    self.assertEqual(u[facts.index_P_m[t]],pf.FACTS_INF_P)
                    self.assertEqual(l[facts.index_P_m[t]],-pf.FACTS_INF_P)
                    self.assertEqual(u[facts.index_P_dc[t]],facts.P_max_dc)
                    self.assertEqual(l[facts.index_P_dc[t]],-facts.P_max_dc)
                    self.assertEqual(u[facts.index_Q_k[t]],pf.FACTS_INF_Q)
                    self.assertEqual(l[facts.index_Q_k[t]],-pf.FACTS_INF_Q)
                    self.assertEqual(u[facts.index_Q_m[t]],pf.FACTS_INF_Q)
                    self.assertEqual(l[facts.index_Q_m[t]],-pf.FACTS_INF_Q)
                    self.assertEqual(u[facts.index_Q_s[t]],facts.Q_max_s)
                    self.assertEqual(l[facts.index_Q_s[t]],facts.Q_min_s)
                    self.assertEqual(u[facts.index_Q_sh[t]],facts.Q_max_sh)
                    self.assertEqual(l[facts.index_Q_sh[t]],facts.Q_min_sh)

                for bus in net.dc_buses:
                    self.assertEqual(u[bus.index_v[t]], pf.BUSDC_INF_V)
                    self.assertEqual(l[bus.index_v[t]], -pf.BUSDC_INF_V)

                for csc in net.csc_converters:
                    self.assertEqual(u[csc.index_P[t]], pf.CONVCSC_INF_P)
                    self.assertEqual(l[csc.index_P[t]], -pf.CONVCSC_INF_P)
                    self.assertEqual(u[csc.index_Q[t]], pf.CONVCSC_INF_Q)
                    self.assertEqual(l[csc.index_Q[t]], -pf.CONVCSC_INF_Q)
                    self.assertEqual(u[csc.index_P_dc[t]], pf.CONVCSC_INF_PDC)
                    self.assertEqual(l[csc.index_P_dc[t]], -pf.CONVCSC_INF_PDC)
                    self.assertEqual(u[csc.index_i_dc[t]], pf.CONVCSC_INF_PDC)
                    self.assertEqual(l[csc.index_i_dc[t]], -pf.CONVCSC_INF_PDC)
                    self.assertEqual(u[csc.index_angle[t]], pf.CONVCSC_INF_ANGLE)
                    self.assertEqual(l[csc.index_angle[t]], -pf.CONVCSC_INF_ANGLE)
                    self.assertEqual(u[csc.index_ratio[t]], pf.CONVCSC_INF_RATIO)
                    self.assertEqual(l[csc.index_ratio[t]], -pf.CONVCSC_INF_RATIO)
            
            # Sensitivities
            mu = np.random.randn(net.num_vars)
            pi = np.random.randn(net.num_vars)

            net.clear_sensitivities()
            
            constr.store_sensitivities(None,None,mu,pi)
            
            for t in range(self.T):
                        
                # Branch sens
                for branch in net.branches:
                    if branch.is_tap_changer():
                        self.assertEqual(branch.sens_ratio_u_bound[t], mu[branch.index_ratio[t]])
                        self.assertEqual(branch.sens_ratio_l_bound[t], pi[branch.index_ratio[t]])
                    else:
                        self.assertEqual(branch.sens_ratio_u_bound[t], 0.)
                        self.assertEqual(branch.sens_ratio_l_bound[t], 0.)
                    if branch.is_phase_shifter():
                        self.assertEqual(branch.sens_phase_u_bound[t], mu[branch.index_phase[t]])
                        self.assertEqual(branch.sens_phase_l_bound[t], pi[branch.index_phase[t]])
                    else:
                        self.assertEqual(branch.sens_phase_u_bound[t], 0.)
                        self.assertEqual(branch.sens_phase_l_bound[t], 0.)

                # Bus sens
                for bus in net.buses:
                    self.assertEqual(bus.sens_P_balance[t],0.)
                    self.assertEqual(bus.sens_Q_balance[t],0.)
                    self.assertEqual(bus.sens_v_mag_u_bound[t], mu[bus.index_v_mag[t]])
                    self.assertEqual(bus.sens_v_mag_l_bound[t], pi[bus.index_v_mag[t]])
                    self.assertEqual(bus.sens_v_ang_u_bound[t], mu[bus.index_v_ang[t]])
                    self.assertEqual(bus.sens_v_ang_l_bound[t], pi[bus.index_v_ang[t]])

                # Gen sens
                for gen in net.generators:
                    self.assertEqual(gen.sens_P_u_bound[t], mu[gen.index_P[t]])
                    self.assertEqual(gen.sens_P_l_bound[t], pi[gen.index_P[t]])
                    self.assertEqual(gen.sens_Q_u_bound[t], mu[gen.index_Q[t]])
                    self.assertEqual(gen.sens_Q_l_bound[t], pi[gen.index_Q[t]])

                # Load sens
                for load in net.loads:
                    self.assertEqual(load.sens_P_u_bound[t], mu[load.index_P[t]])
                    self.assertEqual(load.sens_P_l_bound[t], pi[load.index_P[t]])

                # Shunts
                for shunt in net.shunts:
                    if shunt.is_switched_v():
                        self.assertEqual(shunt.sens_b_u_bound[t], mu[shunt.index_b[t]])
                        self.assertEqual(shunt.sens_b_l_bound[t], pi[shunt.index_b[t]])

    def test_constr_LBOUND_with_outages(self):

        # Multiperiod
        for case in test_cases.CASES:
            
            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)

            net.clear_outages()

            gen = net.get_generator(0)
            branch = net.get_branch(0)

            gen.outage = True
            branch.outage = True

            self.assertTrue(gen.is_on_outage())
            self.assertTrue(branch.is_on_outage())

            gen.P_min = np.random.rand()
            gen.Q_min = np.random.rand()
            branch.ratio_min = np.random.randn()
            branch.phase_min = np.random.randn()
            gen.P_max = gen.P_min + 3.
            gen.Q_max = gen.Q_min + 4.
            branch.ratio_max = branch.ratio_min + 5.
            branch.phase_max = branch.phase_min + 2.
            
            net.set_flags('generator',
                          ['variable','bounded'],
                          'any',
                          ['active power', 'reactive power'])
            net.set_flags('branch',
                          ['variable','bounded'],
                          'any',
                          ['tap ratio', 'phase shift'])
            self.assertEqual(net.num_vars,
                             self.T*(2*net.num_generators + 2*net.num_branches))
            self.assertEqual(net.num_vars, net.num_bounded)

            constr = pf.Constraint('variable bounds', net)
            constr.analyze()

            l = constr.l
            u = constr.u
            G = constr.G

            for t in range(self.T):

                # gen P
                k = np.where(G.col == gen.index_P[t])[0]
                self.assertEqual(k.size, 1)
                k = k[0]
                i = G.row[k]
                self.assertEqual(G.data[k], 1.)
                self.assertEqual(l[i], gen.P_min)
                self.assertEqual(u[i], gen.P_max)
                self.assertEqual(u[i], l[i] + 3.)

                # gen Q
                k = np.where(G.col == gen.index_Q[t])[0]
                self.assertEqual(k.size, 1)
                k = k[0]
                i = G.row[k]
                self.assertEqual(G.data[k], 1.)
                self.assertEqual(l[i], gen.Q_min)
                self.assertEqual(u[i], gen.Q_max)
                self.assertEqual(u[i], l[i] + 4.)

                # branch ratio
                k = np.where(G.col == branch.index_ratio[t])[0]
                self.assertEqual(k.size, 1)
                k = k[0]
                i = G.row[k]
                self.assertEqual(G.data[k], 1.)
                self.assertEqual(l[i], branch.ratio_min)
                self.assertEqual(u[i], branch.ratio_max)
                self.assertEqual(u[i], l[i] + 5.)

                # branch phase
                k = np.where(G.col == branch.index_phase[t])[0]
                self.assertEqual(k.size, 1)
                k = k[0]
                i = G.row[k]
                self.assertEqual(G.data[k], 1.)
                self.assertEqual(l[i], branch.phase_min)
                self.assertEqual(u[i], branch.phase_max)
                self.assertEqual(u[i], l[i] + 2.)

            # Disconnect
            net.clear_outages()
            net.clear_flags()
            self.assertEqual(net.num_vars, 0)
            for bus in net.buses:
                if bus.degree == 1:
                    self.assertEqual(len(bus.branches), 1)
                    bus.branches[0].outage = True
                    self.assertTrue(bus.branches[0].is_on_outage())
                    net.set_flags_of_component(bus,
                                               ['variable', 'bounded'],
                                               ['voltage magnitude', 'voltage angle'])
                    self.assertEqual(net.num_vars, 2*self.T)
                    self.assertEqual(net.num_vars, net.num_bounded)
                    self.assertTrue(bus.has_flags('variable', ['voltage magnitude',
                                                               'voltage angle']))
                    self.assertTrue(bus.has_flags('bounded', ['voltage magnitude',
                                                              'voltage angle']))
                    constr = pf.Constraint('variable bounds', net)
                    constr.analyze()
                    G = constr.G
                    l = constr.l
                    u = constr.u

                    self.assertEqual(G.shape[0], 2*self.T)
                    for t in range(self.T):
                        k = np.where(G.col == bus.index_v_mag[t])[0]
                        self.assertEqual(k.size, 1)
                        k = k[0]
                        self.assertEqual(l[G.row[k]], bus.v_min)
                        self.assertEqual(u[G.row[k]], bus.v_max)
                        k = np.where(G.col == bus.index_v_ang[t])[0]
                        self.assertEqual(k.size, 1)
                        k = k[0]
                        self.assertEqual(l[G.row[k]], -100.)
                        self.assertEqual(u[G.row[k]], 100.)
                    break                                     

    def test_constr_PAR_GEN_P(self):

        # Multiperiod
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)
            self.assertEqual(net.num_vars,0)

            # Vars
            net.set_flags('generator',
                          'variable',
                          'slack',
                          ['active power','reactive power'])
            net.set_flags('generator',
                          'variable',
                          'regulator',
                          'reactive power')
            self.assertGreater(net.num_vars,0)
            self.assertEqual(net.num_vars,(net.get_num_slack_gens()+net.get_num_reg_gens())*self.T)

            x0 = net.get_var_values()
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            # Constraint
            constr = pf.Constraint('generator active power participation',net)
            self.assertEqual(constr.name,'generator active power participation')

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b

            # Before
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,0))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,0))
            self.assertEqual(A.nnz,0)

            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)

            # Manual count
            nnz = 0
            num_constr = 0
            for i in range(net.num_buses):
                bus = net.get_bus(i)
                if bus.is_slack():
                    num_constr += len(bus.generators)-1 # P participation
                    nnz += 2*(len(bus.generators)-1)

            constr.analyze()
            self.assertEqual(nnz*self.T,constr.A_nnz)
            constr.eval(x0)
            self.assertEqual(0,constr.A_nnz)

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b

            # After
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(num_constr*self.T,))
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(num_constr*self.T,net.num_vars))
            self.assertEqual(A.nnz,nnz*self.T)
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,net.num_vars))
            self.assertEqual(J.nnz,0)

            self.assertTrue(not np.any(np.isinf(b)))
            self.assertTrue(not np.any(np.isnan(b)))

            # Detailed check
            Ai = A.row
            Aj = A.col
            Ad = A.data
            self.assertEqual(Ai.size,nnz*self.T)
            self.assertEqual(Aj.size,nnz*self.T)
            self.assertEqual(Ad.size,nnz*self.T)
            i = 0
            row = 0
            for t in range(self.T):
                for bus in net.buses:
                    if bus.is_slack():
                        gens = bus.generators
                        self.assertGreater(len(gens),0)
                        g1 = gens[0]
                        for g2 in gens[1:]:
                            self.assertEqual(b[row],0.)
                            self.assertEqual(Ai[i],row)
                            self.assertEqual(Aj[i],g1.index_P[t])
                            self.assertEqual(Ad[i],1.)
                            i += 1
                            self.assertEqual(Ai[i],row)
                            self.assertEqual(Aj[i],g2.index_P[t])
                            self.assertEqual(Ad[i],-1.)
                            i += 1
                            row += 1
            self.assertEqual(i,nnz*self.T)

            # Last check
            x = np.zeros(net.num_vars)
            for t in range(self.T):
                for i in range(net.num_buses):
                    bus = net.get_bus(i)
                    if bus.is_slack():
                        self.assertGreater(len(bus.generators),0)
                        for g in bus.generators:
                            self.assertTrue(g.has_flags('variable','active power'))
                            x[g.index_P[t]] = 10.
            self.assertGreater(norm(x),0)
            self.assertTrue(norm(A*x-b) < 1e-10)

    def test_constr_PAR_GEN_P_with_outages(self):

        # Multiperiod
        for case in test_cases.CASES:
            
            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)

            net.clear_outages()
            net.clear_flags()

            for bus in net.buses:
                if bus.is_slack():
                    for branch in net.branches:
                        branch.outage = True
                    for gen in net.generators:
                        gen.outage = True

            net.set_flags('generator',
                          'variable',
                          'any',
                          'active power')
            self.assertEqual(net.num_vars, self.T*net.num_generators)

            constr = pf.Constraint('generator active power participation', net)
            constr.analyze()
            A = constr.A
            b = constr.b

            self.assertEqual(A.shape[0], 0)
            self.assertEqual(b.shape[0], 0)

            net.clear_outages()

            constr.analyze()
            A = constr.A
            b = constr.b

            check = False
            for bus in net.buses:
                if bus.is_slack() and len(bus.generators) > 1:
                    check = True
            if check:
                self.assertGreater(A.shape[0], 0)
                self.assertGreater(b.shape[0], 0)
            
    def test_constr_PVPQ_SWITCHING(self):

        # Multiperiod
        for case in test_cases.CASES:
            
            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)
            self.assertEqual(net.num_vars,0)

            # Vars
            net.set_flags('bus',
                          'variable',
                          'regulated by generator',
                          'voltage magnitude')
            net.set_flags('generator',
                          'variable',
                          'slack',
                          ['active power','reactive power'])
            net.set_flags('generator',
                          'variable',
                          'regulator',
                          'reactive power')
            self.assertGreater(net.num_vars,0)
            self.assertEqual(net.num_vars,
                             (net.get_num_buses_reg_by_gen()+net.get_num_slack_gens()+net.get_num_reg_gens())*self.T)
            
            x0 = net.get_var_values()
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            # Make it iteresting
            for gen in net.generators:
                gen.Q_par = np.random.rand()            
            
            # Constraint
            constr = pf.Constraint('PVPQ switching',net)
            self.assertEqual(constr.name,'PVPQ switching')
            
            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            
            # Before
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,0))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,0))
            self.assertEqual(A.nnz,0)
            
            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            
            # Manual count
            nnz = 0
            num_constr = 0
            for i in range(net.num_buses):
                bus = net.get_bus(i)
                if bus.is_regulated_by_gen():
                    num_constr += len(bus.reg_generators)
                    nnz += len(bus.reg_generators)*(len(bus.reg_generators)+1)

            constr.analyze()
            self.assertEqual(nnz*self.T,constr.A_nnz)
            constr.eval(x0)
            self.assertEqual(0,constr.A_nnz)
                
            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            
            # After
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(num_constr*self.T,))
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(num_constr*self.T,net.num_vars))
            self.assertEqual(A.nnz,nnz*self.T)
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,net.num_vars))
            self.assertEqual(J.nnz,0)
            
            self.assertTrue(not np.any(np.isinf(b)))
            self.assertTrue(not np.any(np.isnan(b)))
                
            # Detailed check
            Ai = A.row
            Aj = A.col
            Ad = A.data
            self.assertEqual(Ai.size,nnz*self.T)
            self.assertEqual(Aj.size,nnz*self.T)
            self.assertEqual(Ad.size,nnz*self.T)
            nnz = 0
            row = 0
            for t in range(self.T):
                for bus in net.buses:
                    if bus.is_regulated_by_gen():
                        self.assertEqual(b[row], bus.v_set[t])
                        self.assertEqual(Ai[nnz], row)
                        self.assertEqual(Aj[nnz], bus.index_v_mag[t])
                        self.assertEqual(Ad[nnz], 1.)
                        nnz += 1
                        for gen in bus.reg_generators:
                            self.assertEqual(Ai[nnz], row)
                            self.assertEqual(Aj[nnz], gen.index_Q[t])
                            self.assertEqual(Ad[nnz], 0.)
                            nnz += 1
                        row += 1
                        for i in range(len(bus.reg_generators)-1):
                            gen1 = bus.reg_generators[i]
                            gen2 = bus.reg_generators[i+1]
                            self.assertEqual(b[row], 0.)
                            self.assertEqual(Ai[nnz], row)
                            self.assertEqual(Aj[nnz], bus.index_v_mag[t])
                            self.assertEqual(Ad[nnz], 0.)
                            nnz += 1                                    
                            for gen3 in bus.reg_generators:
                                self.assertEqual(Ai[nnz], row)
                                self.assertEqual(Aj[nnz], gen3.index_Q[t])
                                if gen3.index == gen1.index:
                                    self.assertEqual(Ad[nnz], np.maximum(gen2.Q_par,1e-4))
                                elif gen3.index == gen2.index:
                                    self.assertEqual(Ad[nnz], -np.maximum(gen1.Q_par,1e-4))
                                else:
                                    self.assertEqual(Ad[nnz], 0.)
                                nnz += 1
                            row += 1
            self.assertEqual(nnz,A.nnz)

            # Now with no Q vars
            net.clear_flags()

            # Vars
            net.set_flags('bus',
                          'variable',
                          'any',
                          'voltage magnitude')
            self.assertEqual(net.num_vars, net.get_num_buses()*self.T)
                        
            # Analyze
            constr.analyze()
            A = constr.A
            b = constr.b
            self.assertEqual(A.shape[0], 0)
            self.assertEqual(A.nnz, 0)
            self.assertEqual(b.size, 0)

            # Now with no v vars
            net.clear_flags()

            # Vars
            net.set_flags('generator',
                          'variable',
                          'any',
                          'reactive power')
            self.assertEqual(net.num_vars, net.get_num_generators()*self.T)
                        
            # Analyze
            constr.analyze()
            A = constr.A
            b = constr.b
            nnz = 0
            m = 0
            for bus in net.buses:
                if bus.is_regulated_by_gen():
                    n = len(bus.reg_generators)
                    m += n-1
                    nnz += n*(n-1)
            self.assertEqual(A.shape[0], m*self.T)
            self.assertEqual(A.nnz, nnz*self.T)

    def test_constr_PVPQ_SWITCHING_with_outages(self):

        # Multiperiod
        for case in test_cases.CASES:
            
            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)
            self.assertEqual(net.num_vars,0)

            # Vars
            net.set_flags('bus',
                          'variable',
                          'regulated by generator',
                          'voltage magnitude')
            net.set_flags('generator',
                          'variable',
                          'slack',
                          ['active power','reactive power'])
            net.set_flags('generator',
                          'variable',
                          'regulator',
                          'reactive power')
            self.assertGreater(net.num_vars,0)
            self.assertEqual(net.num_vars,
                             (net.get_num_buses_reg_by_gen()+net.get_num_slack_gens()+net.get_num_reg_gens())*self.T)

            constr = pf.Constraint('PVPQ switching', net)
            constr.analyze()
            A0 = constr.A.copy()
            b0 = constr.b.copy()

            self.assertEqual(net.get_num_branches_on_outage(), 0)
            self.assertEqual(net.get_num_generators_on_outage(), 0)

            for bus in net.buses:
                if bus.is_regulated_by_gen():
                    for branch in net.branches:
                        branch.outage = True

            self.assertNotEqual(net.get_num_branches_on_outage(), 0)
            self.assertEqual(net.get_num_generators_on_outage(), 0)

            constr = pf.Constraint('PVPQ switching', net)
            constr.analyze()
            A1 = constr.A.copy()
            b1 = constr.b.copy()

            self.assertEqual((A1-A0).tocoo().nnz, 0)
            self.assertLess(norm(b1-b0), 1e-8)

            for bus in net.buses:
                if bus.is_regulated_by_gen():
                    for gen in bus.reg_generators:
                        gen.outage = True
                    self.assertFalse(bus.is_regulated_by_gen())

            self.assertNotEqual(net.get_num_generators_on_outage(), 0)

            constr = pf.Constraint('PVPQ switching', net)
            constr.analyze()
            A2 = constr.A.copy()
            b2 = constr.b.copy()

            self.assertEqual(A2.shape[0], 0)
            self.assertEqual(b2.size, 0)
            
    def test_constr_ACPF(self):

        # Constants
        h = 1e-10

        # Multiperiods
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)
            
            # Add vargens
            load_buses = net.get_load_buses()
            net.add_var_generators_from_parameters(load_buses,80.,50.,30.,5,0.05)
            self.assertGreater(net.num_var_generators,0)
            self.assertEqual(net.num_var_generators,len(load_buses))
            for vargen in net.var_generators:
                vargen.P = np.random.rand(net.num_periods)
                vargen.Q = np.random.randn(net.num_periods)

            # Add batteries
            gen_buses = net.get_generator_buses()
            net.add_batteries_from_parameters(gen_buses,20.,40.,0.8,0.9)
            self.assertGreater(net.num_batteries,0)
            self.assertEqual(net.num_batteries,len(gen_buses))
            for bat in net.batteries:
                bat.P = np.random.randn(net.num_periods)

            # No vars
            self.assertEqual(net.num_vars,0)
            
            # Constraint
            constr = pf.Constraint('AC power balance',net)
            self.assertEqual(constr.name,'AC power balance')

            x0 = net.get_var_values()
            self.assertEqual(x0.size,0)
            constr.analyze()
            constr.eval(x0)
            
            f = constr.f
            self.assertEqual(f.size,2*net.num_buses*net.num_periods)

            # Check mismatches (no vars)
            for t in range(net.num_periods):
                for bus in net.buses:
                    P_mis = 0.
                    Q_mis = 0.
                    for branch in bus.branches_k:
                        P_mis -= branch.get_P_km()[t]
                        Q_mis -= branch.get_Q_km()[t]
                    for branch in bus.branches_m:
                        P_mis -= branch.get_P_mk()[t]
                        Q_mis -= branch.get_Q_mk()[t]
                    for gen in bus.generators:
                        P_mis += gen.P[t]
                        Q_mis += gen.Q[t]
                    for vargen in bus.var_generators:
                        P_mis += vargen.P[t]
                        Q_mis += vargen.Q[t]
                    for load in bus.loads:
                        P_mis -= load.P[t]
                        Q_mis -= load.Q[t]
                    for bat in bus.batteries:
                        P_mis -= bat.P[t]
                    for shunt in bus.shunts:
                        P_mis -= shunt.g*(bus.v_mag[t]**2.)
                        Q_mis -= -shunt.b[t]*(bus.v_mag[t]**2.)
                    for conv in bus.csc_converters:
                        P_mis += conv.P[t]
                        Q_mis += conv.Q[t]
                    for conv in bus.vsc_converters:
                        P_mis += conv.P[t]
                        Q_mis += conv.Q[t]
                    for facts in bus.facts_k:
                        P_mis += facts.P_k[t]
                        Q_mis += facts.Q_k[t]
                    for facts in bus.facts_m:
                        P_mis += facts.P_m[t]
                        Q_mis += facts.Q_m[t]
                    self.assertAlmostEqual(P_mis,f[bus.index_P[t]])
                    self.assertAlmostEqual(Q_mis,f[bus.index_Q[t]])

            # Cross check mismatches with net properties (no vars)
            net.update_properties()
            dP_list = dict([(t,list()) for t in range(self.T)])
            dQ_list = dict([(t,list()) for t in range(self.T)])
            for t in range(self.T):
                for i in range(net.num_buses):
                    bus = net.get_bus(i)
                    dP = f[bus.index_P[t]]
                    dQ = f[bus.index_Q[t]]
                    dP_list[t].append(dP)
                    dQ_list[t].append(dQ)
                    self.assertAlmostEqual(dP,bus.P_mismatch[t])
                    self.assertAlmostEqual(dQ,bus.Q_mismatch[t])
            self.assertAlmostEqual(net.bus_P_mis[t],np.max(np.abs(dP_list[t]))*net.base_power)
            self.assertAlmostEqual(net.bus_Q_mis[t],np.max(np.abs(dQ_list[t]))*net.base_power)

            # Vars
            net.set_flags('bus',
                          'variable',
                          'any',
                          ['voltage magnitude','voltage angle'])
            net.set_flags('generator',
                          'variable',
                          'any',
                          ['active power','reactive power'])
            net.set_flags('load',
                          'variable',
                          'any',
                          ['active power','reactive power'])
            net.set_flags('branch',
                          'variable',
                          'tap changer',
                          'tap ratio')
            net.set_flags('branch',
                          'variable',
                          'phase shifter',
                          'phase shift')
            net.set_flags('shunt',
                          'variable',
                          'switching - v',
                          'susceptance')
            net.set_flags('variable generator',
                          'variable',
                          'any',
                          ['active power','reactive power'])
            net.set_flags('battery',
                          'variable',
                          'any',
                          ['charging power','energy level'])
            net.set_flags('vsc converter',
                          'variable',
                          'any',
                          ['active power', 'reactive power', 'dc power'])
            net.set_flags('facts',
                          'variable',
                          'any',
                          ['active power', 'reactive power'])
            self.assertEqual(net.num_vars,
                             (2*net.num_buses +
                              2*net.num_generators +
                              2*net.num_loads +
                              net.get_num_tap_changers() +
                              net.get_num_phase_shifters() +
                              net.get_num_switched_v_shunts() +
                              3*net.num_batteries +
                              net.num_var_generators*2 +
                              net.num_vsc_converters*4 +
                              net.num_facts*7)*self.T)

            # Check facts
            for facts in net.facts:
                self.assertTrue(facts.has_flags('variable', 'active power'))
                self.assertTrue(facts.has_flags('variable', 'reactive power'))
            
            x0 = net.get_var_values()
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))
            
            # Constraint
            constr = pf.Constraint('AC power balance',net)
            self.assertEqual(constr.name,'AC power balance')

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            G = constr.G

            # Before
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,0))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,0))
            self.assertEqual(A.nnz,0)
            self.assertTrue(type(G) is coo_matrix)
            self.assertTupleEqual(G.shape,(0,0))
            self.assertEqual(G.nnz,0)

            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.G_nnz,0)

            num_statcom = len([f for f in net.facts if f.is_STATCOM()])

            num_Jnnz = (net.num_buses*4 +
                        net.num_branches*8 +                        
                        net.get_num_tap_changers()*4 +
                        net.get_num_phase_shifters()*4 +
                        net.get_num_switched_v_shunts() +
                        net.num_generators*2 +
                        net.num_loads*2 +
                        net.num_batteries*2 +
                        net.num_var_generators*2 +
                        net.num_vsc_converters*2 +
                        (net.num_facts-num_statcom)*4+num_statcom*2)*self.T

            constr.analyze()
            self.assertEqual(num_Jnnz,constr.J_nnz)
            constr.eval(x0)
            self.assertEqual(num_Jnnz,constr.J_nnz)

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            G = constr.G
            constr.combine_H(np.ones(f.size),False)
            Hcomb = constr.H_combined

            # After
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(2*net.num_buses*self.T,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(2*net.num_buses*self.T,net.num_vars))
            self.assertEqual(J.nnz,num_Jnnz)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,net.num_vars))
            self.assertEqual(A.nnz,0)
            self.assertTrue(type(G) is coo_matrix)
            self.assertTupleEqual(G.shape,(0,net.num_vars))
            self.assertEqual(G.nnz,0)
            self.assertTupleEqual(Hcomb.shape,(net.num_vars,net.num_vars))
            self.assertEqual(Hcomb.nnz,2*(net.get_num_buses()*3 +
                                          net.get_num_branches()*12 +
                                          net.get_num_tap_changers()*9 +
                                          net.get_num_phase_shifters()*10 +
                                          net.get_num_switched_v_shunts())*self.T)

            self.assertTrue(not np.any(np.isinf(f)))
            self.assertTrue(not np.any(np.isnan(f)))

            # Check mismatches
            x1 = x0+np.random.randn(net.num_vars)
            constr.eval(x1)
            for t in range(net.num_periods):
                for bus in net.buses:
                    P_mis = 0.
                    Q_mis = 0.
                    for branch in bus.branches_k:
                        P_mis -= branch.get_P_km(x1)[t]
                        Q_mis -= branch.get_Q_km(x1)[t]
                    for branch in bus.branches_m:
                        P_mis -= branch.get_P_mk(x1)[t]
                        Q_mis -= branch.get_Q_mk(x1)[t]
                    for gen in bus.generators:
                        P_mis += x1[gen.index_P[t]]
                        Q_mis += x1[gen.index_Q[t]]
                    for vargen in bus.var_generators:
                        P_mis += x1[vargen.index_P[t]]
                        Q_mis += x1[vargen.index_Q[t]]
                    for load in bus.loads:
                        P_mis -= x1[load.index_P[t]]
                        Q_mis -= x1[load.index_Q[t]]
                    for bat in bus.batteries:
                        P_mis -= x1[bat.index_Pc[t]]-x1[bat.index_Pd[t]]
                    for shunt in bus.shunts:
                        if shunt.has_flags('variable','susceptance'):
                            b = x1[shunt.index_b[t]]
                        else:
                            b = shunt.b[t]
                        if bus.has_flags('variable','voltage magnitude'):
                            v = x1[bus.index_v_mag[t]]
                        else:
                            v = bus.v_mag[t]
                        P_mis -= shunt.g*v*v
                        Q_mis -= -b*v*v
                    for conv in bus.vsc_converters:
                        if conv.has_flags('variable', 'active power'):
                            P_mis += x1[conv.index_P[t]]
                        else:
                            P_mis += conv.P[t]
                        if conv.has_flags('variable', 'reactive power'):
                            Q_mis += x1[conv.index_Q[t]]
                        else:
                            Q_mis += conv.Q[t]
                    for conv in bus.csc_converters:
                        P_mis += conv.P[t]
                        Q_mis += conv.Q[t]
                    for facts in bus.facts_k:
                        self.assertTrue(facts.has_flags('variable', 'active power'))
                        P_mis += x1[facts.index_P_k[t]]
                        self.assertTrue(facts.has_flags('variable', 'reactive power'))
                        Q_mis += x1[facts.index_Q_k[t]]
                    for facts in bus.facts_m:
                        self.assertTrue(facts.has_flags('variable', 'active power'))
                        P_mis += x1[facts.index_P_m[t]]
                        self.assertTrue(facts.has_flags('variable', 'reactive power'))
                        Q_mis += x1[facts.index_Q_m[t]]
                    self.assertAlmostEqual(P_mis,f[bus.index_P[t]])
                    self.assertAlmostEqual(Q_mis,f[bus.index_Q[t]])

            # Cross check mismatches with net properties
            constr.eval(x1)
            net.update_properties(x1)
            dP_list = dict([(t,list()) for t in range(self.T)])
            dQ_list = dict([(t,list()) for t in range(self.T)])
            for t in range(self.T):
                for i in range(net.num_buses):
                    bus = net.get_bus(i)
                    dP = f[bus.index_P[t]]
                    dQ = f[bus.index_Q[t]]
                    dP_list[t].append(dP)
                    dQ_list[t].append(dQ)
                    self.assertAlmostEqual(dP,bus.P_mismatch[t])
                    self.assertAlmostEqual(dQ,bus.Q_mismatch[t])
            self.assertAlmostEqual(net.bus_P_mis[t],np.max(np.abs(dP_list[t]))*net.base_power)
            self.assertAlmostEqual(net.bus_Q_mis[t],np.max(np.abs(dQ_list[t]))*net.base_power)

            # Check mismatches across time
            for vargen in net.var_generators:
                vargen.P = np.ones(net.num_periods)*0.2 # static
                vargen.Q = np.ones(net.num_periods)*0.1 # static
            for bat in net.batteries:
                bat.P = np.ones(net.num_periods)*0.1    # static
            x0 = net.get_var_values()
            constr.eval(x0)
            f = constr.f
            J = constr.J
            P_list = []
            for t in range(self.T):
                P_list.append(net.get_var_projection('all','any','all',t_start=t,t_end=t))
            fp_list = [f[t*net.num_buses:(t+1)*net.num_buses] for t in range(self.T)]
            fq_list = [f[(t+self.T)*net.num_buses:(t+1+self.T)*net.num_buses] for t in range(self.T)]
            for t in range(self.T-1):
                self.assertLess(norm(fp_list[t]-fp_list[t+1]),1e-12*norm(fp_list[t]))
                self.assertLess(norm(fq_list[t]-fq_list[t+1]),1e-12*norm(fq_list[t]))
            Jx = J*x0
            Jxp_list = [Jx[t*net.num_buses:(t+1)*net.num_buses] for t in range(self.T)]
            Jxq_list = [Jx[(t+self.T)*net.num_buses:(t+1+self.T)*net.num_buses] for t in range(self.T)]
            for t in range(self.T-1):
                self.assertLess(norm(Jxp_list[t]-Jxp_list[t+1]),1e-12*norm(Jxp_list[t]))
                self.assertLess(norm(Jxq_list[t]-Jxq_list[t+1]),1e-12*norm(Jxq_list[t]))
            for i in range(10):
                Hp_list = []
                Hq_list = []
                j = np.random.randint(0,net.num_buses)
                for t in range(self.T):
                    Hp_list.append(coo_matrix(P_list[t]*constr.get_H_single(t*net.num_buses+j)*P_list[t].T))
                    Hq_list.append(coo_matrix(P_list[t]*constr.get_H_single((t+self.T)*net.num_buses+j)*P_list[t].T))
                for t in range(self.T-1):
                    self.assertTrue(np.all(Hp_list[t].row == Hp_list[t+1].row))
                    self.assertTrue(np.all(Hp_list[t].col == Hp_list[t+1].col))
                    self.assertLessEqual(norm(Hp_list[t].data-Hp_list[t+1].data),1e-12*norm(Hp_list[t].data))
                    self.assertTrue(np.all(Hq_list[t].row == Hq_list[t+1].row))
                    self.assertTrue(np.all(Hq_list[t].col == Hq_list[t+1].col))
                    self.assertLessEqual(norm(Hq_list[t].data-Hq_list[t+1].data),1e-12*norm(Hq_list[t].data))

            # Jacobian check
            pf.tests.utils.check_constraint_Jacobian(self,
                                                     constr,
                                                     x0,
                                                     np.zeros(0),
                                                     NUM_TRIALS,
                                                     TOL,
                                                     EPS,
                                                     h)

            # Sigle Hessian check
            pf.tests.utils.check_constraint_single_Hessian(self,
                                                           constr,
                                                           x0,
                                                           np.zeros(0),
                                                           NUM_TRIALS,
                                                           TOL,
                                                           EPS,
                                                           h)

            # Combined Hessian check
            pf.tests.utils.check_constraint_combined_Hessian(self,
                                                             constr,
                                                             x0,
                                                             np.zeros(0),
                                                             NUM_TRIALS,
                                                             TOL,
                                                             EPS,
                                                             h)
            

            # Sensitivities
            net.clear_sensitivities()
            for t in range(self.T):
                for i in range(net.num_buses):
                    bus = net.get_bus(i)
                    self.assertEqual(bus.sens_P_balance[t],0.)
                    self.assertEqual(bus.sens_Q_balance[t],0.)
            sens = np.zeros(2*net.num_buses*self.T)
            for t in range(self.T):
                for i in range(net.num_buses):
                    bus = net.get_bus(i)
                    sens[bus.index_P[t]] = 3.5*bus.index_P[t]+0.33+t*2*net.num_buses
                    sens[bus.index_Q[t]] = 3.4*bus.index_Q[t]+0.32+t*2*net.num_buses
            constr.store_sensitivities(None,sens,None,None)
            for t in range(self.T):
                for i in range(net.num_buses):
                    bus = net.get_bus(i)
                    self.assertEqual(bus.sens_P_balance[t],3.5*bus.index_P[t]+0.33+t*2*net.num_buses)
                    self.assertEqual(bus.sens_Q_balance[t],3.4*bus.index_Q[t]+0.32+t*2*net.num_buses)

    def test_constr_ACPF_with_outages(self):

        # Constants
        h = 1e-10

        # Multiperiods
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)
            
            # Vars
            net.set_flags('bus',
                          'variable',
                          'any',
                          ['voltage magnitude','voltage angle'])
            net.set_flags('generator',
                          'variable',
                          'any',
                          ['active power','reactive power'])
            net.set_flags('load',
                          'variable',
                          'any',
                          ['active power','reactive power'])
            net.set_flags('branch',
                          'variable',
                          'tap changer',
                          'tap ratio')
            net.set_flags('branch',
                          'variable',
                          'phase shifter',
                          'phase shift')
            net.set_flags('shunt',
                          'variable',
                          'switching - v',
                          'susceptance')
            net.set_flags('variable generator',
                          'variable',
                          'any',
                          ['active power','reactive power'])
            net.set_flags('battery',
                          'variable',
                          'any',
                          ['charging power','energy level'])
            self.assertEqual(net.num_vars,
                             (2*net.num_buses +
                              2*net.num_generators +
                              2*net.num_loads +
                              net.get_num_tap_changers() +
                              net.get_num_phase_shifters() +
                              net.get_num_switched_v_shunts() +
                              3*net.num_batteries +
                              net.num_var_generators*2)*self.T)

            x0 = net.get_var_values()
            
            constr0 = pf.Constraint('AC power balance', net)
            constr0.analyze()
            constr0.eval(x0)

            buses = net.buses[:10]
            side = []
            for bus in buses:
                for gen in bus.generators:
                    gen.outage = True
                for br in bus.branches_k:
                    self.assertTrue(bus.is_equal(br.bus_k))
                    br.outage = True
                    side.append(br.bus_m)
                for br in bus.branches_m:
                    self.assertTrue(bus.is_equal(br.bus_m))
                    br.outage = True
                    side.append(br.bus_k)

            constr1 = pf.Constraint('AC power balance', net)
            constr1.analyze()
            constr1.eval(x0)

            f0 = constr0.f
            f1 = constr1.f

            for bus in net.buses:
                if bus not in buses+side:
                    for t in range(self.T):
                        i = bus.index_P[t]
                        j = bus.index_Q[t]
                        self.assertLess(np.abs(f0[i]-f1[i]), 1e-8)
                        self.assertLess(np.abs(f0[j]-f1[j]), 1e-8)

            for bus in buses:
                for t in range(self.T):
                    i = bus.index_P[t]
                    j = bus.index_Q[t]
                    dp = 0.
                    dq = 0.
                    for gen in bus.generators:
                        self.assertTrue(gen.is_on_outage())
                        dp += gen.P[t]
                        dq += gen.Q[t]
                    for br in bus.branches_k:
                        dp -= br.P_km[t]
                        dq -= br.Q_km[t]
                    for br in bus.branches_m:
                        dp -= br.P_mk[t]
                        dq -= br.Q_mk[t]
                    self.assertLess(np.abs(f1[i]+dp-f0[i]), 1e-8)
                    self.assertLess(np.abs(f1[j]+dq-f0[j]), 1e-8)                 

            # Jacobian check
            pf.tests.utils.check_constraint_Jacobian(self,
                                                     constr1,
                                                     x0,
                                                     np.zeros(0),
                                                     NUM_TRIALS,
                                                     TOL,
                                                     EPS,
                                                     h)

            # Sigle Hessian check
            pf.tests.utils.check_constraint_single_Hessian(self,
                                                           constr1,
                                                           x0,
                                                           np.zeros(0),
                                                           NUM_TRIALS,
                                                           TOL,
                                                           EPS,
                                                           h)

            # Combined Hessian check
            pf.tests.utils.check_constraint_combined_Hessian(self,
                                                             constr1,
                                                             x0,
                                                             np.zeros(0),
                                                             NUM_TRIALS,
                                                             TOL,
                                                             EPS,
                                                             h)
                    
    def test_constr_REG_VSET(self):

        # Constants
        h = 1e-8

        # Multiperiod
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)

            # Vars
            net.set_flags('bus',
                          'variable',
                          'not slack',
                          ['voltage magnitude','voltage angle'])
            net.set_flags('generator',
                          'variable',
                          'slack',
                          'active power')
            net.set_flags('generator',
                          'variable',
                          'regulator',
                          'reactive power')
            self.assertEqual(net.num_vars,
                             (2*(net.num_buses-net.get_num_slack_buses()) +
                              net.get_num_slack_gens() +
                              net.get_num_reg_gens())*self.T)

            x0 = net.get_var_values()
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            # Constraint
            constr = pf.Constraint('voltage set point regulation',net)
            self.assertEqual(constr.name,'voltage set point regulation')

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b

            # Before
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,0))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,0))
            self.assertEqual(A.nnz,0)

            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.J_row,0)
            self.assertEqual(constr.A_row,0)
            self.assertEqual(constr.num_extra_vars,0)

            Jnnz = 0
            for i in range(net.num_buses):
                bus = net.get_bus(i)
                if bus.is_regulated_by_gen() and not bus.is_slack():
                    for gen in bus.reg_generators:
                        Jnnz += 4

            Annz = 3*(net.get_num_reg_gens()-net.get_num_slack_gens())

            rowsJ = 2*(net.get_num_reg_gens()-net.get_num_slack_gens())
            rowsA = net.get_num_reg_gens()-net.get_num_slack_gens()

            constr.analyze()
            self.assertEqual(constr.J_nnz,Jnnz*self.T)
            self.assertEqual(constr.A_nnz,Annz*self.T)
            self.assertEqual(constr.J_row,rowsJ*self.T)
            self.assertEqual(constr.A_row,rowsA*self.T)
            self.assertEqual(constr.num_extra_vars,rowsJ*self.T)

            y_init = constr.init_extra_vars
            self.assertEqual(y_init.size,constr.num_extra_vars)
            self.assertTrue(np.all(y_init == 0.))
            
            y0 = np.random.rand(constr.num_extra_vars)
            constr.eval(x0,y0)
            self.assertEqual(constr.J_nnz,Jnnz*self.T)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.J_row,rowsJ*self.T)
            self.assertEqual(constr.A_row,0)

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b

            # After
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(rowsJ*self.T,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(rowsA*self.T,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(rowsJ*self.T,net.num_vars+constr.num_extra_vars))
            self.assertEqual(J.nnz,Jnnz*self.T)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(rowsA*self.T,net.num_vars+constr.num_extra_vars))
            self.assertEqual(A.nnz,Annz*self.T)

            self.assertTrue(not np.any(np.isinf(b)))
            self.assertTrue(not np.any(np.isnan(b)))
            self.assertTrue(not np.any(np.isinf(f)))
            self.assertTrue(not np.any(np.isnan(f)))
            self.assertTrue(not np.any(np.isinf(J.data)))
            self.assertTrue(not np.any(np.isnan(J.data)))
            self.assertTrue(not np.any(np.isinf(A.data)))
            self.assertTrue(not np.any(np.isnan(A.data)))

            # Ax=b check
            self.assertEqual(norm(A.data,1),rowsA*3*self.T)
            self.assertEqual(np.sum(A.data),(net.get_num_reg_gens()-net.get_num_slack_gens())*self.T)
            for k in range(J.shape[0]//2):
                index1 = np.where(A.col == net.num_vars+2*k)[0]
                index2 = np.where(A.col == net.num_vars+2*k+1)[0]
                self.assertEqual(index1.size,1)
                self.assertEqual(index2.size,1)
                self.assertEqual(A.row[index1[0]],A.row[index2[0]])
                index3 = np.where(A.row == A.row[index1[0]])[0]
                self.assertEqual(index3.size,3)
                for i in index3:
                    if A.col[i] == net.num_vars+2*k:   # y
                        self.assertEqual(A.data[i],-1.)
                    elif A.col[i] == net.num_vars+2*k+1:
                        self.assertEqual(A.data[i],1.) # z
                    else:
                        self.assertEqual(A.data[i],1.) # v

            # f check
            flags = {}
            eps = 1e-8
            J_row = 0
            for t in range(self.T):
                for bus in net.buses:
                    if bus.is_regulated_by_gen() and not bus.is_slack():
                        for gen in bus.reg_generators:
                            y = y0[J_row]
                            z = y0[J_row+1]
                            Q = gen.Q[t]
                            Qmax = gen.Q_max
                            Qmin = gen.Q_min
                            CompY = (Q-Qmin)+y-np.sqrt((Q-Qmin)**2.+y**2.+2*eps)
                            CompZ = (Qmax-Q)+z-np.sqrt((Qmax-Q)**2.+z**2.+2*eps)
                            self.assertAlmostEqual(CompY,f[J_row])
                            self.assertAlmostEqual(CompZ,f[J_row+1])
                            J_row += 2

            # Jacobian check
            pf.tests.utils.check_constraint_Jacobian(self,
                                                     constr,
                                                     x0,
                                                     y0,
                                                     NUM_TRIALS,
                                                     TOL,
                                                     EPS,
                                                     h)

            # Sigle Hessian check
            pf.tests.utils.check_constraint_single_Hessian(self,
                                                           constr,
                                                           x0,
                                                           y0,
                                                           NUM_TRIALS,
                                                           TOL,
                                                           EPS,
                                                           h)
            
            # Combined Hessian check
            pf.tests.utils.check_constraint_combined_Hessian(self,
                                                             constr,
                                                             x0,
                                                             y0,
                                                             NUM_TRIALS,
                                                             TOL,
                                                             EPS,
                                                             h)

            # Sensitivities
            net.clear_sensitivities()
            for t in range(self.T):
                for i in range(net.num_buses):
                    bus = net.get_bus(i)
                    self.assertEqual(bus.sens_v_set_reg[t],0.)
            sensf = np.zeros(constr.f.size)
            sensA = np.ones(constr.b.size)*10.5
            self.assertEqual(sensf.size,rowsJ*self.T)
            Ji = constr.J.row
            Jj = constr.J.col
            Ai = constr.A.row
            Aj = constr.A.col
            
            for t in range(self.T):
                for i in range(net.num_buses):
                    bus = net.get_bus(i)
                    if bus.is_regulated_by_gen() and not bus.is_slack():
                        indices = Ji[np.where(Jj == bus.reg_generators[-1].index_Q[t])[0]]
                        self.assertEqual(indices.size,2)
                        sensf[indices[0]] = -bus.index-10
                        sensf[indices[1]] = bus.index+11*(bus.index % 2)
            constr.store_sensitivities(sensA,sensf,None,None)
            for t in range(self.T):
                for i in range(net.num_buses):
                    bus = net.get_bus(i)
                    if bus.is_regulated_by_gen() and not bus.is_slack():
                        if bus.index % 2 == 1:
                            self.assertEqual(bus.sens_v_set_reg[t],bus.index+11)
                        else:
                            self.assertEqual(bus.sens_v_set_reg[t],-bus.index-10 if bus.index != 0 else 10.5)

    def test_constr_REG_VSET_with_outages(self):

        # Multiperiod
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)

            # Vars
            net.set_flags('bus',
                          'variable',
                          'not slack',
                          ['voltage magnitude','voltage angle'])
            net.set_flags('generator',
                          'variable',
                          'slack',
                          'active power')
            net.set_flags('generator',
                          'variable',
                          'regulator',
                          'reactive power')
            self.assertEqual(net.num_vars,
                             (2*(net.num_buses-net.get_num_slack_buses()) +
                              net.get_num_slack_gens() +
                              net.get_num_reg_gens())*self.T)

            x0 = net.get_var_values()
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            constr0 = pf.Constraint('voltage set point regulation', net)
            constr0.analyze()
            constr0.eval(x0)
            
            for bus in net.buses:
                if bus.is_regulated_by_gen():
                    for branch in bus.branches:
                        branch.outage = True

            constr1 = pf.Constraint('voltage set point regulation', net)
            constr1.analyze()
            constr1.eval(x0)

            self.assertEqual((constr0.A-constr1.A).tocoo().nnz, 0)
            self.assertLess(norm(constr0.b-constr1.b), 1e-8)
            self.assertEqual((constr0.J-constr1.J).tocoo().nnz, 0)
            self.assertLess(norm(constr0.f-constr1.f), 1e-8)

            for bus in net.buses:
                if bus.is_regulated_by_gen():
                    for gen in bus.reg_generators:
                        gen.outage = True
                    self.assertFalse(bus.is_regulated_by_gen())

            constr2 = pf.Constraint('voltage set point regulation', net)
            constr2.analyze()
            constr2.eval(x0)

            self.assertEqual(constr2.A.shape[0], 0)
            self.assertEqual(constr2.J.shape[0], 0)
             
    def test_constr_REG_TRAN(self):

        # Constants
        h = 1e-8
        normal = 1e0
        eta = 1e-8

        # Multiperiod
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)

            # Vars
            net.set_flags('bus',
                          'variable',
                          'regulated by transformer',
                          'voltage magnitude')
            net.set_flags('branch',
                          'variable',
                          'tap changer - v',
                          'tap ratio')
            self.assertEqual(net.num_vars,
                             (net.get_num_buses_reg_by_tran() +
                              net.get_num_tap_changers_v())*self.T)

            x0 = net.get_var_values()
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            # Constraint
            constr = pf.Constraint('voltage regulation by transformers',net)
            self.assertEqual(constr.name,'voltage regulation by transformers')

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b

            # Before
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,0))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,0))
            self.assertEqual(A.nnz,0)

            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.J_row,0)
            self.assertEqual(constr.A_row,0)

            Jnnz = 10*net.get_num_tap_changers_v()
            Annz = 3*net.get_num_tap_changers_v()
            self.assertGreaterEqual(Jnnz,0)
            self.assertGreaterEqual(Annz,0)

            rowsJ = 4*net.get_num_tap_changers_v()
            rowsA = net.get_num_tap_changers_v()
            self.assertGreaterEqual(rowsJ,0)
            self.assertGreaterEqual(rowsA,0)

            constr.analyze()
            self.assertEqual(constr.J_nnz,Jnnz*self.T)
            self.assertEqual(constr.A_nnz,Annz*self.T)
            self.assertEqual(constr.J_row,rowsJ*self.T)
            self.assertEqual(constr.A_row,rowsA*self.T)

            y_init = constr.init_extra_vars
            self.assertEqual(y_init.size,constr.num_extra_vars)
            self.assertTrue(np.all(y_init == 0.))

            constr.eval(x0)
            self.assertEqual(constr.J_nnz,Jnnz*self.T)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.J_row,rowsJ*self.T)
            self.assertEqual(constr.A_row,0)

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            
            # After
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(rowsJ*self.T,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(rowsA*self.T,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(rowsJ*self.T,net.num_vars+constr.num_extra_vars))
            self.assertEqual(J.nnz,Jnnz*self.T)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(rowsA*self.T,net.num_vars+constr.num_extra_vars))
            self.assertEqual(A.nnz,Annz*self.T)
            self.assertEqual(constr.num_extra_vars,rowsJ*self.T)

            self.assertTrue(not np.any(np.isinf(b)))
            self.assertTrue(not np.any(np.isnan(b)))
            self.assertTrue(not np.any(np.isinf(f)))
            self.assertTrue(not np.any(np.isnan(f)))
            self.assertTrue(not np.any(np.isinf(J.data)))
            self.assertTrue(not np.any(np.isnan(J.data)))
            self.assertTrue(not np.any(np.isinf(A.data)))
            self.assertTrue(not np.any(np.isnan(A.data)))

            y0 = np.random.rand(constr.num_extra_vars)

            # Ax=b check
            self.assertEqual(norm(A.data,1),rowsA*3*self.T)
            self.assertEqual(np.sum(A.data),net.get_num_tap_changers_v()*self.T)

            # f check
            index = 0
            for t in range(self.T):
                for bus in net.buses:
                    for br in bus.branches_k:
                        if br.is_tap_changer_v():
                            self.assertTrue(br.has_flags('variable','tap ratio'))
                            bus = br.reg_bus
                            fvmin = ((bus.v_mag[t]-bus.v_min_reg) - np.sqrt((bus.v_mag[t]-bus.v_min_reg)**2. + 2*eta))*normal
                            fvmax = ((bus.v_max_reg-bus.v_mag[t]) - np.sqrt((bus.v_max_reg-bus.v_mag[t])**2. + 2*eta))*normal
                            ftmax = ((br.ratio_max-br.ratio[t]) - np.sqrt((br.ratio_max-br.ratio[t])**2. + 2*eta))*normal
                            ftmin = ((br.ratio[t]-br.ratio_min) - np.sqrt((br.ratio[t]-br.ratio_min)**2. + 2*eta))*normal
                            self.assertLess(np.abs(fvmin-f[index]),1e-10*(1+np.abs(fvmin)))
                            self.assertLess(np.abs(fvmax-f[index+1]),1e-10*(1+np.abs(fvmax)))
                            self.assertLess(np.abs(ftmax-f[index+2]),1e-10*(1+np.abs(ftmax)))
                            self.assertLess(np.abs(ftmin-f[index+3]),1e-10*(1+np.abs(ftmin)))
                            index += 4

            # Jacobian check
            pf.tests.utils.check_constraint_Jacobian(self,
                                                     constr,
                                                     x0,
                                                     y0,
                                                     NUM_TRIALS,
                                                     TOL,
                                                     EPS,
                                                     h)

            # Sigle Hessian check
            pf.tests.utils.check_constraint_single_Hessian(self,
                                                           constr,
                                                           x0,
                                                           y0,
                                                           NUM_TRIALS,
                                                           TOL,
                                                           EPS,
                                                           h)
            
            # Combined Hessian check
            pf.tests.utils.check_constraint_combined_Hessian(self,
                                                             constr,
                                                             x0,
                                                             y0,
                                                             NUM_TRIALS,
                                                             TOL,
                                                             EPS,
                                                             h)

            # Sensitivities
            net.clear_sensitivities()
            for t in range(self.T):
                for i in range(net.num_buses):
                    bus = net.get_bus(i)
                    self.assertEqual(bus.sens_v_reg_by_tran[t],0.)
            sens = np.zeros(constr.f.size)
            counter = 0
            for t in range(self.T):
                for bus in net.buses:
                    for branch in bus.branches_k:
                        if branch.is_tap_changer_v():
                            sens[counter:counter+4] = branch.reg_bus.index*t
                            counter += 4
            self.assertEqual(counter,constr.f.size)
            constr.store_sensitivities(np.zeros(constr.A.shape[0]),sens,None,None)
            for t in range(self.T):
                for bus in net.buses:
                    for branch in bus.branches_k:
                        if branch.is_tap_changer_v():
                            self.assertEqual(branch.reg_bus.sens_v_reg_by_tran[t],branch.reg_bus.index*t)

    def test_constr_REG_TRAN_with_outages(self):

        # Multiperiod
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)

            # Vars
            net.set_flags('bus',
                          'variable',
                          'regulated by transformer',
                          'voltage magnitude')
            net.set_flags('branch',
                          'variable',
                          'tap changer - v',
                          'tap ratio')
            self.assertEqual(net.num_vars,
                             (net.get_num_buses_reg_by_tran() +
                              net.get_num_tap_changers_v())*self.T)

            x0 = net.get_var_values()
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            constr0 = pf.Constraint('voltage regulation by transformers', net)
            constr0.analyze()
            constr0.eval(x0)

            for bus in net.buses:
                if bus.is_regulated_by_tran():
                    for gen in bus.generators:
                        gen.outage = True
            
            constr1 = pf.Constraint('voltage regulation by transformers', net)
            constr1.analyze()
            constr1.eval(x0)

            self.assertEqual((constr0.A-constr1.A).tocoo().nnz, 0)
            self.assertLess(norm(constr0.b-constr1.b), 1e-8)
            self.assertEqual((constr0.J-constr1.J).tocoo().nnz, 0)
            self.assertLess(norm(constr0.f-constr1.f), 1e-8)

            for bus in net.buses:
                if bus.is_regulated_by_tran():
                    for branch in bus.reg_trans:
                        branch.outage = True
                    self.assertFalse(bus.is_regulated_by_tran())

            constr2 = pf.Constraint('voltage regulation by transformers', net)
            constr2.analyze()
            constr2.eval(x0)

            self.assertEqual(constr2.A.shape[0], 0)
            self.assertEqual(constr2.J.shape[0], 0)
                        
    def test_constr_REG_SHUNT(self):

        # Constants
        h = 1e-8
        normal = 1e0
        eta = 1e-8

        # Multiperiod
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)

            # Vars
            net.set_flags('bus',
                          'variable',
                          'regulated by shunt',
                          'voltage magnitude')
            net.set_flags('shunt',
                          'variable',
                          'switching - v',
                          'susceptance')
            self.assertEqual(net.num_vars,
                             (net.get_num_buses_reg_by_shunt() +
                              net.get_num_switched_v_shunts())*self.T)

            x0 = net.get_var_values()
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            # Constraint
            constr = pf.Constraint('voltage regulation by shunts',net)
            self.assertEqual(constr.name,'voltage regulation by shunts')

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b

            # Before
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,0))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,0))
            self.assertEqual(A.nnz,0)

            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.J_row,0)
            self.assertEqual(constr.A_row,0)

            Jnnz = 10*net.get_num_switched_v_shunts()
            Annz = 3*net.get_num_switched_v_shunts()
            self.assertGreaterEqual(Jnnz,0)
            self.assertGreaterEqual(Annz,0)

            rowsJ = 4*net.get_num_switched_v_shunts()
            rowsA = net.get_num_switched_v_shunts()
            self.assertGreaterEqual(rowsJ,0)
            self.assertGreaterEqual(rowsA,0)

            constr.analyze()
            self.assertEqual(constr.J_nnz,Jnnz*self.T)
            self.assertEqual(constr.A_nnz,Annz*self.T)
            self.assertEqual(constr.J_row,rowsJ*self.T)
            self.assertEqual(constr.A_row,rowsA*self.T)

            y_init = constr.init_extra_vars
            self.assertEqual(y_init.size,constr.num_extra_vars)
            self.assertTrue(np.all(y_init == 0.))

            constr.eval(x0)
            self.assertEqual(constr.J_nnz,Jnnz*self.T)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.J_row,rowsJ*self.T)
            self.assertEqual(constr.A_row,0)

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b

            # After
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(rowsJ*self.T,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(rowsA*self.T,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(rowsJ*self.T,net.num_vars+constr.num_extra_vars))
            self.assertEqual(J.nnz,Jnnz*self.T)
            self.assertTrue(np.all(J.row <= rowsJ*self.T-1))
            self.assertTrue(np.all(J.col <= net.num_vars+constr.num_extra_vars-1))
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(rowsA*self.T,net.num_vars+constr.num_extra_vars))
            self.assertEqual(A.nnz,Annz*self.T)
            self.assertTrue(np.all(A.row <= rowsA*self.T-1))
            self.assertTrue(np.all(A.col <= net.num_vars+constr.num_extra_vars-1))
            self.assertEqual(constr.num_extra_vars,rowsJ*self.T)

            self.assertTrue(not np.any(np.isinf(b)))
            self.assertTrue(not np.any(np.isnan(b)))
            self.assertTrue(not np.any(np.isinf(f)))
            self.assertTrue(not np.any(np.isnan(f)))
            self.assertTrue(not np.any(np.isinf(J.data)))
            self.assertTrue(not np.any(np.isnan(J.data)))
            self.assertTrue(not np.any(np.isinf(A.data)))
            self.assertTrue(not np.any(np.isnan(A.data)))

            y0 = np.random.rand(constr.num_extra_vars)

            # Ax=b check
            self.assertEqual(norm(A.data,1),rowsA*3*self.T)
            self.assertEqual(np.sum(A.data),net.get_num_switched_v_shunts()*self.T)

            # f check
            index = 0
            for t in range(self.T):
                for bus in net.buses:
                    for s in bus.reg_shunts:
                        self.assertEqual(bus.number,s.reg_bus.number)
                        self.assertTrue(bus.has_flags('variable','voltage magnitude'))
                        self.assertTrue(s.has_flags('variable','susceptance'))
                        fvmin = ((bus.v_mag[t]-bus.v_min_reg) - np.sqrt((bus.v_mag[t]-bus.v_min_reg)**2. + 2.*eta))*normal
                        fvmax = ((bus.v_max_reg-bus.v_mag[t]) - np.sqrt((bus.v_max_reg-bus.v_mag[t])**2. + 2.*eta))*normal
                        fbmax = ((s.b_max-s.b[t]) - np.sqrt((s.b_max-s.b[t])**2. + 2*eta))*normal
                        fbmin = ((s.b[t]-s.b_min) - np.sqrt((s.b[t]-s.b_min)**2. + 2*eta))*normal
                        self.assertLess(np.abs(fvmin-f[index]),1e-10*(1+np.abs(fvmin)))
                        self.assertLess(np.abs(fvmax-f[index+1]),1e-10*(1+np.abs(fvmax)))
                        self.assertLess(np.abs(fbmax-f[index+2]),1e-10*(1+np.abs(fbmax)))
                        self.assertLess(np.abs(fbmin-f[index+3]),1e-10*(1+np.abs(fbmin)))
                        index += 4

            # Jacobian check
            pf.tests.utils.check_constraint_Jacobian(self,
                                                     constr,
                                                     x0,
                                                     y0,
                                                     NUM_TRIALS,
                                                     TOL,
                                                     EPS,
                                                     h)

            # Sigle Hessian check
            pf.tests.utils.check_constraint_single_Hessian(self,
                                                           constr,
                                                           x0,
                                                           y0,
                                                           NUM_TRIALS,
                                                           TOL,
                                                           EPS,
                                                           h)

            # Combined Hessian check
            pf.tests.utils.check_constraint_combined_Hessian(self,
                                                             constr,
                                                             x0,
                                                             y0,
                                                             NUM_TRIALS,
                                                             TOL,
                                                             EPS,
                                                             h)

            # Sensitivities
            net.clear_sensitivities()
            for t in range(self.T):
                for i in range(net.num_buses):
                    bus = net.get_bus(i)
                    self.assertEqual(bus.sens_v_reg_by_shunt[t],0.)
            sens = np.zeros(constr.f.size)
            counter = 0
            for t in range(self.T):
                for bus in net.buses:
                    for shunt in bus.reg_shunts:
                        sens[counter:counter+4] = bus.index*t
                        counter += 4
            self.assertEqual(counter,constr.f.size)
            constr.store_sensitivities(np.zeros(constr.A.shape[0]),sens,None,None)
            for t in range(self.T):
                for bus in net.buses:
                    for shunt in bus.reg_shunts:
                        self.assertEqual(bus.sens_v_reg_by_shunt[t],bus.index*t)

    def test_constr_REG_SHUNT_with_outages(self):

        # Multiperiod
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)

            # Vars
            net.set_flags('bus',
                          'variable',
                          'regulated by shunt',
                          'voltage magnitude')
            net.set_flags('shunt',
                          'variable',
                          'switching - v',
                          'susceptance')
            self.assertEqual(net.num_vars,
                             (net.get_num_buses_reg_by_shunt() +
                              net.get_num_switched_v_shunts())*self.T)

            x0 = net.get_var_values()

            constr0 = pf.Constraint('voltage regulation by shunts', net)
            constr0.analyze()
            constr0.eval(x0)

            for bus in net.buses:
                if bus.is_regulated_by_shunt():
                    for gen in bus.generators:
                        gen.outage = True
                    for branch in bus.branches:
                        branch.outage = True
            
            constr1 = pf.Constraint('voltage regulation by shunts', net)
            constr1.analyze()
            constr1.eval(x0)

            self.assertEqual((constr0.A-constr1.A).tocoo().nnz, 0)
            self.assertLess(norm(constr0.b-constr1.b), 1e-8)
            self.assertEqual((constr0.J-constr1.J).tocoo().nnz, 0)
            self.assertLess(norm(constr0.f-constr1.f), 1e-8)
                        
    def test_robustness(self):

        for case in test_cases.CASES:

            net = pf.Network(self.T)

            constraints = [pf.Constraint('variable fixing',net),
                           pf.Constraint('generator active power participation',net),
                           pf.Constraint('PVPQ switching',net),
                           pf.Constraint('AC power balance',net),
                           pf.Constraint('DC power balance',net),
                           pf.Constraint('voltage set point regulation',net),
                           pf.Constraint('voltage regulation by transformers',net),
                           pf.Constraint('voltage regulation by shunts',net),
                           pf.Constraint('AC branch flow limits',net)]

            x0 = net.get_var_values()

            for c in constraints:
                self.assertTrue(isinstance(c.b,np.ndarray))
                self.assertTrue(isinstance(c.A,coo_matrix))
                self.assertTrue(isinstance(c.f,np.ndarray))
                self.assertTrue(isinstance(c.J,coo_matrix))
                self.assertEqual(c.b.size,0)
                self.assertEqual(c.A.nnz,0)
                self.assertTupleEqual(c.A.shape,(0,0))
                self.assertEqual(c.f.size,0)
                self.assertEqual(c.J.nnz,0)
                self.assertTupleEqual(c.J.shape,(0,0))

            list(map(lambda c: c.eval(x0),constraints))
            list(map(lambda c: c.analyze(),constraints))
            list(map(lambda c: c.eval(x0),constraints))

            for c in constraints:
                self.assertTrue(isinstance(c.b,np.ndarray))
                self.assertTrue(isinstance(c.A,coo_matrix))
                self.assertTrue(isinstance(c.f,np.ndarray))
                self.assertTrue(isinstance(c.J,coo_matrix))
                self.assertEqual(c.b.size,0)
                self.assertEqual(c.A.nnz,0)
                self.assertTupleEqual(c.A.shape,(0,0))
                self.assertEqual(c.f.size,0)
                self.assertEqual(c.J.nnz,0)
                self.assertTupleEqual(c.J.shape,(0,0))

            # Network changes
            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)

            
            constraints = [pf.Constraint('variable fixing',net),
                           pf.Constraint('generator active power participation',net),
                           pf.Constraint('PVPQ switching',net),
                           pf.Constraint('AC power balance',net),
                           pf.Constraint('DC power balance',net),
                           pf.Constraint('voltage set point regulation',net),
                           pf.Constraint('voltage regulation by transformers',net),
                           pf.Constraint('voltage regulation by shunts',net),
                           pf.Constraint('AC branch flow limits',net)]
            
            # After updating network
            list(map(lambda c: c.analyze(),constraints))
            list(map(lambda c: c.eval(x0),constraints))

            for c in constraints:
                self.assertTrue(isinstance(c.b,np.ndarray))
                self.assertTrue(isinstance(c.A,coo_matrix))
                self.assertTrue(isinstance(c.f,np.ndarray))
                self.assertTrue(isinstance(c.J,coo_matrix))

            # Add variables
            net.set_flags('bus',
                          'variable',
                          'any',
                          ['voltage magnitude','voltage angle'])
            net.set_flags('generator',
                          'variable',
                          'any',
                          ['active power','reactive power'])
            net.set_flags('branch',
                          'variable',
                          'tap changer',
                          'tap ratio')
            net.set_flags('branch',
                          'variable',
                          'phase shifter',
                          'phase shift')
            net.set_flags('shunt',
                          'variable',
                          'switching - v',
                          'susceptance')
            net.set_flags('battery',
                          'variable',
                          'any',
                          ['charging power','energy level'])
            self.assertEqual(net.num_vars,
                             (2*net.num_buses +
                              2*net.num_generators +
                              net.get_num_tap_changers()+
                              net.get_num_phase_shifters()+
                              net.get_num_switched_v_shunts()+
                              3*net.num_batteries)*self.T)

            x0 = net.get_var_values()

            # Before analyzing
            list(map(lambda c: c.clear_error(),constraints))
            for c in constraints:
                self.assertRaises(pf.ConstraintError,c.eval,x0)
            list(map(lambda c: c.clear_error(),constraints))
            
            # Do it right
            list(map(lambda c: c.analyze(),constraints))
            list(map(lambda c: c.eval(x0),constraints))

            for c in constraints:
                self.assertTrue(isinstance(c.b,np.ndarray))
                self.assertTrue(isinstance(c.A,coo_matrix))
                self.assertTrue(isinstance(c.f,np.ndarray))
                self.assertTrue(isinstance(c.J,coo_matrix))
                self.assertEqual(c.A.shape[1],net.num_vars+c.num_extra_vars)
                self.assertEqual(c.J.shape[1],net.num_vars+c.num_extra_vars)
                if c.f.size:
                    self.assertTupleEqual(c.get_H_single(0).shape,
                                          (net.num_vars+c.num_extra_vars,net.num_vars+c.num_extra_vars))
                else:
                    self.assertTupleEqual(c.get_H_single(0).shape,(0,0))

    def test_constr_DCPF(self):

        # Single period
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case)
            self.assertEqual(net.num_periods,1)

            self.assertEqual(net.num_vars,0)

            # Add vargens
            load_buses = net.get_load_buses()
            net.add_var_generators_from_parameters(load_buses,80.,50.,30.,5,0.05)
            self.assertGreater(net.num_var_generators,0)
            self.assertEqual(net.num_var_generators,len([b for b in net.buses if b.loads]))
            for b in net.buses:
                if b.loads:
                    self.assertGreater(len(b.var_generators),0)
                    for vargen in b.var_generators:
                        self.assertEqual(vargen.bus,b)

            # batteries
            for bat in net.batteries:
                if bat.index % 2 == 0:
                    bat.P *= -1.

            # Variables
            net.set_flags('bus',
                          'variable',
                          'not slack',
                          'voltage angle')
            net.set_flags('generator',
                          'variable',
                          'any',
                          'active power')
            net.set_flags('load',
                          'variable',
                          'any',
                          'active power')
            net.set_flags('variable generator',
                          'variable',
                          'any',
                          'active power')
            net.set_flags('branch',
                          'variable',
                          'phase shifter',
                          'phase shift')
            net.set_flags('battery',
                          'variable',
                          'any',
                          'charging power')
            self.assertEqual(net.num_vars,
                             (net.num_buses-net.get_num_slack_buses() +
                              net.num_generators +
                              net.num_loads +
                              net.num_var_generators +
                              net.get_num_phase_shifters()+
                              2*net.num_batteries))

            x0 = net.get_var_values()
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            # Constraint
            constr = pf.Constraint('DC power balance',net)
            self.assertEqual(constr.name,'DC power balance')
            
            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b

            # Before
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,0))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,0))
            self.assertEqual(A.nnz,0)
            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.J_row,0)
            self.assertEqual(constr.A_row,0)

            r = 0
            for b in net.buses:
                if b.is_slack():
                    r += len(b.branches)

            # Analyze
            constr.analyze()
            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.J_row,0)
            self.assertEqual(constr.A_row,net.num_buses*1)
            self.assertEqual(constr.A_nnz,
                             (net.num_generators +
                              net.num_loads +
                              net.num_var_generators +
                              4*net.num_branches -
                              2*r +
                              2*net.get_num_phase_shifters()+
                              2*net.num_batteries))
            self.assertTupleEqual(b.shape,(net.num_buses,))
            self.assertTupleEqual(f.shape,(0,))
            self.assertTupleEqual(A.shape,(net.num_buses,net.num_vars))
            self.assertEqual(A.nnz,constr.A_nnz)
            self.assertTupleEqual(J.shape,(0,net.num_vars))

            constr.eval(x0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(A.nnz,
                             (net.num_generators +
                              net.num_loads +
                              net.num_var_generators +
                              4*net.num_branches -
                              2*r +
                              2*net.get_num_phase_shifters()+
                              2*net.num_batteries))

            # Extract pieces
            P1 = net.get_var_projection('bus','any','voltage angle')
            P2 = net.get_var_projection('generator','any','active power')
            P3 = net.get_var_projection('variable generator','any','active power')
            P4 = net.get_var_projection('branch','any','phase shift')
            P5 = net.get_var_projection('load','any','active power')
            P6 = net.get_var_projection('battery','any','charging power')

            G = A*P2.T
            R = A*P3.T
            Atheta = -A*P1.T
            Aphi = -A*P4.T
            L = -A*P5.T
            B = -A*P6.T
            x = np.random.randn(net.num_vars)
            p = P2*x
            r = P3*x
            theta = P1*x
            phi = P4*x
            l = P5*x
            Pb = P6*x
            self.assertLess(norm((G*p+R*r-Atheta*theta-Aphi*phi-L*l-B*Pb)-A*x),1e-10)

            # Sensitivities
            for bus in net.buses:
                self.assertEqual(bus.sens_P_balance,0.)
                self.assertEqual(bus.sens_Q_balance,0.)
            new_sens = np.random.randn(net.num_buses)
            constr.store_sensitivities(new_sens,None,None,None)
            for bus in net.buses:
                self.assertNotEqual(bus.sens_P_balance,0.)
                self.assertEqual(bus.sens_Q_balance,0.)
                self.assertEqual(bus.sens_P_balance,new_sens[bus.index])

            # mismatches
            mismatches = A*x0-b
            for bus in net.buses:
                mis = 0
                for gen in bus.generators:
                    mis += gen.P
                for vargen in bus.var_generators:
                    mis += vargen.P
                for load in bus.loads:
                    mis -= load.P
                for bat in bus.batteries:
                    mis -= bat.P
                for br in bus.branches_k:
                    mis -= br.P_km_DC
                for br in bus.branches_m:
                    mis += br.P_km_DC
                self.assertLess(np.abs(mismatches[bus.index]-mis),1e-8)

            # No variables
            net.clear_flags()
            self.assertEqual(net.num_vars,0)
            constr.del_matvec()
            constr.analyze()
            f1 = constr.f
            J1 = constr.J
            A1 = constr.A
            b1 = constr.b
            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.J_row,0)
            self.assertEqual(constr.A_row,net.num_buses)
            self.assertEqual(constr.A_nnz,0)
            self.assertTupleEqual(b1.shape,(net.num_buses,))
            self.assertTupleEqual(f1.shape,(0,))
            self.assertTupleEqual(A1.shape,(net.num_buses,net.num_vars))
            self.assertEqual(A1.nnz,constr.A_nnz)
            self.assertTupleEqual(J1.shape,(0,net.num_vars))
            x1 = net.get_var_values()
            self.assertTrue(type(x1) is np.ndarray)
            self.assertTupleEqual(x1.shape,(net.num_vars,))

            mismatches1 = A1*x1-b1
            for bus in net.buses:
                mis = 0
                for gen in bus.generators:
                    mis += gen.P
                for vargen in bus.var_generators:
                    mis += vargen.P
                for load in bus.loads:
                    mis -= load.P
                for bat in bus.batteries:
                    mis -= bat.P
                for br in bus.branches_k:
                    mis -= br.P_km_DC
                for br in bus.branches_m:
                    mis -= br.P_mk_DC
                self.assertLess(np.abs(mismatches1[bus.index_P]-mis),1e-8)

        # Multi period
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)

            self.assertEqual(net.num_vars,0)

            # Add vargens
            load_buses = net.get_load_buses()
            net.add_var_generators_from_parameters(load_buses,80.,50.,30.,5,0.05)

            # batteries
            for bat in net.batteries:
                bat.P = np.random.randn(self.T)*10

            # Variables
            net.set_flags('bus',
                          'variable',
                          'not slack',
                          'voltage angle')
            net.set_flags('generator',
                          'variable',
                          'any',
                          'active power')
            net.set_flags('load',
                          'variable',
                          'any',
                          'active power')
            net.set_flags('variable generator',
                          'variable',
                          'any',
                          'active power')
            net.set_flags('branch',
                          'variable',
                          'phase shifter',
                          'phase shift')
            net.set_flags('battery',
                          'variable',
                          'any',
                          'charging power')
            self.assertEqual(net.num_vars,
                             (net.num_buses-net.get_num_slack_buses() +
                              net.num_generators +
                              net.num_loads +
                              net.num_var_generators +
                              net.get_num_phase_shifters()+
                              2*net.num_batteries)*self.T)
            x0 = net.get_var_values()

            # Count something
            r = 0
            for b in net.buses:
                if b.is_slack():
                    r += len(b.branches)

            # Constraint
            constr = pf.Constraint('DC power balance',net)
            self.assertEqual(constr.name,'DC power balance')

            # Analyze
            constr.analyze()
            A = constr.A
            b = constr.b
            self.assertEqual(constr.A_row, net.num_buses*self.T)
            self.assertEqual(constr.A_nnz,
                             (net.num_generators +
                              net.num_loads +
                              net.num_var_generators +
                              4*net.num_branches -
                              2*r +
                              2*net.get_num_phase_shifters()+
                              2*net.num_batteries)*self.T)
            self.assertTupleEqual(b.shape,(net.num_buses*self.T,))
            self.assertTupleEqual(A.shape,(net.num_buses*self.T,net.num_vars))
            self.assertEqual(A.nnz,constr.A_nnz)

            # Eval
            constr.eval(x0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(A.nnz,
                             (net.num_generators +
                              net.num_loads +
                              net.num_var_generators +
                              4*net.num_branches -
                              2*r +
                              2*net.get_num_phase_shifters()+
                              2*net.num_batteries)*self.T)

            # Mismatches
            mismatches = A*x0-b
            for t in range(self.T):
                for bus in net.buses:
                    mis = 0
                    for gen in bus.generators:
                        mis += gen.P[t]
                    for vargen in bus.var_generators:
                        mis += vargen.P[t]
                    for load in bus.loads:
                        mis -= load.P[t]
                    for bat in bus.batteries:
                        mis -= bat.P[t]
                    for br in bus.branches_k:
                        mis -= br.P_km_DC[t]
                    for br in bus.branches_m:
                        mis -= br.P_mk_DC[t]
                    self.assertLess(np.abs(mismatches[bus.index_t[t]]-mis),1e-8)

            # No variables
            net.clear_flags()
            self.assertEqual(net.num_vars,0)
            constr.del_matvec()
            constr.analyze()
            A1 = constr.A
            b1 = constr.b
            x1 = net.get_var_values()
            self.assertTupleEqual(x1.shape,(0,))

            mismatches1 = A1*x1-b1
            for t in range(self.T):
                for bus in net.buses:
                    mis = 0
                    for gen in bus.generators:
                        mis += gen.P[t]
                    for vargen in bus.var_generators:
                        mis += vargen.P[t]
                    for load in bus.loads:
                        mis -= load.P[t]
                    for bat in bus.batteries:
                        mis -= bat.P[t]
                    for br in bus.branches_k:
                        mis -= br.P_km_DC[t]
                    for br in bus.branches_m:
                        mis -= br.P_mk_DC[t]
                    self.assertLess(np.abs(mismatches1[bus.index_P[t]]-mis),1e-8)

            # Sensitivities
            net.clear_sensitivities()

            lam = np.random.randn(net.num_buses*net.num_periods)
            self.assertEqual(lam.size, constr.A.shape[0])

            for t in range(net.num_periods):
                for bus in net.buses:
                    self.assertEqual(bus.sens_P_balance[t], 0.)
                    self.assertEqual(bus.sens_Q_balance[t], 0.)

            constr.store_sensitivities(lam, None, None, None)

            for t in range(net.num_periods):
                for bus in net.buses:
                    self.assertEqual(bus.sens_P_balance[t], lam[bus.index_P[t]])
                    self.assertNotEqual(bus.sens_P_balance[t], 0.)
                    self.assertEqual(bus.sens_Q_balance[t], 0.)

    def test_constr_DCPF_with_outages(self):

        # Multiperiods
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)
            
            # Vars
            net.set_flags('bus',
                          'variable',
                          'any',
                          'voltage angle')
            net.set_flags('generator',
                          'variable',
                          'any',
                          'active power')
            net.set_flags('load',
                          'variable',
                          'any',
                          'active power')
            net.set_flags('branch',
                          'variable',
                          'phase shifter',
                          'phase shift')
            self.assertEqual(net.num_vars,
                             (net.num_buses +
                              net.num_generators +
                              net.num_loads +
                              net.get_num_phase_shifters())*self.T)

            x0 = net.get_var_values()
            
            constr0 = pf.Constraint('DC power balance', net)
            constr0.analyze()
            constr0.eval(x0)

            buses = net.buses[:10]
            side = []
            for bus in buses:
                for gen in bus.generators:
                    gen.outage = True
                for br in bus.branches_k:
                    self.assertTrue(bus.is_equal(br.bus_k))
                    br.outage = True
                    side.append(br.bus_m)
                for br in bus.branches_m:
                    self.assertTrue(bus.is_equal(br.bus_m))
                    br.outage = True
                    side.append(br.bus_k)

            constr1 = pf.Constraint('DC power balance', net)
            constr1.analyze()
            constr1.eval(x0)
            
            f0 = constr0.A*x0-constr0.b
            f1 = constr1.A*x0-constr1.b

            for bus in net.buses:
                if bus not in buses+side:
                    for t in range(self.T):
                        i = bus.index_P[t]
                        self.assertLess(np.abs(f0[i]-f1[i]), 1e-8)

            for bus in buses:
                for t in range(self.T):
                    i = bus.index_P[t]
                    dp = 0.
                    for gen in bus.generators:
                        self.assertTrue(gen.is_on_outage())
                        dp += gen.P[t]
                    for br in bus.branches_k:
                        dp -= br.P_km_DC[t]
                    for br in bus.branches_m:
                        dp -= br.P_mk_DC[t]
                    self.assertLess(np.abs(f1[i]+dp-f0[i]), 1e-8)
                    
    def test_constr_DC_FLOW_LIM(self):

        # Single period
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case)
            self.assertEqual(net.num_periods,1)

            self.assertEqual(net.num_vars,0)

            # Variables
            net.set_flags('bus',
                          'variable',
                          'not slack',
                          'voltage angle')
            self.assertEqual(net.num_vars,net.num_buses-net.get_num_slack_buses())

            x0 = net.get_var_values()
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            # Constraint
            constr = pf.Constraint('DC branch flow limits',net)
            self.assertEqual(constr.name,'DC branch flow limits')

            # Num constr
            num_constr = len([br for br in net.branches if br.ratingA != 0.])

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            G = constr.G
            l = constr.l
            u = constr.u

            # Before
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(l) is np.ndarray)
            self.assertTupleEqual(l.shape,(0,))
            self.assertTrue(type(u) is np.ndarray)
            self.assertTupleEqual(u.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,0))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,0))
            self.assertEqual(A.nnz,0)
            self.assertTrue(type(G) is coo_matrix)
            self.assertTupleEqual(G.shape,(0,0))
            self.assertEqual(G.nnz,0)
            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.G_nnz,0)
            self.assertEqual(constr.J_row,0)
            self.assertEqual(constr.A_row,0)
            self.assertEqual(constr.G_row,0)

            # Analyze
            constr.analyze()
            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            l = constr.l
            u = constr.u
            G = constr.G
            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.J_row,0)
            self.assertEqual(constr.A_row,0)
            self.assertEqual(constr.G_row,num_constr)

            self.assertTupleEqual(b.shape,(0,))
            self.assertTupleEqual(f.shape,(0,))
            self.assertTupleEqual(l.shape,(num_constr,))
            self.assertTupleEqual(u.shape,(num_constr,))

            self.assertTupleEqual(A.shape,(0,net.num_vars))
            self.assertTupleEqual(J.shape,(0,net.num_vars))
            self.assertTupleEqual(G.shape,(num_constr,net.num_vars))
            self.assertEqual(G.nnz,constr.G_nnz)

            self.assertTrue(np.all(l <= u))

            num = 0
            for br in net.branches:
                if br.ratingA == 0.:
                    continue
                if not br.bus_k.is_slack():
                    num += 1
                if not br.bus_m.is_slack():
                    num += 1
            self.assertEqual(num,constr.G_nnz)

            counter = 0
            index = 0
            for bus in net.buses:
                for br in bus.branches_k:
                    if br.ratingA == 0.:
                        continue
                    off = 0
                    if br.bus_k.is_slack():
                        off = br.b*br.bus_k.v_ang
                    else:
                        self.assertEqual(G.row[counter],index)
                        self.assertEqual(G.col[counter],br.bus_k.index_v_ang)
                        self.assertEqual(G.data[counter],-br.b)
                        counter += 1
                    if br.bus_m.is_slack():
                        off = -br.b*br.bus_m.v_ang
                    else:
                        self.assertEqual(G.row[counter],index)
                        self.assertEqual(G.col[counter],br.bus_m.index_v_ang)
                        self.assertEqual(G.data[counter],br.b)
                        counter += 1
                    rating = br.ratingA
                    self.assertEqual(l[index],-rating+off-br.b*br.phase)
                    self.assertEqual(u[index],rating+off-br.b*br.phase)
                    index += 1
            self.assertEqual(counter,G.nnz)
            self.assertEqual(index,G.shape[0])

            # Flow
            Gx0 = constr.G*x0
            self.assertTupleEqual(Gx0.shape,(num_constr,))
            index = 0
            for bus in net.buses:
                for branch in bus.branches_k:
                    if branch.ratingA == 0.:
                        continue
                    bus1 = branch.bus_k
                    bus2 = branch.bus_m
                    if bus1.is_slack():
                        flow = Gx0[index]-branch.b*(bus1.v_ang-branch.phase)
                    elif bus2.is_slack():
                        flow = Gx0[index]-branch.b*(-bus2.v_ang-branch.phase)
                    else:
                        flow = Gx0[index]-branch.b*(-branch.phase)
                    self.assertLess(np.abs(branch.P_km_DC-flow),1e-10)
                    index += 1

            # Sensitivities
            index = 0
            for branch in net.branches:
                self.assertEqual(branch.sens_P_u_bound,0.)
                self.assertEqual(branch.sens_P_l_bound,0.)
            mu = np.random.randn(num_constr)
            pi = np.random.randn(num_constr)
            self.assertEqual(constr.G.shape[0],num_constr)
            constr.store_sensitivities(None,None,mu,pi)
            for bus in net.buses:
                for branch in bus.branches_k:
                    if branch.ratingA == 0.:
                        continue
                    self.assertEqual(branch.sens_P_u_bound,mu[index])
                    self.assertEqual(branch.sens_P_l_bound,pi[index])
                    index += 1
            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.G_nnz,0)
            self.assertEqual(constr.J_row,0)
            self.assertEqual(constr.A_row,0)
            self.assertEqual(constr.G_row,num_constr)

        # Multi period
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)

            self.assertEqual(net.num_vars,0)

            # Nonzero angles
            for bus in net.buses:
                bus.v_ang = np.random.randn()*np.ones(self.T)

            # Variables
            net.set_flags('bus',
                          'variable',
                          'not slack',
                          'voltage angle')
            self.assertEqual(net.num_vars,(net.num_buses-net.get_num_slack_buses())*self.T)

            x0 = net.get_var_values()
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            # Num constr
            num_constr = len([br for br in net.branches if br.ratingA != 0.])

            # Constraint
            constr = pf.Constraint('DC branch flow limits',net)
            self.assertEqual(constr.name,'DC branch flow limits')
            constr.analyze()
            G = constr.G
            l = constr.l
            u = constr.u
            self.assertTupleEqual(l.shape,(num_constr*self.T,))
            self.assertTupleEqual(u.shape,(num_constr*self.T,))
            self.assertTupleEqual(G.shape,(num_constr*self.T,net.num_vars))
            Projs = []
            for t in range(self.T):
                Projs.append(net.get_var_projection('all','any','all',t,t))
            Gs = [G*P.T for P in Projs]
            x0s = [P*x0 for P in Projs]
            Gx0s = [(Gs[t]*x0s[t])[t*num_constr:(t+1)*num_constr] for t in range(self.T)]
            ls = [l[t*num_constr:(t+1)*num_constr] for t in range(self.T)]
            us = [u[t*num_constr:(t+1)*num_constr] for t in range(self.T)]
            for t in range(self.T):
                self.assertLessEqual(norm(Gx0s[t]-Gx0s[0]),1e-10*norm(Gx0s[0]))
                self.assertLessEqual(norm(ls[t]-ls[0]),1e-10*norm(ls[0]))
                self.assertLessEqual(norm(us[t]-us[0]),1e-10*norm(us[0]))

    def test_constr_DC_FLOW_LIM_with_outages(self):

        # Multi period
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case, self.T)
            self.assertEqual(net.num_periods, self.T)

            self.assertEqual(net.num_vars,0)

            # Variables
            net.set_flags('bus',
                          'variable',
                          'not slack',
                          'voltage angle')
            self.assertEqual(net.num_vars,(net.num_buses-net.get_num_slack_buses())*self.T)

            x0 = net.get_var_values()

            constr = pf.Constraint('DC branch flow limits', net)
            constr.analyze()
            constr.eval(x0)

            num_constr = len([br for br in net.branches if br.ratingA != 0.])*self.T

            self.assertEqual(constr.G.shape[0], num_constr)

            for branch in net.branches:
                branch.outage = True

            constr.analyze()

            self.assertEqual(constr.G.shape[0], 0)
        
    def test_constr_LINPF(self):

        # Multiperiod
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)

            # load
            if sum([l.P[0] for l in net.loads]) < 0:
                lmin = np.min([l.P for l in net.loads])
                for l in net.loads:
                    l.P = l.P + np.abs(lmin)

            # add vargens
            load_buses = net.get_load_buses()
            net.add_var_generators_from_parameters(load_buses,80.,50.,30.,5,0.05)
            self.assertGreater(net.num_var_generators,0)
            self.assertEqual(net.num_var_generators,len(load_buses))
            for vargen in net.var_generators:
                vargen.Q = np.abs(vargen.P)
                for t in range(self.T):
                    self.assertGreater(vargen.Q[t],0.)

            # Vars
            net.set_flags('bus',
                          'variable',
                          'any',
                          ['voltage magnitude','voltage angle'])
            net.set_flags('generator',
                          'variable',
                          'slack',
                          'active power')
            net.set_flags('generator',
                          'variable',
                          'regulator',
                          'reactive power')
            net.set_flags('branch',
                          'variable',
                          'tap changer',
                          'tap ratio')
            net.set_flags('branch',
                          'variable',
                          'phase shifter',
                          'phase shift')
            net.set_flags('shunt',
                          'variable',
                          'switching - v',
                          'susceptance')
            net.set_flags('variable generator',
                          'variable',
                          'any',
                          ['active power','reactive power'])
            self.assertEqual(net.num_vars,
                             (2*net.get_num_buses() +
                              net.get_num_slack_gens() +
                              net.get_num_reg_gens() +
                              net.get_num_tap_changers() +
                              net.get_num_phase_shifters() +
                              net.get_num_switched_v_shunts() +
                              net.num_var_generators*2)*self.T)

            x0 = net.get_var_values()
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            # Constraint
            constr = pf.Constraint('linearized AC power balance',net)
            self.assertEqual(constr.name,'linearized AC power balance')

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            G = constr.G

            # Before
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,0))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,0))
            self.assertEqual(A.nnz,0)
            self.assertTrue(type(G) is coo_matrix)
            self.assertTupleEqual(G.shape,(0,0))
            self.assertEqual(G.nnz,0)

            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.G_nnz,0)

            num_Annz = (net.num_buses*4 +
                        net.get_num_branches()*8 +
                        net.get_num_tap_changers()*4 +
                        net.get_num_phase_shifters()*4 +
                        net.get_num_switched_v_shunts() +
                        net.get_num_slack_gens() +
                        net.get_num_reg_gens()+
                        net.num_var_generators*2)

            constr.analyze()
            self.assertEqual(constr.A_nnz,0)
            constr.eval(x0)
            self.assertEqual(constr.A_nnz,0)

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            G = constr.G

            # After
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(2*net.num_buses*self.T,))
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(2*net.num_buses*self.T,net.num_vars))
            self.assertEqual(A.nnz,num_Annz*self.T)
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,net.num_vars))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(G) is coo_matrix)
            self.assertTupleEqual(G.shape,(0,net.num_vars))
            self.assertEqual(G.nnz,0)
            self.assertTrue(not np.any(np.isinf(b)))
            self.assertTrue(not np.any(np.isnan(b)))

            # Check with ACPF
            constrPF = pf.Constraint('AC power balance',net)
            self.assertEqual(constrPF.name,'AC power balance')
            constrPF.analyze()
            constrPF.eval(x0)
            self.assertEqual(A.nnz,constrPF.J.nnz)
            self.assertTrue(np.all(A.row == constrPF.J.row))
            self.assertTrue(np.all(A.col == constrPF.J.col))
            self.assertTrue(np.all(A.data == constrPF.J.data))
            self.assertGreater(norm(A.row),0)
            self.assertGreater(norm(A.col),0)
            self.assertGreater(norm(A.data),0)
            self.assertGreater(norm(b),0)
            self.assertLess(norm(b-(constrPF.J*x0-constrPF.f)),1e-10*(norm(b)+1))

            # After eval
            constr.eval(np.zeros(x0.size))
            self.assertEqual(constr.A.nnz,constrPF.J.nnz)
            self.assertTrue(np.all(constr.A.row == constrPF.J.row))
            self.assertTrue(np.all(constr.A.col == constrPF.J.col))
            self.assertTrue(np.all(constr.A.data == constrPF.J.data))
            self.assertGreater(norm(constr.A.row),0)
            self.assertGreater(norm(constr.A.col),0)
            self.assertGreater(norm(constr.A.data),0)
            self.assertGreater(norm(constr.b),0)
            self.assertLess(norm(constr.b-(constrPF.J*x0-constrPF.f)),1e-10*(norm(b)+1))

    def test_constr_LINPF_with_outages(self):

        # Multiperiod
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)

            for gen in net.generators:
                gen.outage = True
            for branch in net.branches:
                branch.outage = True

            # Vars
            net.set_flags('bus',
                          'variable',
                          'any',
                          ['voltage magnitude','voltage angle'])
            net.set_flags('generator',
                          'variable',
                          'slack',
                          'active power')
            net.set_flags('generator',
                          'variable',
                          'regulator',
                          'reactive power')
            net.set_flags('branch',
                          'variable',
                          'tap changer',
                          'tap ratio')
            net.set_flags('branch',
                          'variable',
                          'phase shifter',
                          'phase shift')
            net.set_flags('shunt',
                          'variable',
                          'switching - v',
                          'susceptance')
            net.set_flags('variable generator',
                          'variable',
                          'any',
                          ['active power','reactive power'])
            self.assertEqual(net.num_vars,
                             (2*net.get_num_buses() +
                              net.get_num_slack_gens() +
                              net.get_num_reg_gens() +
                              net.get_num_tap_changers() +
                              net.get_num_phase_shifters() +
                              net.get_num_switched_v_shunts() +
                              net.num_var_generators*2)*self.T)

            constr = pf.Constraint('linearized AC power balance',net)
            constr.analyze()

            x0 = net.get_var_values()

            constrPF = pf.Constraint('AC power balance',net)
            constrPF.analyze()
            constrPF.eval(x0)
            
            self.assertEqual(constr.A.nnz,constrPF.J.nnz)
            self.assertTrue(np.all(constr.A.row == constrPF.J.row))
            self.assertTrue(np.all(constr.A.col == constrPF.J.col))
            self.assertTrue(np.all(constr.A.data == constrPF.J.data))            
            if net.num_shunts:
                self.assertGreater(norm(constr.A.row),0)
                self.assertGreater(norm(constr.A.col),0)
                self.assertGreater(norm(constr.A.data),0)
            self.assertGreater(norm(constr.b),0)
            self.assertLess(norm(constr.b-(constrPF.J*x0-constrPF.f)),1e-10*(norm(constr.b)+1))
            
    def test_constr_GEN_RAMP(self):

        # Multi period
        for case in test_cases.CASES:
            
            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)
            self.assertEqual(net.num_vars,0)

            # Gens
            for gen in net.generators:
                gen.dP_max = np.random.rand()*100.
                gen.P_prev = np.random.rand()*10.
                gen.P = np.random.rand()*20

            # Vars
            net.set_flags('generator',
                          'variable',
                          'not slack',
                          'active power')
            num = net.num_generators-net.get_num_slack_gens()
            self.assertEqual(net.num_vars,num*self.T)

            x0 = net.get_var_values()
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            # Constraint
            constr = pf.Constraint('generator ramp limits',net)
            self.assertEqual(constr.name,'generator ramp limits')

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            l = constr.l
            G = constr.G
            u = constr.u

            # Before
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,0))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,0))
            self.assertEqual(A.nnz,0)
            self.assertTrue(type(l) is np.ndarray)
            self.assertTupleEqual(l.shape,(0,))
            self.assertTrue(type(u) is np.ndarray)
            self.assertTupleEqual(u.shape,(0,))
            self.assertTrue(type(G) is coo_matrix)
            self.assertTupleEqual(G.shape,(0,0))
            self.assertEqual(G.nnz,0)
            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.G_nnz,0)

            constr.analyze()
            self.assertEqual(constr.A_nnz,0)
            constr.eval(x0)
            self.assertEqual(constr.A_nnz,0)

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            l = constr.l
            G = constr.G
            u = constr.u

            # After
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,net.num_vars))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,net.num_vars))
            self.assertEqual(A.nnz,0)
            self.assertTrue(type(l) is np.ndarray)
            self.assertTupleEqual(l.shape,(num*self.T,))
            self.assertTrue(type(u) is np.ndarray)
            self.assertTupleEqual(u.shape,(num*self.T,))
            self.assertTrue(type(G) is coo_matrix)
            self.assertTupleEqual(G.shape,(num*self.T,net.num_vars))
            self.assertEqual(G.nnz,num*(1 + (self.T-1)*2))
            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.G_nnz,0)

            for t in range(self.T):
                for gen in net.generators:
                    if not gen.is_slack():
                        ac = np.where(G.col == gen.index_P[t])[0]

                        # Last time
                        if t == self.T-1:
                            self.assertEqual(ac.size,1)
                            i = G.row[ac[0]]
                            self.assertEqual(G.data[ac[0]],1.)
                            self.assertEqual(l[i],-gen.dP_max)
                            self.assertEqual(u[i],gen.dP_max)
                            ar = np.where(G.row == i)[0]
                            self.assertEqual(ar.size,2)
                            for j in ar:
                                if G.col[j] == gen.index_P[t]:
                                    pass
                                else:
                                    self.assertEqual(G.col[j],gen.index_P[t-1])
                                    self.assertEqual(G.data[j],-1.)

                        # Not last time
                        else:
                            self.assertEqual(ac.size,2)
                            for i in ac:
                                self.assertEqual(G.col[i],gen.index_P[t])

                                # added
                                if G.data[i] == -1.:
                                    self.assertEqual(l[G.row[i]],-gen.dP_max)
                                    self.assertEqual(u[G.row[i]],gen.dP_max)
                                    ar = np.where(G.row == G.row[i])[0]
                                    self.assertEqual(ar.size,2)
                                    for j in ar:
                                        if G.col[j] == gen.index_P[t]:
                                            pass
                                        else:
                                            self.assertEqual(G.col[j],gen.index_P[t+1])
                                            self.assertEqual(G.data[j],1.)

                                # subtracted
                                else:
                                    if t == 0:
                                        self.assertEqual(l[G.row[i]],-gen.dP_max+gen.P_prev)
                                        self.assertEqual(u[G.row[i]],gen.dP_max+gen.P_prev)
                                    else:
                                        self.assertEqual(l[G.row[i]],-gen.dP_max)
                                        self.assertEqual(u[G.row[i]],gen.dP_max)
                                        ar = np.where(G.row == G.row[i])[0]
                                        self.assertEqual(ar.size,2)
                                        for j in ar:
                                            if G.col[j] == gen.index_P[t]:
                                                pass
                                            else:
                                                self.assertEqual(G.col[j],gen.index_P[t-1])
                                                self.assertEqual(G.data[j],-1.)

    def test_constr_GEN_RAMP_with_outages(self):

        # Multi period
        for case in test_cases.CASES:
            
            net = pf.Parser(case).parse(case,self.T)

            # Vars
            net.set_flags('generator',
                          'variable',
                          'not slack',
                          'active power')
            num = net.num_generators-net.get_num_slack_gens()
            self.assertEqual(net.num_vars,num*self.T)

            x0 = net.get_var_values()

            # Constraint
            constr = pf.Constraint('generator ramp limits',net)
            constr.analyze()

            self.assertEqual(constr.A.shape[0], 0)
            self.assertGreater(constr.G.shape[0], 0)

            for gen in net.generators:
                gen.outage = True

            constr.analyze()

            self.assertEqual(constr.A.shape[0], 0)
            self.assertEqual(constr.G.shape[0], 0)
                                                
    def test_constr_AC_FLOW_LIM(self):

        # Constants
        h = 1e-11
        tol = 1e-2
        eps = 1.1 # %
        param = 1e-6

        # Multiperiod
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)

            # Vars
            net.set_flags('bus',
                          'variable',
                          'any',
                          ['voltage magnitude','voltage angle'])
            net.set_flags('branch',
                          'variable',
                          'tap changer',
                          'tap ratio')
            net.set_flags('branch',
                          'variable',
                          'phase shifter',
                          'phase shift')
            self.assertEqual(net.num_vars,
                             (2*net.get_num_buses() +
                              net.get_num_tap_changers() +
                              net.get_num_phase_shifters())*self.T)

            x0 = net.get_var_values()
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            # Constr
            constr = pf.Constraint('AC branch flow limits',net)
            self.assertEqual(constr.name,'AC branch flow limits')
            constr.analyze()
            num_constr = len([br for br in net.branches if br.ratingA != 0.])*2*net.num_periods
            self.assertTupleEqual(constr.f.shape,(num_constr,))
            self.assertEqual(constr.J_row,num_constr)

            # zero ratings
            for br in net.branches:
                if br.ratingA == 0.:
                    br.ratingA = 100.

            # Constraint
            constr = pf.Constraint('AC branch flow limits',net)
            self.assertEqual(constr.name,'AC branch flow limits')

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            G = constr.G
            l = constr.l
            u = constr.u

            # Before
            self.assertEqual(constr.num_extra_vars,0)
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,0))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,0))
            self.assertEqual(A.nnz,0)
            self.assertTrue(type(G) is coo_matrix)
            self.assertTupleEqual(G.shape,(0,0))
            self.assertEqual(G.nnz,0)
            self.assertTrue(type(u) is np.ndarray)
            self.assertTupleEqual(u.shape,(0,))
            self.assertTrue(type(l) is np.ndarray)
            self.assertTupleEqual(l.shape,(0,))
            self.assertEqual(constr.J_row,0)
            self.assertEqual(constr.A_row,0)
            self.assertEqual(constr.G_row,0)
            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.G_nnz,0)
            self.assertEqual(constr.num_extra_vars,0)

            num_constr = net.get_num_branches()*2*self.T
            num_Jnnz = (net.get_num_branches()*8 +
                        net.get_num_tap_changers()*2 +
                        net.get_num_phase_shifters()*2)*self.T+num_constr

            constr.analyze()
            self.assertEqual(num_Jnnz,constr.J_nnz)
            self.assertEqual(num_constr,constr.G_nnz)
            self.assertEqual(num_constr,constr.J_row)
           
            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            G = constr.G
            l = constr.l
            u = constr.u
            
            # After analyze
            self.assertEqual(constr.num_extra_vars,num_constr)
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(num_constr,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(num_constr,net.num_vars+num_constr))
            self.assertEqual(J.nnz,num_Jnnz)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,net.num_vars+num_constr))
            self.assertEqual(A.nnz,0)
            self.assertTrue(type(G) is coo_matrix)
            self.assertTupleEqual(G.shape,(num_constr,net.num_vars+num_constr))
            self.assertEqual(G.nnz,num_constr)
            self.assertTrue(np.all(G.row == np.array(range(num_constr))))
            self.assertTrue(np.all(G.col == np.array(range(net.num_vars,net.num_vars+num_constr))))
            self.assertTrue(np.all(G.row == G.col-net.num_vars))
            self.assertTrue(np.all(G.data == 1.))
            self.assertTrue(type(u) is np.ndarray)
            self.assertTupleEqual(u.shape,(num_constr,))
            self.assertTrue(type(l) is np.ndarray)
            self.assertTupleEqual(l.shape,(num_constr,))
            J_row = 0
            for t in range(net.num_periods):
                for bus in net.buses:
                    for branch in bus.branches_k:
                        #i = t*net.num_branches*2+2*branch.index
                        self.assertEqual(u[J_row],branch.ratingA)
                        self.assertEqual(u[J_row+1],branch.ratingA)
                        self.assertEqual(l[J_row],-branch.ratingA)
                        self.assertEqual(l[J_row+1],-branch.ratingA)
                        J_row += 2

            # Row info
            index = 0
            for t in range(net.num_periods):
                for bus in net.buses:
                    for branch in bus.branches_k:
                        if branch.ratingA != 0:
                            skmJ = constr.get_J_row_info_string(index)
                            smkJ = constr.get_J_row_info_string(index+1)
                            self.assertEqual(skmJ,"AC branch flow limits:branch:%d:%s:%d" %(branch.index,"km",t))
                            self.assertEqual(smkJ,"AC branch flow limits:branch:%d:%s:%d" %(branch.index,"mk",t))
                            skmG = constr.get_G_row_info_string(index)
                            smkG = constr.get_G_row_info_string(index+1)
                            self.assertEqual(skmG,"AC branch flow limits:branch:%d:%s:%d" %(branch.index,"km",t))
                            self.assertEqual(smkG,"AC branch flow limits:branch:%d:%s:%d" %(branch.index,"mk",t))
                            index += 2

            # Hessian structure
            for i in range(constr.J.shape[0]):
                H = constr.get_H_single(i)
                self.assertTupleEqual(H.shape,(net.num_vars+num_constr,net.num_vars+num_constr))
                self.assertTrue(np.all(H.row >= H.col))
            Hcomb = constr.H_combined
            H_comb_nnz = 2*(net.num_branches*10 +
                            net.get_num_tap_changers()*5+
                            net.get_num_phase_shifters()*5)*self.T
            self.assertTupleEqual(Hcomb.shape,(net.num_vars+num_constr,net.num_vars+num_constr))
            self.assertTrue(np.all(Hcomb.row >= Hcomb.col))
            self.assertEqual(Hcomb.nnz,H_comb_nnz)

            y_init = constr.init_extra_vars
            self.assertEqual(y_init.size,constr.num_extra_vars)
            self.assertEqual(y_init.size,constr.f.size)
            self.assertTrue(np.all(y_init == 0.))
            constr.eval(x0)

            y0 = np.random.randn(num_constr)

            constr.eval(x0,y0)
            self.assertEqual(num_constr,constr.J_row)
            self.assertEqual(0,constr.G_nnz)
            self.assertEqual(num_Jnnz,constr.J_nnz)
            
            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            G = constr.G
            l = constr.l
            u = constr.u
            constr.combine_H(np.ones(f.size),False)
            Hcomb = constr.H_combined

            # After eval
            self.assertTrue(not np.any(np.isinf(f)))
            self.assertTrue(not np.any(np.isnan(f)))

            # Projections
            P1 = constr.get_var_projection()
            P2 = constr.get_extra_var_projection()
            self.assertTrue(isinstance(P1,coo_matrix))
            self.assertTrue(isinstance(P2,coo_matrix))
            self.assertEqual(P1.shape[0],net.num_vars)
            self.assertEqual(P2.shape[0],constr.num_extra_vars)
            self.assertEqual(P1.shape[1],net.num_vars+constr.num_extra_vars)
            self.assertEqual(P2.shape[1],net.num_vars+constr.num_extra_vars)
            self.assertEqual(P1.nnz,net.num_vars)
            self.assertEqual(P2.nnz,constr.num_extra_vars)
            self.assertLess(np.linalg.norm(x0-P1*np.hstack((x0,y0))),1e-12)
            self.assertLess(np.linalg.norm(y0-P2*np.hstack((x0,y0))),1e-12)
            
            # Cross check current magnitudes
            J_row = 0
            for t in range(net.num_periods):
                for bus in net.buses:
                    for branch in bus.branches_k:
                        Pkm = branch.get_P_km()[t]
                        Qkm = branch.get_Q_km()[t]
                        Pmk = branch.get_P_mk()[t]
                        Qmk = branch.get_Q_mk()[t]
                        vk = branch.bus_k.v_mag[t]
                        vm = branch.bus_m.v_mag[t]
                        ikmmag = branch.get_i_km_mag(eps=param)[t]
                        imkmag = branch.get_i_mk_mag(eps=param)[t]
                        error_km = 100.*np.abs(ikmmag-f[J_row]-y0[J_row])/max([ikmmag,tol])
                        error_mk = 100.*np.abs(imkmag-f[J_row+1]-y0[J_row+1])/max([imkmag,tol])
                        self.assertLess(error_km,eps)
                        self.assertLess(error_mk,eps)
                        J_row += 2

            # Jacobian check
            pf.tests.utils.check_constraint_Jacobian(self,
                                                     constr,
                                                     x0,
                                                     y0,
                                                     NUM_TRIALS,
                                                     TOL,
                                                     EPS,
                                                     h)

            # Sigle Hessian check
            pf.tests.utils.check_constraint_single_Hessian(self,
                                                           constr,
                                                           x0,
                                                           y0,
                                                           NUM_TRIALS,
                                                           TOL,
                                                           EPS,
                                                           h)

            # Combined Hessian check 1
            h = 1e-12
            pf.tests.utils.check_constraint_combined_Hessian(self,
                                                             constr,
                                                             x0,
                                                             y0,
                                                             NUM_TRIALS,
                                                             TOL,
                                                             EPS,
                                                             h)

            # Combined Hessian check 2
            coeff = np.random.randn(constr.f.shape[0])
            constr.eval(x0,y0)
            constr.combine_H(coeff,False)
            H = constr.H_combined.copy()
            H_manual = 0
            for i in range(constr.f.size):
                Hi = constr.get_H_single(i)
                H_manual = H_manual + coeff[i]*Hi
            diff = coo_matrix(H_manual-H)
            self.assertLess(norm(diff.data)/norm(H.data),1e-12)

            # Sensitivities
            net.clear_sensitivities()
            for t in range(net.num_periods):
                for branch in net.branches:
                    self.assertEqual(branch.sens_i_mag_u_bound[t], 0.)

            mu = np.random.randn(constr.J.shape[0])
            self.assertEqual(mu.size, constr.G.shape[0])

            constr.store_sensitivities(None, np.zeros(mu.size), mu, np.zeros(mu.size))

            G_row = 0
            for t in range(net.num_periods):
                for bus in net.buses:
                    for branch in bus.branches_k:
                        if np.abs(mu[G_row]) > np.abs(mu[G_row+1]):
                            self.assertEqual(branch.sens_i_mag_u_bound[t], mu[G_row])
                        else:
                            self.assertEqual(branch.sens_i_mag_u_bound[t], mu[G_row+1])
                        G_row += 2

        # Single period
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,1)
            self.assertEqual(net.num_periods,1)
            
            net.set_flags('bus',['variable','bounded'],'any','voltage magnitude')
            net.set_flags('bus','variable','not slack','voltage angle')
            self.assertEqual(net.num_vars,2*net.num_buses-net.get_num_slack_buses())

            if len([b for b in net.branches if b.ratingA != 0.]) == 0:
                continue
            
            constr = pf.Constraint('AC branch flow limits',net)
            constr.analyze()
            self.assertGreater(constr.num_extra_vars,0)

            # Single Hessian check
            x0 = net.get_var_values()
            y0 = np.zeros(constr.num_extra_vars)
            constr.eval(x0,y0)
            for i in range(10):
                
                j = np.random.randint(0,constr.f.size)

                constr.eval(x0,y0)

                g0 = constr.J.tocsr()[j,:].toarray().flatten()
                H0lt = constr.get_H_single(j).copy()

                self.assertTrue(np.all(H0lt.row >= H0lt.col)) # lower triangular
                H0 = (H0lt + H0lt.T) - triu(H0lt)

                d = np.random.randn(net.num_vars+constr.num_extra_vars)

                x = x0 + h*d[:net.num_vars]
                y = y0 + h*d[net.num_vars:]

                constr.eval(x,y)
                
                g1 = constr.J.tocsr()[j,:].toarray().flatten()

                Hd_exact = H0*d
                Hd_approx = (g1-g0)/h
                error = 100.*norm(Hd_exact-Hd_approx)/np.maximum(norm(Hd_exact),tol)
                self.assertLessEqual(error,EPS)
                
            # Combined Hessian check
            x0 = net.get_var_values()
            y0 = np.zeros(constr.num_extra_vars)
            lam = np.random.randn(constr.f.size)
            constr.eval(x0,y0)
            constr.combine_H(lam)
            
            h = 1e-11
            F0 = np.dot(constr.f,lam)
            GradF0 = constr.J.T*lam
            HessF0lt = constr.H_combined.copy()
            self.assertTrue(np.all(HessF0lt.row >= HessF0lt.col)) # lower triangular
            HessF0 = (HessF0lt + HessF0lt.T - triu(HessF0lt))
            for i in range(10):
                
                d = np.random.randn(x0.size+y0.size)
                
                x = x0 + h*d[:x0.size]
                y = y0 + h*d[x0.size:]
                
                constr.eval(x,y)
                
                F1 = np.dot(constr.f,lam)
                GradF1 = constr.J.T*lam
                
                Jd_exact = np.dot(GradF0,d)
                Jd_approx = (F1-F0)/h
                
                Hd_exact = HessF0*d
                Hd_approx = (GradF1-GradF0)/h
                
                errorJ = 100.*norm(Jd_exact-Jd_approx)/norm(Jd_exact) 
                errorH = 100.*norm(Hd_exact-Hd_approx)/norm(Hd_exact) 
        
                self.assertLess(errorJ,EPS)
                self.assertLess(errorH,EPS)

    def test_constr_AC_FLOW_LIM_with_outages(self):

        # Constants
        h = 1e-11
        tol = 1e-2
        eps = 1.1 # %

        # Multiperiod
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)

            # Vars
            net.set_flags('bus',
                          'variable',
                          'any',
                          ['voltage magnitude','voltage angle'])
            net.set_flags('branch',
                          'variable',
                          'tap changer',
                          'tap ratio')
            net.set_flags('branch',
                          'variable',
                          'phase shifter',
                          'phase shift')
            self.assertEqual(net.num_vars,
                             (2*net.get_num_buses() +
                              net.get_num_tap_changers() +
                              net.get_num_phase_shifters())*self.T)

            x0 = net.get_var_values()
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            for branch in net.branches:
                branch.outage = True

            # Constr
            constr = pf.Constraint('AC branch flow limits',net)

            constr.analyze()
            constr.eval(x0)

            self.assertEqual(constr.f.size, 0)
            self.assertTupleEqual(constr.J.shape, (0, net.num_vars))
            self.assertEqual(constr.l.size, 0)
            self.assertEqual(constr.u.size, 0)
            self.assertTupleEqual(constr.G.shape, (0, net.num_vars))

            # Jacobian check
            pf.tests.utils.check_constraint_Jacobian(self,
                                                     constr,
                                                     x0,
                                                     np.zeros(0),
                                                     NUM_TRIALS,
                                                     TOL,
                                                     EPS,
                                                     h)

            # Sigle Hessian check
            pf.tests.utils.check_constraint_single_Hessian(self,
                                                           constr,
                                                           x0,
                                                           np.zeros(0),
                                                           NUM_TRIALS,
                                                           TOL,
                                                           EPS,
                                                           h)

            # Combined Hessian check 1
            h = 1e-12
            pf.tests.utils.check_constraint_combined_Hessian(self,
                                                             constr,
                                                             x0,
                                                             np.zeros(0),
                                                             NUM_TRIALS,
                                                             TOL,
                                                             EPS,
                                                             h)
                
    def test_constr_DUMMY(self):

        # Multiperiod
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)

            # Too big
            if net.num_buses > 1000:
                continue

            # Add vargens
            load_buses = net.get_load_buses()
            net.add_var_generators_from_parameters(load_buses,80.,50.,30.,5,0.05)
            self.assertGreater(net.num_var_generators,0)
            self.assertEqual(net.num_var_generators,len([b for b in net.buses if b.loads]))
            for b in net.buses:
                if b.loads:
                    self.assertGreater(len(b.var_generators),0)
                    for vargen in b.var_generators:
                        self.assertEqual(vargen.bus,b)

            # batteries
            for bat in net.batteries:
                if bat.index % 2 == 0:
                    bat.P *= -1.

            # Variables
            net.set_flags('bus',
                          'variable',
                          'not slack',
                          'voltage angle')
            net.set_flags('generator',
                          'variable',
                          'any',
                          'active power')
            net.set_flags('load',
                          'variable',
                          'any',
                          'active power')
            net.set_flags('variable generator',
                          'variable',
                          'any',
                          'active power')
            net.set_flags('branch',
                          'variable',
                          'phase shifter',
                          'phase shift')
            net.set_flags('battery',
                          'variable',
                          'any',
                          'charging power')
            self.assertEqual(net.num_vars,
                             (net.num_buses-net.get_num_slack_buses() +
                              net.num_generators +
                              net.num_loads +
                              net.num_var_generators +
                              net.get_num_phase_shifters()+
                              2*net.num_batteries)*net.num_periods)

            x0 = net.get_var_values()
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            # Ref constraint
            constrREF = pf.Constraint('DC power balance',net)
            self.assertEqual(constrREF.name,'DC power balance')

            # Dummy constraint
            constr = pf.constraints.DummyDCPF(net)
            self.assertEqual(constr.name,'dummy DC power balance')
            
            self.assertEqual(constr.A_row,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.A_row,constrREF.A_row)
            self.assertEqual(constr.A_nnz,constrREF.A_nnz)
            
            self.assertEqual(constr.b.size,0)
            self.assertEqual(constr.A.shape[0],0)
            self.assertEqual(constr.A.shape[1],0)
            self.assertEqual(constr.A.nnz,0)
           
            constrREF.analyze()
            constr.analyze()

            self.assertEqual(constr.A_row,net.num_buses*self.T)
            self.assertGreater(constr.A_nnz,0)
            self.assertEqual(constr.A_row,constrREF.A_row)
            self.assertEqual(constr.A_nnz,constrREF.A_nnz)
            
            self.assertTrue(np.all(constr.b == constrREF.b))
            self.assertTrue(np.all(constr.A.row == constrREF.A.row))
            self.assertTrue(np.all(constr.A.col == constrREF.A.col))
            self.assertTrue(np.all(constr.A.data == constrREF.A.data))
            
            self.assertTupleEqual(constr.l.shape,(0,))
            self.assertTupleEqual(constr.u.shape,(0,))
            self.assertTupleEqual(constr.f.shape,(0,))
            self.assertTupleEqual(constr.G.shape,(0,net.num_vars))
            self.assertTupleEqual(constr.J.shape,(0,net.num_vars))

            constrREF.eval(net.get_var_values())
            constr.eval(net.get_var_values())

            self.assertTrue(np.all(constr.b == constrREF.b))
            self.assertTrue(np.all(constr.A.row == constrREF.A.row))
            self.assertTrue(np.all(constr.A.col == constrREF.A.col))
            self.assertTrue(np.all(constr.A.data == constrREF.A.data))

    def test_constr_BAT_DYN(self):

        # Multi period
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,5)
            self.assertEqual(net.num_periods,5)
            self.assertEqual(net.num_vars,0)

            # Add battries
            gen_buses = net.get_generator_buses()
            net.add_batteries_from_parameters(gen_buses,20.,40.,0.8,0.7)
            self.assertEqual(net.num_batteries,len(gen_buses))
            self.assertGreater(net.num_batteries,0)

            # Vars
            net.set_flags('battery',
                          'variable',
                          'any',
                          ['charging power','energy level'])
            self.assertEqual(net.num_vars,5*3*net.num_batteries)

            x0 = net.get_var_values()
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            # Constraint
            constr = pf.Constraint('battery dynamics',net)
            self.assertEqual(constr.name,'battery dynamics')

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            l = constr.l
            G = constr.G
            u = constr.u

            # Before
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,0))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,0))
            self.assertEqual(A.nnz,0)
            self.assertTrue(type(l) is np.ndarray)
            self.assertTupleEqual(l.shape,(0,))
            self.assertTrue(type(u) is np.ndarray)
            self.assertTupleEqual(u.shape,(0,))
            self.assertTrue(type(G) is coo_matrix)
            self.assertTupleEqual(G.shape,(0,0))
            self.assertEqual(G.nnz,0)
            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.G_nnz,0)

            constr.analyze()
            self.assertEqual(constr.A_row,(5+1)*net.num_batteries)
            self.assertEqual(constr.A_nnz,5*4*net.num_batteries)
            self.assertEqual(constr.G_nnz,0)
            constr.eval(x0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.G_nnz,0)

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            l = constr.l
            G = constr.G
            u = constr.u

            # After
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(6*net.num_batteries,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,net.num_vars))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(6*net.num_batteries,net.num_vars))
            self.assertEqual(A.nnz,5*4*net.num_batteries)
            self.assertTrue(type(l) is np.ndarray)
            self.assertTupleEqual(l.shape,(0,))
            self.assertTrue(type(u) is np.ndarray)
            self.assertTupleEqual(u.shape,(0,))
            self.assertTrue(type(G) is coo_matrix)
            self.assertTupleEqual(G.shape,(0,net.num_vars))
            self.assertEqual(G.nnz,0)
            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.G_nnz,0)
            
            for t in range(5):
                for bat in net.batteries:
                    self.assertTrue(bat.has_flags('variable',['charging power','energy level']))
                    
                    aPc = np.where(A.col == bat.index_Pc[t])[0]
                    aPd = np.where(A.col == bat.index_Pd[t])[0]
                    aE = np.where(A.col == bat.index_E[t])[0]
                    if t < 5-1:
                        aEE = np.where(A.col == bat.index_E[t+1])[0]
                    
                    self.assertEqual(aPc.size,1)
                    self.assertEqual(aPd.size,1)
                   
                    eq_row = A.row[aPc[0]]
                    self.assertEqual(eq_row,A.row[aPd[0]])
                    self.assertEqual(A.data[aPc[0]],-bat.eta_c)
                    self.assertEqual(A.data[aPd[0]],1./bat.eta_d)
 
                    if t == 0:
                        self.assertEqual(aE.size,2)
                        
                        # init eq
                        j = aE[0] 
                        self.assertEqual(A.data[j],1.)
                        self.assertEqual(b[A.row[j]],bat.E_init)
                        self.assertEqual(np.where(A.row == A.row[j])[0].size,1)
                        
                        # update eq E_{t+1} - E_t - eta_c Pc_t + (1/eta_d) Pd_t = 0
                        j = aE[1]
                        self.assertEqual(A.data[j],-1.)
                        self.assertEqual(b[A.row[j]],0.)
                        self.assertEqual(np.where(A.row == A.row[j])[0].size,4)
                        self.assertEqual(A.row[j],eq_row)
                        self.assertEqual(A.row[j],A.row[aEE[0]])
                        
                    elif t < 5-1:
                        self.assertEqual(aE.size,2)

                        # update eq E_t - E_{t-1} - eta_c Pc_{t-1} + (1/eta_d) Pd_{t-1} = 0
                        j = aE[0]
                        self.assertEqual(A.data[j],1.)
                        self.assertEqual(b[A.row[j]],0.)
                        self.assertEqual(np.where(A.row == A.row[j])[0].size,4)
                        self.assertNotEqual(A.row[j],eq_row)
                        self.assertNotEqual(A.row[j],A.row[aEE[0]])

                        # update eq E_{t+1} - E_t - eta_c Pc_t + (1/eta_d) Pd_t = 0
                        j = aE[1]
                        self.assertEqual(A.data[j],-1.)
                        self.assertEqual(b[A.row[j]],0.)
                        self.assertEqual(np.where(A.row == A.row[j])[0].size,4)
                        self.assertEqual(A.row[j],eq_row)
                        self.assertEqual(A.row[j],A.row[aEE[0]])

                    else:
                        self.assertEqual(aE.size,2)

                        # update eq E_t - E_{t-1} - eta_c Pc_{t-1} + (1/eta_d) Pd_{t-1} = 0
                        j = aE[0]
                        self.assertEqual(A.data[j],1.)
                        self.assertEqual(b[A.row[j]],0.)
                        self.assertEqual(np.where(A.row == A.row[j])[0].size,4)
                        self.assertNotEqual(A.row[j],eq_row)

                        # update eq - E_t - eta_c Pc_t + (1/eta_d) Pd_t = -E_final
                        j = aE[1]
                        self.assertEqual(A.data[j],-1.)
                        self.assertEqual(b[A.row[j]],-bat.E_final)
                        self.assertEqual(np.where(A.row == A.row[j])[0].size,3)
                        self.assertEqual(A.row[j],eq_row)

    def test_constr_BAT_DYN_with_outages(self):

        # Multi period
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,5)

            # Add battries
            gen_buses = net.get_generator_buses()
            net.add_batteries_from_parameters(gen_buses,20.,40.,0.8,0.7)
            self.assertEqual(net.num_batteries,len(gen_buses))
            self.assertGreater(net.num_batteries,0)

            # Vars
            net.set_flags('battery',
                          'variable',
                          'any',
                          ['charging power','energy level'])
            self.assertEqual(net.num_vars,5*3*net.num_batteries)

            x0 = net.get_var_values()
            
            # Constraint
            constr0 = pf.Constraint('battery dynamics',net)
            constr0.analyze()

            for branch in net.branches:
                branch.outage = True
            for gen in net.generators:
                gen.outage = True

            constr1 = pf.Constraint('battery dynamics',net)
            constr1.analyze()

            self.assertEqual((constr1.A-constr0.A).tocoo().nnz, 0)
            self.assertEqual((constr1.G-constr0.G).tocoo().nnz, 0)
            self.assertLess(norm(constr1.b-constr0.b), 1e-8)
            self.assertLess(norm(constr1.l-constr0.u), 1e-8)
            self.assertLess(norm(constr1.l-constr0.u), 1e-8)
                        
    def test_constr_LOAD_PF(self):

        # Multi period
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)
            self.assertEqual(net.num_vars,0)

            # Powers
            for load in net.loads:
                load.P = np.random.rand(net.num_periods)
                self.assertTrue(np.all(load.P > 0))

            # Target power factors
            for load in net.loads:
                load.target_power_factor = np.random.rand()
                self.assertTrue(0 < load.target_power_factor < 1.)

            # Vars
            net.set_flags('load',
                          'variable',
                          'any',
                          ['active power','reactive power'])
            self.assertEqual(net.num_vars,2*net.num_loads*self.T)

            x0 = net.get_var_values()
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            # Constraint
            constr = pf.Constraint('load constant power factor',net)
            self.assertEqual(constr.name,'load constant power factor')

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            l = constr.l
            G = constr.G
            u = constr.u

            # Before
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,0))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,0))
            self.assertEqual(A.nnz,0)
            self.assertTrue(type(l) is np.ndarray)
            self.assertTupleEqual(l.shape,(0,))
            self.assertTrue(type(u) is np.ndarray)
            self.assertTupleEqual(u.shape,(0,))
            self.assertTrue(type(G) is coo_matrix)
            self.assertTupleEqual(G.shape,(0,0))
            self.assertEqual(G.nnz,0)
            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.G_nnz,0)

            constr.analyze()
            self.assertEqual(constr.A_nnz,2*net.num_loads*self.T)
            self.assertEqual(constr.A_row,net.num_loads*self.T)
            constr.eval(x0)
            self.assertEqual(constr.A_nnz,0)

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            l = constr.l
            G = constr.G
            u = constr.u

            # After
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(net.num_loads*self.T,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,net.num_vars))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(net.num_loads*self.T,net.num_vars))
            self.assertEqual(A.nnz,2*net.num_loads*self.T)
            self.assertTrue(type(l) is np.ndarray)
            self.assertTupleEqual(l.shape,(0,))
            self.assertTrue(type(u) is np.ndarray)
            self.assertTupleEqual(u.shape,(0,))
            self.assertTrue(type(G) is coo_matrix)
            self.assertTupleEqual(G.shape,(0,net.num_vars))
            self.assertEqual(G.nnz,0)
            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.G_nnz,0)

            for load in net.loads:
                for t in range(net.num_periods):
                    indices = np.where(A.col == load.index_P[t])[0]
                    self.assertEqual(indices.size,1)
                    row = A.row[indices[0]]
                    indices = np.where(A.row == row)[0]
                    self.assertEqual(indices.size,2)
                    for i in indices:
                        if A.col[i] == load.index_P[t]:
                            gamma = load.target_power_factor
                            factor = np.sqrt((1.-gamma**2.)/(gamma**2.))
                            load.Q[t] = np.abs(load.P[t])*factor*(1. if load.Q[t] >= 0 else -1.)
                            self.assertLess(np.abs(gamma-load.power_factor[t]),1e-12)
                            if load.P[t]*load.Q[t] >= 0:
                                self.assertAlmostEqual(A.data[i],-factor)
                                self.assertLess(np.abs(-factor*load.P[t]+load.Q[t]),1e-12)
                            else:
                                self.assertAlmostEqual(A.data[i],factor)
                                self.assertLess(np.abs(factor*load.P[t]+load.Q[t]),1e-12)
                        else:
                            self.assertEqual(A.col[i],load.index_Q[t])
                            self.assertEqual(A.data[i],1.)
            
            x = net.get_var_values()
            self.assertLess(np.linalg.norm(constr.A*x-constr.b),1e-10)
            
            for load in net.loads:
                for t in range(net.num_periods):
                    self.assertAlmostEqual(load.power_factor[t],load.target_power_factor)

    def test_constr_LOAD_PF_with_outages(self):

        # Multi period
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,self.T)


            # Vars
            net.set_flags('load',
                          'variable',
                          'any',
                          ['active power','reactive power'])
            self.assertEqual(net.num_vars,2*net.num_loads*self.T)

            # Constraint
            constr0 = pf.Constraint('load constant power factor',net)
            constr0.analyze()
            
            for branch in net.branches:
                branch.outage = True
            for gen in net.generators:
                gen.outage = True

            constr1 = pf.Constraint('load constant power factor',net)
            constr1.analyze()

            self.assertEqual((constr1.A-constr0.A).tocoo().nnz, 0)
            self.assertEqual((constr1.G-constr0.G).tocoo().nnz, 0)
            self.assertLess(norm(constr1.b-constr0.b), 1e-8)
            self.assertLess(norm(constr1.l-constr0.u), 1e-8)
            self.assertLess(norm(constr1.l-constr0.u), 1e-8)
                    
    def test_constr_AC_LIN_FLOW_LIM(self):

        # Multiperiod
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case,self.T)
            self.assertEqual(net.num_periods,self.T)

            # Vars
            net.set_flags('bus',
                          'variable',
                          'any',
                          'voltage magnitude')
            net.set_flags('bus',
                          'variable',
                          'not slack',
                          'voltage angle')
            net.set_flags('branch',
                          'variable',
                          'tap changer',
                          'tap ratio')
            net.set_flags('branch',
                          'variable',
                          'phase shifter',
                          'phase shift')
            self.assertEqual(net.num_vars,
                             (2*net.get_num_buses()-net.get_num_slack_buses() +
                              net.get_num_tap_changers() +
                              net.get_num_phase_shifters())*self.T)

            # Zero ratings
            for br in net.branches:
                if br.ratingA == 0.:
                    br.ratingA = 100.

            x0 = net.get_var_values()
            self.assertTrue(type(x0) is np.ndarray)
            self.assertTupleEqual(x0.shape,(net.num_vars,))

            # Constraint
            constr = pf.Constraint('linearized AC branch flow limits',net)
            self.assertEqual(constr.name,'linearized AC branch flow limits')

            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            G = constr.G
            l = constr.l
            u = constr.u

            # Before
            self.assertEqual(constr.num_extra_vars,0)
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,0))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,0))
            self.assertEqual(A.nnz,0)
            self.assertTrue(type(G) is coo_matrix)
            self.assertTupleEqual(G.shape,(0,0))
            self.assertEqual(G.nnz,0)
            self.assertTrue(type(u) is np.ndarray)
            self.assertTupleEqual(u.shape,(0,))
            self.assertTrue(type(l) is np.ndarray)
            self.assertTupleEqual(l.shape,(0,))
            self.assertEqual(constr.J_row,0)
            self.assertEqual(constr.A_row,0)
            self.assertEqual(constr.G_row,0)
            self.assertEqual(constr.J_nnz,0)
            self.assertEqual(constr.A_nnz,0)
            self.assertEqual(constr.G_nnz,0)
            self.assertEqual(constr.num_extra_vars,0)
      
            # Tap ratios and phase shifts
            if net.get_num_tap_changers()+net.get_num_phase_shifters() > 0:
                self.assertRaises(pf.ConstraintError,constr.analyze)
                constr.clear_error()
                continue

            # No voltage magnitude bounds
            self.assertRaises(pf.ConstraintError,constr.analyze)
            self.assertRaisesRegexp(pf.ConstraintError,
                                    "AC_LIN_FLOW_LIM constraint requires variable voltage magnitudes to be bounded",
                                    constr.analyze)
            constr.clear_error()

            net.set_flags('bus',
                          'bounded',
                          'any',
                          'voltage magnitude')

            self.assertEqual(net.num_bounded,net.num_buses*self.T)
        
            constr.analyze()

            self.assertGreaterEqual(constr.G_nnz,constr.G_row)
           
            f = constr.f
            J = constr.J
            A = constr.A
            b = constr.b
            G = constr.G
            l = constr.l
            u = constr.u
            
            # After analyze
            self.assertEqual(constr.num_extra_vars,0)
            self.assertTrue(type(f) is np.ndarray)
            self.assertTupleEqual(f.shape,(0,))
            self.assertTrue(type(b) is np.ndarray)
            self.assertTupleEqual(b.shape,(0,))
            self.assertTrue(type(J) is coo_matrix)
            self.assertTupleEqual(J.shape,(0,net.num_vars))
            self.assertEqual(J.nnz,0)
            self.assertTrue(type(A) is coo_matrix)
            self.assertTupleEqual(A.shape,(0,net.num_vars))
            self.assertEqual(A.nnz,0)
            self.assertTrue(type(G) is coo_matrix)
            self.assertTupleEqual(G.shape,(constr.G_row,net.num_vars))
            self.assertFalse(np.any(np.isnan(G.data)))
            self.assertTrue(type(u) is np.ndarray)
            self.assertTupleEqual(u.shape,(constr.G_row,))
            self.assertFalse(np.any(np.isnan(u)))
            self.assertTrue(type(l) is np.ndarray)
            self.assertTupleEqual(l.shape,(constr.G_row,))
            self.assertTrue(np.all(l == -1e8))

    def test_constr_AC_LIN_FLOW_LIM_with_outages(self):

        pass

    def test_nonlinear_constr_creation(self):
        
        # Single period
        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case)
            self.assertEqual(net.num_periods,1)

            constr = pf.Constraint("variable fixing",net)

            # J row
            self.assertEqual(constr.J_row,0)
            constr.J_row = 19
            self.assertEqual(constr.J_row,19)

            # J_nnz
            self.assertEqual(constr.J_nnz,0)
            constr.J_nnz = 17
            self.assertEqual(constr.J_nnz,17)

            # f
            f = constr.f
            self.assertEqual(f.size,0)
            a = np.random.randn(15)
            constr.set_f(a)
            self.assertEqual(constr.f.size,15)
            self.assertTrue(np.all(constr.f == a))

            # J
            J = constr.J
            self.assertTupleEqual(J.shape,(0,0))
            self.assertEqual(J.nnz,0)
            Jm = coo_matrix(np.random.randn(4,3))
            constr.set_J(Jm)
            self.assertTrue(isinstance(constr.J,coo_matrix))
            self.assertTupleEqual(constr.J.shape,Jm.shape)
            self.assertTrue(np.all(constr.J.row == Jm.row))
            self.assertEqual(constr.J.nnz,Jm.nnz)
            self.assertTrue(np.all(constr.J.col == Jm.col))
            self.assertTrue(np.all(constr.J.data == Jm.data))

            # H array
            self.assertEqual(constr.H_array_size,0)
            constr.allocate_H_array(100)
            self.assertEqual(constr.H_array_size,100)

            # H single
            H = constr.get_H_single(5)
            self.assertTrue(isinstance(H,coo_matrix))
            self.assertEqual(H.nnz,0)
            self.assertTupleEqual(H.shape,(0,0))
            A = coo_matrix(np.random.randn(5,4))
            constr.set_H_single(5,A)
            H = constr.get_H_single(5)
            self.assertTrue(isinstance(H,coo_matrix))
            self.assertTupleEqual(A.shape,H.shape)
            self.assertTrue(np.all(A.row == H.row))
            self.assertEqual(A.nnz,H.nnz)
            self.assertTrue(np.all(A.col == H.col))
            self.assertTrue(np.all(A.data == H.data))

            # H_nnz
            constr.set_H_nnz(np.zeros(50,dtype='int32'))
            H_nnz = constr.H_nnz
            self.assertTrue(isinstance(H_nnz,np.ndarray))
            self.assertEqual(H_nnz.dtype,np.dtype('int32'))
            self.assertEqual(H_nnz.size,50)
            for i in range(50):
                self.assertEqual(H_nnz[i],0)
            constr.H_nnz[10] = 2
            self.assertEqual(H_nnz[10],2)

    def test_robustness_with_outages(self):

        for case in test_cases.CASES:

            net = pf.Parser(case).parse(case, self.T)

            constraints = [pf.Constraint('variable bounds', net),
                           pf.Constraint('variable fixing', net),
                           pf.Constraint('battery dynamics', net),
                           pf.Constraint('generator active power participation', net),
                           pf.Constraint('PVPQ switching', net),
                           pf.Constraint('AC power balance', net), # nonlinear
                           pf.Constraint('DC power balance', net),
                           pf.Constraint('linearized AC power balance', net),
                           pf.Constraint('voltage set point regulation', net), # nonlinear
                           pf.Constraint('voltage regulation by transformers', net), # nonlinear
                           pf.Constraint('voltage regulation by shunts', net), # nonlinear
                           pf.Constraint('AC branch flow limits', net), # nolinear
                           pf.Constraint('DC branch flow limits', net),
                           pf.Constraint('generator ramp limits', net),
                           pf.Constraint('load constant power factor', net)]

            # Add variables
            net.set_flags('bus',
                          'variable',
                          'any',
                          ['voltage magnitude','voltage angle'])
            net.set_flags('generator',
                          'variable',
                          'any',
                          ['active power','reactive power'])
            net.set_flags('branch',
                          'variable',
                          'tap changer',
                          'tap ratio')
            net.set_flags('branch',
                          'variable',
                          'phase shifter',
                          'phase shift')
            net.set_flags('shunt',
                          'variable',
                          'switching - v',
                          'susceptance')
            net.set_flags('battery',
                          'variable',
                          'any',
                          ['charging power','energy level'])
            self.assertEqual(net.num_vars,
                             (2*net.num_buses +
                              2*net.num_generators +
                              net.get_num_tap_changers()+
                              net.get_num_phase_shifters()+
                              net.get_num_switched_v_shunts()+
                              3*net.num_batteries)*self.T)

            x0 = net.get_var_values()

            net.clear_outages()

            # Analyze without outages
            for c in constraints:
                c.analyze()

            # Eval without outages
            for c in constraints:
                self.assertEqual(c.state_tag, net.state_tag)
                c.eval(x0)

            for gen in net.generators:
                gen.outage = True
            for branch in net.branches:
                branch.outage = True

            # Eval with outages
            for c in constraints:
                self.assertNotEqual(c.state_tag, net.state_tag)
                self.assertRaises(pf.ConstraintError,
                                  c.eval,
                                  x0)

            # Analyze with outages
            for c in constraints:
                c.analyze()

            # Eval with outages
            for c in constraints:
                self.assertEqual(c.state_tag, net.state_tag)
                c.eval(x0)

            net.clear_outages()

            # Eval without outages
            for c in constraints:
                self.assertNotEqual(c.state_tag, net.state_tag)
                self.assertRaises(pf.ConstraintError,
                                  c.eval,
                                  x0)
                
    def tearDown(self):

        pass
