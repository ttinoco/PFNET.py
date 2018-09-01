#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

cimport cvec

cdef extern from "pfnet/branch.h":

    ctypedef struct BranchDC
    ctypedef struct BusDC
    ctypedef double REAL
    ctypedef char BOOL

    cdef char BRANCHDC_PROP_ANY

    char BRANCHDC_get_flags_vars(BranchDC* br)
    char BRANCHDC_get_flags_fixed(BranchDC* br)
    char BRANCHDC_get_flags_bounded(BranchDC* br)
    char BRANCHDC_get_flags_sparse(BranchDC* br)

    char* BRANCHDC_get_name(BranchDC* br)

    char BRANCHDC_get_obj_type(void* br)
    int BRANCHDC_get_num_periods(BranchDC* br)
    int BRANCHDC_get_index(BranchDC* br)
    BusDC* BRANCHDC_get_bus_k(BranchDC* br)
    BusDC* BRANCHDC_get_bus_m(BranchDC* br)
    REAL BRANCHDC_get_r(BranchDC* br)
    REAL BRANCHDC_get_i_km(BranchDC* br, cvec.Vec* var_values, int t)
    REAL BRANCHDC_get_i_mk(BranchDC* br, cvec.Vec* var_values, int t)
    REAL BRANCHDC_get_P_km(BranchDC* br, cvec.Vec* values, int t)
    REAL BRANCHDC_get_P_mk(BranchDC* br, cvec.Vec* values, int t)
    BranchDC* BRANCHDC_get_next_k(BranchDC* br)
    BranchDC* BRANCHDC_get_next_m(BranchDC* br)
    char* BRANCHDC_get_json_string(BranchDC* br, char* output)
    char* BRANCHDC_get_var_info_string(BranchDC* br, int index)  
    bint BRANCHDC_is_equal(BranchDC* br, BranchDC* other)
    bint BRANCHDC_has_flags(BranchDC* br, char flag_type, char mask)
    BranchDC* BRANCHDC_new(int num_periods)
    BranchDC* BRANCHDC_array_new(int size, int num_periods)
    void BRANCHDC_array_del(BranchDC* br_array, int size)
    void BRANCHDC_set_name(BranchDC* br, char* name)
    void BRANCHDC_set_bus_k(BranchDC* br, BusDC* bus_k)
    void BRANCHDC_set_bus_m(BranchDC* br, BusDC* bus_m)
    void BRANCHDC_set_r(BranchDC* br, REAL r)
