#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

cdef extern from "pfnet/load.h":

    ctypedef struct Load
    ctypedef struct Bus
    ctypedef double REAL

    cdef char LOAD_VAR_P
    cdef char LOAD_VAR_Q

    cdef double LOAD_INF_P
    cdef double LOAD_INF_Q

    cdef double LOAD_MIN_TARGET_PF
       
    cdef char LOAD_PROP_ANY
    cdef char LOAD_PROP_P_ADJUST
    cdef char LOAD_PROP_VDEP

    char LOAD_get_flags_vars(Load* load)
    char LOAD_get_flags_fixed(Load* load)
    char LOAD_get_flags_bounded(Load* load)
    char LOAD_get_flags_sparse(Load* load)
    char* LOAD_get_name(Load* load)
    REAL LOAD_get_power_factor(Load* load, int t)
    REAL LOAD_get_target_power_factor(Load* load)
    REAL* LOAD_get_sens_P_u_bound_array(Load* load)
    REAL* LOAD_get_sens_P_l_bound_array(Load* load)  
    REAL LOAD_get_P_util(Load* load, int t)
    REAL LOAD_get_util_coeff_Q0(Load* load)
    REAL LOAD_get_util_coeff_Q1(Load* load)
    REAL LOAD_get_util_coeff_Q2(Load* load)
    char LOAD_get_obj_type(void* load)
    int LOAD_get_num_periods(Load* load)
    int LOAD_get_index(Load* load)
    int* LOAD_get_index_P_array(Load* load)
    int* LOAD_get_index_Q_array(Load* load)
    Bus* LOAD_get_bus(Load* load)
    REAL* LOAD_get_P_array(Load* load)
    REAL* LOAD_get_P_max_array(Load* load)
    REAL* LOAD_get_P_min_array(Load* load)
    REAL* LOAD_get_Q_array(Load* load)
    REAL* LOAD_get_Q_max_array(Load* load)
    REAL* LOAD_get_Q_min_array(Load* load)
    REAL* LOAD_get_comp_cp_array(Load* load)
    REAL* LOAD_get_comp_cq_array(Load* load)
    REAL* LOAD_get_comp_ci_array(Load* load)
    REAL* LOAD_get_comp_cj_array(Load* load)
    REAL LOAD_get_comp_cg(Load* load)
    REAL LOAD_get_comp_cb(Load* load)
    Load* LOAD_get_next(Load* load)
    char* LOAD_get_json_string(Load* load, char* output)
    char* LOAD_get_var_info_string(Load* load, int index)
    bint LOAD_is_in_service(Load* load)
    bint LOAD_is_equal(Load* load, Load* other)
    bint LOAD_is_P_adjustable(Load* load)
    bint LOAD_is_vdep(Load* load)
    bint LOAD_has_flags(Load* load, char flag_type, char mask)
    Load* LOAD_new(int num_periods)
    Load* LOAD_array_new(int size, int num_periods)
    void LOAD_array_del(Load* load_array, int size)
    void LOAD_set_name(Load* load, char* name)
    void LOAD_set_in_service(Load* load, bint in_service)
    void LOAD_set_bus(Load* load, Bus* bus)
    void LOAD_set_util_coeff_Q0(Load* load, REAL c)
    void LOAD_set_util_coeff_Q1(Load* load, REAL c)
    void LOAD_set_util_coeff_Q2(Load* load, REAL c)
    void LOAD_set_target_power_factor(Load* load, REAL pf)
    void LOAD_set_comp_cg(Load* load, REAL comp)
    void LOAD_set_comp_cb(Load* load, REAL comp)
    void LOAD_update_P_components(Load* load, REAL weight_cp, REAL weight_ci, REAL weight_cg, int t)
    void LOAD_update_Q_components(Load* load, REAL weight_cq, REAL weight_cj, REAL weight_cb, int t)
