#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

cimport cvec

cdef extern from "pfnet/facts.h":

    ctypedef struct Facts
    ctypedef struct Bus
    ctypedef double REAL
    ctypedef char BOOL

    cdef char FACTS_VAR_VMAG_S
    cdef char FACTS_VAR_VANG_S
    cdef char FACTS_VAR_P
    cdef char FACTS_VAR_Q

    cdef double FACTS_INF_VMAG_S
    cdef double FACTS_INF_VANG_S
    cdef double FACTS_INF_P
    cdef double FACTS_INF_Q

    cdef char FACTS_SERIES_MODE_DISABLED
    cdef char FACTS_SERIES_MODE_NORMAL
    cdef char FACTS_SERIES_MODE_BYPASS
    cdef char FACTS_SERIES_MODE_CZ
    cdef char FACTS_SERIES_MODE_CV

    cdef char FACTS_PROP_ANY

    void FACTS_array_del(Facts* facts_array, int size)
    Facts* FACTS_array_new(int size, int num_periods)

    char FACTS_get_flags_vars(Facts* facts)
    char FACTS_get_flags_fixed(Facts* facts)
    char FACTS_get_flags_bounded(Facts* facts)
    char FACTS_get_flags_sparse(Facts* facts)

    char* FACTS_get_name(Facts* facts)
    int FACTS_get_num_periods(Facts* facts)
    char FACTS_get_obj_type(void* facts)
    Bus* FACTS_get_reg_bus(Facts* facts)
    Bus* FACTS_get_bus_k(Facts* facts)
    Bus* FACTS_get_bus_m(Facts* facts)
    int FACTS_get_index(Facts* facts)

    char FACTS_get_mode_s(Facts* facts)
    REAL FACTS_get_Q_par(Facts* facts)
    REAL FACTS_get_Q_max_s(Facts* facts)
    REAL FACTS_get_Q_max_sh(Facts* facts)
    REAL FACTS_get_Q_min_s(Facts* facts)
    REAL FACTS_get_Q_min_sh(Facts* facts)
    REAL FACTS_get_i_max_s(Facts* facts)
    REAL FACTS_get_i_max_sh(Facts* facts)
    REAL FACTS_get_P_max_dc(Facts* facts)
    REAL FACTS_get_v_min_m(Facts* facts)
    REAL FACTS_get_v_max_m(Facts* facts)
    REAL FACTS_get_v_max_s(Facts* facts)
    REAL FACTS_get_g(Facts* facts)
    REAL FACTS_get_b(Facts* facts)
 
    int* FACTS_get_index_v_mag_s_array(Facts* facts)
    int* FACTS_get_index_v_ang_s_array(Facts* facts)
    int* FACTS_get_index_P_k_array(Facts* facts)
    int* FACTS_get_index_P_m_array(Facts* facts)
    int* FACTS_get_index_Q_k_array(Facts* facts)
    int* FACTS_get_index_Q_m_array(Facts* facts)
    int* FACTS_get_index_P_dc_array(Facts* facts)
    int* FACTS_get_index_Q_s_array(Facts* facts)
    int* FACTS_get_index_Q_sh_array(Facts* facts)

    REAL* FACTS_get_v_mag_s_array(Facts* facts)
    REAL* FACTS_get_v_ang_s_array(Facts* facts)
    REAL* FACTS_get_P_k_array(Facts* facts)
    REAL* FACTS_get_P_m_array(Facts* facts)
    REAL* FACTS_get_Q_k_array(Facts* facts)
    REAL* FACTS_get_Q_m_array(Facts* facts)
    REAL* FACTS_get_Q_sh_array(Facts* facts)
    REAL* FACTS_get_Q_s_array(Facts* facts)
    REAL* FACTS_get_P_dc_array(Facts* facts)
    REAL* FACTS_get_P_set_array(Facts* facts)
    REAL* FACTS_get_Q_set_array(Facts* facts)

    Facts* FACTS_get_reg_next(Facts* facts)
    Facts* FACTS_get_next_k(Facts* facts)
    Facts* FACTS_get_next_m(Facts* facts)
    char* FACTS_get_var_info_string(Facts* facts, int index)
    char* FACTS_get_json_string(Facts* facts, char* output)
    bint FACTS_has_flags(void* facts, char flag_type, unsigned char mask)

    bint FACTS_is_equal(Facts* facts, Facts* other)
    bint FACTS_is_regulator(Facts* facts)
    bint FACTS_is_STATCOM(Facts* facts)
    bint FACTS_is_SSSC(Facts* facts)
    bint FACTS_is_UPFC(Facts* facts)
    bint FACTS_is_series_link_disabled(Facts* facts)
    bint FACTS_is_series_link_bypassed(Facts* facts)
    bint FACTS_is_in_normal_series_mode(Facts* facts)
    bint FACTS_is_in_constant_series_z_mode(Facts* facts)
    bint FACTS_is_in_constant_series_v_mode(Facts* facts)

    Facts* FACTS_new(int num_periods)

    void FACTS_set_name(Facts* facts, char* name)
    void FACTS_set_reg_bus(Facts* facts, Bus* bus)
    void FACTS_set_bus_k(Facts* facts, Bus* bus)
    void FACTS_set_bus_m(Facts* facts, Bus* bus)
    void FACTS_set_v_max_s(Facts* facts, REAL v_max)
    void FACTS_set_g(Facts* facts, REAL g)
    void FACTS_set_b(Facts* facts, REAL b)
    void FACTS_set_mode_s(Facts* facts, char mode)
    void FACTS_set_Q_par(Facts* facts, REAL Q_par)
    void FACTS_set_Q_max_s(Facts* facts, REAL Q_max)
    void FACTS_set_Q_max_sh(Facts* facts, REAL Q_max)
    void FACTS_set_Q_min_s(Facts* facts, REAL Q_min)
    void FACTS_set_Q_min_sh(Facts* facts, REAL Q_min)
    void FACTS_set_i_max_s(Facts* facts, REAL i_max)
    void FACTS_set_i_max_sh(Facts* facts, REAL i_max)
    void FACTS_set_P_max_dc(Facts* facts, REAL P_max)
    void FACTS_set_v_min_m(Facts* facts, REAL v_min)
    void FACTS_set_v_max_m(Facts* facts, REAL v_max)
