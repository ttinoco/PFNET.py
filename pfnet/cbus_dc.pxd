#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

cimport cnet

cdef extern from "pfnet/bus_dc.h":

    ctypedef struct BusDC
    ctypedef struct BranchDC
    ctypedef struct ConvCSC
    ctypedef struct ConvVSC
    ctypedef double REAL
    ctypedef char BOOL

    cdef char BUSDC_VAR_V

    cdef double BUSDC_INF_V

    cdef char BUSDC_PROP_ANY

    char BUSDC_get_flags_vars(BusDC* bus)
    char BUSDC_get_flags_fixed(BusDC* bus)
    char BUSDC_get_flags_bounded(BusDC* bus)
    char BUSDC_get_flags_sparse(BusDC* bus)

    char BUSDC_get_obj_type(void* bus)
    int BUSDC_get_num_periods(BusDC* bus)
    int BUSDC_get_index(BusDC* bus)
    int BUSDC_get_index_t(BusDC* bus, int t)
    int BUSDC_get_index_v(BusDC* bus, int t)
    int BUSDC_get_number(BusDC* bus)
    char* BUSDC_get_name(BusDC* bus)
    int BUSDC_get_num_vars(void* bus, char var, int t_start, int t_end)
    BranchDC* BUSDC_get_branch_k(BusDC* bus)
    BranchDC* BUSDC_get_branch_m(BusDC* bus)
    ConvCSC* BUSDC_get_csc_conv(BusDC* bus)
    ConvVSC* BUSDC_get_vsc_conv(BusDC* bus)

    REAL BUSDC_get_v(BusDC* bus, int t)
    REAL BUSDC_get_v_base(BusDC* bus)
    REAL BUSDC_get_P_mis(BusDC* bus, int t)

    char* BUSDC_get_json_string(BusDC* bus, char* output)
    char* BUSDC_get_var_info_string(BusDC* bus, int index)  
    bint BUSDC_is_equal(BusDC* bus, BusDC* other)
    bint BUSDC_has_flags(BusDC* bus, char flag_type, char mask)
    BusDC* BUSDC_new(int num_periods)
    BusDC* BUSDC_array_new(int size, int num_periods)
    void BUSDC_array_del(BusDC* bus_array, int size)
    void BUSDC_set_number(BusDC* bus, REAL num)
    void BUSDC_set_name(BusDC* bus, char* name)
    void BUSDC_set_v(BusDC* bus, REAL v_mag, int t)
    void BUSDC_set_v_base(BusDC* bus, REAL v_base) 

    void BUSDC_add_csc_conv(BusDC* bus, ConvCSC* conv)
    void BUSDC_del_csc_conv(BusDC* bus, ConvCSC* conv)
    void BUSDC_add_vsc_conv(BusDC* bus, ConvVSC* conv)
    void BUSDC_del_vsc_conv(BusDC* bus, ConvVSC* conv)
    void BUSDC_add_branch_k(BusDC* bus, BranchDC* branch)
    void BUSDC_del_branch_k(BusDC* bus, BranchDC* branch)
    void BUSDC_add_branch_m(BusDC* bus, BranchDC* branch)
    void BUSDC_del_branch_m(BusDC* bus, BranchDC* branch)
    void BUSDC_del_all_connections(BusDC* bus)
