#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

cdef extern from "pfnet/conv_csc.h":

    ctypedef struct ConvCSC
    ctypedef struct BusDC
    ctypedef struct Bus
    ctypedef double REAL

    cdef char CONVCSC_VAR_P
    cdef char CONVCSC_VAR_Q
    cdef char CONVCSC_VAR_PDC
    cdef char CONVCSC_VAR_RATIO
    cdef char CONVCSC_VAR_ANGLE

    cdef char CONVCSC_MODE_DC_NC
    cdef char CONVCSC_MODE_DC_CP
    cdef char CONVCSC_MODE_DC_CC
    cdef char CONVCSC_MODE_DC_CV    

    cdef double CONVCSC_INF_P
    cdef double CONVCSC_INF_Q
    cdef double CONVCSC_INF_PDC
    cdef double CONVCSC_INF_RATIO
    cdef double CONVCSC_INF_ANGLE

    cdef char CONVCSC_TYPE_INV
    cdef char CONVCSC_TYPE_REC
       
    cdef char CONVCSC_PROP_ANY

    ConvCSC* CONVCSC_array_new(int size, int num_periods)
    void CONVCSC_array_del(ConvCSC* conv_array, int size)
  
    char CONVCSC_get_flags_vars(ConvCSC* conv)
    char CONVCSC_get_flags_fixed(ConvCSC* conv)
    char CONVCSC_get_flags_bounded(ConvCSC* conv)
    char CONVCSC_get_flags_sparse(ConvCSC* conv)

    Bus* CONVCSC_get_ac_bus(ConvCSC* conv)
    BusDC* CONVCSC_get_dc_bus(ConvCSC* conv)
    int CONVCSC_get_num_periods(ConvCSC* conv)
    char* CONVCSC_get_name(ConvCSC* conv)
    char CONVCSC_get_obj_type(void* conv)
    int CONVCSC_get_num_bridges(ConvCSC* conv)
    REAL CONVCSC_get_x_cap(ConvCSC* conv)
    REAL CONVCSC_get_x(ConvCSC* conv)
    REAL CONVCSC_get_r(ConvCSC* conv)
    REAL CONVCSC_get_angle_max(ConvCSC* conv)
    REAL CONVCSC_get_angle_min(ConvCSC* conv)
    REAL CONVCSC_get_ratio_max(ConvCSC* conv)
    REAL CONVCSC_get_ratio_min(ConvCSC* conv)
    REAL CONVCSC_get_v_base_p(ConvCSC* conv)
    REAL CONVCSC_get_v_base_s(ConvCSC* conv)

    int CONVCSC_get_index(ConvCSC* conv)
    ConvCSC* CONVCSC_get_next_ac(ConvCSC* conv)
    ConvCSC* CONVCSC_get_next_dc(ConvCSC* conv)
    char* CONVCSC_get_json_string(ConvCSC* conv, char* output)
    char* CONVCSC_get_var_info_string(ConvCSC* conv, int index)

    int* CONVCSC_get_index_P_array(ConvCSC* conv)
    int* CONVCSC_get_index_Q_array(ConvCSC* conv)
    int* CONVCSC_get_index_P_dc_array(ConvCSC* conv)
    int* CONVCSC_get_index_i_dc_array(ConvCSC* conv)
    int* CONVCSC_get_index_ratio_array(ConvCSC* conv)
    int* CONVCSC_get_index_angle_array(ConvCSC* conv)

    REAL* CONVCSC_get_angle_array(ConvCSC* conv)
    REAL* CONVCSC_get_P_array(ConvCSC* conv)
    REAL* CONVCSC_get_P_dc_array(ConvCSC* conv)
    REAL CONVCSC_get_i_dc(ConvCSC* conv, int t)
    REAL* CONVCSC_get_P_dc_set_array(ConvCSC* conv)
    REAL* CONVCSC_get_v_dc_set_array(ConvCSC* conv)
    REAL* CONVCSC_get_i_dc_set_array(ConvCSC* conv)
    REAL* CONVCSC_get_Q_array(ConvCSC* conv)
    REAL* CONVCSC_get_ratio_array(ConvCSC* conv)

    bint CONVCSC_is_equal(ConvCSC* conv, ConvCSC* other)
    bint CONVCSC_is_inverter(ConvCSC* conv)
    bint CONVCSC_is_rectifier(ConvCSC* conv)
    bint CONVCSC_is_in_P_dc_mode(ConvCSC* conv)
    bint CONVCSC_is_in_i_dc_mode(ConvCSC* conv)
    bint CONVCSC_is_in_v_dc_mode(ConvCSC* conv)
    bint CONVCSC_has_flags(ConvCSC* conv, char flag_type, char mask)
    ConvCSC* CONVCSC_new(int num_periods)

    void CONVCSC_set_ac_bus(ConvCSC* conv, Bus* bus)
    void CONVCSC_set_dc_bus(ConvCSC* conv, BusDC* bus)
    void CONVCSC_set_name(ConvCSC* conv, char* name)
    void CONVCSC_set_angle_max(ConvCSC* conv, REAL angle_max)
    void CONVCSC_set_angle_min(ConvCSC* conv, REAL angle_min)
    void CONVCSC_set_ratio_max(ConvCSC* conv, REAL ratio_max)
    void CONVCSC_set_ratio_min(ConvCSC* conv, REAL ratio_min)
    void CONVCSC_set_num_bridges(ConvCSC* conv, int num)
    void CONVCSC_set_v_base_p(ConvCSC* conv, REAL v_base_p)
    void CONVCSC_set_v_base_s(ConvCSC* conv, REAL v_base_s)
    void CONVCSC_set_x_cap(ConvCSC* conv, REAL x_cap)
    void CONVCSC_set_x(ConvCSC* conv, REAL x)
    void CONVCSC_set_r(ConvCSC* conv, REAL r)
    void CONVCSC_set_type(ConvCSC* conv, char type)
    void CONVCSC_set_mode_dc(ConvCSC* conv, char mode)  

