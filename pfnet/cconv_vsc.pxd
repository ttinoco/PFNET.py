#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

cdef extern from "pfnet/conv_vsc.h":

    ctypedef struct ConvVSC
    ctypedef struct BusDC
    ctypedef struct Bus
    
    ctypedef double REAL

    cdef char CONVVSC_VAR_P
    cdef char CONVVSC_VAR_Q
    cdef char CONVVSC_VAR_PDC

    cdef double CONVVSC_INF_P
    cdef double CONVVSC_INF_Q
    cdef double CONVVSC_INF_PDC

    cdef char CONVVSC_MODE_AC_NC
    cdef char CONVVSC_MODE_AC_CV
    cdef char CONVVSC_MODE_AC_CF

    cdef char CONVVSC_MODE_DC_NC
    cdef char CONVVSC_MODE_DC_CV
    cdef char CONVVSC_MODE_DC_CP
       
    cdef char CONVVSC_PROP_ANY

    ConvVSC* CONVVSC_array_new(int size, int num_periods)
    void CONVVSC_array_del(ConvVSC* conv_array, int size)

    Bus* CONVVSC_get_ac_bus(ConvVSC* conv)
    BusDC* CONVVSC_get_dc_bus(ConvVSC* conv)
    Bus* CONVVSC_get_reg_bus(ConvVSC* conv)
    char CONVVSC_get_flags_vars(ConvVSC* bus)
    char CONVVSC_get_flags_fixed(ConvVSC* bus)
    char CONVVSC_get_flags_bounded(ConvVSC* bus)
    char CONVVSC_get_flags_sparse(ConvVSC* bus)

    int CONVVSC_get_index(ConvVSC* conv)

    int* CONVVSC_get_index_P_array(ConvVSC* conv)
    int* CONVVSC_get_index_Q_array(ConvVSC* conv)
    int* CONVVSC_get_index_P_dc_array(ConvVSC* conv)
    int* CONVVSC_get_index_i_dc_array(ConvVSC* conv)
     
    char* CONVVSC_get_json_string(ConvVSC* conv, char* output)
    char* CONVVSC_get_name(ConvVSC* conv)
    ConvVSC* CONVVSC_get_next_ac(ConvVSC* conv)
    ConvVSC* CONVVSC_get_next_dc(ConvVSC* conv)
    ConvVSC* CONVVSC_get_reg_next(ConvVSC* conv)
    int CONVVSC_get_num_periods(ConvVSC* conv)
    char CONVVSC_get_obj_type(void* conv)
    REAL CONVVSC_get_i_dc(ConvVSC* conv, int t)

    REAL* CONVVSC_get_P_array(ConvVSC* conv)
    REAL* CONVVSC_get_P_dc_array(ConvVSC* conv)
    REAL* CONVVSC_get_Q_array(ConvVSC* conv)
    REAL* CONVVSC_get_v_dc_set_array(ConvVSC* conv)
    REAL* CONVVSC_get_P_dc_set_array(ConvVSC* conv)

    REAL CONVVSC_get_P_max(ConvVSC* conv)
    REAL CONVVSC_get_P_min(ConvVSC* conv)
    REAL CONVVSC_get_Q_max(ConvVSC* conv)
    REAL CONVVSC_get_Q_min(ConvVSC* conv)
    REAL CONVVSC_get_Q_par(ConvVSC* conv)
    REAL CONVVSC_get_loss_coeff_A(ConvVSC* conv)
    REAL CONVVSC_get_loss_coeff_B(ConvVSC* conv)
    REAL CONVVSC_get_target_power_factor(ConvVSC* conv)
    char* CONVVSC_get_var_info_string(ConvVSC* conv, int index)

    bint CONVVSC_has_flags(void* vconv, char flag_type, unsigned char mask)

    bint CONVVSC_is_equal(ConvVSC* conv, ConvVSC* other)
    bint CONVVSC_is_in_f_ac_mode(ConvVSC* conv)
    bint CONVVSC_is_in_v_ac_mode(ConvVSC* conv)
    bint CONVVSC_is_in_v_dc_mode(ConvVSC* conv)
    bint CONVVSC_is_in_P_dc_mode(ConvVSC* conv)

    ConvVSC* CONVVSC_new(int num_periods)

    void CONVVSC_set_ac_bus(ConvVSC* conv, Bus* bus)
    void CONVVSC_set_dc_bus(ConvVSC* conv, BusDC* bus)
    void CONVVSC_set_reg_bus(ConvVSC* conv, Bus* reg_bus)
    void CONVVSC_set_name(ConvVSC* conv, char* name)
   
    void CONVVSC_set_P_max(ConvVSC* conv, REAL P_max)
    void CONVVSC_set_P_min(ConvVSC* conv, REAL P_min)
    void CONVVSC_set_Q_max(ConvVSC* conv, REAL Q_max)
    void CONVVSC_set_Q_min(ConvVSC* conv, REAL Q_min)
    void CONVVSC_set_Q_par(ConvVSC* conv, REAL Q_par)
    void CONVVSC_set_loss_coeff_A(ConvVSC* conv, REAL A)
    void CONVVSC_set_loss_coeff_B(ConvVSC* conv, REAL B)
    void CONVVSC_set_target_power_factor(ConvVSC* conv, REAL pf)
    void CONVVSC_set_mode_ac(ConvVSC* conv, char mode)
    void CONVVSC_set_mode_dc(ConvVSC* conv, char mode) 
    
