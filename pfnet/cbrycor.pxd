#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

cdef extern from "pfnet/brycor.h":

    ctypedef struct BrYCor
    ctypedef double REAL

    char* BRYCOR_get_name(BrYCor* b)
    int BRYCOR_get_num_values(BrYCor* b);
    int BRYCOR_get_max_num_values(BrYCor* b);
    REAL* BRYCOR_get_values(BrYCor* b);
    REAL* BRYCOR_get_corrections(BrYCor* b);
    bint BRYCOR_is_based_on_tap_ratio(BrYCor* b);
    bint BRYCOR_is_based_on_phase_shift(BrYCor* b);
