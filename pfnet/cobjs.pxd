#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

cdef extern from "pfnet/obj_types.h":

    cdef char OBJ_ALL
    cdef char OBJ_BUS
    cdef char OBJ_GEN
    cdef char OBJ_BRANCH
    cdef char OBJ_SHUNT
    cdef char OBJ_LOAD
    cdef char OBJ_VARGEN
    cdef char OBJ_BAT
    cdef char OBJ_CONVCSC
    cdef char OBJ_CONVVSC
    cdef char OBJ_BUSDC
    cdef char OBJ_BRANCHDC
    cdef char OBJ_FACTS
    cdef char OBJ_UNKNOWN



    
