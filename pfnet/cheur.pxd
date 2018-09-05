#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

cdef extern from "pfnet/heur.h":

    ctypedef struct Heur
    ctypedef struct Net
    ctypedef struct Vec
    ctypedef struct Constr
    ctypedef double REAL

    void HEUR_del(Heur* h)
    char* HEUR_get_name(Heur* h)
    Net* HEUR_get_network(Heur* h)
    void HEUR_apply(Heur* h, Constr** cptrs, int cnum, Vec* var_values)
    void HEUR_clear_error(Heur * h)
    bint HEUR_has_error(Heur* h)
    char* HEUR_get_error_string(Heur* h)
    Heur* HEUR_get_next(Heur* h)

    Heur* HEUR_PVPQ_SWITCHING_new(Net* net)
    Heur* HEUR_REG_PF_SWITCH_new(Net* net)
