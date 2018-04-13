#cython: embedsignature=True

#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

cimport cheur
cimport cconstr

class HeuristicError(Exception):
    """
    Heuristic error exception.
    """

    pass

cdef class HeuristicBase:
    """
    Base heuristic class.
    """

    cdef cheur.Heur* _c_heur
    cdef bint _alloc
    cdef Network _net

    def __init__(self):
        """
        Base heuristic class.
        """

        pass

    def __cinit__(self):

        self._c_heur = NULL
        self._alloc = False
        self._net = None

    def __dealloc__(self):
        """
        Frees heuristic C data structure.
        """

        if self._alloc:
            cheur.HEUR_del(self._c_heur)
            self._c_heur = NULL
            self._alloc = False
            self._net = None

    def clear_error(self):
        """
        Clears error flag and string.
        """

        cheur.HEUR_clear_error(self._c_heur)

    def apply(self, constraints, values):
        """
        Applies heuristic.
        
        Parameters
        ----------
        constraints : List of |Constraint| objects
        values : |Array|
        """

        cdef cconstr.Constr** ptr_array
        cdef Constraint constr
        cdef np.ndarray[double,mode='c'] x = values
        cdef cvec.Vec* v = cvec.VEC_new_from_array(<cfunc.REAL*>(x.data),x.size)
        ptr_array = <cconstr.Constr**>malloc(len(constraints)*sizeof(cconstr.Constr*))
        for i in range(len(constraints)):
            constr = constraints[i]
            ptr_array[i] = constr._c_constr
        cheur.HEUR_apply(self._c_heur,
                         ptr_array,
                         len(constraints),
                         v)
        free(v)
        free(ptr_array)
        if cheur.HEUR_has_error(self._c_heur):
            error_str = cheur.HEUR_get_error_string(self._c_heur).decode('UTF-8')
            self.clear_error()
            raise HeuristicError(error_str)
