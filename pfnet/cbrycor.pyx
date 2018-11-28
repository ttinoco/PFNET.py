#cython: embedsignature=True

#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

cimport cbrycor

class BranchYCorrectionError(Exception):
    """
    Branch Y correction exception.
    """

    pass

cdef class BranchYCorrection:
    """
    Branch Y correction class.
    """

    cdef cbrycor.BrYCor* _c_ptr

    def __init__(self):
        """
        Branch Y correction class.
        """

        pass

    def __cinit__(self):

        self._c_ptr = NULL

    def is_based_on_tap_ratio(self):
        """
        Determines whether the corrections are for changes in tap ratio.

        Returns
        -------
        flag : |TrueFalse|
        """

        return cbrycor.BRYCOR_is_based_on_tap_ratio(self._c_ptr)

    def is_based_on_phase_shift(self):
        """
        Determines whether the corrections are for changes in phase shift.

        Returns
        -------
        flag : |TrueFalse|
        """

        return cbrycor.BRYCOR_is_based_on_phase_shift(self._c_ptr)

    property name:
        """ Name (string). """
        def __get__(self):
            return cbrycor.BRYCOR_get_name(self._c_ptr).decode('UTF-8')

    property num_values:
        """ Number of correction values (int). """
        def __get__(self):
            return cbrycor.BRYCOR_get_num_values(self._c_ptr)

    property max_num_values:
        """ Maximum number of correction values (int). """
        def __get__(self):
            return cbrycor.BRYCOR_get_max_num_values(self._c_ptr)

    property values:
        """ Phase shift or tap ratio values (|Array|). """
        def __get__(self): return DoubleArray(cbrycor.BRYCOR_get_values(self._c_ptr),
                                              cbrycor.BRYCOR_get_num_values(self._c_ptr))

    property corrections:
        """ Branch addmittance scaling factors (|Array|). """
        def __get__(self): return DoubleArray(cbrycor.BRYCOR_get_corrections(self._c_ptr),
                                              cbrycor.BRYCOR_get_num_values(self._c_ptr))

cdef new_BranchYCorrection(cbrycor.BrYCor* ptr):
    if ptr is not NULL:
        y_cor = BranchYCorrection()
        y_cor._c_ptr = ptr
        return y_cor
    else:
        raise BranchYCorrectionError('no branch Y correction data')
