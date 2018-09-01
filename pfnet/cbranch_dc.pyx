#cython: embedsignature=True

#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

cimport cbranch_dc

class BranchDCError(Exception):
    """
    DC branch error exception.
    """

    pass

cdef class BranchDC:
    """
    DC branch class.
    """

    cdef cbranch_dc.BranchDC* _c_ptr
    cdef bint alloc

    def __init__(self, num_periods=1, alloc=True):
        """
        DC branch class.

        Parameters
        ----------
        num_periods : int
        alloc : |TrueFalse|
        """

        pass

    def __cinit__(self, num_periods=1, alloc=True):

        if alloc:
            self._c_ptr = cbranch_dc.BRANCHDC_new(num_periods)
        else:
            self._c_ptr = NULL
        self.alloc = alloc

    def __dealloc__(self):

        if self.alloc:
            cbranch_dc.BRANCHDC_array_del(self._c_ptr,1)
            self._c_ptr = NULL

    def _get_c_ptr(self):

        return new_CPtr(self._c_ptr)

    def is_equal(self, other):
        """
        Determines whether DC branch is equal to given DC branch.

        Parameters
        ----------
        other : |BranchDC|
        """

        cdef BranchDC b_other

        if not isinstance(other,BranchDC):
            return False

        b_other = other

        return cbranch_dc.BRANCHDC_is_equal(self._c_ptr,b_other._c_ptr)
        
    def __richcmp__(self, other, op):
        """
        Compares two DC branches.

        Parameters
        ----------
        other : |BranchDC|
        op : comparison type

        Returns
        -------
        flag : |TrueFalse|
        """

        if op == 2:
            return self.is_equal(other)
        elif op == 3:
            return not self.is_equal(other)
        else:
            return False

    def has_flags(self, flag_type, q):
        """
        Determines whether the DC branch has the flags associated with
        specific quantities set.

        Parameters
        ----------
        flag_type : string (|RefFlags|)
        q : string or list of strings (|RefBranchDCQuantities|)

        Returns
        -------
        flag : |TrueFalse|
        """

        q = q if isinstance(q,list) else [q]

        return cbranch_dc.BRANCHDC_has_flags(self._c_ptr,
                                        str2flag[flag_type],
                                        reduce(lambda x,y: x|y,[str2q[self.obj_type][qq] for qq in q],0))

    def get_var_info_string(self, index):
        """
        Gets info string of variable associated with index.

        Parameters
        ----------
        index : int

        Returns
        -------
        info : string
        """

        cdef char* info_string = cbranch_dc.BRANCHDC_get_var_info_string(self._c_ptr, index)
        if info_string:
            s = info_string.decode('UTF-8')
            free(info_string)
            return s
        else:
            raise BranchDCError('index does not correspond to any variable')

    def get_i_km(self, var_values=None):
        """
        Gets the DC branch current at bus "k" torwards bus "m" (p.u.).
        
        Parameters
        ----------
        var_values : |Array|

        Returns
        -------
        i : float or |Array|
        """

        cdef np.ndarray[double,mode='c'] x = var_values
        cdef cvec.Vec* v = cvec.VEC_new_from_array(<cnet.REAL*>(x.data),x.size) if var_values is not None else NULL
        r = [cbranch_dc.BRANCHDC_get_i_km(self._c_ptr,v,t) for t in range(self.num_periods)]
        free(v)
        if self.num_periods == 1:
            return AttributeFloat(r[0])
        else:
            return np.array(r)
        
    def get_i_mk(self, var_values=None):
        """
        Gets the DC branch current magnitude at bus "m" torwards bus "k" (p.u.).
        
        Parameters
        ----------
        var_values : |Array|

        Returns
        -------
        i : float or |Array|
        """

        cdef np.ndarray[double,mode='c'] x = var_values
        cdef cvec.Vec* v = cvec.VEC_new_from_array(<cnet.REAL*>(x.data),x.size) if var_values is not None else NULL
        r = [cbranch_dc.BRANCHDC_get_i_mk(self._c_ptr,v,t) for t in range(self.num_periods)]
        free(v)
        if self.num_periods == 1:
            return AttributeFloat(r[0])
        else:
            return np.array(r)

    def get_P_km(self, var_values=None):
        """
        Gets the DC power flow at bus "k" towards bus "m" (p.u.)

        Parameters
        ----------
        var_values : |Array|

        Returns
        -------
        P_km : float or |Array|
        """

        cdef np.ndarray[double,mode='c'] x = var_values
        cdef cvec.Vec* v = cvec.VEC_new_from_array(<cnet.REAL*>(x.data),x.size) if var_values is not None else NULL
        r = [cbranch_dc.BRANCHDC_get_P_km(self._c_ptr,v,t) for t in range(self.num_periods)]
        free(v)
        if self.num_periods == 1:
            return AttributeFloat(r[0])
        else:
            return np.array(r)

    def get_P_mk(self, var_values=None):
        """
        Gets the DC power flow at bus "m" towards bus "k" (p.u.).

        Parameters
        ----------
        var_values : :|Array|

        Returns
        -------
        P_mk : float or |Array|
        """

        cdef np.ndarray[double,mode='c'] x = var_values
        cdef cvec.Vec* v = cvec.VEC_new_from_array(<cnet.REAL*>(x.data),x.size) if var_values is not None else NULL
        r = [cbranch_dc.BRANCHDC_get_P_mk(self._c_ptr,v,t) for t in range(self.num_periods)]
        free(v)
        if self.num_periods == 1:
            return AttributeFloat(r[0])
        else:
            return np.array(r)

    property name:
        """ DC branch name (string). """
        def __get__(self):
            return cbranch_dc.BRANCHDC_get_name(self._c_ptr).decode('UTF-8')
        def __set__(self,name):
            name = name.encode('UTF-8')
            cbranch_dc.BRANCHDC_set_name(self._c_ptr,name)

    property num_periods:
        """ Number of time periods (int). """
        def __get__(self): return cbranch_dc.BRANCHDC_get_num_periods(self._c_ptr)

    property obj_type:
        """ Object type (string). """
        def __get__(self): return obj2str[cbranch_dc.BRANCHDC_get_obj_type(self._c_ptr)]

    property index:
        """ DC branch index (int). """
        def __get__(self): return cbranch_dc.BRANCHDC_get_index(self._c_ptr)

    property bus_k:
        """ |BusDC| connected to the "k" side. """
        def __get__(self):
            return new_BusDC(cbranch_dc.BRANCHDC_get_bus_k(self._c_ptr))
        def __set__(self,bus): 
            cdef BusDC cbus
            if not isinstance(bus,BusDC) and bus is not None:
                raise BranchDCError('Not a BusDC type object')
            cbus = bus
            cbranch_dc.BRANCHDC_set_bus_k(self._c_ptr,cbus._c_ptr if bus is not None else NULL)

    property bus_m:
        """ |BusDC| connected to the "m" side. """
        def __get__(self):
            return new_BusDC(cbranch_dc.BRANCHDC_get_bus_m(self._c_ptr))
        def __set__(self,bus): 
            cdef BusDC cbus
            if not isinstance(bus,BusDC) and bus is not None:
                raise BranchDCError('Not a BusDC type object')
            cbus = bus
            cbranch_dc.BRANCHDC_set_bus_m(self._c_ptr,cbus._c_ptr if bus is not None else NULL)

    property r:
        """ DC branch resistance (p.u.) (float). """
        def __get__(self): return cbranch_dc.BRANCHDC_get_r(self._c_ptr)
        def __set__(self,value): cbranch_dc.BRANCHDC_set_r(self._c_ptr,value)

    property i_km:
        """ DC branch current at bus "k" towards bus "m" (p.u.) (float or |Array|). """
        def __get__(self):
            return self.get_i_km()

    property i_mk:
        """ DC branch current at bus "m" towards bus "k" (p.u.) (float or |Array|). """
        def __get__(self):
            return self.get_i_mk()

    property P_km:
        """ DC power flow at bus "k" towards bus "m" (p.u.) (float or |Array|). """
        def __get__(self):
            return self.get_P_km()

    property P_mk:
        """ DC power flow at bus "m" towards bus "k" (p.u.) (float or |Array|). """
        def __get__(self):
            return self.get_P_mk()

    property json_string:
        """ JSON string (string). """
        def __get__(self): 
            cdef char* json_string = cbranch_dc.BRANCHDC_get_json_string(self._c_ptr, NULL)
            s = json_string.decode('UTF-8')
            free(json_string)
            return s

    property flags_vars:
        """ Flags associated with variable quantities (byte). """
        def __get__(self): return cbranch_dc.BRANCHDC_get_flags_vars(self._c_ptr)

    property flags_fixed:
        """ Flags associated with fixed quantities (byte). """
        def __get__(self): return cbranch_dc.BRANCHDC_get_flags_fixed(self._c_ptr)

    property flags_bounded:
        """ Flags associated with bounded quantities (byte). """
        def __get__(self): return cbranch_dc.BRANCHDC_get_flags_bounded(self._c_ptr)

    property flags_sparse:
        """ Flags associated with sparse quantities (byte). """
        def __get__(self): return cbranch_dc.BRANCHDC_get_flags_sparse(self._c_ptr)

cdef new_BranchDC(cbranch_dc.BranchDC* b):
    if b is not NULL:
        branch = BranchDC(alloc=False)
        branch._c_ptr = b
        return branch
    else:
        return None
        
