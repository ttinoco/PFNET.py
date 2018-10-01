#cython: embedsignature=True

#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

cimport cshunt

# Infinite
SHUNT_INF_SUSC = cshunt.SHUNT_INF_SUSC

class ShuntError(Exception):
    """
    Shunt error exception.
    """

    pass

cdef class Shunt:
    """
    Shunt class.
    """

    cdef cshunt.Shunt* _c_ptr
    cdef bint alloc

    def __init__(self, num_periods=1, alloc=True):
        """
        Shunt class.

        Parameters
        ----------
        num_periods : int
        alloc : |TrueFalse|
        """

        pass

    def __cinit__(self, num_periods=1, alloc=True):

        if alloc:
            self._c_ptr = cshunt.SHUNT_new(num_periods)
        else:
            self._c_ptr = NULL
        self.alloc = alloc

    def __dealloc__(self):

        if self.alloc:
            cshunt.SHUNT_array_del(self._c_ptr,1)
            self._c_ptr = NULL

    def _get_c_ptr(self):

        return new_CPtr(self._c_ptr)

    def is_equal(self, other):
        """
        Determines whether shunt is equal to given shunt.

        Parameters
        ----------
        other : |Shunt|
        """

        cdef Shunt s_other

        if not isinstance(other,Shunt):
            return False

        s_other = other

        return cshunt.SHUNT_is_equal(self._c_ptr, s_other._c_ptr)

    def is_fixed(self):
        """
        Determines whether the shunt device is fixed (as opposed to switching).

        Returns
        -------
        flag : |TrueFalse|
        """

        return cshunt.SHUNT_is_fixed(self._c_ptr)

    def is_switched(self):
        """
        Determines whether the shunt is switchable.

        Returns
        -------
        flag : |TrueFalse|
        """

        return cshunt.SHUNT_is_switched(self._c_ptr)

    def is_switched_locked(self):
        """
        Determines whether the shunt is switchable but currently locked.

        Returns
        -------
        flag : |TrueFalse|
        """

        return cshunt.SHUNT_is_switched_locked(self._c_ptr)

    def is_switched_v(self):
        """
        Determines whether the shunt is switchable and regulates
        bus voltage magnitude.

        Returns
        -------
        flag : |TrueFalse|
        """

        return cshunt.SHUNT_is_switched_v(self._c_ptr)

    def is_continuous(self):
        """
        Determines whether the shunt has continuous 
        susceptance adjustments. 

        Returns
        -------
        flag : |TrueFalse|
        """

        return cshunt.SHUNT_is_continuous(self._c_ptr)

    def is_discrete(self):
        """
        Determines whether the shunt has discrete
        susceptance adjustments. 

        Returns
        -------
        flag : |TrueFalse|
        """

        return cshunt.SHUNT_is_discrete(self._c_ptr)

    def has_flags(self, flag_type, q):
        """
        Determines whether the shunt devices has flags associated with
        certain quantities set.

        Parameters
        ----------
        flag_type : string (|RefFlags|)
        q : string or list of strings (|RefShuntQuantities|)

        Returns
        -------
        flag : |TrueFalse|
        """

        q = q if isinstance(q,list) else [q]

        return cshunt.SHUNT_has_flags(self._c_ptr,
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

        cdef char* info_string = cshunt.SHUNT_get_var_info_string(self._c_ptr, index)
        if info_string:
            s = info_string.decode('UTF-8')
            free(info_string)
            return s
        else:
            raise ShuntError('index does not correspond to any variable')

    def round_b(self, t=None):
        """
        Rounds susceptance to nearest valid discrete value.

        Parameters
        ----------
        t : int (None for all)

        Returns
        -------
        num : int (number of significant changes)
        """

        if t is not None:
            return cshunt.SHUNT_round_b(self._c_ptr, t)
        else:
            num = 0
            for t in range(self.num_periods):
                num += cshunt.SHUNT_round_b(self._c_ptr, t)
            return num

    def set_as_fixed(self):
        """
        Sets shunt as fixed.
        """

        cshunt.SHUNT_set_type(self._c_ptr,cshunt.SHUNT_TYPE_FIXED)

    def set_as_switched(self):
        """
        Sets shunt as switched.
        """

        cshunt.SHUNT_set_type(self._c_ptr,cshunt.SHUNT_TYPE_SWITCHED)

    def set_as_switched_v(self):
        """
        Sets shunt as switched proviging voltage regulation.
        """

        cshunt.SHUNT_set_type(self._c_ptr,cshunt.SHUNT_TYPE_SWITCHED_V)

    def set_as_continuous(self):
        """
        Sets shunt as having continuous susceptance adjustments.
        """

        cshunt.SHUNT_set_mode(self._c_ptr,cshunt.SHUNT_MODE_CONT)

    def set_as_discrete(self):
        """
        Sets shunt as having discrete susceptance adjustments.
        """

        cshunt.SHUNT_set_mode(self._c_ptr,cshunt.SHUNT_MODE_DIS)

    def set_b_values(self, values):
        """
        Sets the block susceptance values.
        
        Parameters
        ----------
        values : |Array|
        """
        
        cdef np.ndarray[double,mode='c'] x = values
        PyArray_CLEARFLAGS(x,np.NPY_OWNDATA)
        cshunt.SHUNT_set_b_values(self._c_ptr,<cnet.REAL*>(x.data),x.size)

    property name:
        """ Shunt name (string). """
        def __get__(self):
            return cshunt.SHUNT_get_name(self._c_ptr).decode('UTF-8')
        def __set__(self,name):
            name = name.encode('UTF-8')
            cshunt.SHUNT_set_name(self._c_ptr,name)

    property num_periods:
        """ Number of time periods (int). """
        def __get__(self): return cshunt.SHUNT_get_num_periods(self._c_ptr)

    property obj_type:
        """ Object type (string). """
        def __get__(self): return obj2str[cshunt.SHUNT_get_obj_type(self._c_ptr)]

    property index:
        """ Shunt index (int). """
        def __get__(self): return cshunt.SHUNT_get_index(self._c_ptr)

    property index_b:
        """ Index of shunt susceptance variable (int or |Array|). """
        def __get__(self):
            return IntArray(cshunt.SHUNT_get_index_b_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)

    property bus:
        """ |Bus| to which the shunt devices is connected. """
        def __get__(self):
            return new_Bus(cshunt.SHUNT_get_bus(self._c_ptr))
        def __set__(self,bus):
            cdef Bus cbus
            if not isinstance(bus,Bus) and bus is not None:
                raise ShuntError('Not a Bus type object')
            cbus = bus
            cshunt.SHUNT_set_bus(self._c_ptr,cbus._c_ptr if bus is not None else NULL)

    property reg_bus:
        """ |Bus| whose voltage magnitude is regulated by this shunt device. """
        def __get__(self): 
            return new_Bus(cshunt.SHUNT_get_reg_bus(self._c_ptr))
        def __set__(self,bus):
            cdef Bus creg_bus
            if not isinstance(bus,Bus) and bus is not None:
                raise ShuntError('Not a Bus type object')
            creg_bus = bus
            cshunt.SHUNT_set_reg_bus(self._c_ptr,creg_bus._c_ptr if bus is not None else NULL)
            
    property g:
        """ Shunt conductance (p.u.) (float). """
        def __get__(self): return cshunt.SHUNT_get_g(self._c_ptr)
        def __set__(self,value): cshunt.SHUNT_set_g(self._c_ptr,value)

    property b:
        """ Shunt susceptance (p.u.) (float or |Array|). """
        def __get__(self):
            return DoubleArray(cshunt.SHUNT_get_b_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        def __set__(self, v):
            DoubleArray(cshunt.SHUNT_get_b_array(self._c_ptr), self.num_periods)[:] = v

    property b_max:
        """ Shunt susceptance upper limit (p.u.) (float). """
        def __get__(self): return cshunt.SHUNT_get_b_max(self._c_ptr)
        def __set__(self,value): cshunt.SHUNT_set_b_max(self._c_ptr,value)

    property b_min:
        """ Shunt susceptance lower limit (p.u.) (float). """
        def __get__(self): return cshunt.SHUNT_get_b_min(self._c_ptr)
        def __set__(self,value): cshunt.SHUNT_set_b_min(self._c_ptr,value)
        
    property b_values:
        """ Valid shunt susceptance values if switchable (p.u.) (|Array|). """
        def __get__(self): return DoubleArray(cshunt.SHUNT_get_b_values(self._c_ptr),
                                              cshunt.SHUNT_get_num_b_values(self._c_ptr))
        def __set__(self,x):
            self.b_values[:] = x

    property json_string:
        """ JSON string (string). """
        def __get__(self): 
            cdef char* json_string = cshunt.SHUNT_get_json_string(self._c_ptr, NULL)
            s = json_string.decode('UTF-8')
            free(json_string)
            return s

    property sens_b_u_bound:
        """ Objective function sensitivity with respect to susceptance upper bound (float or |Array|). """
        def __get__(self): return DoubleArray(cshunt.SHUNT_get_sens_b_u_bound_array(self._c_ptr),
                                              cshunt.SHUNT_get_num_periods(self._c_ptr))
        def __set__(self,x):
            self.sens_b_u_bound[:] = x

    property sens_b_l_bound:
        """ Objective function sensitivity with respect to susceptance lower bound (float or |Array|). """
        def __get__(self): return DoubleArray(cshunt.SHUNT_get_sens_b_l_bound_array(self._c_ptr),
                                              cshunt.SHUNT_get_num_periods(self._c_ptr))
        def __set__(self,x):
            self.sens_b_l_bound[:] = x

    property flags_vars:
        """ Flags associated with variable quantities (byte). """
        def __get__(self): return cshunt.SHUNT_get_flags_vars(self._c_ptr)

    property flags_fixed:
        """ Flags associated with fixed quantities (byte). """
        def __get__(self): return cshunt.SHUNT_get_flags_fixed(self._c_ptr)

    property flags_bounded:
        """ Flags associated with bounded quantities (byte). """
        def __get__(self): return cshunt.SHUNT_get_flags_bounded(self._c_ptr)

    property flags_sparse:
        """ Flags associated with sparse quantities (byte). """
        def __get__(self): return cshunt.SHUNT_get_flags_sparse(self._c_ptr)

cdef new_Shunt(cshunt.Shunt* s):
    if s is not NULL:
        shunt = Shunt(alloc=False)
        shunt._c_ptr = s
        return shunt
    else:
        raise ShuntError('no shunt data')
