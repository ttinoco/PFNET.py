#cython: embedsignature=True

#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

cimport cconv_csc

# Infinity
CONVCSC_INF_P = cconv_csc.CONVCSC_INF_P
CONVCSC_INF_Q = cconv_csc.CONVCSC_INF_Q
CONVCSC_INF_PDC = cconv_csc.CONVCSC_INF_PDC
CONVCSC_INF_RATIO = cconv_csc.CONVCSC_INF_RATIO
CONVCSC_INF_ANGLE = cconv_csc.CONVCSC_INF_ANGLE

class ConverterCSCError(Exception):
    """
    CSC converter error exception.
    """

    pass

cdef class ConverterCSC:
    """
    CSC converter class.
    """

    cdef cconv_csc.ConvCSC* _c_ptr
    cdef bint alloc

    def __init__(self, num_periods=1, alloc=True):
        """
        CSC converter class.

        Parameters
        ----------
        num_periods : int
        alloc : |TrueFalse|
        """

        pass

    def __cinit__(self, num_periods=1, alloc=True):

        if alloc:
            self._c_ptr = cconv_csc.CONVCSC_new(num_periods)
        else:
            self._c_ptr = NULL
        self.alloc = alloc

    def __dealloc__(self):

        if self.alloc:
            cconv_csc.CONVCSC_array_del(self._c_ptr,1)
            self._c_ptr = NULL    

    def _get_c_ptr(self):

        return new_CPtr(self._c_ptr)

    def is_equal(self, other):
        """
        Determines whether CSC converter is equal to given CSC converter.

        Parameters
        ----------
        other : |ConverterCSC|

        Returns
        -------
        flag : |TrueFalse|
        """

        cdef ConverterCSC c_other

        if not isinstance(other,ConverterCSC):
            return False

        c_other = other

        return cconv_csc.CONVCSC_is_equal(self._c_ptr, c_other._c_ptr)

    def is_inverter(self):
        """
        Determines whether CSC converter is an inverter.
        
        Returns
        -------
        flag : |TrueFalse|
        """

        return cconv_csc.CONVCSC_is_inverter(self._c_ptr)

    def is_rectifier(self):
        """
        Determines whether CSC converter is a rectifier.
        
        Returns
        -------
        flag : |TrueFalse|
        """

        return cconv_csc.CONVCSC_is_rectifier(self._c_ptr)

    def is_in_P_dc_mode(self):
        """
        Determines whether CSC converter is in constant DC power mode.
        
        Returns
        -------
        flag : |TrueFalse|
        """

        return cconv_csc.CONVCSC_is_in_P_dc_mode(self._c_ptr)

    def is_in_i_dc_mode(self):
        """
        Determines whether CSC converter is in constant DC current mode.
        
        Returns
        -------
        flag : |TrueFalse|
        """

        return cconv_csc.CONVCSC_is_in_i_dc_mode(self._c_ptr)

    def is_in_v_dc_mode(self):
        """
        Determines whether CSC converter is in constant DC voltage mode.
        
        Returns
        -------
        flag : |TrueFalse|
        """

        return cconv_csc.CONVCSC_is_in_v_dc_mode(self._c_ptr)

    def set_as_inverter(self):
        """
        Sets CSC converter to be inverter.
        """

        cconv_csc.CONVCSC_set_type(self._c_ptr, cconv_csc.CONVCSC_TYPE_INV)

    def set_as_rectifier(self):
        """
        Sets CSC converter to be rectifier.
        """

        cconv_csc.CONVCSC_set_type(self._c_ptr, cconv_csc.CONVCSC_TYPE_REC)

    def set_in_P_dc_mode(self):
        """
        Sets CSC converter to be in constant DC power mode.
        """

        cconv_csc.CONVCSC_set_mode_dc(self._c_ptr, cconv_csc.CONVCSC_MODE_DC_CP)

    def set_in_v_dc_mode(self):
        """
        Sets CSC converter to be in constant DC voltage mode.
        """

        cconv_csc.CONVCSC_set_mode_dc(self._c_ptr, cconv_csc.CONVCSC_MODE_DC_CV)

    def set_in_i_dc_mode(self):
        """
        Sets CSC converter to be in constant DC current mode.
        """

        cconv_csc.CONVCSC_set_mode_dc(self._c_ptr, cconv_csc.CONVCSC_MODE_DC_CC)

    def has_flags(self, flag_type, q):
        """
        Determines whether the CSC converter has the flags associated with
        certain quantities set.

        Parameters
        ----------
        flag_type : string (|RefFlags|)
        q : string or list of strings (|RefConverterCSCQuantities|)

        Returns
        -------
        flag : |TrueFalse|
        """

        q = q if isinstance(q,list) else [q]

        return cconv_csc.CONVCSC_has_flags(self._c_ptr,
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

        cdef char* info_string = cconv_csc.CONVCSC_get_var_info_string(self._c_ptr, index)
        if info_string:
            s = info_string.decode('UTF-8')
            free(info_string)
            return s
        else:
            raise ConverterCSCError('index does not correspond to any variable')
        
    property name:
        """ CSC converter name (string). """
        def __get__(self):
            return cconv_csc.CONVCSC_get_name(self._c_ptr).decode('UTF-8')
        def __set__(self,name):
            name = name.encode('UTF-8')
            cconv_csc.CONVCSC_set_name(self._c_ptr,name)

    property num_periods:
        """ Number of time periods (int). """
        def __get__(self): return cconv_csc.CONVCSC_get_num_periods(self._c_ptr)

    property num_bridges:
        """ Number of bridges in series (int). """
        def __get__(self): return cconv_csc.CONVCSC_get_num_bridges(self._c_ptr)
        def __set__(self, n): cconv_csc.CONVCSC_set_num_bridges(self._c_ptr, n)

    property x_cap:
        """ Commutating capacitor reactance as seen by each individual bridge (p.u. DC base) (float). """
        def __get__(self): return cconv_csc.CONVCSC_get_x_cap(self._c_ptr)
        def __set__(self, x): cconv_csc.CONVCSC_set_x_cap(self._c_ptr, x)

    property x:
        """ Commutating transformer reactance as seen by each individual bridge (p.u. DC base) (float). """
        def __get__(self): return cconv_csc.CONVCSC_get_x(self._c_ptr)
        def __set__(self, x): cconv_csc.CONVCSC_set_x(self. _c_ptr, x)

    property r:
        """ Commutating transformer resistance as seen by each individual bridge (p.u. DC base) (float). """
        def __get__(self): return cconv_csc.CONVCSC_get_r(self._c_ptr)
        def __set__(self, r): cconv_csc.CONVCSC_set_r(self._c_ptr, r)

    property obj_type:
        """ Object type (string). """
        def __get__(self): return obj2str[cconv_csc.CONVCSC_get_obj_type(self._c_ptr)]

    property index:
        """ CSC converter index (int). """
        def __get__(self): return cconv_csc.CONVCSC_get_index(self._c_ptr)

    property index_P:
        """ Index of active power variable (int or |Array|). """
        def __get__(self):
            return IntArray(cconv_csc.CONVCSC_get_index_P_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)

    property index_Q:
        """ Index of reactive power variable (int or |Array|). """
        def __get__(self):
            return IntArray(cconv_csc.CONVCSC_get_index_Q_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
    
    property index_P_dc:
        """ Index of DC power variable (int or |Array|). """
        def __get__(self):
            return IntArray(cconv_csc.CONVCSC_get_index_P_dc_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)

    property index_i_dc:
        """ Index of DC current variable (int or |Array|). """
        def __get__(self):
            return IntArray(cconv_csc.CONVCSC_get_index_i_dc_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        
    property index_ratio:
        """ Index of commutating transformer turns ratio variable (int or |Array|). """
        def __get__(self):
            return IntArray(cconv_csc.CONVCSC_get_index_ratio_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)

    property index_angle:
        """ Index of ignition delay angle if rectifier or extinction advance angle if inverter (int or |Array|). """
        def __get__(self):
            return IntArray(cconv_csc.CONVCSC_get_index_angle_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)

    property ac_bus:
        """ |Bus| to which CSC converter is connected. """
        def __get__(self): 
            return new_Bus(cconv_csc.CONVCSC_get_ac_bus(self._c_ptr))
        def __set__(self,bus):
            cdef Bus cbus
            if not isinstance(bus,Bus) and bus is not None:
                raise ConverterCSCError('Not a Bus type object')
            cbus = bus
            cconv_csc.CONVCSC_set_ac_bus(self._c_ptr,cbus._c_ptr if bus is not None else NULL)

    property dc_bus:
        """ |BusDC| to which CSC converter is connected. """
        def __get__(self): 
            return new_BusDC(cconv_csc.CONVCSC_get_dc_bus(self._c_ptr))
        def __set__(self,bus):
            cdef BusDC cbus
            if not isinstance(bus,BusDC) and bus is not None:
                raise ConverterCSCError('Not a BusDC type object')
            cbus = bus
            cconv_csc.CONVCSC_set_dc_bus(self._c_ptr,cbus._c_ptr if bus is not None else NULL)

    property P:
        """ Active power injection into AC bus (p.u. system base MVA) (float or |Array|). """
        def __get__(self):
            return DoubleArray(cconv_csc.CONVCSC_get_P_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        def __set__(self, v):
            DoubleArray(cconv_csc.CONVCSC_get_P_array(self._c_ptr), self.num_periods)[:] = v

    property Q:
        """ Reactive power injection into AC bus (p.u. system base MVA) (float or |Array|). """
        def __get__(self):
            return DoubleArray(cconv_csc.CONVCSC_get_Q_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        def __set__(self, v):
            DoubleArray(cconv_csc.CONVCSC_get_Q_array(self._c_ptr), self.num_periods)[:] = v

    property P_dc:
        """ DC power injection into DC bus (p.u. system base MVA) (float or |Array|). """
        def __get__(self):
            return DoubleArray(cconv_csc.CONVCSC_get_P_dc_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        def __set__(self, v):
            DoubleArray(cconv_csc.CONVCSC_get_P_dc_array(self._c_ptr), self.num_periods)[:] = v

    property i_dc:
        """ DC current injection into DC bus (p.u. system base MVA) (float or |Array|). """
        def __get__(self):
            r = [cconv_csc.CONVCSC_get_i_dc(self._c_ptr,t) for t in range(self.num_periods)]
            if self.num_periods == 1:
                return AttributeFloat(r[0])
            else:
                return AttributeArray(r)
            
    property P_dc_set:
        """ DC power set point (p.u. system base MVA) (float or |Array|). """
        def __get__(self):
            return DoubleArray(cconv_csc.CONVCSC_get_P_dc_set_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        def __set__(self, v):
            DoubleArray(cconv_csc.CONVCSC_get_P_dc_set_array(self._c_ptr), self.num_periods)[:] = v

    property i_dc_set:
        """ DC current set point (p.u. DC base) (float or |Array|). """
        def __get__(self):
            return DoubleArray(cconv_csc.CONVCSC_get_i_dc_set_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        def __set__(self, v):
            DoubleArray(cconv_csc.CONVCSC_get_i_dc_set_array(self._c_ptr), self.num_periods)[:] = v

    property v_dc_set:
        """ DC voltage set point (p.u. DC base) (float or |Array|). """
        def __get__(self):
            return DoubleArray(cconv_csc.CONVCSC_get_v_dc_set_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        def __set__(self, v):
            DoubleArray(cconv_csc.CONVCSC_get_v_dc_set_array(self._c_ptr), self.num_periods)[:] = v

    property angle:
        """ Ignition delay angle if rectifier or extinction advance angle if inverter (radians) (float or |Array|). """
        def __get__(self):
            return DoubleArray(cconv_csc.CONVCSC_get_angle_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        def __set__(self, v):
            DoubleArray(cconv_csc.CONVCSC_get_angle_array(self._c_ptr), self.num_periods)[:] = v

    property angle_max:
        """ Maximum angle (float). """
        def __get__(self): return cconv_csc.CONVCSC_get_angle_max(self._c_ptr)
        def __set__(self, a): cconv_csc.CONVCSC_set_angle_max(self._c_ptr, a)

    property angle_min:
        """ Minimum angle (float). """
        def __get__(self): return cconv_csc.CONVCSC_get_angle_min(self._c_ptr)
        def __set__(self, a): cconv_csc.CONVCSC_set_angle_min(self._c_ptr, a)

    property ratio:
        """ Commutating transformer turns ratio (AC bus base voltage / DC bus base voltage) (float or |Array|). """
        def __get__(self):
            return DoubleArray(cconv_csc.CONVCSC_get_ratio_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        def __set__(self, v):
            DoubleArray(cconv_csc.CONVCSC_get_ratio_array(self._c_ptr), self.num_periods)[:] = v

    property ratio_max:
        """ Maximum turns ratio (float). """
        def __get__(self): return cconv_csc.CONVCSC_get_ratio_max(self._c_ptr)
        def __set__(self, r): cconv_csc.CONVCSC_set_ratio_max(self._c_ptr, r)

    property ratio_min:
        """ Minimum ratio (float). """
        def __get__(self): return cconv_csc.CONVCSC_get_ratio_min(self._c_ptr)
        def __set__(self, r): cconv_csc.CONVCSC_set_ratio_min(self._c_ptr, r)

    property v_base_p:
        """ Primary side bus base AC voltage (kv) (float). """
        def __get__(self): return cconv_csc.CONVCSC_get_v_base_p(self._c_ptr)
        def __set__(self, v): cconv_csc.CONVCSC_set_v_base_p(self._c_ptr, v)

    property v_base_s:
        """ Secondary side bus base AC voltage (kv) (float). """
        def __get__(self): return cconv_csc.CONVCSC_get_v_base_s(self._c_ptr)
        def __set__(self, v): cconv_csc.CONVCSC_set_v_base_s(self._c_ptr, v)

    property json_string:
        """ JSON string (string). """
        def __get__(self): 
            cdef char* json_string = cconv_csc.CONVCSC_get_json_string(self._c_ptr, NULL)
            s = json_string.decode('UTF-8')
            free(json_string)
            return s

    property flags_vars:
        """ Flags associated with variable quantities (byte). """
        def __get__(self): return cconv_csc.CONVCSC_get_flags_vars(self._c_ptr)

    property flags_fixed:
        """ Flags associated with fixed quantities (byte). """
        def __get__(self): return cconv_csc.CONVCSC_get_flags_fixed(self._c_ptr)

    property flags_bounded:
        """ Flags associated with bounded quantities (byte). """
        def __get__(self): return cconv_csc.CONVCSC_get_flags_bounded(self._c_ptr)

    property flags_sparse:
        """ Flags associated with sparse quantities (byte). """
        def __get__(self): return cconv_csc.CONVCSC_get_flags_sparse(self._c_ptr)
            
cdef new_ConverterCSC(cconv_csc.ConvCSC* c):
    if c is not NULL:
        conv = ConverterCSC(alloc=False)
        conv._c_ptr = c
        return conv
    else:
        raise ConverterCSCError('no CSC converter data')
