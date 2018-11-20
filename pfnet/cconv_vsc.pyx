#cython: embedsignature=True

#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

cimport cconv_vsc

# Infinity
CONVVSC_INF_P = cconv_vsc.CONVVSC_INF_P
CONVVSC_INF_Q = cconv_vsc.CONVVSC_INF_Q
CONVVSC_INF_PDC = cconv_vsc.CONVVSC_INF_PDC

class ConverterVSCError(Exception):
    """
    VSC converter error exception.
    """

    pass

cdef class ConverterVSC:
    """
    VSC converter class.
    """

    cdef cconv_vsc.ConvVSC* _c_ptr
    cdef bint alloc

    def __init__(self, num_periods=1, alloc=True):
        """
        VSC converter class.

        Parameters
        ----------
        num_periods : int
        alloc : |TrueFalse|
        """

        pass

    def __cinit__(self, num_periods=1, alloc=True):

        if alloc:
            self._c_ptr = cconv_vsc.CONVVSC_new(num_periods)
        else:
            self._c_ptr = NULL
        self.alloc = alloc

    def __dealloc__(self):

        if self.alloc:
            cconv_vsc.CONVVSC_array_del(self._c_ptr,1)
            self._c_ptr = NULL    

    def _get_c_ptr(self):

        return new_CPtr(self._c_ptr)

    def is_equal(self, other):
        """
        Determines whether VSC converter is equal to given VSC converter.

        Parameters
        ----------
        other : |ConverterVSC|

        Returns
        -------
        flag : |TrueFalse|
        """

        cdef ConverterVSC c_other

        if not isinstance(other,ConverterVSC):
            return False

        c_other = other

        return cconv_vsc.CONVVSC_is_equal(self._c_ptr, c_other._c_ptr)

    def is_in_P_dc_mode(self):
        """
        Determines whether VSC converter is in constant DC power mode.
        
        Returns
        -------
        flag : |TrueFalse|
        """

        return cconv_vsc.CONVVSC_is_in_P_dc_mode(self._c_ptr)

    def is_in_v_dc_mode(self):
        """
        Determines whether VSC converter is in constant DC voltage mode.
        
        Returns
        -------
        flag : |TrueFalse|
        """

        return cconv_vsc.CONVVSC_is_in_v_dc_mode(self._c_ptr)

    def is_in_f_ac_mode(self):
        """
        Determines whether VSC converter is in constant AC power factor mode.
        
        Returns
        -------
        flag : |TrueFalse|
        """

        return cconv_vsc.CONVVSC_is_in_f_ac_mode(self._c_ptr)

    def is_in_v_ac_mode(self):
        """
        Determines whether VSC converter is in constant AC voltage magnitude mode.
        
        Returns
        -------
        flag : |TrueFalse|
        """

        return cconv_vsc.CONVVSC_is_in_v_ac_mode(self._c_ptr)

    def set_in_P_dc_mode(self):
        """
        Sets VSC converter to be in constant DC power mode.
        """

        cconv_vsc.CONVVSC_set_mode_dc(self._c_ptr, cconv_vsc.CONVVSC_MODE_DC_CP)

    def set_in_v_dc_mode(self):
        """
        Sets VSC converter to be in constant DC voltage mode.
        """

        cconv_vsc.CONVVSC_set_mode_dc(self._c_ptr, cconv_vsc.CONVVSC_MODE_DC_CV)

    def set_in_f_ac_mode(self):
        """
        Sets VSC converter to be in constant AC power factor mode.
        """

        cconv_vsc.CONVVSC_set_mode_ac(self._c_ptr, cconv_vsc.CONVVSC_MODE_AC_CF)

    def set_in_v_ac_mode(self):
        """
        Sets VSC converter to be in constant AC voltage mode.
        """

        cconv_vsc.CONVVSC_set_mode_ac(self._c_ptr, cconv_vsc.CONVVSC_MODE_AC_CV)

    def has_flags(self, flag_type, q):
        """
        Determines whether the VSC converter has the flags associated with
        certain quantities set.

        Parameters
        ----------
        flag_type : string (|RefFlags|)
        q : string or list of strings (|RefConverterVSCQuantities|)

        Returns
        -------
        flag : |TrueFalse|
        """

        q = q if isinstance(q,list) else [q]

        return cconv_vsc.CONVVSC_has_flags(self._c_ptr,
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

        cdef char* info_string = cconv_vsc.CONVVSC_get_var_info_string(self._c_ptr, index)
        if info_string:
            s = info_string.decode('UTF-8')
            free(info_string)
            return s
        else:
            raise ConverterVSCError('index does not correspond to any variable')
        
    property name:
        """ VSC converter name (string). """
        def __get__(self):
            return cconv_vsc.CONVVSC_get_name(self._c_ptr).decode('UTF-8')
        def __set__(self,name):
            name = name.encode('UTF-8')
            cconv_vsc.CONVVSC_set_name(self._c_ptr,name)

    property num_periods:
        """ Number of time periods (int). """
        def __get__(self): return cconv_vsc.CONVVSC_get_num_periods(self._c_ptr)

    property obj_type:
        """ Object type (string). """
        def __get__(self): return obj2str[cconv_vsc.CONVVSC_get_obj_type(self._c_ptr)]

    property index:
        """ VSC converter index (int). """
        def __get__(self): return cconv_vsc.CONVVSC_get_index(self._c_ptr)

    property index_P:
        """ Index of active power variable (int or |Array|). """
        def __get__(self):
            return IntArray(cconv_vsc.CONVVSC_get_index_P_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)

    property index_Q:
        """ Index of reactive power variable (int or |Array|). """
        def __get__(self):
            return IntArray(cconv_vsc.CONVVSC_get_index_Q_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)

    property index_P_dc:
        """ Index of DC power variable (int or |Array|). """
        def __get__(self):
            return IntArray(cconv_vsc.CONVVSC_get_index_P_dc_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)

    property index_i_dc:
        """ Index of DC current variable (int or |Array|). """
        def __get__(self):
            return IntArray(cconv_vsc.CONVVSC_get_index_i_dc_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)

    property ac_bus:
        """ |Bus| to which VSC converter is connected. """
        def __get__(self): 
            return new_Bus(cconv_vsc.CONVVSC_get_ac_bus(self._c_ptr))
        def __set__(self,bus):
            cdef Bus cbus
            if not isinstance(bus,Bus) and bus is not None:
                raise ConverterVSCError('Not a Bus type object')
            cbus = bus
            cconv_vsc.CONVVSC_set_ac_bus(self._c_ptr,cbus._c_ptr if bus is not None else NULL)

    property dc_bus:
        """ |BusDC| to which VSC converter is connected. """
        def __get__(self): 
            return new_BusDC(cconv_vsc.CONVVSC_get_dc_bus(self._c_ptr))
        def __set__(self,bus):
            cdef BusDC cbus
            if not isinstance(bus,BusDC) and bus is not None:
                raise ConverterVSCError('Not a BusDC type object')
            cbus = bus
            cconv_vsc.CONVVSC_set_dc_bus(self._c_ptr,cbus._c_ptr if bus is not None else NULL)

    property reg_bus:
        """ |Bus| whose voltage is regulated by this generator. """
        def __get__(self): 
            return new_Bus(cconv_vsc.CONVVSC_get_reg_bus(self._c_ptr))
        def __set__(self, reg_bus):
            cdef Bus creg_bus
            if not isinstance(reg_bus, Bus) and reg_bus is not None:
                raise ConverterVSCError('Not a Bus type object')
            creg_bus = reg_bus
            cconv_vsc.CONVVSC_set_reg_bus(self._c_ptr,creg_bus._c_ptr if reg_bus is not None else NULL)
            
    property P:
        """ Active power injection into AC bus (p.u. system base MVA) (float or |Array|). """
        def __get__(self):
            return DoubleArray(cconv_vsc.CONVVSC_get_P_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        def __set__(self, v):
            DoubleArray(cconv_vsc.CONVVSC_get_P_array(self._c_ptr), self.num_periods)[:] = v

    property Q:
        """ Reactive power injection into AC bus (p.u. system base MVA) (float or |Array|). """
        def __get__(self):
            return DoubleArray(cconv_vsc.CONVVSC_get_Q_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        def __set__(self, v):
            DoubleArray(cconv_vsc.CONVVSC_get_Q_array(self._c_ptr), self.num_periods)[:] = v

    property P_max:
        """ Maximum active power injection into AC bus (float). """
        def __get__(self): return cconv_vsc.CONVVSC_get_P_max(self._c_ptr)
        def __set__(self, v): cconv_vsc.CONVVSC_set_P_max(self._c_ptr, v)

    property P_min:
        """ Minimum active power injection into AC bus (float). """
        def __get__(self): return cconv_vsc.CONVVSC_get_P_min(self._c_ptr)
        def __set__(self, v): cconv_vsc.CONVVSC_set_P_min(self._c_ptr, v)
                
    property Q_max:
        """ Maximum reactive power injection into AC bus (float). """
        def __get__(self): return cconv_vsc.CONVVSC_get_Q_max(self._c_ptr)
        def __set__(self, v): cconv_vsc.CONVVSC_set_Q_max(self._c_ptr, v)

    property Q_min:
        """ Minimum reactive power injection into AC bus (float). """
        def __get__(self): return cconv_vsc.CONVVSC_get_Q_min(self._c_ptr)
        def __set__(self, v): cconv_vsc.CONVVSC_set_Q_min(self._c_ptr, v)

    property Q_par:
        """ Reactive power participation (unitless) (float). """
        def __get__(self): return cconv_vsc.CONVVSC_get_Q_par(self._c_ptr)
        def __set__(self, v): cconv_vsc.CONVVSC_set_Q_par(self._c_ptr, v)

    property P_dc:
        """ DC power injection into DC bus (p.u. system base MVA) (float or |Array|). """
        def __get__(self):
            return DoubleArray(cconv_vsc.CONVVSC_get_P_dc_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        def __set__(self, v):
            DoubleArray(cconv_vsc.CONVVSC_get_P_dc_array(self._c_ptr), self.num_periods)[:] = v

    property i_dc:
        """ DC current injection into DC bus (p.u. system base MVA and DC bus v base) (float or |Array|). """
        def __get__(self):
            r = [cconv_vsc.CONVVSC_get_i_dc(self._c_ptr,t) for t in range(self.num_periods)]
            if self.num_periods == 1:
                return AttributeFloat(r[0])
            else:
                return AttributeArray(r)
                
    property P_dc_set:
        """ DC power set point (p.u. system base MVA) (float or |Array|). """
        def __get__(self):
            return DoubleArray(cconv_vsc.CONVVSC_get_P_dc_set_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        def __set__(self, v):
            DoubleArray(cconv_vsc.CONVVSC_get_P_dc_set_array(self._c_ptr), self.num_periods)[:] = v

    property v_dc_set:
        """ DC voltage set point (p.u. DC base) (float or |Array|). """
        def __get__(self):
            return DoubleArray(cconv_vsc.CONVVSC_get_v_dc_set_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        def __set__(self, v):
            DoubleArray(cconv_vsc.CONVVSC_get_v_dc_set_array(self._c_ptr), self.num_periods)[:] = v

    property loss_coeff_A:
        """ Loss coefficient for constant term (p.u. system base power) (float). """
        def __get__(self): return cconv_vsc.CONVVSC_get_loss_coeff_A(self._c_ptr)
        def __set__(self, A): cconv_vsc.CONVVSC_set_loss_coeff_A(self._c_ptr, A)

    property loss_coeff_B:
        """ Loss coefficient for linear term (p.u. system base power over DC current in p.u.) (float). """
        def __get__(self): return cconv_vsc.CONVVSC_get_loss_coeff_B(self._c_ptr)
        def __set__(self, B): cconv_vsc.CONVVSC_set_loss_coeff_B(self._c_ptr, B)

    property target_power_factor:
        """ Target AC power factor (float). """
        def __get__(self): return cconv_vsc.CONVVSC_get_target_power_factor(self._c_ptr)
        def __set__(self,pf): cconv_vsc.CONVVSC_set_target_power_factor(self._c_ptr,pf)

    property json_string:
        """ JSON string (string). """
        def __get__(self): 
            cdef char* json_string = cconv_vsc.CONVVSC_get_json_string(self._c_ptr, NULL)
            s = json_string.decode('UTF-8')
            free(json_string)
            return s

    property flags_vars:
        """ Flags associated with variable quantities (byte). """
        def __get__(self): return cconv_vsc.CONVVSC_get_flags_vars(self._c_ptr)

    property flags_fixed:
        """ Flags associated with fixed quantities (byte). """
        def __get__(self): return cconv_vsc.CONVVSC_get_flags_fixed(self._c_ptr)

    property flags_bounded:
        """ Flags associated with bounded quantities (byte). """
        def __get__(self): return cconv_vsc.CONVVSC_get_flags_bounded(self._c_ptr)

    property flags_sparse:
        """ Flags associated with sparse quantities (byte). """
        def __get__(self): return cconv_vsc.CONVVSC_get_flags_sparse(self._c_ptr)
            
cdef new_ConverterVSC(cconv_vsc.ConvVSC* c):
    if c is not NULL:
        conv = ConverterVSC(alloc=False)
        conv._c_ptr = c
        return conv
    else:
        raise ConverterVSCError('no VSC converter data')
