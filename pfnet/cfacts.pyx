#cython: embedsignature=True

#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

cimport cfacts

# Infinite
FACTS_INF_VMAG_S = cfacts.FACTS_INF_VMAG_S
FACTS_INF_VANG_S = cfacts.FACTS_INF_VANG_S
FACTS_INF_P = cfacts.FACTS_INF_P
FACTS_INF_Q = cfacts.FACTS_INF_Q

class FactsError(Exception):
    """
    Facts error exception.
    """

    pass

cdef class Facts:
    """
    Facts class.
    """

    cdef cfacts.Facts* _c_ptr
    cdef bint alloc

    def __init__(self, num_periods=1, alloc=True):
        """
        Facts class.

        Parameters
        ----------
        num_periods : int
        alloc : |TrueFalse|
        """
        
        pass

    def __cinit__(self, num_periods=1, alloc=True):

        if alloc:
            self._c_ptr = cfacts.FACTS_new(num_periods)
        else:
            self._c_ptr = NULL
        self.alloc = alloc

    def __dealloc__(self):

        if self.alloc:
            cfacts.FACTS_array_del(self._c_ptr,1)
            self._c_ptr = NULL

    def _get_c_ptr(self):

        return new_CPtr(self._c_ptr)

    def is_equal(self, other):
        """
        Determines whether facts is equal to given facts.

        Parameters
        ----------
        other : |Facts|

        Returns
        -------
        flag : |TrueFalse|
        """

        cdef Facts f_other

        if not isinstance(other,Facts):
            return False

        f_other = other

        return cfacts.FACTS_is_equal(self._c_ptr,f_other._c_ptr)

    def is_regulator(self):
        """
        Determines whether FACTS provides voltage set point regulation with shunt converter.

        Returns
        -------
        flag : |TrueFalse|
        """

        return cfacts.FACTS_is_regulator(self._c_ptr)
    
    def is_STATCOM(self):
        """
        Determines whether device is STATCOM.

        Returns
        -------
        flag : |TrueFalse|
        """

        return cfacts.FACTS_is_STATCOM(self._c_ptr)

    def is_SSSC(self):
        """
        Determines whether device is SSSC.

        Returns
        -------
        flag : |TrueFalse|
        """

        return cfacts.FACTS_is_SSSC(self._c_ptr)

    def is_UPFC(self):
        """
        Determines whether device is UPFC.

        Returns
        -------
        flag : |TrueFalse|
        """

        return cfacts.FACTS_is_UPFC(self._c_ptr)

    def is_series_link_disabled(self):
        """
        Determines whether series link is disabled.

        Returns
        -------
        flag : |TrueFalse|
        """

        return cfacts.FACTS_is_series_link_disabled(self._c_ptr)
    
    def is_series_link_bypassed(self):
        """
        Determines whether series link is bypassed.

        Returns
        -------
        flag : |TrueFalse|
        """

        return cfacts.FACTS_is_series_link_bypassed(self._c_ptr)

    def is_in_normal_series_mode(self):
        """
        Determines whether series link operates in normal mode.

        Returns
        -------
        flag : |TrueFalse|
        """

        return cfacts.FACTS_is_in_normal_series_mode(self._c_ptr)
    
    def is_in_constant_series_z_mode(self):
        """
        Determines whether series link operates in constant
        impedance mode.

        Returns
        -------
        flag : |TrueFalse|
        """

        return cfacts.FACTS_is_in_constant_series_z_mode(self._c_ptr)

    def is_in_constant_series_v_mode(self):
        """
        Determines whether series link operates in constant
        voltage mode.

        Returns
        -------
        flag : |TrueFalse|
        """

        return cfacts.FACTS_is_in_constant_series_v_mode(self._c_ptr)

    def set_series_link_disabled(self):
        """
        Sets series link to disabled.
        """
        
        cfacts.FACTS_set_mode_s(self._c_ptr, cfacts.FACTS_SERIES_MODE_DISABLED)

    def set_series_link_bypassed(self):

        """
        Sets series link to bypassed.
        """
        
        cfacts.FACTS_set_mode_s(self._c_ptr, cfacts.FACTS_SERIES_MODE_BYPASS)

    def set_in_normal_series_mode(self):
        """
        Sets FACTS to normal series mode.
        """
        
        cfacts.FACTS_set_mode_s(self._c_ptr, cfacts.FACTS_SERIES_MODE_NORMAL)

    def set_in_constant_series_z_mode(self):
        """
        Sets FACTS to constant series z mode.
        """
        
        cfacts.FACTS_set_mode_s(self._c_ptr, cfacts.FACTS_SERIES_MODE_CZ)

    def set_in_constant_series_v_mode(self):
        """
        Sets FACTS to normal series v mode.
        """
        
        cfacts.FACTS_set_mode_s(self._c_ptr, cfacts.FACTS_SERIES_MODE_CV)

    def __richcmp__(self, other, op):
        """
        Compares two facts devices.

        Parameters
        ----------
        other : |Facts|
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
        Determines whether the facts has the flags associated with
        specific quantities set.

        Parameters
        ----------
        flag_type : string (|RefFlags|)
        q : string or list of strings (|RefFactsQuantities|)

        Returns
        -------
        flag : |TrueFalse|
        """

        q = q if isinstance(q,list) else [q]

        return cfacts.FACTS_has_flags(self._c_ptr,
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

        cdef char* info_string = cfacts.FACTS_get_var_info_string(self._c_ptr, index)
        if info_string:
            s = info_string.decode('UTF-8')
            free(info_string)
            return s
        else:
            raise FactsError('index does not correspond to any variable')

    property name:
        """ Facts name (string). """
        def __get__(self):
            return cfacts.FACTS_get_name(self._c_ptr).decode('UTF-8')
        def __set__(self,name):
            name = name.encode('UTF-8')
            cfacts.FACTS_set_name(self._c_ptr,name)

    property num_periods:
        """ Number of time periods (int). """
        def __get__(self): return cfacts.FACTS_get_num_periods(self._c_ptr)

    property obj_type:
        """ Object type (string). """
        def __get__(self): return obj2str[cfacts.FACTS_get_obj_type(self._c_ptr)]

    property index:
        """ Facts index (int). """
        def __get__(self): return cfacts.FACTS_get_index(self._c_ptr)

    property index_v_mag_s:
        """ Index of series voltage magnitude variable (int or |Array|). """
        def __get__(self):
            return IntArray(cfacts.FACTS_get_index_v_mag_s_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)

    property index_v_ang_s:
        """ Index of series voltage angle variable (int or |Array|). """
        def __get__(self):
            return IntArray(cfacts.FACTS_get_index_v_ang_s_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)

    property index_P_k:
        """ Index of active power injection into "k" bus (int or |Array|). """
        def __get__(self):
            return IntArray(cfacts.FACTS_get_index_P_k_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)

    property index_P_m:
        """ Index of active power injection into "m" bus (int or |Array|). """
        def __get__(self):
            return IntArray(cfacts.FACTS_get_index_P_m_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)

    property index_P_dc:
        """ Index of DC power exchanged from shunt to series converter (int or |Array|). """
        def __get__(self):
            return IntArray(cfacts.FACTS_get_index_P_dc_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)

    property index_Q_k:
        """ Index of reactive power injection into "k" bus (int or |Array|). """
        def __get__(self):
            return IntArray(cfacts.FACTS_get_index_Q_k_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)

    property index_Q_m:
        """ Index of reactive power injection into "m" bus (int or |Array|). """
        def __get__(self):
            return IntArray(cfacts.FACTS_get_index_Q_m_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)

    property index_Q_s:
        """ Index of reactive power provided by series converter (int or |Array|). """
        def __get__(self):
            return IntArray(cfacts.FACTS_get_index_Q_s_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)

    property index_Q_sh:
        """ Index of reactive power provided by shunt converter (int or |Array|). """
        def __get__(self):
            return IntArray(cfacts.FACTS_get_index_Q_sh_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)

    property v_mag_s:
        """ Series voltage magnitude (p.u.) (float or |Array|). """
        def __get__(self):
            return DoubleArray(cfacts.FACTS_get_v_mag_s_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        def __set__(self, v):
            DoubleArray(cfacts.FACTS_get_v_mag_s_array(self._c_ptr), self.num_periods)[:] = v

    property v_ang_s:
        """ Series voltage angle (radians with respect to bus_k voltage angle) (float or |Array|). """
        def __get__(self):
            return DoubleArray(cfacts.FACTS_get_v_ang_s_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        def __set__(self, v):
            DoubleArray(cfacts.FACTS_get_v_ang_s_array(self._c_ptr), self.num_periods)[:] = v

    property v_max_s:
        """ Maximum series voltage magnitude (p.u.) (float). """
        def __get__(self): return cfacts.FACTS_get_v_max_s(self._c_ptr)
        def __set__(self,v): cfacts.FACTS_set_v_max_s(self._c_ptr,v)

    property g:
        """ Series conductance set-point for constant impedance mode (p.u.) (float). """
        def __get__(self): return cfacts.FACTS_get_g(self._c_ptr)
        def __set__(self,g): cfacts.FACTS_set_g(self._c_ptr,g)

    property b:
        """ Series susceptance set-point for constant impedance mode (p.u.) (float). """
        def __get__(self): return cfacts.FACTS_get_b(self._c_ptr)
        def __set__(self,b): cfacts.FACTS_set_b(self._c_ptr,b)

    property P_k:
        """ Active power injected into the "k" bus (p.u.) (float or |Array|). """
        def __get__(self):
            return DoubleArray(cfacts.FACTS_get_P_k_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        def __set__(self, v):
            DoubleArray(cfacts.FACTS_get_P_k_array(self._c_ptr), self.num_periods)[:] = v

    property P_m:
        """ Active power injected into the "m" bus (p.u.) (float or |Array|). """
        def __get__(self):
            return DoubleArray(cfacts.FACTS_get_P_m_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        def __set__(self, v):
            DoubleArray(cfacts.FACTS_get_P_m_array(self._c_ptr), self.num_periods)[:] = v

    property Q_k:
        """ Reactive power injected into the "k" bus (p.u.) (float or |Array|). """
        def __get__(self):
            return DoubleArray(cfacts.FACTS_get_Q_k_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        def __set__(self, v):
            DoubleArray(cfacts.FACTS_get_Q_k_array(self._c_ptr), self.num_periods)[:] = v

    property Q_m:
        """ Reactive power injected into the "m" bus (p.u.) (float or |Array|). """
        def __get__(self):
            return DoubleArray(cfacts.FACTS_get_Q_m_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        def __set__(self, v):
            DoubleArray(cfacts.FACTS_get_Q_m_array(self._c_ptr), self.num_periods)[:] = v

    property Q_sh:
        """ Reactive power provided by shunt converter (p.u.) (float or |Array|). """
        def __get__(self):
            return DoubleArray(cfacts.FACTS_get_Q_sh_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        def __set__(self, v):
            DoubleArray(cfacts.FACTS_get_Q_sh_array(self._c_ptr), self.num_periods)[:] = v

    property Q_s:
        """ Reactive power provided by series converter (p.u.) (float or |Array|). """
        def __get__(self):
            return DoubleArray(cfacts.FACTS_get_Q_s_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        def __set__(self, v):
            DoubleArray(cfacts.FACTS_get_Q_s_array(self._c_ptr), self.num_periods)[:] = v

    property P_dc:
        """ DC power exchanged from shunt to series converter (p.u.) (float or |Array|). """
        def __get__(self):
            return DoubleArray(cfacts.FACTS_get_P_dc_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        def __set__(self, v):
            DoubleArray(cfacts.FACTS_get_P_dc_array(self._c_ptr), self.num_periods)[:] = v

    property P_set:
        """ Active power set-point at the "m" bus (p.u.) (float or |Array|). """
        def __get__(self):
            return DoubleArray(cfacts.FACTS_get_P_set_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        def __set__(self, v):
            DoubleArray(cfacts.FACTS_get_P_set_array(self._c_ptr), self.num_periods)[:] = v

    property Q_set:
        """ Reactive power set-point at the "m" bus (p.u.) (float or |Array|). """
        def __get__(self):
            return DoubleArray(cfacts.FACTS_get_Q_set_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        def __set__(self, v):
            DoubleArray(cfacts.FACTS_get_Q_set_array(self._c_ptr), self.num_periods)[:] = v

    property Q_par:
        """ Reactive power participation factor of shunt converter for regulating bus voltage magnitude (unitless) (float). """
        def __get__(self): return cfacts.FACTS_get_Q_par(self._c_ptr)
        def __set__(self, Q_par): cfacts.FACTS_set_Q_par(self._c_ptr, Q_par)

    property Q_max_s:
        """ Maximum series converter reactive power (p.u.) (float). """
        def __get__(self): return cfacts.FACTS_get_Q_max_s(self._c_ptr)
        def __set__(self, Q_max): cfacts.FACTS_set_Q_max_s(self._c_ptr, Q_max)

    property Q_max_sh:
        """ Maximum shunt converter reactive power (p.u.) (float). """
        def __get__(self): return cfacts.FACTS_get_Q_max_sh(self._c_ptr)
        def __set__(self, Q_max): cfacts.FACTS_set_Q_max_sh(self._c_ptr, Q_max)

    property Q_min_s:
        """ Minimum series converter reactive power (p.u.) (float). """
        def __get__(self): return cfacts.FACTS_get_Q_min_s(self._c_ptr)
        def __set__(self, Q_min): cfacts.FACTS_set_Q_min_s(self._c_ptr, Q_min)

    property Q_min_sh:
        """ Minimum shunt converter reactive power (p.u.) (float). """
        def __get__(self): return cfacts.FACTS_get_Q_min_sh(self._c_ptr)
        def __set__(self, Q_min): cfacts.FACTS_set_Q_min_sh(self._c_ptr, Q_min)
        
    property i_max_s:
        """ Maximum series converter current (p.u.) (float). """
        def __get__(self): return cfacts.FACTS_get_i_max_s(self._c_ptr)
        def __set__(self, i_max): cfacts.FACTS_set_i_max_s(self._c_ptr, i_max)

    property i_max_sh:
        """ Maximum shunt converter current (p.u.) (float). """
        def __get__(self): return cfacts.FACTS_get_i_max_sh(self._c_ptr)
        def __set__(self, i_max): cfacts.FACTS_set_i_max_sh(self._c_ptr, i_max)

    property P_max_dc:
        """ Maximum DC power transfer (p.u.) (float). """
        def __get__(self): return cfacts.FACTS_get_P_max_dc(self._c_ptr)
        def __set__(self, P_max): cfacts.FACTS_set_P_max_dc(self._c_ptr, P_max)

    property v_min_m:
        """ Minimum voltage magnitude for bus "m" (p.u.) (float). """
        def __get__(self): return cfacts.FACTS_get_v_min_m(self._c_ptr)
        def __set__(self, v_min): cfacts.FACTS_set_v_min_m(self._c_ptr, v_min)

    property v_max_m:
        """ Maximum voltage magnitude for bus "m" (p.u.) (float). """
        def __get__(self): return cfacts.FACTS_get_v_max_m(self._c_ptr)
        def __set__(self, v_max): cfacts.FACTS_set_v_max_m(self._c_ptr, v_max)

    property bus_k:
        """ |Bus| connected to the "k" side. """
        def __get__(self):
            return new_Bus(cfacts.FACTS_get_bus_k(self._c_ptr))
        def __set__(self,bus): 
            cdef Bus cbus
            if not isinstance(bus,Bus) and bus is not None:
                raise FactsError('Not a Bus type object')
            cbus = bus
            cfacts.FACTS_set_bus_k(self._c_ptr,cbus._c_ptr if bus is not None else NULL)

    property bus_m:
        """ |Bus| connected to the "m" side. """
        def __get__(self):
            return new_Bus(cfacts.FACTS_get_bus_m(self._c_ptr))
        def __set__(self,bus): 
            cdef Bus cbus
            if not isinstance(bus,Bus) and bus is not None:
                raise FactsError('Not a Bus type object')
            cbus = bus
            cfacts.FACTS_set_bus_m(self._c_ptr,cbus._c_ptr if bus is not None else NULL)

    property reg_bus:
        """ |Bus| whose voltage is regulated by this FACTS device. """
        def __get__(self):
            return new_Bus(cfacts.FACTS_get_reg_bus(self._c_ptr))
        def __set__(self,bus): 
            cdef Bus cbus
            if not isinstance(bus,Bus) and bus is not None:
                raise FactsError('Not a Bus type object')
            cbus = bus
            cfacts.FACTS_set_reg_bus(self._c_ptr,cbus._c_ptr if bus is not None else NULL)

    property json_string:
        """ JSON string (string). """
        def __get__(self): 
            cdef char* json_string = cfacts.FACTS_get_json_string(self._c_ptr, NULL)
            s = json_string.decode('UTF-8')
            free(json_string)
            return s

    property flags_vars:
        """ Flags associated with variable quantities (byte). """
        def __get__(self): return cfacts.FACTS_get_flags_vars(self._c_ptr)

    property flags_fixed:
        """ Flags associated with fixed quantities (byte). """
        def __get__(self): return cfacts.FACTS_get_flags_fixed(self._c_ptr)

    property flags_bounded:
        """ Flags associated with bounded quantities (byte). """
        def __get__(self): return cfacts.FACTS_get_flags_bounded(self._c_ptr)

    property flags_sparse:
        """ Flags associated with sparse quantities (byte). """
        def __get__(self): return cfacts.FACTS_get_flags_sparse(self._c_ptr)

cdef new_Facts(cfacts.Facts* b):
    if b is not NULL:
        facts = Facts(alloc=False)
        facts._c_ptr = b
        return facts
    else:
        return None
        
