#cython: embedsignature=True

#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

cimport cbranch

# Infinite
BRANCH_INF_RATIO = cbranch.BRANCH_INF_RATIO
BRANCH_INF_PHASE = cbranch.BRANCH_INF_PHASE

class BranchError(Exception):
    """
    Branch error exception.
    """

    pass

cdef class Branch:
    """
    Branch class.
    """

    cdef cbranch.Branch* _c_ptr
    cdef bint alloc

    def __init__(self, num_periods=1, alloc=True):
        """
        Branch class.

        Parameters
        ----------
        num_periods : int
        alloc : |TrueFalse|
        """

        pass

    def __cinit__(self, num_periods=1, alloc=True):

        if alloc:
            self._c_ptr = cbranch.BRANCH_new(num_periods)
        else:
            self._c_ptr = NULL
        self.alloc = alloc

    def __dealloc__(self):

        if self.alloc:
            cbranch.BRANCH_array_del(self._c_ptr,1)
            self._c_ptr = NULL

    def _get_c_ptr(self):

        return new_CPtr(self._c_ptr)

    def has_pos_ratio_v_sens(self):
        """
        Determines whether tap-changing transformer has positive
        sensitivity between tap ratio and controlled bus voltage magnitude.

        Returns
        -------
        flag : |TrueFalse|
        """

        return cbranch.BRANCH_has_pos_ratio_v_sens(self._c_ptr)

    def is_equal(self, other):
        """
        Determines whether branch is equal to given branch.

        Parameters
        ----------
        other : |Branch|
        """

        cdef Branch b_other

        if not isinstance(other,Branch):
            return False

        b_other = other

        return cbranch.BRANCH_is_equal(self._c_ptr,b_other._c_ptr)
        
    def set_pos_ratio_v_sens(self, flag):
        """
        Sets the flag for positive ratio-voltage sensitivity.
        
        Parameters
        ----------
        flag : |TrueFalse|
        """
        
        cbranch.BRANCH_set_pos_ratio_v_sens(self._c_ptr, flag);

    def get_rating(self, code):
        """
        Gets branch thermal rating.

        Parameters
        ----------
        code : string ('A', 'B', 'C')

        Returns
        -------
        rating : float
        """

        if code == 'A':
            return self.ratingA
        if code == 'B':
            return self.ratingB
        if code == 'C':
            return self.ratingC
        raise BranchError('thermal rating code must be A, B, or C')

    def set_as_fixed_tran(self):
        """ 
        Sets branch as a fixed transformer. 
        """
        
        cbranch.BRANCH_set_type(self._c_ptr,cbranch.BRANCH_TYPE_TRAN_FIXED)

    def set_as_line(self):
        """ 
        Sets branch as a line. 
        """

        cbranch.BRANCH_set_type(self._c_ptr,cbranch.BRANCH_TYPE_LINE)

    def set_as_phase_shifter(self):
        """ 
        Sets branch as a phase shifter. 
        """

        cbranch.BRANCH_set_type(self._c_ptr,cbranch.BRANCH_TYPE_TRAN_PHASE)

    def set_as_tap_changer_v(self):
        """ 
        Sets branch as a tap changer regulating voltage. 
        """

        cbranch.BRANCH_set_type(self._c_ptr,cbranch.BRANCH_TYPE_TRAN_TAP_V)

    def set_as_tap_changer_Q(self):
        """ 
        Sets branch as a tap changer regulating reactive power. 
        """

        cbranch.BRANCH_set_type(self._c_ptr,cbranch.BRANCH_TYPE_TRAN_TAP_Q)

    def __richcmp__(self, other, op):
        """
        Compares two branches.

        Parameters
        ----------
        other : |Branch|
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

    def is_on_outage(self):
        """
        Determines whether branch in on outage.

        Returns
        -------
        flag : |TrueFalse|
        """

        return cbranch.BRANCH_is_on_outage(self._c_ptr)

    def is_fixed_tran(self):
        """
        Determines whether branch is fixed transformer.

        Returns
        -------
        flag : |TrueFalse|
        """

        return cbranch.BRANCH_is_fixed_tran(self._c_ptr)

    def is_line(self):
        """
        Determines whether branch is transmission line.

        Returns
        -------
        flag : |TrueFalse|
        """

        return cbranch.BRANCH_is_line(self._c_ptr)

    def is_phase_shifter(self):
        """
        Determines whether branch is phase shifter.

        Returns
        -------
        flag : |TrueFalse|
        """

        return cbranch.BRANCH_is_phase_shifter(self._c_ptr)

    def is_tap_changer(self):
        """
        Determines whether branch is tap-changing transformer.

        Returns
        -------
        flag : |TrueFalse|
        """

        return cbranch.BRANCH_is_tap_changer(self._c_ptr)

    def is_tap_changer_v(self):
        """
        Determines whether branch is tap-changing transformer
        that regulates bus voltage magnitude.

        Returns
        -------
        flag : |TrueFalse|
        """

        return cbranch.BRANCH_is_tap_changer_v(self._c_ptr)

    def is_tap_changer_Q(self):
        """
        Determines whether branch is tap-changing transformer
        that regulates reactive power flow.

        Returns
        -------
        flag : |TrueFalse|
        """

        return cbranch.BRANCH_is_tap_changer_Q(self._c_ptr)

    def is_part_of_3_winding_transformer(self):
        """
        Determines whether branch is part of 3-winding
        transformer.

        Returns
        -------
        flag : |TrueFalse|
        """

        return cbranch.BRANCH_is_part_of_3_winding_transformer(self._c_ptr)

    def has_y_correction(self):
        """
        Determines whether branch has y correction table.

        Returns
        -------
        flag : |TrueFalse|
        """

        return cbranch.BRANCH_has_y_correction(self._c_ptr)

    def has_flags(self, flag_type, q):
        """
        Determines whether the branch has the flags associated with
        specific quantities set.

        Parameters
        ----------
        flag_type : string (|RefFlags|)
        q : string or list of strings (|RefBranchQuantities|)

        Returns
        -------
        flag : |TrueFalse|
        """

        q = q if isinstance(q,list) else [q]

        return cbranch.BRANCH_has_flags(self._c_ptr,
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

        cdef char* info_string = cbranch.BRANCH_get_var_info_string(self._c_ptr, index)
        if info_string:
            s = info_string.decode('UTF-8')
            free(info_string)
            return s
        else:
            raise BranchError('index does not correspond to any variable')

    def get_i_km_mag(self, var_values=None, eps=0.):
        """
        Gets the branch current magnitude at bus "k" torwards bus "m" (p.u.).
        
        Parameters
        ----------
        var_values : |Array|
        eps : float

        Returns
        -------
        i_mag : float or |Array|
        """

        cdef np.ndarray[double,mode='c'] x = var_values
        cdef cvec.Vec* v = cvec.VEC_new_from_array(<cnet.REAL*>(x.data),x.size) if var_values is not None else NULL
        r = [cbranch.BRANCH_get_i_km_mag(self._c_ptr,v,t,eps) for t in range(self.num_periods)]
        free(v)
        if self.num_periods == 1:
            return AttributeFloat(r[0])
        else:
            return np.array(r)
        
    def get_i_mk_mag(self, var_values=None, eps=0.):
        """
        Gets the branch current magnitude at bus "m" torwards bus "k" (p.u.).
        
        Parameters
        ----------
        var_values : |Array|
        eps : float

        Returns
        -------
        i_mag : float or |Array|
        """

        cdef np.ndarray[double,mode='c'] x = var_values
        cdef cvec.Vec* v = cvec.VEC_new_from_array(<cnet.REAL*>(x.data),x.size) if var_values is not None else NULL
        r = [cbranch.BRANCH_get_i_mk_mag(self._c_ptr,v,t,eps) for t in range(self.num_periods)]
        free(v)
        if self.num_periods == 1:
            return AttributeFloat(r[0])
        else:
            return np.array(r)

    def get_S_km_mag(self, var_values=None):
        """
        Gets the branch apparent power magnitude at bus "k" torwards bus "m" (p.u.).
        
        Parameters
        ----------
        var_values : |Array|

        Returns
        -------
        S_mag : float or |Array|
        """

        cdef np.ndarray[double,mode='c'] x = var_values
        cdef cvec.Vec* v = cvec.VEC_new_from_array(<cnet.REAL*>(x.data),x.size) if var_values is not None else NULL
        r = [cbranch.BRANCH_get_S_km_mag(self._c_ptr,v,t) for t in range(self.num_periods)]
        free(v)
        if self.num_periods == 1:
            return AttributeFloat(r[0])
        else:
            return np.array(r)

    def get_S_mk_mag(self, var_values=None):
        """
        Gets the branch apparent power magnitude at bus "m" torwards bus "k" (p.u.).
        
        Parameters
        ----------
        var_values : |Array|

        Returns
        -------
        S_mag : float or |Array|
        """

        cdef np.ndarray[double,mode='c'] x = var_values
        cdef cvec.Vec* v = cvec.VEC_new_from_array(<cnet.REAL*>(x.data),x.size) if var_values is not None else NULL
        r = [cbranch.BRANCH_get_S_mk_mag(self._c_ptr,v,t) for t in range(self.num_periods)]
        free(v)
        if self.num_periods == 1:
            return AttributeFloat(r[0])
        else:
            return np.array(r)

    def get_P_km(self, var_values=None):
        """
        Gets the real power flow at bus "k" towards bus "m" (p.u.)

        Parameters
        ----------
        var_values : |Array|

        Returns
        -------
        P_km : float or |Array|
        """

        cdef np.ndarray[double,mode='c'] x = var_values
        cdef cvec.Vec* v = cvec.VEC_new_from_array(<cnet.REAL*>(x.data),x.size) if var_values is not None else NULL
        r = [cbranch.BRANCH_get_P_km(self._c_ptr,v,t) for t in range(self.num_periods)]
        free(v)
        if self.num_periods == 1:
            return AttributeFloat(r[0])
        else:
            return np.array(r)

    def get_Q_km(self, var_values=None):
        """
        Gets the reactive power flow at bus "k" towards bus "m" (p.u.).

        Parameters
        ----------
        var_values : |Array|

        Returns
        -------
        Q_km : float or |Array|
        """

        cdef np.ndarray[double,mode='c'] x = var_values
        cdef cvec.Vec* v = cvec.VEC_new_from_array(<cnet.REAL*>(x.data),x.size) if var_values is not None else NULL
        r = [cbranch.BRANCH_get_Q_km(self._c_ptr,v,t) for t in range(self.num_periods)]
        free(v)
        if self.num_periods == 1:
            return AttributeFloat(r[0])
        else:
            return np.array(r)

    def get_P_mk(self, var_values=None):
        """
        Gets the real power flow at bus "m" towards bus "k" (p.u.).

        Parameters
        ----------
        var_values : :|Array|

        Returns
        -------
        P_mk : float or |Array|
        """

        cdef np.ndarray[double,mode='c'] x = var_values
        cdef cvec.Vec* v = cvec.VEC_new_from_array(<cnet.REAL*>(x.data),x.size) if var_values is not None else NULL
        r = [cbranch.BRANCH_get_P_mk(self._c_ptr,v,t) for t in range(self.num_periods)]
        free(v)
        if self.num_periods == 1:
            return AttributeFloat(r[0])
        else:
            return np.array(r)

    def get_Q_mk(self, var_values=None):
        """
        Gets the reactive power flow at bus "m" towards bus "k" (p.u.).

        Parameters
        ----------
        var_values : |Array|

        Returns
        -------
        Q_mk : float or |Array|
        """

        cdef np.ndarray[double,mode='c'] x = var_values
        cdef cvec.Vec* v = cvec.VEC_new_from_array(<cnet.REAL*>(x.data),x.size) if var_values is not None else NULL
        r = [cbranch.BRANCH_get_Q_mk(self._c_ptr,v,t) for t in range(self.num_periods)]
        free(v)
        if self.num_periods == 1:
            return AttributeFloat(r[0])
        else:
            return np.array(r)

    def get_P_km_series(self, var_values=None):
        """
        Gets the real power flow at bus "k" towards bus "m" over the series impedance of the line (p.u.).

        Parameters
        ----------
        var_values : |Array|

        Returns
        -------
        P_km_series : float or |Array|
        """

        cdef np.ndarray[double,mode='c'] x = var_values
        cdef cvec.Vec* v = cvec.VEC_new_from_array(<cnet.REAL*>(x.data),x.size) if var_values is not None else NULL
        r = [cbranch.BRANCH_get_P_km_series(self._c_ptr,v,t) for t in range(self.num_periods)]
        free(v)
        if self.num_periods == 1:
            return AttributeFloat(r[0])
        else:
            return np.array(r)

    def get_Q_km_series(self, var_values=None):
        """
        Gets the reactive power flow at bus "k" towards bus "m" over the series impedance of the line (p.u.).

        Parameters
        ----------
        var_values : |Array|

        Returns
        -------
        Q_km_series : float or |Array|
        """

        cdef np.ndarray[double,mode='c'] x = var_values
        cdef cvec.Vec* v = cvec.VEC_new_from_array(<cnet.REAL*>(x.data),x.size) if var_values is not None else NULL
        r = [cbranch.BRANCH_get_Q_km_series(self._c_ptr,v,t) for t in range(self.num_periods)]
        free(v)
        if self.num_periods == 1:
            return AttributeFloat(r[0])
        else:
            return np.array(r)

    def get_P_mk_series(self, var_values=None):
        """
        Gets the real power flow at bus "m" towards bus "k" over the series impedance of the line (p.u.).

        Parameters
        ----------
        var_values : |Array|

        Returns
        -------
        P_mk_series : float or |Array|
        """

        cdef np.ndarray[double,mode='c'] x = var_values
        cdef cvec.Vec* v = cvec.VEC_new_from_array(<cnet.REAL*>(x.data),x.size) if var_values is not None else NULL
        r = [cbranch.BRANCH_get_P_mk_series(self._c_ptr,v,t) for t in range(self.num_periods)]
        free(v)
        if self.num_periods == 1:
            return AttributeFloat(r[0])
        else:
            return np.array(r)

    def get_Q_mk_series(self, var_values=None):
        """
        Gets the reactive power flow at bus "m" towards bus "k" over the series impedance of the line (p.u.).

        Parameters
        ----------
        var_values : |Array|

        Returns
        -------
        Q_mk_series : float or |Array|
        """

        cdef np.ndarray[double,mode='c'] x = var_values
        cdef cvec.Vec* v = cvec.VEC_new_from_array(<cnet.REAL*>(x.data),x.size) if var_values is not None else NULL
        r = [cbranch.BRANCH_get_Q_mk_series(self._c_ptr,v,t) for t in range(self.num_periods)]
        free(v)
        if self.num_periods == 1:
            return AttributeFloat(r[0])
        else:
            return np.array(r)

    def get_P_k_shunt(self, var_values=None):
        """
        Gets the real power flow into the shunt element at bus "k" (p.u.).

        Parameters
        ----------
        var_values : |Array|

        Returns
        -------
        P_k_shunt : float or |Array|
        """

        cdef np.ndarray[double,mode='c'] x = var_values
        cdef cvec.Vec* v = cvec.VEC_new_from_array(<cnet.REAL*>(x.data),x.size) if var_values is not None else NULL
        r = [cbranch.BRANCH_get_P_k_shunt(self._c_ptr,v,t) for t in range(self.num_periods)]
        free(v)
        if self.num_periods == 1:
            return AttributeFloat(r[0])
        else:
            return np.array(r)

    def get_Q_k_shunt(self, var_values=None):
        """
        Gets the reactive power flow into the shunt element bus "k" (p.u.).

        Parameters
        ----------
        var_values : |Array|

        Returns
        -------
        Q_k_shunt : float or |Array|
        """

        cdef np.ndarray[double,mode='c'] x = var_values
        cdef cvec.Vec* v = cvec.VEC_new_from_array(<cnet.REAL*>(x.data),x.size) if var_values is not None else NULL
        r = [cbranch.BRANCH_get_Q_k_shunt(self._c_ptr,v,t) for t in range(self.num_periods)]
        free(v)
        if self.num_periods == 1:
            return AttributeFloat(r[0])
        else:
            return np.array(r)

    def get_P_m_shunt(self, var_values=None):
        """
        Gets the real power flow into the shunt element at bus "m" (p.u.).

        Parameters
        ----------
        var_values : |Array|

        Returns
        -------
        P_m_shunt : float or |Array|
        """

        cdef np.ndarray[double,mode='c'] x = var_values
        cdef cvec.Vec* v = cvec.VEC_new_from_array(<cnet.REAL*>(x.data),x.size) if var_values is not None else NULL
        r = [cbranch.BRANCH_get_P_m_shunt(self._c_ptr,v,t) for t in range(self.num_periods)]
        free(v)
        if self.num_periods == 1:
            return AttributeFloat(r[0])
        else:
            return np.array(r)

    def get_Q_m_shunt(self, var_values=None):
        """
        Gets the reactive power flow into the shunt element at bus "m" (p.u.).

        Parameters
        ----------
        var_values : |Array|

        Returns
        -------
        Q_m_shunt : float or |Array|
        """

        cdef np.ndarray[double,mode='c'] x = var_values
        cdef cvec.Vec* v = cvec.VEC_new_from_array(<cnet.REAL*>(x.data),x.size) if var_values is not None else NULL
        r = [cbranch.BRANCH_get_Q_m_shunt(self._c_ptr,v,t) for t in range(self.num_periods)]
        free(v)
        if self.num_periods == 1:
            return AttributeFloat(r[0])
        else:
            return np.array(r)

    def power_flow_Jacobian_km(self, var_values, t=0):
        """
        Constructs Jacobian of (Pkm, Qkm) at the given 
        point and time.

        Parameters
        ----------
        var_values : |Array|
        t : int

        Returns
        -------
        J : |CooMatrix|
        """

        cdef np.ndarray[double,mode='c'] x = var_values
        cdef cvec.Vec* v = cvec.VEC_new_from_array(<cnet.REAL*>(x.data),x.size)
        m = Matrix(cbranch.BRANCH_power_flow_Jacobian(self._c_ptr, v, t, True), owndata=True)
        free(v)
        return m

    def power_flow_Jacobian_mk(self, var_values, t=0):
        """
        Constructs Jacobian of (Pmk, Qmk) at the given 
        point and time.

        Parameters
        ----------
        var_values : |Array|
        t : int

        Returns
        -------
        J : |CooMatrix|
        """

        cdef np.ndarray[double,mode='c'] x = var_values
        cdef cvec.Vec* v = cvec.VEC_new_from_array(<cnet.REAL*>(x.data),x.size)
        m = Matrix(cbranch.BRANCH_power_flow_Jacobian(self._c_ptr, v, t, False), owndata=True)
        free(v)
        return m

    property name:
        """ Branch name (string). """
        def __get__(self):
            return cbranch.BRANCH_get_name(self._c_ptr).decode('UTF-8')
        def __set__(self,name):
            name = name.encode('UTF-8')
            cbranch.BRANCH_set_name(self._c_ptr,name)

    property num_periods:
        """ Number of time periods (int). """
        def __get__(self): return cbranch.BRANCH_get_num_periods(self._c_ptr)

    property obj_type:
        """ Object type (string). """
        def __get__(self): return obj2str[cbranch.BRANCH_get_obj_type(self._c_ptr)]

    property index:
        """ Branch index (int). """
        def __get__(self): return cbranch.BRANCH_get_index(self._c_ptr)

    property index_ratio:
        """ Index of transformer tap ratio variable (int or |Array|). """
        def __get__(self):
            return IntArray(cbranch.BRANCH_get_index_ratio_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        
    property index_phase:
        """ Index of transformer phase shift variable (int or |Array|). """
        def __get__(self):
            return IntArray(cbranch.BRANCH_get_index_phase_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        
    property ratio:
        """ Transformer tap ratio (float or |Array|). """
        def __get__(self):
            return DoubleArray(cbranch.BRANCH_get_ratio_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        def __set__(self, v):
            DoubleArray(cbranch.BRANCH_get_ratio_array(self._c_ptr), self.num_periods)[:] = v

    property ratio_max:
        """ Transformer tap ratio upper limit (float). """
        def __get__(self): return cbranch.BRANCH_get_ratio_max(self._c_ptr)
        def __set__(self,value): cbranch.BRANCH_set_ratio_max(self._c_ptr,value)

    property ratio_min:
        """ Transformer tap ratio lower limit (float). """
        def __get__(self): return cbranch.BRANCH_get_ratio_min(self._c_ptr)
        def __set__(self,value): cbranch.BRANCH_set_ratio_min(self._c_ptr,value)

    property bus_k:
        """ |Bus| connected to the "k" side. """
        def __get__(self):
            return new_Bus(cbranch.BRANCH_get_bus_k(self._c_ptr))
        def __set__(self,bus): 
            cdef Bus cbus
            if not isinstance(bus,Bus) and bus is not None:
                raise BranchError('Not a Bus type object')
            cbus = bus
            cbranch.BRANCH_set_bus_k(self._c_ptr,cbus._c_ptr if bus is not None else NULL)

    property bus_m:
        """ |Bus| connected to the "m" side. """
        def __get__(self):
            return new_Bus(cbranch.BRANCH_get_bus_m(self._c_ptr))
        def __set__(self,bus): 
            cdef Bus cbus
            if not isinstance(bus,Bus) and bus is not None:
                raise BranchError('Not a Bus type object')
            cbus = bus
            cbranch.BRANCH_set_bus_m(self._c_ptr,cbus._c_ptr if bus is not None else NULL)

    property reg_bus:
        """ |Bus| whose voltage is regulated by this tap-changing transformer. """
        def __get__(self):
            return new_Bus(cbranch.BRANCH_get_reg_bus(self._c_ptr))
        def __set__(self,bus): 
            cdef Bus cbus
            if not isinstance(bus,Bus) and bus is not None:
                raise BranchError('Not a Bus type object')
            cbus = bus
            cbranch.BRANCH_set_reg_bus(self._c_ptr,cbus._c_ptr if bus is not None else NULL)

    property b:
        """ Branch series susceptance (p.u.) (float). """
        def __get__(self): return cbranch.BRANCH_get_b(self._c_ptr)
        def __set__(self,value): cbranch.BRANCH_set_b(self._c_ptr,value)

    property b_k:
        """ Branch shunt susceptance at the "k" side (p.u.) (float). """
        def __get__(self): return cbranch.BRANCH_get_b_k(self._c_ptr)
        def __set__(self,value): cbranch.BRANCH_set_b_k(self._c_ptr,value)

    property b_m:
        """ Branch shunt susceptance at the "m" side (p.u.) (float). """
        def __get__(self): return cbranch.BRANCH_get_b_m(self._c_ptr)
        def __set__(self,value): cbranch.BRANCH_set_b_m(self._c_ptr,value)

    property g:
        """ Branch series conductance (p.u.) (float). """
        def __get__(self): return cbranch.BRANCH_get_g(self._c_ptr)
        def __set__(self,value): cbranch.BRANCH_set_g(self._c_ptr,value)

    property g_k:
        """ Branch shunt conductance at the "k" side (p.u.) (float). """
        def __get__(self): return cbranch.BRANCH_get_g_k(self._c_ptr)
        def __set__(self,value): cbranch.BRANCH_set_g_k(self._c_ptr,value)

    property g_m:
        """ Branch shunt conductance at the "m" side (p.u.) (float). """
        def __get__(self): return cbranch.BRANCH_get_g_m(self._c_ptr)
        def __set__(self,value): cbranch.BRANCH_set_g_m(self._c_ptr,value)

    property phase:
        """ Transformer phase shift (radians) (float or |Array|). """
        def __get__(self):
            return DoubleArray(cbranch.BRANCH_get_phase_array(self._c_ptr), self.num_periods, owndata=False, toscalar=True)
        def __set__(self, v):
            DoubleArray(cbranch.BRANCH_get_phase_array(self._c_ptr), self.num_periods)[:] = v

    property phase_max:
        """ Transformer phase shift upper limit (radians) (float). """
        def __get__(self): return cbranch.BRANCH_get_phase_max(self._c_ptr)
        def __set__(self,value): cbranch.BRANCH_set_phase_max(self._c_ptr,value)

    property phase_min:
        """ Transformer phase shift lower limit (radians) (float). """
        def __get__(self): return cbranch.BRANCH_get_phase_min(self._c_ptr)
        def __set__(self,value): cbranch.BRANCH_set_phase_min(self._c_ptr,value)
        
    property P_max:
        """ Maximum active power flow (p.u.) (float). """
        def __get__(self): return cbranch.BRANCH_get_P_max(self._c_ptr)
        def __set__(self,value): cbranch.BRANCH_set_P_max(self._c_ptr,value)
        
    property P_min:
        """ Minimum active power flow (p.u.) (float). """
        def __get__(self): return cbranch.BRANCH_get_P_min(self._c_ptr)
        def __set__(self,value): cbranch.BRANCH_set_P_min(self._c_ptr,value)
        
    property Q_max:
        """ Maximum reactive power flow (p.u.) (float). """
        def __get__(self): return cbranch.BRANCH_get_Q_max(self._c_ptr)
        def __set__(self,value): cbranch.BRANCH_set_Q_max(self._c_ptr,value)
        
    property Q_min:
        """ Minimum reactive power flow (p.u.) (float). """
        def __get__(self): return cbranch.BRANCH_get_Q_min(self._c_ptr)
        def __set__(self,value): cbranch.BRANCH_set_Q_min(self._c_ptr,value)

    property i_km_mag:
        """ Branch current magnitude at bus "k" towards bus "m" (p.u.) (float or |Array|). """
        def __get__(self):
            return self.get_i_km_mag()

    property i_mk_mag:
        """ Branch current magnitude at bus "m" towards bus "k" (p.u.) (float or |Array|). """
        def __get__(self):
            return self.get_i_mk_mag()

    property S_km_mag:
        """ Branch apparent power magnitude at bus "k" towards bus "m" (p.u.) (float or |Array|). """
        def __get__(self):
            return self.get_S_km_mag()

    property S_mk_mag:
        """ Branch apparent power magnitude at bus "m" towards bus "k" (p.u.) (float or |Array|). """
        def __get__(self):
            return self.get_S_mk_mag()

    property P_km:
        """ Real power flow at bus "k" towards bus "m" (p.u.) (float or |Array|). """
        def __get__(self):
            return self.get_P_km()

    property Q_km:
        """ Reactive power flow at bus "k" towards bus "m" (p.u.) (float or |Array|). """
        def __get__(self):
            return self.get_Q_km()

    property P_mk:
        """ Real power flow at bus "m" towards bus "k" (p.u.) (float or |Array|). """
        def __get__(self):
            return self.get_P_mk()

    property Q_mk:
        """ Reactive power flow at bus "m" towards bus "k" (p.u.) (float or |Array|). """
        def __get__(self):
             return self.get_Q_mk()

    property P_km_series:
        """ Real power flow at bus "k" towards bus "m" over the series impedance of the line (p.u.) (float or |Array|). """
        def __get__(self):
            return self.get_P_km_series()

    property Q_km_series:
        """ Reactive power flow at bus "k" towards bus "m" over the series impedance of the line (p.u.) (float or |Array|). """
        def __get__(self):
            return self.get_Q_km_series()

    property P_mk_series:
        """ Real power flow at bus "m" towards bus "k" over the series impedance of the line (p.u.) (float or |Array|). """
        def __get__(self):
            return self.get_P_mk_series()

    property Q_mk_series:
        """ Reactive power flow at bus "m" towards bus "k" over the series impedance of the line (p.u.) (float or |Array|). """
        def __get__(self):
            return self.get_Q_mk_series()

    property P_k_shunt:
        """ Real power flow into the shunt element at bus "k" (p.u.) (float or |Array|). """
        def __get__(self):
            return self.get_P_k_shunt()

    property Q_k_shunt:
        """ Reactive power flow into the shunt element bus "k" (p.u.) (float or |Array|). """
        def __get__(self):
            return self.get_Q_k_shunt()

    property P_m_shunt:
        """ Real power flow into the shunt element at bus "m" (p.u.) (float or |Array|). """
        def __get__(self):
            return self.get_P_m_shunt()

    property Q_m_shunt:
        """ Reactive power flow into the shunt element at bus "m" (p.u.) (float or |Array|). """
        def __get__(self):
            return self.get_Q_m_shunt()

    property ratingA:
        """ Branch thermal rating A (p.u. system base power) (float). """
        def __get__(self): return cbranch.BRANCH_get_ratingA(self._c_ptr)
        def __set__(self,r): cbranch.BRANCH_set_ratingA(self._c_ptr,r)

    property ratingB:
        """ Branch thermal rating B (p.u. system base power) (float). """
        def __get__(self): return cbranch.BRANCH_get_ratingB(self._c_ptr)
        def __set__(self,r): cbranch.BRANCH_set_ratingB(self._c_ptr,r)

    property ratingC:
        """ Branch thermal rating C (p.u. system base power) (float). """
        def __get__(self): return cbranch.BRANCH_get_ratingC(self._c_ptr)
        def __set__(self,r): cbranch.BRANCH_set_ratingC(self._c_ptr,r)

    property P_km_DC:
        """ Active power flow (DC approx.) from bus "k" to bus "m" (float or |Array|). """
        def __get__(self):
            r = [cbranch.BRANCH_get_P_km_DC(self._c_ptr,t) for t in range(self.num_periods)]
            if self.num_periods == 1:
                return AttributeFloat(r[0])
            else:
                return np.array(r)

    property P_mk_DC:
        """ Active power flow (DC approx.) from bus "m" to bus "k" (float or |Array|). """
        def __get__(self):
            r = [cbranch.BRANCH_get_P_mk_DC(self._c_ptr,t) for t in range(self.num_periods)]
            if self.num_periods == 1:
                return AttributeFloat(r[0])
            else:
                return np.array(r)

    property sens_P_u_bound:
        """ Objective function sensitivity with respect to active power flow upper bound (float or |Array|). """
        def __get__(self): return DoubleArray(cbranch.BRANCH_get_sens_P_u_bound_array(self._c_ptr),
                                              cbranch.BRANCH_get_num_periods(self._c_ptr))
        def __set__(self,x):
            self.sens_P_u_bound[:] = x

    property sens_P_l_bound:
        """ Objective function sensitivity with respect to active power flow lower bound (float or |Array|). """
        def __get__(self): return DoubleArray(cbranch.BRANCH_get_sens_P_l_bound_array(self._c_ptr),
                                              cbranch.BRANCH_get_num_periods(self._c_ptr))
        def __set__(self,x):
            self.sens_P_l_bound[:] = x

    property sens_ratio_u_bound:
        """ Objective function sensitivity with respect to tap ratio upper bound (float or |Array|). """
        def __get__(self): return DoubleArray(cbranch.BRANCH_get_sens_ratio_u_bound_array(self._c_ptr),
                                              cbranch.BRANCH_get_num_periods(self._c_ptr))
        def __set__(self,x):
            self.sens_ratio_u_bound[:] = x

    property sens_ratio_l_bound:
        """ Objective function sensitivity with respect to tap ratio lower bound (float or |Array|). """
        def __get__(self): return DoubleArray(cbranch.BRANCH_get_sens_ratio_l_bound_array(self._c_ptr),
                                              cbranch.BRANCH_get_num_periods(self._c_ptr))
        def __set__(self,x):
            self.sens_ratio_l_bound[:] = x

    property sens_phase_u_bound:
        """ Objective function sensitivity with respect to phase shift upper bound (float or |Array|). """
        def __get__(self): return DoubleArray(cbranch.BRANCH_get_sens_phase_u_bound_array(self._c_ptr),
                                              cbranch.BRANCH_get_num_periods(self._c_ptr))
        def __set__(self,x):
            self.sens_phase_u_bound[:] = x

    property sens_phase_l_bound:
        """ Objective function sensitivity with respect to phase shift lower bound (float or |Array|). """
        def __get__(self): return DoubleArray(cbranch.BRANCH_get_sens_phase_l_bound_array(self._c_ptr),
                                              cbranch.BRANCH_get_num_periods(self._c_ptr))
        def __set__(self,x):
            self.sens_phase_l_bound[:] = x

    property sens_i_mag_u_bound:
        """ Objective function sensitivity with respect to current magnitude upper bound (float or |Array|). """
        def __get__(self): return DoubleArray(cbranch.BRANCH_get_sens_i_mag_u_bound_array(self._c_ptr),
                                              cbranch.BRANCH_get_num_periods(self._c_ptr))
        def __set__(self,x):
            self.sens_i_mag_u_bound[:] = x

    property outage:
        """ Flag that indicates whether branch is on outage (boolean). """
        def __get__(self): return cbranch.BRANCH_is_on_outage(self._c_ptr)
        def __set__(self, o): cbranch.BRANCH_set_outage(self._c_ptr, o);

    property json_string:
        """ JSON string (string). """
        def __get__(self): 
            cdef char* json_string = cbranch.BRANCH_get_json_string(self._c_ptr, NULL)
            s = json_string.decode('UTF-8')
            free(json_string)
            return s

    property y_correction:
        """ Y correction table (|BranchYCorrection|). """
        def __get__(self):
            return new_BranchYCorrection(cbranch.BRANCH_get_y_correction(self._c_ptr)) 

    property flags_vars:
        """ Flags associated with variable quantities (byte). """
        def __get__(self): return cbranch.BRANCH_get_flags_vars(self._c_ptr)

    property flags_fixed:
        """ Flags associated with fixed quantities (byte). """
        def __get__(self): return cbranch.BRANCH_get_flags_fixed(self._c_ptr)

    property flags_bounded:
        """ Flags associated with bounded quantities (byte). """
        def __get__(self): return cbranch.BRANCH_get_flags_bounded(self._c_ptr)

    property flags_sparse:
        """ Flags associated with sparse quantities (byte). """
        def __get__(self): return cbranch.BRANCH_get_flags_sparse(self._c_ptr)

cdef new_Branch(cbranch.Branch* b):
    if b is not NULL:
        branch = Branch(alloc=False)
        branch._c_ptr = b
        return branch
    else:
        raise BranchError('no branch data')
        
