#cython: embedsignature=True

#***************************************************#
# This file is part of PFNET.                       #
#                                                   #
# Copyright (c) 2015, Tomas Tinoco De Rubira.       #
#                                                   #
# PFNET is released under the BSD 2-clause license. #
#***************************************************#

cimport cbus_dc

# Infinite
BUSDC_INF_V = cbus_dc.BUSDC_INF_V

class BusDCError(Exception):
    """
    Bus error exception.
    """

    pass

cdef class BusDC:
    """
    DC bus class.
    """

    cdef cbus_dc.BusDC* _c_ptr
    cdef bint alloc

    def __init__(self, num_periods=1, alloc=True):
        """
        DC bus class.

        Parameters
        ----------
        num_periods : int
        alloc : |TrueFalse|
        """

        pass

    def __cinit__(self, num_periods=1, alloc=True):

        if alloc:
            self._c_ptr = cbus_dc.BUSDC_new(num_periods)
        else:
            self._c_ptr = NULL
        self.alloc = alloc

    def __dealloc__(self):

        if self.alloc:
            cbus_dc.BUSDC_array_del(self._c_ptr,1)
            self._c_ptr = NULL        
        
    def _get_c_ptr(self):

        return new_CPtr(self._c_ptr)

    def is_equal(self, other):
        """
        Determines whether the DC bus is equal to given DC bus.

        Parameters
        ----------
        other : |BusDC|

        Returns
        -------
        flag : |TrueFalse|
        """

        cdef BusDC b_other

        if not isinstance(other,BusDC):
            return False

        b_other = other

        return cbus_dc.BUSDC_is_equal(self._c_ptr,b_other._c_ptr)

    def has_flags(self, flag_type, q):
        """
        Determines whether the DC bus has the flags associated with
        certain quantities set.

        Parameters
        ----------
        flag_type : string (|RefFlags|)
        q : string or list of strings (|RefBusDCQuantities|)

        Returns
        -------
        flag : |TrueFalse|
        """

        q = q if isinstance(q,list) else [q]

        return cbus_dc.BUSDC_has_flags(self._c_ptr,
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

        cdef char* info_string = cbus_dc.BUSDC_get_var_info_string(self._c_ptr, index)
        if info_string:
            s = info_string.decode('UTF-8')
            free(info_string)
            return s
        else:
            raise BusDCError('index does not correspond to any variable')

    def get_num_vars(self, q, t_start=0, t_end=None):
        """
        Gets number of variables associated with the
        given quantities.

        Parameters
        ----------
        q : string or list of strings (|RefBusDCQuantities|)
        t_start : int
        t_end : int

        Returns
        -------
        num : int
        """

        q = q if isinstance(q,list) else [q]

        if t_end is None:
            t_end = self.num_periods-1
        return cbus_dc.BUSDC_get_num_vars(self._c_ptr,
                                     reduce(lambda x,y: x|y,[str2q[self.obj_type][qq] for qq in q],0),
                                     t_start,
                                     t_end)

    def set_v(self, v, t=0):
        """
        Sets DC bus voltage.

        Parameters
        ----------
        v : float
        t : int
        """

        cbus_dc.BUSDC_set_v(self._c_ptr,v,t)

    def add_csc_converter(self, conv):
        """
        Adds a CSC converter connection to this bus.
        
        Parameters
        ----------
        conv : |ConverterCSC|
        """
        
        cdef ConverterCSC cconv
        if not isinstance(conv,ConverterCSC):
            raise BusDCError('Not a ConverterCSC type object')
        cconv = conv
        cbus_dc.BUSDC_add_csc_conv(self._c_ptr, cconv._c_ptr)

    def remove_csc_converter(self, conv):
        """
        Removes a CSC converter connection to this bus.
        
        Parameters
        ----------
        conv : |ConverterCSC|
        """
        
        cdef ConverterCSC cconv
        if not isinstance(conv,ConverterCSC):
            raise BusDCError('Not a ConverterCSC type object')
        cconv = conv
        cbus_dc.BUSDC_del_csc_conv(self._c_ptr, cconv._c_ptr)

    def add_vsc_converter(self, conv):
        """
        Adds a VSC converter connection to this bus.
        
        Parameters
        ----------
        conv : |ConverterVSC|
        """
        
        cdef ConverterVSC cconv
        if not isinstance(conv,ConverterVSC):
            raise BusDCError('Not a ConverterVSC type object')
        cconv = conv
        cbus_dc.BUSDC_add_vsc_conv(self._c_ptr, cconv._c_ptr)

    def remove_vsc_converter(self, conv):
        """
        Removes a VSC converter connection to this bus.
        
        Parameters
        ----------
        conv : |ConverterVSC|
        """
        
        cdef ConverterVSC cconv
        if not isinstance(conv,ConverterVSC):
            raise BusDCError('Not a ConverterVSC type object')
        cconv = conv
        cbus_dc.BUSDC_del_vsc_conv(self._c_ptr, cconv._c_ptr)
    
    def add_branch_k(self, branch):
        """
        Adds a "k" branch connection to this bus.
        
        Parameters
        ----------
        branch : |BranchDC|
        """
        
        cdef BranchDC cbranch
        if not isinstance(branch, BranchDC):
            raise BusDCError('Not a BranchDC type object')
        cbranch = branch
        cbus_dc.BUSDC_add_branch_k(self._c_ptr, cbranch._c_ptr)

    def remove_branch_k(self, branch):
        """
        Removes a "k" branch connection to this bus.
        
        Parameters
        ----------
        branch : |BranchDC|
        """
        
        cdef BranchDC cbranch
        if not isinstance(branch, BranchDC):
            raise BusDCError('Not a BranchDC type object')
        cbranch = branch
        cbus_dc.BUSDC_del_branch_k(self._c_ptr, cbranch._c_ptr)
    
    def add_branch_m(self, branch):
        """
        Adds an "m" branch connection to this bus.
        
        Parameters
        ----------
        branch : |BranchDC|
        """
        
        cdef BranchDC cbranch
        if not isinstance(branch,BranchDC):
            raise BusDCError('Not a BranchDC type object')
        cbranch = branch
        cbus_dc.BUSDC_add_branch_m(self._c_ptr, cbranch._c_ptr)

    def remove_branch_m(self, branch):
        """
        Removes an "m" branch connection to this bus.
        
        Parameters
        ----------
        branch : |BranchDC|
        """
        
        cdef BranchDC cbranch
        if not isinstance(branch,BranchDC):
            raise BusDCError('Not a BranchDC type object')
        cbranch = branch
        cbus_dc.BUSDC_del_branch_m(self._c_ptr, cbranch._c_ptr)
    
    def remove_all_connections(self):
        """
        Removes all connections to this DC bus.
        """

        cbus_dc.BUSDC_del_all_connections(self._c_ptr)

    def __richcmp__(self, other, op):
        """
        Compares two DC buses.

        Parameters
        ----------
        other : |BusDC|
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
        
    property num_periods:
        """ Number of time periods (int). """
        def __get__(self): return cbus_dc.BUSDC_get_num_periods(self._c_ptr)

    property obj_type:
        """ Object type (string). """
        def __get__(self): return obj2str[cbus_dc.BUSDC_get_obj_type(self._c_ptr)]

    property index:
        """ Bus index (int). """
        def __get__(self): return cbus_dc.BUSDC_get_index(self._c_ptr)

    property index_t:
        """ Unique indices for bus and time (int). """
        def __get__(self):
            r = [cbus_dc.BUSDC_get_index_t(self._c_ptr,t) for t in range(self.num_periods)]
            if self.num_periods == 1:
                return AttributeInt(r[0])
            else:
                return np.array(r)

    property index_v:
        """ Index of voltage variable (int or |Array|). """
        def __get__(self):
            r = [cbus_dc.BUSDC_get_index_v(self._c_ptr,t) for t in range(self.num_periods)]
            if self.num_periods == 1:
                return AttributeInt(r[0])
            else:
                return np.array(r)

    property number:
        """ Bus number (int). """
        def __get__(self):
            return cbus_dc.BUSDC_get_number(self._c_ptr)
        def __set__(self,num):
            cbus_dc.BUSDC_set_number(self._c_ptr,num)

    property name:
        """ Bus name (string). """
        def __get__(self):
            name = cbus_dc.BUSDC_get_name(self._c_ptr)
            if name != NULL:
                return name.decode('UTF-8')
            else:
                return ""
        def __set__(self,name):
            name = name.encode('UTF-8')
            cbus_dc.BUSDC_set_name(self._c_ptr,name)
            
    property json_string:
        """ JSON string (string). """
        def __get__(self): 
            cdef char* json_string = cbus_dc.BUSDC_get_json_string(self._c_ptr, NULL)
            s = json_string.decode('UTF-8')
            free(json_string)
            return s

    property v_base:
        """ DC bus base voltage (kilo-volts) (float). """
        def __get__(self):
            return cbus_dc.BUSDC_get_v_base(self._c_ptr)
        def __set__(self,value):
            cbus_dc.BUSDC_set_v_base(self._c_ptr,value)

    property v:
        """ DC bus voltage (p.u. bus base kv) (float or |Array|). """
        def __get__(self):
            r = [cbus_dc.BUSDC_get_v(self._c_ptr,t) for t in range(self.num_periods)]
            if self.num_periods == 1:
                return AttributeFloat(r[0])
            else:
                return AttributeArray(r,self.set_v)
        def __set__(self,v):
            cdef int t
            cdef np.ndarray var = np.array(v).flatten()
            for t in range(np.minimum(var.size,self.num_periods)):
                cbus_dc.BUSDC_set_v(self._c_ptr,var[t],t)

    property P_mismatch:
        """ DC bus power mismatch (p.u. system base MVA) (float or |Array|). """
        def __get__(self):
            r = [cbus_dc.BUSDC_get_P_mis(self._c_ptr,t) for t in range(self.num_periods)]
            if self.num_periods == 1:
                return AttributeFloat(r[0])
            else:
                return np.array(r)

    property branches_k:
        """ List of |BranchDC| objects that have this DC bus on the "k" side (list). """
        def __get__(self):
            branches = []
            cdef cbranch_dc.BranchDC* br = cbus_dc.BUSDC_get_branch_k(self._c_ptr)
            while br is not NULL:
                branches.append(new_BranchDC(br))
                br = cbranch_dc.BRANCHDC_get_next_k(br)
            return branches

    property branches_m:
        """ List of |BranchDC| objects that have this bus on the "m" side (list). """
        def __get__(self):
            branches = []
            cdef cbranch_dc.BranchDC* br = cbus_dc.BUSDC_get_branch_m(self._c_ptr)
            while br is not NULL:
                branches.append(new_BranchDC(br))
                br = cbranch_dc.BRANCHDC_get_next_m(br)
            return branches

    property branches:
        """ List of |BranchDC| objects incident on this bus (list). """
        def __get__(self):
            return self.branches_k+self.branches_m

    property csc_converters:
        """ List of |ConverterCSC| objects connected to this bus (list). """
        
        def __get__(self):
            convs = []
            cdef cconv_csc.ConvCSC* c = cbus_dc.BUSDC_get_csc_conv(self._c_ptr)
            while c is not NULL:
                convs.append(new_ConverterCSC(c))
                c = cconv_csc.CONVCSC_get_next_dc(c)
            return convs

    property vsc_converters:
        """ List of |ConverterVSC| objects connected to this bus (list). """
        def __get__(self):
            convs = []
            cdef cconv_vsc.ConvVSC* c = cbus_dc.BUSDC_get_vsc_conv(self._c_ptr)
            while c is not NULL:
                convs.append(new_ConverterVSC(c))
                c = cconv_vsc.CONVVSC_get_next_dc(c)
            return convs

    property flags_vars:
        """ Flags associated with variable quantities (byte). """
        def __get__(self): return cbus_dc.BUSDC_get_flags_vars(self._c_ptr)

    property flags_fixed:
        """ Flags associated with fixed quantities (byte). """
        def __get__(self): return cbus_dc.BUSDC_get_flags_fixed(self._c_ptr)

    property flags_bounded:
        """ Flags associated with bounded quantities (byte). """
        def __get__(self): return cbus_dc.BUSDC_get_flags_bounded(self._c_ptr)

    property flags_sparse:
        """ Flags associated with sparse quantities (byte). """
        def __get__(self): return cbus_dc.BUSDC_get_flags_sparse(self._c_ptr)

cdef new_BusDC(cbus_dc.BusDC* b):
    if b is not NULL:
        bus = BusDC(alloc=False)
        bus._c_ptr = b
        return bus
    else:
        return None
