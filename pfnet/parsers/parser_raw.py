import os
import pfnet
import numpy as np

class PyParserRAW(object):
    """
    Class for parsing .raw files.
    """

    BUS_TYPE_PQ = 1
    BUS_TYPE_PV = 2
    BUS_TYPE_SL = 3
    BUS_TYPE_IS = 4

    def __init__(self):
        """
        Parser for parsing .raw files.
        """

        # grg-pssedata case
        self.case = None 

        # Options
        self.keep_all_oos = False # flag for keeping all out-of-service components

    def set(self, key, value):
        """
        Sets parser parameter.

        Parameters
        ----------
        key : string
        value : float
        """
        
        if key == 'keep_all_out_of_service':
            self.keep_all_oos = value
        else:
            raise ValueError('invalid parser parameter %s' %key)

    def parse(self, filename, num_periods=None):
        """
        Parses .raw file.
        
        Parameters
        ----------
        filename : string
        num_periods : int

        Returns
        -------
        net : |Network|
        """

        import grg_pssedata as pd

        # Check extension
        if os.path.splitext(filename)[-1][1:] != 'raw':
            raise pfnet.ParserError('invalid file extension')

        # Get grg-pssedata case
        case = pd.io.parse_psse_case_file(filename)
        self.case = case

        if num_periods is None:
            num_periods = 1
        
        net = pfnet.Network(num_periods=num_periods)

        # Raw bus hash table (number -> raw bus)
        num2rawbus = {}
        for raw_bus in case.buses:
            num2rawbus[raw_bus.i] = raw_bus
            raw_bus.star = False

        # Create star buses and add to raw buses list
        tran2star = {} # maps (raw transformer index -> raw_bus)
        max_bus_number = max([rb.i for rb in case.buses]+[0])
        for raw_tran in case.transformers:
            if isinstance(raw_tran, pd.io.ThreeWindingTransformer):
                raw_bus_i = num2rawbus[raw_tran.p1.i]
                raw_bus = pd.io.Bus(max_bus_number+1,   # number
                                    '',                 # name
                                    raw_bus_i.basekv,   # base kv
                                    self.BUS_TYPE_PQ if raw_tran.p1.stat > 0 else self.BUS_TYPE_IS, # type
                                    raw_bus_i.area,     # area
                                    raw_bus_i.zone,     # zone
                                    raw_bus_i.owner,    # owner
                                    raw_tran.p2.vmstar, # vm
                                    raw_tran.p2.anstar, # va
                                    1.1,                # nvhi
                                    0.9,                # nvlo
                                    1.1,                # evhi
                                    0.9)                # evlo
                raw_bus.star = True
                tran2star[raw_tran.index] = raw_bus
                case.buses.append(raw_bus)
                max_bus_number += 1

        # PFNET base power
        net.base_power = case.sbase

        # PFNET buses
        raw_buses = []
        for raw_bus in case.buses:
            if self.keep_all_oos or raw_bus.ide != self.BUS_TYPE_IS:
                raw_buses.append(raw_bus)
        net.set_bus_array(len(raw_buses)) # allocate PFNET bus array
        for index, raw_bus in enumerate(reversed(raw_buses)):
            bus = net.get_bus(index)
            bus.number = raw_bus.i
            bus.in_service = raw_bus.ide != self.BUS_TYPE_IS
            bus.area = raw_bus.area
            bus.zone = raw_bus.zone
            bus.name = raw_bus.name
            bus.v_mag = raw_bus.vm
            bus.v_ang = raw_bus.va*np.pi/180.
            bus.v_base = raw_bus.basekv
            bus.v_max_norm = raw_bus.nvhi
            bus.v_min_norm = raw_bus.nvlo
            bus.v_max_emer = raw_bus.evhi
            bus.v_min_emer = raw_bus.evlo
            bus.set_slack_flag(raw_bus.ide == self.BUS_TYPE_SL)
            bus.set_star_flag(raw_bus.star)
        net.update_hash_tables()

        # PFNET loads
        raw_loads = []
        for raw_load in case.loads:
            if self.keep_all_oos or (raw_load.status > 0 and num2rawbus[raw_load.i].ide != self.BUS_TYPE_IS):
                raw_loads.append(raw_load)
        net.set_load_array(len(raw_loads)) # allocate PFNET load array
        for index, raw_load in enumerate(reversed(raw_loads)):
            load = net.get_load(index)
            load.bus = net.get_bus_from_number(raw_load.i)
            load.name = raw_load.id
            load.in_service = raw_load.status > 0 and num2rawbus[raw_load.i].ide != self.BUS_TYPE_IS
            load.P = (raw_load.pl+raw_load.ip+raw_load.yp)/net.base_power
            load.Q = (raw_load.ql+raw_load.iq+raw_load.yq)/net.base_power
            load.P_max = load.P
            load.P_min = load.P
            load.Q_max = load.Q
            load.Q_min = load.Q
            load.comp_cp = raw_load.pl/net.base_power
            load.comp_cq = raw_load.ql/net.base_power
            load.comp_ci = raw_load.ip/net.base_power
            load.comp_cj = raw_load.iq/net.base_power
            load.comp_cg = raw_load.yp/net.base_power
            load.comp_cb = raw_load.yq/net.base_power
            
        # PFNET generators

        # PFNET branches

        # PFNET shunts

        # PFNET DC buses


        # PFNET DC branches

        # PFNET CSC HVDC

        # PFNET VSC HVDC

        # PFNET Facts

        # Update properties
        net.update_properties()

        # Return
        return net

    def show(self):

        if self.case is not None:
            print(self.case)

    def write(self, net, filename):
        """
        Writes network to .raw file.

        Parameters
        ----------
        net : |Network|
        filename : string
        """

        pass
