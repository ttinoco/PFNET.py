import pfnet
import numpy as np

class PyParserMAT(object):

    BUS_TYPE_PQ = 1
    BUS_TYPE_PV = 2
    BUS_TYPE_SL = 3
    BUS_TYPE_IS =4

    def __init__(self):

        pass

    def parse(self, filename, num_periods=1):

        import grg_mpdata as mp

        case = mp.io.parse_mp_case_file(filename)

        net = pfnet.Network(num_periods=num_periods)

        # Base power
        net.base_power = case.baseMVA

        # Buses
        net.set_bus_array(len([b for b in case.bus if b.bus_type != self.BUS_TYPE_IS]))
        index = 0
        for mat_bus in case.bus:
            if mat_bus.bus_type == self.BUS_TYPE_IS:
                continue
            bus = net.get_bus(index)
            bus.number = mat_bus.bus_i
            bus.area = mat_bus.area
            bus.zone = mat_bus.zone
            bus.name = "BUS %d" %mat_bus.bus_i
            bus.v_mag = mat_bus.vm
            bus.v_ang = mat_bus.va*np.pi/180.
            bus.v_base = mat_bus.base_kv
            bus.v_max_norm = mat_bus.vmax
            bus.v_min_norm = mat_bus.vmin
            if mat_bus.bus_type == self.BUS_TYPE_SL:
                bus.set_slack_flag(True)
            index += 1

        # Hashes
        net.update_hashes()
        
        # Load

        # Shunts

        # Gens

        # Branches

        # Costs

        net.show_components()

    def show(self):

        pass

    def write(self, net, filename):

        pass
