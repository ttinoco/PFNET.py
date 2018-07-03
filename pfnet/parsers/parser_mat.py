import pfnet
import numpy as np

class PyParserMAT(object):

    BUS_TYPE_PQ = 1
    BUS_TYPE_PV = 2
    BUS_TYPE_SL = 3
    BUS_TYPE_IS = 4

    def __init__(self):

        pass

    def parse(self, filename, num_periods=1):

        import grg_mpdata as mp

        case = mp.io.parse_mp_case_file(filename)

        net = pfnet.Network(num_periods=num_periods)

        # Base power
        net.base_power = case.baseMVA

        # Numbers
        num_buses = 0
        num_loads = 0
        num_shunts = 0
        for mat_bus in case.bus:
            if mat_bus.bus_type != self.BUS_TYPE_IS:
                num_buses += 1
                if mat_bus.pd != 0. or mat_bus.qd != 0.:
                    num_loads += 1
                if mat_bus.gs != 0. or mat_bus.bs != 0.:
                    num_shunts += 1                
                
        # Buses
        bus_index = 0
        net.set_bus_array(num_buses)
        for mat_bus in case.bus:
            if mat_bus.bus_type != self.BUS_TYPE_IS:
                bus = net.get_bus(bus_index)
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
                bus_index += 1

        # Hashes
        net.update_hashes()
        
        # Load and shunts
        load_index = 0
        shunt_index = 0
        net.set_load_array(num_loads)
        net.set_shunt_array(num_shunts)
        for mat_bus in case.bus:
            if mat_bus.bus_type != self.BUS_TYPE_IS:

                # Load
                if mat_bus.pd != 0. or mat_bus.qd != 0.:
                    bus = net.get_bus_from_number(mat_bus.bus_i)
                    load = net.get_load(load_index)
                    bus.add_load(load)
                    assert(bus.is_equal(load.bus))
                    load.P = mat_bus.pd/net.base_power
                    load.Q = mat_bus.qd/net.base_power
                    load.P_max = load.P
                    load.P_min = load.P
                    load.Q_max = load.Q
                    load.Q_min = load.Q
                    load_index += 1
                
                # Shunt
                if mat_bus.gs != 0. or mat_bus.bs != 0.:
                    bus = net.get_bus_from_number(mat_bus.bus_i)
                    shunt = net.get_shunt(shunt_index)
                    bus.add_shunt(shunt)
                    assert(bus.is_equal(shunt.bus))
                    shunt.g = mat_bus.gs/net.base_power
                    shunt.g = mat_bus.bs/net.base_power
                    shunt.b_max = shunt.b
                    shunt.b_min = shunt.b
                    shunt_index += 1
        # Gens
        net.set_gen_array(len([g for g in case.gen if g.gen_status > 0]))
        gen_index = 0
        for mat_gen in case.gen:
            if mat_gen.gen_status > 0:
                bus = net.get_bus_from_number(mat_gen.gen_bus)
                gen = net.get_generator(gen_index)
                bus.add_generator(gen)
                assert(bus.is_equal(gen.bus))
                gen.P = mat_gen.pg/net.base_power
                gen.P_max = mat_gen.pmax/net.base_power
                gen.P_min = mat_gen.pmin/net.base_power
                gen.Q = mat_gen.qg/net.base_power
                gen.Q_max = mat_gen.qmax/net.base_power
                gen.Q_min = mat_gen.qmin/net.base_power
                if bus.is_slack() or gen.Q_max > gen.Q_min:
                    gen.reg_bus = bus
                    assert(gen.index in [g.index for g in bus.reg_generators])
                    bus.v_set = mat_gen.vg
                gen_index += 1
                
        # Branches
        net.set_branch_array(len([br for br in case.branch if br.br_status > 0]))
        br_index = 0
        for mat_br in case.branch:
            if mat_br.br_status > 0:
                bus_k = net.get_bus_from_number(mat_br.f_bus)
                bus_m = net.get_bus_from_number(mat_br.t_bus)
                branch = net.get_branch(br_index)
                den = mat_br.br_r**2.+mat_br.br_x**2.
                g = mat_br.br_r/den
                b = -mat_br.br_x/den
                if mat_br.tap > 0:
                    t = mat_br.tap
                else:
                    t = 1.
                z = mat_br.shift*np.pi/180.
                if t == 1. and z == 0.:
                    branch.set_as_line()
                else:
                    branch.set_as_fixed_tran()
                branch.bus_k = bus_k
                branch.bus_m = bus_m
                assert(branch.index in [br.index for br in bus_k.branches_k])
                assert(branch.index in [br.index for br in bus_m.branches_m])
                branch.ratio = 1./t
                branch.ratio_max = branch.ratio
                branch.ratio_min = branch.ratio
                branch.phase = z
                branch.phase_max = z
                branch.phase_min = z
                branch.g = g
                branch.b = b
                branch.b_k = mat_br.br_b/2.
                branch.b_m = mat_br.br_b/2.
                branch.ratingA = mat_br.rate_a/net.base_power
                branch.ratingB = mat_br.rate_b/net.base_power
                branch.ratingC = mat_br.rate_c/net.base_power
                br_index += 1
                
        # Gen costs
        
        

        # Return
        return net

    def show(self):

        pass

    def write(self, net, filename):

        pass
