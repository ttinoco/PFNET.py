import os
import pfnet
import numpy as np

class PyParserMAT(object):
    """
    Class for parsing Matpower .m files.
    """
    
    BUS_TYPE_PQ = 1
    BUS_TYPE_PV = 2
    BUS_TYPE_SL = 3
    BUS_TYPE_IS = 4

    def __init__(self):
        """
        Parser for parsing Matpower .m files.
        """

        self.case = None

    def set(self, key, value):
        """
        Sets parser parameter.

        Parameters
        ----------
        key : string
        value : float
        """
        
        pass

    def parse(self, filename, num_periods=None):
        """
        Parses Matpower .m file.
        
        Parameters
        ----------
        filename : string
        num_periods : int

        Returns
        -------
        net : |Network|
        """

        import grg_mpdata as mp

        if os.path.splitext(filename)[-1][1:] != 'm':
            raise pfnet.ParserError('invalid file extension')

        case = mp.io.parse_mp_case_file(filename)
        self.case = case

        if num_periods is None:
            num_periods = 1
        
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
        for mat_bus in reversed(case.bus):
            if mat_bus.bus_type != self.BUS_TYPE_IS:
                bus = net.get_bus(bus_index)
                bus.number = mat_bus.bus_i
                bus.area = mat_bus.area
                bus.zone = mat_bus.zone
                bus.name = "BUS %d" %bus.number
                bus.v_mag = mat_bus.vm
                bus.v_ang = mat_bus.va*np.pi/180.
                bus.v_base = mat_bus.base_kv
                bus.v_max_norm = mat_bus.vmax
                bus.v_min_norm = mat_bus.vmin
                if mat_bus.bus_type == self.BUS_TYPE_SL:
                    bus.set_slack_flag(True)
                bus_index += 1

        # Hashes
        net.update_hash_tables()
        
        # Load and shunts
        load_index = 0
        shunt_index = 0
        net.set_load_array(num_loads)
        net.set_shunt_array(num_shunts)
        for mat_bus in reversed(case.bus):
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
                    load.comp_cp = load.P
                    load.comp_cq = load.Q
                    load_index += 1
                
                # Shunt
                if mat_bus.gs != 0. or mat_bus.bs != 0.:
                    bus = net.get_bus_from_number(mat_bus.bus_i)
                    shunt = net.get_shunt(shunt_index)
                    bus.add_shunt(shunt)
                    assert(bus.is_equal(shunt.bus))
                    shunt.g = mat_bus.gs/net.base_power
                    shunt.b = mat_bus.bs/net.base_power
                    shunt.b_max = shunt.b[0]
                    shunt.b_min = shunt.b[0]
                    shunt_index += 1
        # Gens
        gen_map = {} # mat index -> pfnet index
        gen_index = 0
        net.set_gen_array(len([g for g in case.gen if g.gen_status > 0]))
        for mat_gen in reversed(case.gen):
            if mat_gen.gen_status > 0:
                bus = net.get_bus_from_number(mat_gen.gen_bus)
                gen = net.get_generator(gen_index)
                gen_map[mat_gen.index] = gen_index
                bus.add_generator(gen)
                assert(bus.is_equal(gen.bus))
                gen.name = "%d" %gen.index
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
        for mat_br in reversed(case.branch):
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
                branch.name = "%d" %branch.index
                branch.ratio = 1./t
                branch.ratio_max = branch.ratio[0]
                branch.ratio_min = branch.ratio[0]
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
        for mat_cost in case.gencost:
            if mat_cost.index not in gen_map:
                continue
            gen = net.get_generator(gen_map[mat_cost.index])
            if mat_cost.model != 2:
                print("gen cost model %d not supported" %mat_cost.model)
                continue
            gen.cost_coeff_Q2 = 0.
            gen.cost_coeff_Q1 = 0.
            gen.cost_coeff_Q0 = 0.
            for n, c in zip(range(mat_cost.ncost-1, -1, -1), mat_cost.cost):
                if n == 2:
                    gen.cost_coeff_Q2 = c*(net.base_power**2.)
                if n == 1:
                    gen.cost_coeff_Q1 = c*net.base_power
                if n == 0:
                    gen.cost_coeff_Q0

        # Update props
        net.update_properties()

        # Return
        return net

    def show(self):

        if self.case is not None:
            print(self.case)

    def write(self, net, filename):
        """
        Writes network to Matpower .m file.

        Parameters
        ----------
        net : |Network|
        filename : string
        """

        import grg_mpdata as mp

        case_name = os.path.splitext(os.path.split(filename)[-1])[0]
        case_version = '\'2\''
        case_baseMVA = net.base_power
        case_bus = []
        case_gen = []
        case_branch = []
        case_gencost = []
        case_dcline = []
        case_dclinecost = []
        case_busname = []

        # Buses
        for bus in reversed(net.buses):
            bus_i = bus.number
            if bus.is_slack():
                bus_type = self.BUS_TYPE_SL
            elif bus.is_regulated_by_gen():
                bus_type = self.BUS_TYPE_PV
            else:
                bus_type = self.BUS_TYPE_PQ
            pd, qd = 0., 0.
            for load in bus.loads:
                pd += load.P[0]*net.base_power
                qd += load.Q[0]*net.base_power
            gs, bs = 0., 0.
            for shunt in bus.shunts:
                gs += shunt.g*net.base_power
                bs += shunt.b[0]*net.base_power
            area = bus.area
            vm = bus.v_mag[0]
            va = bus.v_ang[0]*180./np.pi
            base_kv = bus.v_base
            zone = bus.zone
            vmax = bus.v_max_norm
            vmin = bus.v_min_norm
            case_bus.append(mp.struct.Bus(bus_i, bus_type, pd, qd, gs, bs,
                                          area, vm, va, base_kv, zone, vmax, vmin))

        # Generators
        for gen in reversed(net.generators):
            index = gen.index
            gen_bus = gen.bus.number
            pg = gen.P[0]*net.base_power
            qg = gen.Q[0]*net.base_power
            qmax = gen.Q_max*net.base_power
            qmin = gen.Q_min*net.base_power
            vg = gen.bus.v_set[0]
            mbase = net.base_power
            gen_status = int(not gen.is_on_outage())
            pmax = gen.P_max*net.base_power
            pmin = gen.P_min*net.base_power
            case_gen.append(mp.struct.Generator(index, gen_bus, pg, qg, qmax, qmin,
                                                vg, mbase, gen_status, pmax, pmin))

        # Branches
        for branch in reversed(net.branches):
            index = branch.index
            f_bus = branch.bus_k.number
            t_bus = branch.bus_m.number
            br_r = branch.g/(branch.g**2.+branch.b**2.)
            br_x = -branch.b/(branch.g**2.+branch.b**2.)
            br_b = branch.b_k+branch.b_m
            tap = 1./branch.ratio[0]
            shift = branch.phase[0]*180./np.pi
            rate_a = branch.ratingA*net.base_power
            rate_b = branch.ratingB*net.base_power
            rate_c = branch.ratingC*net.base_power
            br_status = int(not branch.is_on_outage())
            case_branch.append(mp.struct.Branch(index, f_bus, t_bus, br_r, br_x, br_b,
                                                rate_a, rate_b, rate_c, tap, shift, br_status))

        # Generator costs
        for gen in reversed(net.generators):
            index = gen.index
            model = 2
            startup = 0
            shutdown = 0
            ncost = 3
            cost = [gen.cost_coeff_Q2/(net.base_power**2.),
                    gen.cost_coeff_Q1/net.base_power,
                    gen.cost_coeff_Q0]
            case_gencost.append(mp.struct.GeneratorCost(index, model, startup, shutdown, ncost, cost))

        case = mp.struct.Case(name=case_name,
                              version=case_version,
                              baseMVA=case_baseMVA,
                              bus=case_bus,
                              gen=case_gen,
                              branch=case_branch,
                              gencost=case_gencost)
        
        f = open(filename, 'w')
        f.write(case.to_matpower())
        f.close()

