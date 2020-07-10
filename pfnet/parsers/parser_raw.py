import os
import pfnet
import numpy as np

class PyParserRAW(object):
    """
    Class for parsing .raw files version 33.
    """

    BUS_TYPE_PQ = 1
    BUS_TYPE_PV = 2
    BUS_TYPE_SL = 3
    BUS_TYPE_IS = 4

    def __init__(self):
        """
        Parser for parsing .raw files version 33.
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
        if os.path.splitext(filename)[-1][1:].lower() != 'raw':
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
            load.in_service = raw_load.status > 0
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
        raw_gens = []
        for raw_gen in case.generators:
            if self.keep_all_oos or (raw_gen.stat > 0 and num2rawbus[raw_gen.i].ide != self.BUS_TYPE_IS):
                raw_gens.append(raw_gen)
        net.set_gen_array(len(raw_gens)) # allocate PFNET gen array
        for index, raw_gen in enumerate(reversed(raw_gens)):          
            gen = net.get_generator(index)           
            gen.in_service = raw_gen.stat > 0
            gen.bus = net.get_bus_from_number(raw_gen.i)      
            gen.name = raw_gen.id
            gen.P = raw_gen.pg/net.base_power
            gen.P_max = raw_gen.pt/net.base_power
            gen.P_min = raw_gen.pb/net.base_power
            gen.Q = raw_gen.qg/net.base_power
            gen.Q_max = raw_gen.qt/net.base_power
            gen.Q_min = raw_gen.qb/net.base_power                          
     
            if gen.bus.is_slack() or raw_gen.ireg == 0:
                gen.reg_bus = gen.bus
                gen.reg_bus.v_set = raw_gen.vs
            else:
                gen.reg_bus = net.get_bus_from_number(raw_gen.ireg)

        # PFNET branches
        raw_branches = []
        
        # Lines
        for raw_line in case.branches:
            if self.keep_all_oos or raw_line.st > 0:
                raw_branches.append(raw_line)
                
        # Transformer
        for raw_tran in case.transformers:
            if self.keep_all_oos or raw_tran.p1.stat > 0:
                if isinstance(raw_tran, pd.struct.TwoWindingTransformer):
                    raw_branches.append(raw_tran)
                elif isinstance(raw_tran, pd.struct.ThreeWindingTransformer): # 3 Times because 3w
                    raw_branches.extend([raw_tran]*3)

        
        net.set_branch_array(len(raw_branches)) # allocate PFNET branch array
        side = 'k' # Index to move in diferent side of 3W transformer

        for index, raw_branch in enumerate(reversed(raw_branches)):

            # Lines
            if isinstance(raw_branch, pd.struct.Branch):
                line = net.get_branch(index)
                line.set_as_line()
                line.name = raw_branch.ckt
                line.in_service = raw_line.st > 0

                line.bus_k = net.get_bus_from_number(raw_branch.i)
                line.bus_m = net.get_bus_from_number(raw_branch.j)

                line.b_k = raw_branch.bi+raw_branch.b/2
                line.b_m = raw_branch.bj+raw_branch.b/2
                line.g_k = raw_branch.gi
                line.g_m = raw_branch.gj
                
                den = raw_branch.r**2 + raw_branch.x**2
                line.g = raw_branch.r / den
                line.b = -raw_branch.x / den

                line.ratingA = raw_branch.ratea/case.sbase
                line.ratingB = raw_branch.rateb/case.sbase
                line.ratingC = raw_branch.ratec/case.sbase

            # 2 Windings Transformers
            elif isinstance(raw_branch, pd.struct.TwoWindingTransformer):
                tr = net.get_branch(index)     
                tr.in_service = raw_branch.p1.stat > 0
                    
                tr.name = raw_branch.p1.name
                tr.bus_k = net.get_bus_from_number(raw_branch.p1.i)
                tr.bus_m = net.get_bus_from_number(raw_branch.p1.j)
                
                tr.ratingA = raw_branch.w1.rata / case.sbase
                tr.ratingB = raw_branch.w1.ratb / case.sbase
                tr.ratingC = raw_branch.w1.ratc / case.sbase
                
                tr.phase = np.deg2rad(raw_branch.w1.ang)
                tr.phase_max = np.deg2rad(raw_branch.w1.ang)
                tr.phase_min = np.deg2rad(raw_branch.w1.ang)

                tr.ratio = raw_branch.w2.windv / raw_branch.w1.windv
                tr.num_ratios = raw_branch.w1.ntp
                cw = raw_branch.p1.cw
                tr.ratio_max = raw_branch.w2.windv / raw_branch.w1.rmi if cw==2 else 1 / raw_branch.w1.rmi
                tr.ratio_min = raw_branch.w2.windv / raw_branch.w1.rma if cw==2 else 1 / raw_branch.w1.rma

                x = raw_branch.p2.x12
                r = raw_branch.p2.r12
                cz = raw_branch.p1.cz
                tbase = raw_branch.p2.sbase12
                sbase = net.base_power
                tr.g, tr.b = self.get_tran_series_parameters(x, r, cz, tbase, sbase)

                if raw_branch.w1.cod == 0:
                    tr.set_as_fixed_tran()
                elif raw_branch.w1.cod == 1:
                    tr.set_as_tap_changer_v()
                    if raw_branch.w1.cont < 0:
                        tr.reg_bus = tr.bus_k
                    else:
                        tr.reg_bus = tr.bus_m
                elif raw_branch.w1.cod == 2:
                    tr.set_as_tap_changer_Q()
                elif raw_branch.w1.cod == 3:
                    tr.set_as_phase_shifter()
                    tr.phase_max = np.deg2rad(raw_branch.w1.rmi)
                    tr.phase_min = np.deg2rad(raw_branch.w1.rma)
                elif raw_branch.w1.cod == 4:
                    pass # DC-Line Control
                elif raw_branch.w1.cod == 5:
                    pass # Asymetric PF          
                
            # 3 Windings Transformers
            elif isinstance(raw_branch, pd.struct.ThreeWindingTransformer):
                tr = net.get_branch(index)
                tr.bus_m = net.get_bus_from_number(tran2star[raw_branch.index].i)
                tr.name = raw_branch.p1.ckt

                x12, r12, tbase12 = raw_branch.p2.x12, raw_branch.p2.x12, raw_branch.p2.sbase12
                x23, r23, tbase23 = raw_branch.p2.x23, raw_branch.p2.x23, raw_branch.p2.sbase23
                x31, r31, tbase31 = raw_branch.p2.x31, raw_branch.p2.x31, raw_branch.p2.sbase31
                cz = raw_branch.p1.cz
                cw = raw_branch.p1.cw
                sbase = case.sbase

                g12, b12 = self.get_tran_series_parameters(x12, r12, cz, tbase12, sbase)
                g23, b23 = self.get_tran_series_parameters(x23, r23, cz, tbase23, sbase)
                g31, b31 = self.get_tran_series_parameters(x31, r31, cz, tbase31, sbase)

                den12 = g12**2 + b12**2
                den23 = g23**2 + b23**2
                den31 = g31**2 + b31**2

                if side == 'k':
                    tr.bus_k = net.get_bus_from_number(raw_branch.p1.k)
                    tr.in_service = raw_branch.p1.stat != 3 and raw_branch.p1.stat != 0
                    tr.g = 2 / (g31/den31 + g23/den23 - g12/den12) if den31*den23*den12 != 0 else 1e6
                    tr.b = -2 / (b31/den31 + b23/den23 - b12/den12) if den31*den23*den12 != 0 else 1e6
                    tr.ratio = 1/raw_branch.w1.windv
                    tr.num_ratios = raw_branch.w1.ntp
                    tr.ratio_max = 1/raw_branch.w1.rmi if cw!=2 else raw_branch.w1.windv/raw_branch.w1.rmi
                    tr.ratio_min = 1/raw_branch.w1.rma if cw!=2 else raw_branch.w1.windv/raw_branch.w1.rma                 
                    tr.phase = np.deg2rad(raw_branch.w1.ang)
                    side = 'j'
                elif side == 'j':
                    tr.bus_k = net.get_bus_from_number(raw_branch.p1.j)
                    tr.in_service = raw_branch.p1.stat != 2 and raw_branch.p1.stat != 0
                    tr.g = 2 / (g12/den12 + g23/den23 - g31/den31) if den31*den23*den12 != 0 else 1e6
                    tr.b = -2 / (b12/den12 + b23/den23 - b31/den31) if den31*den23*den12 != 0 else 1e6
                    tr.ratio = 1/raw_branch.w2.windv
                    tr.num_ratios = raw_branch.w2.ntp
                    tr.ratio_max = 1/raw_branch.w2.rmi if cw!=2 else raw_branch.w1.windv/raw_branch.w2.rmi
                    tr.ratio_min = 1/raw_branch.w2.rma if cw!=2 else raw_branch.w1.windv/raw_branch.w2.rma  
                    tr.phase = np.deg2rad(raw_branch.w2.ang)               
                    side = 'i'
                elif side == 'i':
                    tr.bus_k = net.get_bus_from_number(raw_branch.p1.i)
                    tr.in_service = raw_branch.p1.stat != 4 and raw_branch.p1.stat != 0
                    tr.g = 2 / (g12/den12 + g31/den31 - g23/den23) if den31*den23*den12 != 0 else 1e6
                    tr.b = -2 / (b12/den12 + b31/den31 - b23/den23) if den31*den23*den12 != 0 else 1e6
                    tr.ratio = 1/raw_branch.w3.windv
                    tr.num_ratios = raw_branch.w3.ntp
                    tr.ratio_max = 1/raw_branch.w3.rmi if cw!=2 else raw_branch.w1.windv/raw_branch.w3.rmi
                    tr.ratio_min = 1/raw_branch.w3.rma if cw!=2 else raw_branch.w1.windv/raw_branch.w3.rma  
                    tr.phase = np.deg2rad(raw_branch.w3.ang)   
                    side = 'k'         
				               
                tr.set_as_fixed_tran # TODO set control modes
    
                      
        # PFNET shunts  

        raw_shunts = []
        for raw_shunt in case.fixed_shunts:
            if self.keep_all_oos or (raw_shunt.status > 0 and num2rawbus[raw_shunt.i].ide != self.BUS_TYPE_IS):
                raw_shunts.append(raw_shunt)                    
        for raw_shunt in case.switched_shunts:
            if self.keep_all_oos or (raw_shunt.stat > 0 and num2rawbus[raw_shunt.i].ide != self.BUS_TYPE_IS):
                raw_shunts.append(raw_shunt)
        for raw_tran in case.transformers:
        	if self.keep_all_oos or raw_tran.p1.stat > 0:
        		raw_shunts.append(raw_tran)

        net.set_shunt_array(len(raw_shunts)) # allocate PFNET shunt array
        
        for index, raw_shunt in enumerate(reversed(raw_shunts)):

            # Fixed Shunt
            if isinstance(raw_shunt, pd.struct.FixedShunt):  
                shunt = net.get_shunt(index)           
                shunt.bus = net.get_bus_from_number(raw_shunt.i)
                shunt.in_service = raw_shunt.status > 0
                shunt.b = raw_shunt.bl/net.base_power
                shunt.g = raw_shunt.gl/net.base_power
                shunt.set_as_fixed()
                
            # Switched Shunt
            elif isinstance(raw_shunt, pd.struct.SwitchedShunt):
                
                shunt = net.get_shunt(index)
                shunt.bus = net.get_bus_from_number(raw_shunt.i)
                shunt.in_service = raw_shunt.stat > 0
                shunt.b = raw_shunt.binit/case.sbase 
                shunt.g = 0
               
                b_step = [raw_shunt.b1, raw_shunt.b2, raw_shunt.b3, raw_shunt.b4,
                 			raw_shunt.b5, raw_shunt.b6, raw_shunt.b7, raw_shunt.b8]
                steps = [raw_shunt.n1, raw_shunt.n2, raw_shunt.n3, raw_shunt.n4,
                 		raw_shunt.n5, raw_shunt.n6, raw_shunt.n7, raw_shunt.n8]

                from itertools import product
                b_block = []
                for (b,n) in zip(b_step, steps):
                	if n is not None:
                	    b_step = [b*i for i in range(n+1)] # posibles valores por bloque
                	    b_block.append(b_step)
                b_values = np.array(list(set([sum(i)/net.base_power for i in product(*b_block)])))
                shunt.set_b_values(b_values)
                 
                if raw_shunt.modsw == 0:
                    shunt.set_as_switched()
                    
                elif raw_shunt.modsw == 1:
                    shunt.set_as_switched_v()
                    shunt.set_as_discrete()
                    shunt.reg_bus=shunt.bus
                    
                elif raw_shunt.modsw == 2:
                    shunt.set_as_switched_v()
                    shunt.set_as_continuous()
                    shunt.reg_bus = shunt.bus
                    
                elif raw_shunt.modsw == 3:
                    shunt.set_as_switched()
                    shunt.set_as_discrete()
                    shunt.reg_bus = net.get_bus_from_number(raw_shunt.swrem)
                    
                elif raw_shunt.modsw in [4,5,6]:
                    shunt.set_as_discrete()
                
                # Para Hacer:
                #Una vez pasadas las VSC-DC y los FACTS se podria terminar mejor 4,6
                #   Igualmente hay que terminar MODSW=5
                #   discrete adjustment, controlling the admittance setting of the switched shunt at bus 
                #   SWREM            
            
            # Magnetizing Impedance of Transformers
            elif isinstance(raw_shunt, pd.struct.ThreeWindingTransformer) or isinstance(raw_shunt, pd.struct.TwoWindingTransformer):
                shunt = net.get_shunt(index)

                buses = [raw_shunt.p1.i, raw_shunt.p1.j, raw_shunt.p1.k]
                nmetr = raw_shunt.p1.nmetr

                shunt.bus = net.get_bus_from_number(buses[nmetr-1])
                shunt.in_service = raw_shunt.p1.stat > 0

                g, b = raw_shunt.p1.mag1, raw_shunt.p1.mag2

                if raw_shunt.p1.cm == 1:
                	shunt.g = g
                	shunt.b = b
                elif raw_shunt.p1.cm == 2: # No load loss in W and IO in sbase 12
                	voltage_correction = shunt.bus.v_base/raw_shunt.w1.nomv
                	power_correction = raw_shunt.p2.sbase12/net.base_power
                	shunt.g = 1e-6 * g / net.base_power * voltage_correction**2
                	shunt.b = np.sqrt(b*power_correction**2 - shunt.b**2)  

                if raw_shunt.is_three_winding():
               		shunt.g *= -1.

               	shunt.set_as_fixed()
               	# shunt.is_part_of_three_winding = True	
                # TODO vincularlo a un transformador	
        

        # PFNET DC buses

        # TODO: No se definen buses de continua en PSSE sino
        # 		Lineas de dos terminales o lineas multiterminales
        # 		ac |--converter--| dc-----line-----dc | ---rect--|ac
        # 		habra que iterear entre todos los componentes de continua
        #		va a ser dificil PFNET -> PSSE 

        # PFNET CSC HVDC

        # PFNET VSC HVDC

        # PFNET Facts

        raw_list_facts = []
        for raw_facts in case.facts:
            if self.keep_all_oos or raw_facts.mode != 0:
                raw_list_facts.append(raw_facts)
        net.set_facts_array(len(raw_list_facts))
        for index, raw_facts in enumerate(reversed(raw_list_facts)):
            facts = net.get_facts(index)
            facts.name = raw_facts.name
            facts.bus_k = net.get_bus_from_number(raw_facts.i)
            if raw_facts.j == 0:
                facts.bus_m = None
                facts.set_series_link_disabled()
            else:
                facts.bus_m = net.get_bus_from_number(raw_facts.j)
            facts.in_service = raw_facts.mode != 0

            if raw_facts.remot != 0:
                facts.reg_bus = net.get_bus_from_number(raw_facts.remot)
            else:
                facts.reg_bus = facts.bus_k	 

            facts.P_set = raw_facts.pdes/net.base_power
            facts.Q_set = raw_facts.qdes/net.base_power

            facts.P_max_dc = 9999	# No se en que parte de la documentacion esta
            facts.Q_par = raw_facts.rmpct/net.base_power
            facts.Q_max_s = 9999
            facts.Q_min_s = 9999
            facts.Q_max_sh = 9999
            facts.Q_min_sh = 9999
            
            facts.b = -1 / raw_facts.linx  # Dummy element for PF solutions
            facts.i_max_s = raw_facts.imx/net.base_power
            facts.i_max_sh = raw_facts.shmx/net.base_power

            facts.v_max_m = raw_facts.vtmx
            facts.v_min_m = raw_facts.vtmx

            facts.v_max_s = raw_facts.vsmx

            if raw_facts.mode == 1:
                facts.set_in_normal_series_mode()
            elif raw_facts.mode == 2:
                facts.set_series_link_bypassed()
            elif raw_facts.mode == 3:
                facts.set_in_constant_series_z_mode()
                den = raw_facts.set1**2 + raw_facts.set2**2
                facts.g = raw_facts.set1/den
                facts.b = -raw_facts.set2/den
            elif raw_facts.mode == 4:
                facts.set_in_constant_series_v_mode()
                facts.v_mag_s = raw_facts.set1
                facts.v_ang_s = np.deg2rad(raw_facts.set2)

        # Update properties
        net.update_properties()

        # Return
        return net

    def show(self):
        """
        Shows parsed data.
        """

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

        import grg_pssedata as pd
        
        # Se pierde demasiada informacion en pasar PSSE => PFNET o viceversa
        # lo importante es que sea razonable PFNET => PSSE => PFNET
        
        case = pd.struct.Case(ic = 0,
                              sbase = net.base_power,
                              rev = 0,
                              xfrrat = 0,
                              nxfrat = 0,
                              basfrq = 50,
                              record1 = '',
                              record2 = '',
                              buses = [],
                              loads = [],
                              fixed_shunts = [],
                              generators = [],
                              branches = [],
                              transformers = [],
                              areas = [],
                              tt_dc_lines = [],
                              vsc_dc_lines = [],
                              transformer_corrections = [],
                              mt_dc_lines = [],
                              line_groupings = [],
                              zones = [],
                              transfers = [],
                              owners = [],
                              facts = [],
                              switched_shunts = [],
                              gnes = [],
                              induction_machines = [])
        
        # PSSE Buses        
        for bus in reversed(net.buses):
            
            def get_ide(bus):
                if bus.is_slack():
                    return 3
                elif bus.reg_generators != []:
                    return 2
                elif bus.reg_generators == []:
                    return 1
                else: 
                    return 4
                                        
            i = bus.number
            name = str(bus.name)
            basekv = float(bus.v_base)
            ide = get_ide(bus)
            area = int(bus.area)
            zone = int(bus.zone)
            owner = 1
            vm = bus.v_mag
            va = np.rad2deg(bus.v_ang)
            nvhi = bus.v_max_norm
            nvlo = bus.v_min_norm
            evhi = bus.v_max_emer
            evlo = bus.v_min_emer
                  
            case.buses.append(pd.struct.Bus(i, name, basekv, ide, area, zone,
                                            owner, vm, va, nvhi, nvlo, evhi, evlo))
        
        # PSSE Loads
        for load in reversed(net.loads):
            index  = load.index 
            i = load.bus.number     
            ID = load.name
            status = load.in_service
            area = load.bus.area  
            zone = load.bus.zone 
            pl = load.comp_cp * net.base_power
            ql = load.comp_cq * net.base_power 
            ip = load.comp_ci * net.base_power 
            iq = load.comp_cj * net.base_power 
            yp = load.comp_cg * net.base_power 
            yq = load.comp_cb * net.base_power 
            owner = 1
            scale = 1
            intrpt = 0

            case.loads.append(pd.struct.Load(index, i, ID, status, area, zone,
            	                             pl, ql, ip, iq, yp, yq, owner, scale, intrpt))
            
        # PSSE Generators
        for gen in reversed(net.generators):
            
            index = int(gen.index)
            i = int(gen.bus.number)
            ID = gen.name
            pg = gen.P * net.base_power
            qg = gen.Q * net.base_power
            qt = gen.Q_max * net.base_power
            qb = gen.Q_min * net.base_power
            vs = gen.reg_bus.v_set
            ireg = gen.reg_bus.number
            mbase = net.base_power
            zr = 0.
            zx = 0.
            rt = 0.
            xt = 0.
            gtap = 0.
            stat = int(gen.in_service)
            rmpct = 0.
            pt = gen.P_max*net.base_power
            pb = gen.P_min*net.base_power
            o1 = 1
            f1 = 1.
            o2 = 0
            f2 = 1.
            o3 = 0
            f3 = 1.
            o4 = 0
            f4 = 1.
            wmod = 0
            wpf  = 0
            
            case.generators.append(pd.struct.Generator(index, i ,ID ,pg ,qg ,qt ,qb,
                                                       vs, ireg, mbase, zr, zx, rt, xt,
                                                       gtap, stat, rmpct, pt, pb,
                                                       o1, f1, o2, f2, o3, f3, o4, f4,
                                                       wmod, wpf))
            
        # PSSE Lines
        for branch in reversed(net.branches):
            
            if branch.is_line():
                
               index = branch.index
               i = branch.bus_k.number
               j = branch.bus_m.number
               ckt = branch.name
               den = branch.g**2 + branch.b**2
               r = branch.g/den
               x = -branch.b/den
               b = branch.b_m + branch.b_k
               ratea = branch.ratingA*net.base_power
               rateb = branch.ratingB*net.base_power
               ratec = branch.ratingC*net.base_power
               gi = branch.g_k
               bi = 0.  
               gj = branch.g_m
               bj = 0.
               st = int(branch.in_service)
               met = -1
               length = 0.
               o1 = 1
               f1 = 1.
               o2 = 0
               f2 = 1.
               o3 = 0
               f3 = 1.
               o4 = 0
               f4 = 1.
               
               case.branches.append(pd.struct.Branch(index, i, j, ckt, r, x, b,
                                                     ratea, rateb, ratec, gi, bi,
                                                     gj, bj, st, met, length,
                                                     o1, f1, o2, f2, o3, f3, o4, f4))
               
        # PSSE Transformers         
            if not branch.is_line():
            	# Tendriamos que determinar que parte del trafo de tres arrollamiento es
                if not branch.is_part_of_3_winding_transformer() or True: # 2 Windings
                # p1
                    i = branch.bus_k.number
                    j = branch.bus_m.number
                    k = 0
                    ckt = 1
                    cw = 1
                    cz = 1
                    cm = 1
                    mag1 = 0.0 # TODO asignar al correspondiente shunt
                    mag2 = 0.0 # TODO asignar al correspondiente shunt
                    nmetr = 2
                    name = str (branch.name)
                    stat = int (branch.in_service)
                    o1 = f1 = o2 = f2 = o3 = f3 =o4 = f4 = 1
                    vecgrp= '            '
                    p1 = pd.struct.TransformerParametersFirstLine(i, j, k, ckt, cw, cz, cm,     
                                                              mag1, mag2, nmetr, name, stat,
                                                              o1, f1, o2, f2, o3, f3, o4, f4, vecgrp)
                # p2
                    den = branch.g**2 + branch.b**2
                    r12 = branch.g/den
                    x12 = -branch.b/den
                    sbase12 = net.base_power
                    p2 = pd.struct.TransformerParametersSecondLineShort(r12, x12, sbase12)
                # w1             
                    index_w1 = 1 
                    windv = 1.0
                    nomv = branch.bus_k.v_base
                    ang = branch.phase
                    rata = branch.ratingA*net.base_power
                    ratb = branch.ratingB*net.base_power
                    ratc = branch.ratingC*net.base_power
                
                    if branch.is_fixed_tran():
                        cod = 0
                        rma = 1/branch.ratio_min
                        rmi = 1/branch.ratio_max
                    elif branch.is_tap_changer_v():
                        cod = 1
                        rma = 1/branch.ratio_min
                        rmi = 1/branch.ratio_max
                    elif branch.is_tap_changer_Q():
                        cod = 2
                    elif branch.is_phase_shifter():
                        cod = 3
                        rma = np.rad2deg(branch.phase_max)
                        rmi = np.rad2deg(branch.ratio_min)
                    
                    if branch.reg_bus == None:
                        cont = 0
                    else:
                        cont = branch.reg_bus.number
                 # Faltan definir los de abajo
                 
                    vma = 1.0
                    vmi = 1.0
                    ntp = branch.num_ratios
                    tab = 0 # table of impedance correction
                    cr = 0 # Load drop compensation
                    cx = 0 # Load drop compensation
                    cnxa = 0. # Solo usado cuando COD1 = 5 Asymetric active power flow control
                    
                    w1 = pd.struct.TransformerWinding(index_w1, windv, nomv, ang, rata, ratb, ratc,
                                                      cod, cont, rma, rmi, vma, vmi, ntp, tab, cr,
                                                      cx, cnxa)
                
                # w2
                  
                    index_w2 = 2 
                    windv = branch.ratio
                    nomv = branch.bus_m.v_base
                
                    w2 = pd.struct.TransformerWindingShort(index_w2,windv,nomv)
                 
                    case.transformers.append(pd.struct.TwoWindingTransformer(index,p1,p2,w1,w2))  
            	
                else:  # 3 Winding
            	    pass
        
        # PSSE Shunts

        for shunt in reversed(net.shunts):
            if shunt.is_part_of_transformer():
                continue
            if shunt.is_fixed():
                index = shunt.index
                i = shunt.bus.number
                ID = shunt.name
                status = int(shunt.in_service)
                gl = shunt.g
                bl = shunt.b
                case.fixed_shunts.append(pd.struct.FixedShunt(index, i, ID, status, gl, bl))
            else:
                index = shunt.index
                i = shunt.bus.number
                if shunt.is_switched_locked():
                    modsw = 0
                elif shunt.is_switched_v():
                	modsw = 1 if shunt.is_discrete() else 2             

                adjm = 1
                stat = int(shunt.is_in_service())
                vswhi = 1.0
                vswlo = 1.0
                swrem = shunt.reg_bus.number if shunt.reg_bus else 0
                rmpct = 0
                rmidnt = None
                binit = shunt.b * net.base_power
                # TODO discriminar pasos del Switched Shunt
                b_values = shunt.b_values
                b_values = b_values[b_values != 0.] # Filtra el cero
                n = [] # Numero de pasos por bloque
                b_i = [] # Numero de susceptancia ind. por bloque por paso
                # Ej 3x20 -> n=3, b_i=20

                n1, n2, n3, n4, n5, n6, n7, n8 = [0] * 8
                b1, b2, b3, b4, b5, b6, b7, b8 = [0] * 8

                # Corregir hasta aca--------------------------------------
                case.switched_shunts.append(pd.struct.SwitchedShunt(index, i, modsw, adjm, stat, vswhi,
                												 vswlo, swrem, rmpct, rmidnt, binit,
                												 n1, b1, n2, b2, n3, b3, n4, b4,
                												 n5, b5, n6, b6, n7, b7, n8, b8))
        # PSSE DC-Line

        # PSSE DC VSC-Line

        # PSSE Facts

        for facts in reversed(net.facts):
            index = facts.index
            name = facts.name
            i = facts.bus_k.number
            j = facts.bus_m.number if facts.bus_m else 0
            if facts.is_in_service():
	            if facts.is_in_normal_series_mode():
	                mode = 1
	                set1 = set2 = 0.
	            elif facts.is_series_link_bypassed():
	                mode = 2
	                set1 = set2 = 0.
	            elif facts.is_in_constant_series_z_mode():
	                den = facts.g**2 + facts.b**2
	                set1 = facts.g/den
	                set2 = -facts.b/den
	                mode = 3
	            elif facts.is_in_constant_series_v_mode():
	                set1 = np.rad2deg(facts.v_mag_s)
	                set2 = np.rad2deg(facts.v_ang_s)
	                mode = 4
            else:
	            mode = 0
	            set1 = set2 = 0
            pdes = facts.P_set * net.base_power
            qdes = facts.Q_set * net.base_power
            vset = 1.0 # voltage set point (Set as PSSE Default)
            shmx = facts.i_max_sh * net.base_power
            trmx = 9999 # maximum bridge active power (Set as Default)
            vtmn = facts.v_min_m
            vtmx = facts.v_max_m
            vsmx = facts.v_max_s
            imx = facts.i_max_s * net.base_power
            linx = -1/facts.b
            rmpct = facts.Q_par * net.base_power
            owner = 0
            vsref = 0
            remot = facts.reg_bus.number if facts.reg_bus else 0
            mname = ''
            
            case.facts.append(pd.struct.FACTSDevice(index, name, i, j, mode, pdes, qdes,
            										vset, shmx, trmx, vtmn, vtmx, vsmx,
            										imx, linx, rmpct, owner, set1, set2,
            										vsref, remot, mname))

        # Write         
        f = open(filename, 'w')
        f.write(case.to_psse())
        f.close()

    def get_tran_series_parameters(self, x, r, cz, tbase, sbase):
        
        den = x**2 + r**2

        if cz == 1:
            g, b = r/den, -x/den  # In system base
        else:
            if cz == 3:  # In Pcc: watts and Z: pu 
                g = (1e-6 * r)/(sbase) / x**2		# g = r/z**2
                b = - np.sqrt((1/x)**2 - g**2)
            else:   # Transformer base
                g *= tbase / sbase
                b *= tbase / sbase
        return g, b                  