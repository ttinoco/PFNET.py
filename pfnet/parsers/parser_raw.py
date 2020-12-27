import os
import io
import sys
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
        self.base_freq = 60.      # value of the system frecuency
        self.output_level = 0     # numero para controlar output
        self.no_mag_shunt = True  # flag to parse the magnetizing impedance of transformers

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
        elif key == 'base_freq':
            self.base_freq = value
        elif key == 'output_level':
            self.output_level = value
        elif key == 'no_mag_shunt':
            self.no_mag_shunt = value
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

        # Check extension
        if os.path.splitext(filename)[-1][1:].lower() != 'raw':
            raise pfnet.ParserError('invalid file extension')

        # Get grg-pssedata case
        if self.output_level == 0:
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            import grg_pssedata as pd
            case = pd.io.parse_psse_case_file(filename)
            sys.stderr = old_stderr
        else:
            import grg_pssedata as pd
            case = pd.io.parse_psse_case_file(filename)
        self.case = case
        self.base_freq = case.basfrq
        
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
        three_winding = [] # contains bus numbers that have conecctions with 3W tran
        max_bus_number = max([rb.i for rb in case.buses]+[0])
        for raw_tran in case.transformers:
            if isinstance(raw_tran, pd.io.ThreeWindingTransformer):
                raw_bus_i = num2rawbus[raw_tran.p1.i]
                raw_bus_type = self.BUS_TYPE_PQ if raw_tran.p1.stat > 0 else self.BUS_TYPE_IS
                raw_bus = pd.io.Bus(max_bus_number+1,   # number
                                    '',                 # name
                                    raw_bus_i.basekv,   # base kv
                                    raw_bus_type,       # type
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
                three_winding.extend([raw_tran.p1.i, raw_tran.p1.j, raw_tran.p1.k])
                case.buses.append(raw_bus)
                max_bus_number += 1

        # PFNET base power
        net.base_power = case.sbase

        # PFNET buses
        raw_buses = []
        for raw_bus in case.buses:
            if self.keep_all_oos or raw_bus.ide != self.BUS_TYPE_IS or raw_bus.i in three_winding:
                raw_buses.append(raw_bus)
        net.set_bus_array(len(raw_buses)) # allocate PFNET bus array
        for index, raw_bus in enumerate(reversed(raw_buses)):
            bus = net.get_bus(index)
            bus.number = raw_bus.i
            bus.in_service = raw_bus.ide != self.BUS_TYPE_IS
            bus.area = raw_bus.area
            bus.zone = raw_bus.zone
            bus.name = raw_bus.name.strip()
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
            if (self.keep_all_oos or
                (raw_load.status > 0 and num2rawbus[raw_load.i].ide != self.BUS_TYPE_IS)):
                raw_loads.append(raw_load)
        net.set_load_array(len(raw_loads)) # allocate PFNET load array
        for index, raw_load in enumerate(reversed(raw_loads)):
            load = net.get_load(index)
            load.bus = net.get_bus_from_number(raw_load.i)
            load.name = str(raw_load.id).ljust(2)
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
            if (self.keep_all_oos or
                (raw_gen.stat > 0 and num2rawbus[raw_gen.i].ide != self.BUS_TYPE_IS)):
                raw_gens.append(raw_gen)
        net.set_gen_array(len(raw_gens)) # allocate PFNET gen array
        for index, raw_gen in enumerate(reversed(raw_gens)):          
            gen = net.get_generator(index)           
            gen.in_service = raw_gen.stat > 0
            gen.bus = net.get_bus_from_number(raw_gen.i)      
            gen.name = str(raw_gen.id).ljust(2)
            gen.P = raw_gen.pg/net.base_power
            gen.P_max = raw_gen.pt/net.base_power
            gen.P_min = raw_gen.pb/net.base_power
            gen.Q = raw_gen.qg/net.base_power
            gen.Q_max = raw_gen.qt/net.base_power
            gen.Q_min = raw_gen.qb/net.base_power                          
            if gen.bus.is_slack() or raw_gen.ireg == 0:
                gen.reg_bus = gen.bus
            else:
                gen.reg_bus = net.get_bus_from_number(raw_gen.ireg)
            gen.reg_bus.v_set = raw_gen.vs

        # PFNET branches
        raw_branches = []       
        for raw_line in case.branches: # Lines
            if self.keep_all_oos or raw_line.st > 0:
                raw_branches.append(raw_line)     
        for raw_tran in case.transformers: # Transformers
            if self.keep_all_oos or raw_tran.p1.stat > 0:
                if isinstance(raw_tran, pd.struct.TwoWindingTransformer):
                    raw_branches.append(raw_tran)
                elif isinstance(raw_tran, pd.struct.ThreeWindingTransformer): # 3 Times because 3w
                    raw_branches.extend([raw_tran]*3)
        net.set_branch_array(len(raw_branches)) # allocate PFNET branch array
        side = 'k' # Index to move in diferent side of 3W transformer
        for index, raw_branch in enumerate(reversed(raw_branches)):
            if isinstance(raw_branch, pd.struct.Branch): # Lines
                line = net.get_branch(index)
                line.set_as_line()
                line.name = str(raw_branch.ckt).ljust(2)
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
            elif isinstance(raw_branch, pd.struct.TwoWindingTransformer): # 2 Windings Transformers
                tr = net.get_branch(index)     
                tr.in_service = raw_branch.p1.stat > 0
                tr.name = str(raw_branch.p1.ckt).ljust(2)
                tr.bus_k = net.get_bus_from_number(raw_branch.p1.i)
                tr.bus_m = net.get_bus_from_number(raw_branch.p1.j)
                tr.ratingA = raw_branch.w1.rata / case.sbase
                tr.ratingB = raw_branch.w1.ratb / case.sbase
                tr.ratingC = raw_branch.w1.ratc / case.sbase
                tr.phase = np.deg2rad(raw_branch.w1.ang)
                tr.phase_max = np.deg2rad(raw_branch.w1.ang)
                tr.phase_min = np.deg2rad(raw_branch.w1.ang)
                if raw_branch.p1.cw == 1:
                    tr.ratio = raw_branch.w2.windv / raw_branch.w1.windv
                else:
                    tr.ratio = (raw_branch.w2.windv/raw_branch.w1.windv) / (tr.bus_m.v_base/tr.bus_k.v_base)
                    if raw_branch.p1.cw == 3:
                        tr.ratio *= raw_branch.w2.nomv/raw_branch.w1.nomv
                tr.num_ratios = raw_branch.w1.ntp
                cw = raw_branch.p1.cw
                tr.ratio_max = raw_branch.w2.windv/raw_branch.w1.rmi if cw==2 else 1/raw_branch.w1.rmi
                tr.ratio_min = raw_branch.w2.windv/raw_branch.w1.rma if cw==2 else 1/raw_branch.w1.rma
                x = raw_branch.p2.x12
                r = raw_branch.p2.r12
                cz = raw_branch.p1.cz
                tbase = raw_branch.p2.sbase12
                sbase = net.base_power
                tr.g, tr.b = self.get_tran_series_parameters(x, r, cz, tbase, sbase)                
                self.set_tran_mode(tr, raw_branch.w1)
            elif isinstance(raw_branch, pd.struct.ThreeWindingTransformer): # 3 Windings Transformers
                tr = net.get_branch(index)
                tr.bus_m = net.get_bus_from_number(tran2star[raw_branch.index].i)
                tr.name = str(raw_branch.p1.ckt).ljust(2)
                x12, r12, tbase12 = raw_branch.p2.x12, raw_branch.p2.r12, raw_branch.p2.sbase12
                x23, r23, tbase23 = raw_branch.p2.x23, raw_branch.p2.r23, raw_branch.p2.sbase23
                x31, r31, tbase31 = raw_branch.p2.x31, raw_branch.p2.r31, raw_branch.p2.sbase31
                cz = raw_branch.p1.cz
                cw = raw_branch.p1.cw
                sbase = case.sbase
                g12, b12 = self.get_tran_series_parameters(x12, r12, cz, tbase12, sbase)
                g23, b23 = self.get_tran_series_parameters(x23, r23, cz, tbase23, sbase)
                g31, b31 = self.get_tran_series_parameters(x31, r31, cz, tbase31, sbase)
                z12 = 1/(g12 + 1j* b12) 
                z23 = 1/(g23 + 1j* b23)
                z31 = 1/(g31 + 1j* b31)
                if side == 'k':
                    tr.bus_k = net.get_bus_from_number(raw_branch.p1.k)
                    tr.in_service = raw_branch.p1.stat != 3 and raw_branch.p1.stat != 0
                    tr.g = (2 / (z23 + z31 - z12)).real if abs(z23 + z31 - z12) != 0 else 1e6
                    tr.b = (2 / (z23 + z31 - z12)).imag if abs(z23 + z31 - z12) != 0 else -1e6
                    tr.ratio = 1/raw_branch.w3.windv
                    if raw_branch.p1.cw != 1:
                        tr.ratio *= tr.bus_k.v_base
                        if raw_branch.p1.cw == 3:
                            tr.ratio *= raw_branch.w3.nomv
                    tr.num_ratios = raw_branch.w3.ntp
                    if cw !=2:
                        tr.ratio_max = 1/raw_branch.w3.rmi  
                    else: 
                        tr.ratio_max = raw_branch.w3.windv/raw_branch.w3.rmi
                    if cw !=2:
                        tr.ratio_min = 1/raw_branch.w3.rma
                    else: 
                        tr.ratio_min = raw_branch.w3.windv/raw_branch.w3.rma                 
                    tr.phase = np.deg2rad(raw_branch.w3.ang)
                    self.set_tran_mode(tr, raw_branch.w3)
                    side = 'j'
                elif side == 'j':
                    tr.bus_k = net.get_bus_from_number(raw_branch.p1.j)
                    tr.in_service = raw_branch.p1.stat != 2 and raw_branch.p1.stat != 0
                    tr.g = (2 / (z12 + z23 - z31)).real if abs(z12 + z23 - z31) != 0 else 1e6
                    tr.b = (2 / (z12 + z23 - z31)).imag if abs(z12 + z23 - z31) != 0 else -1e6
                    tr.ratio = 1/raw_branch.w2.windv
                    if raw_branch.p1.cw != 1:
                        tr.ratio *= tr.bus_k.v_base
                        if raw_branch.p1.cw == 3:
                            tr.ratio *= raw_branch.w2.nomv
                    tr.num_ratios = raw_branch.w2.ntp
                    if cw !=2: 
                        tr.ratio_max = 1/raw_branch.w2.rmi
                    else:
                        tr.ratio_max = raw_branch.w1.windv/raw_branch.w2.rmi
                    if cw !=2: 
                        tr.ratio_min = 1/raw_branch.w2.rma
                    else:
                        tr.ratio_min = raw_branch.w1.windv/raw_branch.w2.rma  
                    tr.phase = np.deg2rad(raw_branch.w2.ang)
                    self.set_tran_mode(tr, raw_branch.w2)               
                    side = 'i'
                elif side == 'i':
                    tr.bus_k = net.get_bus_from_number(raw_branch.p1.i)
                    tr.in_service = raw_branch.p1.stat != 4 and raw_branch.p1.stat != 0
                    tr.g = (2 / (z12 + z31 - z23)).real if abs(z12 + z31 - z23) != 0 else 1e6
                    tr.b = (2 / (z12 + z31 - z23)).imag if abs(z12 + z31 - z23) != 0 else -1e6
                    tr.ratio = 1/raw_branch.w1.windv
                    if raw_branch.p1.cw != 1:
                        tr.ratio *= tr.bus_k.v_base
                        if raw_branch.p1.cw == 3:
                            tr.ratio *= raw_branch.w1.nomv
                    tr.num_ratios = raw_branch.w1.ntp
                    if cw !=2: 
                        tr.ratio_max = 1/raw_branch.w1.rmi
                    else:
                        tr.ratio_max = raw_branch.w1.windv/raw_branch.w1.rmi
                    if cw !=2:
                        tr.ratio_min = 1/raw_branch.w1.rma 
                    else:
                        tr.ratio_min = raw_branch.w1.windv/raw_branch.w1.rma  
                    tr.phase = np.deg2rad(raw_branch.w1.ang)
                    self.set_tran_mode(tr, raw_branch.w1)   
                    side = 'k'  

                # TODO: el flag de three_winding_... como se coloca?   
                # tr.set_as_part_of_three_winding_transformer()  

        # PFNET shunts  
        raw_shunts = []
        for raw_shunt in case.fixed_shunts: # Fixed Shunts
            if (self.keep_all_oos or
                (raw_shunt.status > 0 and num2rawbus[raw_shunt.i].ide != self.BUS_TYPE_IS)):
                raw_shunts.append(raw_shunt)                    
        for raw_shunt in case.switched_shunts: # Switched Shunts
            if (self.keep_all_oos or
                (raw_shunt.stat > 0 and num2rawbus[raw_shunt.i].ide != self.BUS_TYPE_IS)):
                raw_shunts.append(raw_shunt)
        for raw_tran in case.transformers: # Magnetizing Impedance como un Shunt
            if self.no_mag_shunt: # Si esta deshabilitada la opcion
                continue
            if self.keep_all_oos or raw_tran.p1.stat > 0:
                raw_shunts.append(raw_tran)

        net.set_shunt_array(len(raw_shunts)) # allocate PFNET shunt array
        for index, raw_shunt in enumerate(reversed(raw_shunts)):            
            if isinstance(raw_shunt, pd.struct.FixedShunt): # Fixed Shunt
                shunt = net.get_shunt(index)           
                shunt.bus = net.get_bus_from_number(raw_shunt.i)
                shunt.name = str(raw_shunt.id).strip().ljust(2)
                shunt.in_service = raw_shunt.status > 0
                shunt.b = raw_shunt.bl/net.base_power
                shunt.g = raw_shunt.gl/net.base_power
                shunt.set_as_fixed()
            elif isinstance(raw_shunt, pd.struct.SwitchedShunt): # Switched Shunt             
                shunt = net.get_shunt(index)
                shunt.bus = net.get_bus_from_number(raw_shunt.i)
                shunt.name = str(raw_shunt.i) # PSSE33 Only allows one sw-shunt per bus
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
                        b_step = [b*i for i in range(n+1)] # Valores de B por bloque
                        b_block.append(b_step)
                b_values = np.array(list(set([sum(i)/net.base_power for i in product(*b_block)])))
                b_values.sort()
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
                
                # TODO: Ver modos 4,5,6 y su relacion con VSC y FACTS
                # por el momento se dejan como shunts discretos     
            elif (isinstance(raw_shunt, pd.struct.ThreeWindingTransformer) or 
                  isinstance(raw_shunt, pd.struct.TwoWindingTransformer)): # Magnetizing Impedance
                shunt = net.get_shunt(index)
                shunt.name = 'TR' + raw_shunt.p1.ckt.strip()
                buses = [raw_shunt.p1.i, raw_shunt.p1.j, raw_shunt.p1.k]
                nmetr = raw_shunt.p1.nmetr
                shunt.bus = net.get_bus_from_number(buses[nmetr-1])
                shunt.in_service = raw_shunt.p1.stat > 0
                g, b = raw_shunt.p1.mag1, raw_shunt.p1.mag2
                if raw_shunt.p1.cm == 1:
                    shunt.g = g
                    shunt.b = b
                elif raw_shunt.p1.cm == 2: # No load loss in W and IO in sbase_12
                    voltage_correction = shunt.bus.v_base/raw_shunt.w1.nomv
                    power_correction = raw_shunt.p2.sbase12/net.base_power
                    shunt.g = 1e-6 * g / net.base_power * voltage_correction**2
                    shunt.b = np.sqrt(b*power_correction**2 - shunt.b**2)  
                if raw_shunt.is_three_winding():
                    shunt.g *= -1.
                shunt.set_as_fixed()
               	# shunt.is_part_of_transformer = True	
                # TODO vincularlo a un transformador
        
        # PFNET DC buses
        # TODO : Los elementos de DC seran solo de dos terminales por el momento

        raw_DC_buses = []
        for raw_DC in case.vsc_dc_lines + case.tt_dc_lines:
            if self.keep_all_oos or raw_DC.params.mdc != 0:
            	raw_DC_buses.extend([raw_DC]*2) # Dos Buses por linea (k - m)
        net.set_dc_bus_array(len(raw_DC_buses))
        for index, raw_bus_DC in enumerate(reversed(raw_DC_buses)):
            busDC = net.get_dc_bus(index)
            busDC.name = raw_bus_DC.params.name.strip('\"').rstrip()
            if isinstance(raw_bus_DC, pd.struct.VSCDCLine):
                converter = raw_bus_DC.c1 if index%2 == 0 else raw_bus_DC.c2
                busDC.number = converter.ibus
                busDC.v_base = 1 # TODO PSSE doesnt have a DC-Voltage basis
            elif isinstance(raw_bus_DC, pd.struct.TwoTerminalDCLine):
                if index%2 == 0: # rectifier
                    rectifier = raw_bus_DC.rectifier
                    busDC.number = rectifier.ipr
                else: # inverter
                    inverter = raw_bus_DC.inverter
                    busDC.number = inverter.ipi
                busDC.v_base = raw_bus_DC.params.vschd
            busDC.in_service = bool(raw_bus_DC.params.mdc)
        net.update_hash_tables()
            
        # PFNET DC line
        raw_DC_branches = []
        for raw_DC in case.vsc_dc_lines + case.tt_dc_lines:
            if self.keep_all_oos or raw_DC.params.mdc != 0:
                raw_DC_branches.append(raw_DC)
        net.set_dc_branch_array(len(raw_DC_branches))            	
        for index, raw_branch_DC in enumerate(reversed(raw_DC_branches)):
            branchDC = net.get_dc_branch(index)
            if isinstance(raw_branch_DC, pd.struct.VSCDCLine):
                branchDC.bus_k = net.get_dc_bus_from_number(raw_branch_DC.c1.ibus)
                branchDC.bus_m = net.get_dc_bus_from_number(raw_branch_DC.c2.ibus)
            elif isinstance(raw_branch_DC, pd.struct.TwoTerminalDCLine):
                branchDC.bus_k = net.get_dc_bus_from_number(raw_branch_DC.rectifier.ipr)
                branchDC.bus_m = net.get_dc_bus_from_number(raw_branch_DC.inverter.ipi)
            branchDC.in_service = bool(raw_branch_DC.params.mdc)
            branchDC.name =  raw_branch_DC.params.name.strip('\"').rstrip()
            branchDC.r = raw_branch_DC.params.rdc * net.base_power / branchDC.bus_m.v_base**2

        # PFNET CSC HVDC
        # solo conversores AC/DC o DC/AC
        raw_csc_converters = []
        for raw_DC in case.tt_dc_lines:
            if self.keep_all_oos or raw_DC.params.mdc != 0:
                raw_csc_converters.extend([raw_DC]*2) # Un conversor por lado (k-m)
        net.set_csc_converter_array(len(raw_csc_converters))
        for index, raw_csc_converter in enumerate(reversed(raw_csc_converters)):
            csc = net.get_csc_converter(index)
            if index % 2 == 0: # rectifier
                raw_csc = raw_csc_converter.rectifier
                csc.set_as_rectifier()
                csc.name = raw_csc_converter.params.name.strip('\"').rstrip()
                csc.ac_bus = net.get_bus_from_number(raw_csc.ipr)
                csc.dc_bus = net.get_dc_bus_from_number(raw_csc.ipr)
                csc.num_bridges = raw_csc.nbr
                csc.angle = np.deg2rad(raw_csc.icr)
                csc.angle_max = np.deg2rad(raw_csc.anmxr)
                csc.angle_min = np.deg2rad(raw_csc.anmnr)
                csc.r = raw_csc.rcr * net.base_power / csc.dc_bus.v_base**2
                csc.x = raw_csc.xcr * net.base_power / csc.dc_bus.v_base**2
                csc.v_base_p = raw_csc.ebasr
                csc.ratio = raw_csc.trr
                csc.ratio_max = raw_csc.tmxr
                csc.ratio_min = raw_csc.tmnr
                csc.x_cap = raw_csc.xcapr * net.base_power / csc.dc_bus.v_base**2
            else: # inverter
                raw_csc = raw_csc_converter.inverter
                csc.set_as_inverter()
                csc.name = raw_csc_converter.params.name.strip('\"').rstrip()
                csc.ac_bus = net.get_bus_from_number(raw_csc.ipi)
                csc.dc_bus = net.get_dc_bus_from_number(raw_csc.ipi)
                csc.num_bridges = raw_csc.nbi  
                csc.angle = np.deg2rad(raw_csc.ici)
                csc.angle_max = np.deg2rad(raw_csc.anmxi)
                csc.angle_min = np.deg2rad(raw_csc.anmni)
                csc.r = raw_csc.rci * net.base_power / csc.dc_bus.v_base**2
                csc.x = raw_csc.xci * net.base_power / csc.dc_bus.v_base**2
                csc.v_base_p = raw_csc.ebasi
                csc.ratio = raw_csc.tri    
                csc.ratio_max = raw_csc.tmxi  
                csc.ratio_min = raw_csc.tmni
                csc.x_cap = raw_csc.xcapi * net.base_power / csc.dc_bus.v_base**2
            if raw_csc_converter.params.mdc == 1:
                csc.set_in_P_dc_mode()
                csc.P_dc_set = raw_csc_converter.params.setvl / net.base_power
            elif raw_csc_converter.params.mdc == 2:
                csc.set_in_i_dc_mode()
                csc.i_dc_set = raw_csc_converter.params.setvl / net.base_power

        # PFNET VSC HVDC
        raw_vsc_converters = []
        for raw_DC in case.vsc_dc_lines:
            if self.keep_all_oos or raw_DC.params.mdc != 0:
                raw_vsc_converters.extend([raw_DC]*2)  # Un conversor por lado (k-m)
        net.set_vsc_converter_array(len(raw_vsc_converters))

        for index, raw_vsc_converter in enumerate(reversed(raw_vsc_converters)):
            vsc = net.get_vsc_converter(index)
            vsc.in_service = bool(raw_vsc_converter.params.mdc)
            converter = raw_vsc_converter.c1 if index%2 == 0 else raw_vsc_converter.c2
            vsc.ac_bus = net.get_bus_from_number(converter.ibus)
            vsc.dc_bus = net.get_dc_bus_from_number(converter.ibus)
            vsc.name = raw_vsc_converter.params.name.strip('\"').rstrip()
            vsc.loss_coeff_A = converter.aloss
            vsc.loss_coeff_A = converter.bloss
            vsc.Q_par = converter.rmpct / net.base_power
            vsc.Q_max = converter.maxq / net.base_power
            vsc.Q_min = converter.minq / net.base_power
            S_max = 1e6 if converter.smax == 0. else converter.smax / net.base_power # 1e6 is big enough for unlimited rating?
            vsc.P_max = 1e6 # TODO: PSSE VSC type does not define active power injection
            vsc.P_min = -1e6# TODO: PSSE VSC type does not define active power injection
            if converter.mode == 1:
                vsc.set_in_v_ac_mode()
                vsc.reg_bus = net.get_bus_from_number(converter.remot) if converter.remot else None
            elif converter.mode == 2:
                vsc.set_in_P_dc_mode()
                vsc.P_dc_set = converter.dcset / net.base_power

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
                if raw_facts.remot == facts.bus_k.number:
                    facts.reg_bus = facts.bus_k
                else:
                    facts.reg_bus = net.get_bus_from_number(raw_facts.remot)
            else:
                facts.reg_bus = facts.bus_k if not facts.bus_k.is_slack() else None	 
            facts.P_set = raw_facts.pdes/net.base_power
            facts.Q_set = raw_facts.qdes/net.base_power
            facts.P_max_dc = 9999	# Default value from PSSE33 (POM 5-59)
            facts.Q_par = raw_facts.rmpct/net.base_power
            facts.Q_max_s = 9999    # Default value from PSSE33 (POM 5-59)
            facts.Q_min_s = 9999    # Default value from PSSE33 (POM 5-59)
            facts.Q_max_sh = 9999   # Default value from PSSE33 (POM 5-59)
            facts.Q_min_sh = 9999   # Default value from PSSE33 (POM 5-59)
            facts.b = -1 / raw_facts.linx  # DOCS PSSE33 POM 5-60
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

        # Only write the 0 time_period
        import grg_pssedata as pd
        
        case_buses = []
        case_loads = []
        case_generators = []
        case_branches = []
        case_transformers = []
        case_switched_shunts = []
        case_fixed_shunts = []
        case_tt_dc_lines = []
        case_vsc_dc_lines = []
        case_zones = []
        case_facts = []

        # PSSE Buses      
        zones = []  
        for bus in reversed(net.buses):
            if bus.is_star():
                continue                                        
            i = bus.number
            name = str(bus.name)
            basekv = float(bus.v_base)
            if bus.is_in_service():
                if bus.is_slack():
                    ide = 3
                elif bus.reg_generators != []:
                    ide = 2
                else:
                    ide = 1
            else: 
                ide = 4
            area = int(bus.area)
            zone = int(bus.zone)
            if zone not in zones:
                zones.append(zone)
            owner = 1
            vm = bus.v_mag[0]
            va = np.rad2deg(bus.v_ang[0])
            nvhi = bus.v_max_norm
            nvlo = bus.v_min_norm
            evhi = bus.v_max_emer
            evlo = bus.v_min_emer  
            case_buses.append(pd.struct.Bus(i, name, basekv, ide, area, zone,
                                            owner, vm, va, nvhi, nvlo, evhi, evlo))
        
        # PSSE Zones
        for i, zone in enumerate(zones):
            case_zones.append(pd.struct.Zone(i+1, zone))

        # PSSE Loads
        for load in reversed(net.loads):
            index  = load.index 
            i = load.bus.number     
            ID = load.name
            status = load.in_service
            area = load.bus.area  
            zone = load.bus.zone 
            pl = load.comp_cp[0] * net.base_power
            ql = load.comp_cq[0] * net.base_power 
            ip = load.comp_ci[0] * net.base_power 
            iq = load.comp_cj[0] * net.base_power 
            yp = load.comp_cg * net.base_power 
            yq = load.comp_cb * net.base_power 
            owner = 1 # Default Value
            scale = 1 # Default Value
            intrpt = 0 # Default Value
            case_loads.append(pd.struct.Load(index, i, ID, status, area, zone,
                                             pl, ql, ip, iq, yp, yq, owner, scale, intrpt))
            
        # PSSE Generators
        for gen in reversed(net.generators):       
            index = int(gen.index)
            i = int(gen.bus.number)
            ID = gen.name
            pg = gen.P[0] * net.base_power
            qg = gen.Q[0] * net.base_power
            qt = gen.Q_max * net.base_power
            qb = gen.Q_min * net.base_power
            vs = gen.reg_bus.v_set[0]
            ireg = gen.reg_bus.number
            mbase = net.base_power
            zr = 0. # Default Value
            zx = 0. # Default Value
            rt = 0. # Default Value
            xt = 0. # Default Value
            gtap = 0. # Default Value
            stat = int(gen.in_service)
            rmpct = 0. # Default Value
            pt = gen.P_max*net.base_power
            pb = gen.P_min*net.base_power
            o1 = o2 = o3 = o4 = 1 # Default Value
            f1 = f2 = f3 = f4 = 1.# Default Value
            wmod = 0 # Default Value
            wpf  = 0 # Default Value
            case_generators.append(pd.struct.Generator(index, i ,ID ,pg ,qg ,qt ,qb,
                                                       vs, ireg, mbase, zr, zx, rt, xt,
                                                       gtap, stat, rmpct, pt, pb,
                                                       o1, f1, o2, f2, o3, f3, o4, f4,
                                                       wmod, wpf))
            
        # PSSE Lines
        for branch in reversed(net.branches):
            if branch.is_line(): # Branches            
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
                length = 0. # Default Value
                o1 = o2 = o3 = o4 = 1 # Default Value
                f1 = f2 = f3 = f4 = 1.# Default Value
                case_branches.append(pd.struct.Branch(index, i, j, ckt, r, x, b,
                                                      ratea, rateb, ratec, gi, bi,
                                                      gj, bj, st, met, length,
                                                      o1, f1, o2, f2, o3, f3, o4, f4))
            else: # PSSE Transformers
                if not branch.is_part_of_3_winding_transformer(): # 2 Windings
                    # p1
                    i = branch.bus_k.number
                    j = branch.bus_m.number
                    k = 0
                    ckt = branch.name
                    cw = 1 # p.u. turns ratio in bus base voltage
                    cz = 1 # Z in system MVA base
                    cm = 1 # MAG in system MVA base
                    mag1 = 0. # TODO ver como vincular el shunt magnetizing impedance
                    mag2 = 0. # net.get_shunt_from_name_and_bus_number('TR'+branch.name, i).b
                    nmetr = 2
                    name = str (branch.name)
                    stat = int (branch.in_service)
                    o1 = o2 = o3 = o4 = 1 # Default Value
                    f1 = f2 = f3 = f4 = 1.# Default Value
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
                    windv = 1/branch.ratio[0]
                    nomv = branch.bus_k.v_base
                    ang = branch.phase[0]
                    rata = branch.ratingA*net.base_power
                    ratb = branch.ratingB*net.base_power
                    ratc = branch.ratingC*net.base_power
                    if branch.is_fixed_tran():
                        cod = 0 # No control
                        rma = 1/branch.ratio_min
                        rmi = 1/branch.ratio_max
                        vma = 1.1 # Default Value
                        vmi = 0.9 # Default Value
                    elif branch.is_tap_changer_v():
                        cod = 1 # Voltage Control
                        rma = 1/branch.ratio_min
                        rmi = 1/branch.ratio_max
                        vma = 1.1 # Default Value
                        vmi = 0.9 # Default Value
                    elif branch.is_tap_changer_Q():
                        cod = 2 # Reactive Control
                        rma = 1/branch.ratio_min
                        rmi = 1/branch.ratio_max
                        vma = 1.1 # Default Value
                        vmi = 0.9 # Default Value
                    elif branch.is_phase_shifter():
                        cod = 3 # Active Control
                        rma = np.rad2deg(branch.phase_max)
                        rmi = np.rad2deg(branch.ratio_min)
                        vma = 1.1 # Default Value
                        vmi = 0.9 # Default Value
                    if branch.reg_bus == None:
                        cont = 0
                    else:
                        cont = branch.reg_bus.number
                    ntp = branch.num_ratios
                    tab = 0 # Default Value
                    cr = 0 # Load drop compensation (Default Value)
                    cx = 0 # Load drop compensation (Default Value)
                    cnxa = 0. # Default value. only used with COD1 = 5 Asymetric active power flow control
                    w1 = pd.struct.TransformerWinding(index_w1, windv, nomv, ang, rata, ratb, ratc,
                                                      cod, cont, rma, rmi, vma, vmi, ntp, tab, cr,
                                                      cx, cnxa)
                    # w2      
                    index_w2 = 2 
                    windv = 1.
                    nomv = branch.bus_m.v_base
                    w2 = pd.struct.TransformerWindingShort(index_w2,windv,nomv)
                 
                    case_transformers.append(pd.struct.TwoWindingTransformer(index,p1,p2,w1,w2))  
            	
        # PSSE 3-Winding Transformer
        for bus in reversed(net.buses):
            if bus.is_star():
                tr_parts = [branch for branch in bus.branches 
                            if branch.is_part_of_3_winding_transformer()]        	            	    
                assert(len(tr_parts) <= 3) # the only conexions are the windings
                # p1
                i, j, k = [tr.bus_k.number for tr in reversed(tr_parts)]
                ckt = tr_parts[0].name
                cw = 1
                cz = 1
                cm = 1
                mag1 = 0. # TODO: vincularlo a traves de un shunt 
                mag2 = 0. # TODO: vincularlo a traves de un shunt
                nmetr = 2
                name = str(branch.name)
                if True in [tr.is_in_service() for tr in reversed(tr_parts)]: # In Service
                    stat = 4 if not tr_parts[0].is_in_service() else 1
                    stat = 2 if not tr_parts[1].is_in_service() else 1
                    stat = 3 if not tr_parts[2].is_in_service() else 1
                else:
                    stat = 0
                o1 = f1 = o2 = f2 = o3 = f3 =o4 = f4 = 1
                vecgrp= '            '
                p1 = pd.struct.TransformerParametersFirstLine(i, j, k, ckt, cw, cz, cm,     
                                                              mag1, mag2, nmetr, name, stat,
                                                              o1, f1, o2, f2, o3, f3, o4, f4, vecgrp)
                # p2
                impedance = [1/(tr.g + tr.b*1j) for tr in reversed(tr_parts)]
                r1, r2, r3 = [z.real for z in impedance]
                x1, x2, x3 = [z.imag for z in impedance]
                r12, x12 = r1+r2, x1+x2
                r23, x23 = r2+r3, x2+x3
                r31, x31 = r3+r1, x3+x1
                sbase12 = sbase23 = sbase31 = net.base_power
                vmstar = bus.v_mag[0]
                anstar = np.rad2deg(bus.v_ang[0])
                p2 = pd.struct.TransformerParametersSecondLine(r12, x12, sbase12,
                                                               r23, x23, sbase23,
                                                               r31, x31, sbase31,
                                                               vmstar, anstar)
                w = []
                for index, tr in enumerate(reversed(tr_parts)): 
                    index_w = len(tr_parts)-index # 3 -> 2 -> 1
                    windv = 1/tr.ratio[0]
                    nomv = tr.bus_k.v_base
                    ang = np.rad2deg(tr.phase[0])
                    rata = tr.ratingA*net.base_power
                    ratb = tr.ratingB*net.base_power
                    ratc = tr.ratingC*net.base_power
                    if tr.is_fixed_tran():
                        cod = 0 # No control
                        rma = 1/tr.ratio_min
                        rmi = 1/tr.ratio_max
                        vma = 1.1 # Default Value
                        vmi = 0.9 # Default Value
                    elif tr.is_tap_changer_v():
                        cod = 1 # Voltage Control
                        rma = 1/tr.ratio_min
                        rmi = 1/tr.ratio_max
                        vma = 1.1 # Default Value
                        vmi = 0.9 # Default Value
                    elif tr.is_tap_changer_Q():
                        cod = 2 # Reactive Control
                        rma = 1/tr.ratio_min
                        rmi = 1/tr.ratio_max
                        vma = 1.1 # Default Value
                        vmi = 0.9 # Default Value
                    elif tr.is_phase_shifter():
                        cod = 3 # Active Control
                        rma = np.rad2deg(tr.phase_max)
                        rmi = np.rad2deg(tr.ratio_min)
                        vma = 1.1 # Default Value
                        vmi = 0.9 # Default Value
                    if tr.reg_bus == None:
                        cont = 0
                    else:
                        cont = tr.reg_bus.number
                    ntp = tr.num_ratios
                    tab = 0 # Default Value
                    cr = 0 # Load drop compensation (Default Value)
                    cx = 0 # Load drop compensation (Default Value)
                    cnxa = 0. # Default value. only used with COD1 = 5 Asymetric active power flow control
                    w.append(pd.struct.TransformerWinding(index_w, windv, nomv, ang, rata, ratb, ratc,
                                                          cod, cont, rma, rmi, vma, vmi, ntp, tab, cr,
                                                          cx, cnxa))
                case_transformers.append(pd.struct.ThreeWindingTransformer(index,p1,p2,w[0],w[1],w[2]))   

        # PSSE Shunts
        for shunt in reversed(net.shunts):
            if shunt.is_part_of_transformer() or 'TR' in shunt.name:
            	# TODO: no se como indicar que un shunt es parte de un transformador. 
                # Sabiendo esto podria eliminar la parte del or.
                continue # Skip Magnetizing Impedance
            if shunt.is_fixed():
                index = shunt.index
                i = shunt.bus.number
                ID = str(shunt.name).strip()
                status = int(shunt.in_service)
                gl = shunt.g * net.base_power
                bl = shunt.b[0] * net.base_power
                case_fixed_shunts.append(pd.struct.FixedShunt(index, i, ID, status, gl, bl))
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
                binit = shunt.b[0] * net.base_power
                
                # Calculation of steps of the blocks
                block_values = {} # b -> n
                shunt.b_values.sort()
                for b in shunt.b_values*net.base_power:
                    if b == 0:
                        continue
                    if len(block_values.keys()) == 0:
                        block_values[b] = 1
                        continue
                    for block in block_values.keys():
                        if b%block <= 1e-4:
                            block_values[block] = int(b//block)
                        else: # is not in block_values
                            block_values[block] = 1

                n1, n2, n3, n4, n5, n6, n7, n8 = list(block_values.values())+[0.]*(8-len(block_values))
                b1, b2, b3, b4, b5, b6, b7, b8 = list(block_values.keys())+[0.]*(8-len(block_values))
                case_switched_shunts.append(pd.struct.SwitchedShunt(index, i, modsw, adjm, stat, vswhi,
                                                                    vswlo, swrem, rmpct, rmidnt, binit,
                                                                    n1, b1, n2, b2, n3, b3, n4, b4,
                                                                    n5, b5, n6, b6, n7, b7, n8, b8))
        # PSSE DC-Line
        for dc_line in reversed(net.dc_branches):
            if dc_line.bus_k.csc_converters:
                converter_k = net.get_csc_converter_from_name_and_ac_bus_number(dc_line.name,
                                                                                dc_line.bus_k.number)
                converter_m = net.get_csc_converter_from_name_and_ac_bus_number(dc_line.name,
                                                                                dc_line.bus_m.number)
                rectifier = converter_k if converter_k.is_rectifier() else converter_m
                inverter = converter_k if converter_k.is_inverter() else converter_m
                # index
                index = dc_line.index
                # params
                name = dc_line.name
                if inverter.is_in_P_dc_mode():
                    mdc = 1
                    setvl = inverter.P_dc_set[0] * net.base_power
                elif inverter.is_in_i_dc_mode():
                    mdc = 2
                    setvl = inverter.i_dc_set[0] * net.base_power
                else:
                    mdc = 0
                    setvl = 0
                rdc = dc_line.r * dc_line.bus_k.v_base**2 / net.base_power
                rcomp = 0
                vschd = dc_line.bus_k.v_base
                vcmod = 0.
                delti = 0.
                meter = "I"
                dcvmin = 0
                cccitmx = 20 
                cccacc = 1.
                params = pd.struct.TwoTerminalDCLineParameters(name,
                                                               mdc,
                                                               rdc,
                                                               setvl,
                                                               vschd,
                                                               vcmod,
                                                               rcomp,
                                                               delti,
                                                               meter,
                                                               dcvmin,
                                                               cccitmx,
                                                               cccacc)
                # rectifier
                ipr = rectifier.ac_bus.number
                nbr = rectifier.num_bridges
                anmxr = np.rad2deg(rectifier.angle_max)
                anmnr = np.rad2deg(rectifier.angle_min)
                rcr = rectifier.r * dc_line.bus_k.v_base**2 / net.base_power 
                xcr = rectifier.x * dc_line.bus_k.v_base**2 / net.base_power
                ebasr = rectifier.v_base_p
                trr = rectifier.ratio[0]
                tapr = rectifier.ratio[0]
                tmxr = rectifier.ratio_max
                tmnr = rectifier.ratio_min
                stpr = 0.00625
                icr = np.rad2deg(rectifier.angle[0])
                ifr = 0
                itr = 0
                idr = 1
                xcapr = rectifier.x_cap * dc_line.bus_k.v_base**2 / net.base_power
                rectifier = pd.struct.TwoTerminalDCLineRectifier(ipr,
                                                                 nbr,
                                                                 anmxr,
                                                                 anmnr,
                                                                 rcr,
                                                                 xcr,
                                                                 ebasr,
                                                                 trr,
                                                                 tapr,
                                                                 tmxr,
                                                                 tmnr,
                                                                 stpr,
                                                                 icr,
                                                                 ifr,
                                                                 itr,
                                                                 idr,
                                                                 xcapr)
                # inverter
                ipi = inverter.ac_bus.number
                nbi = inverter.num_bridges
                anmxi = np.rad2deg(inverter.angle_max)
                anmni = np.rad2deg(inverter.angle_min)
                rci = inverter.r * dc_line.bus_m.v_base**2 / net.base_power 
                xci = inverter.x * dc_line.bus_m.v_base**2 / net.base_power
                ebasi = inverter.v_base_p
                tri = inverter.ratio[0]
                tapi = inverter.ratio[0]
                tmxi = inverter.ratio_max
                tmni = inverter.ratio_min
                stpi = 0.00625
                ici = np.rad2deg(inverter.angle[0])
                ifi = 0
                iti = 0
                idi = 1
                xcapi = inverter.x_cap * dc_line.bus_m.v_base**2 / net.base_power
                inverter = pd.struct.TwoTerminalDCLineInverter(ipi, 
                                                               nbi, 
                                                               anmxi, 
                                                               anmni, 
                                                               rci, 
                                                               xci, 
                                                               ebasi, 
                                                               tri,
                                                               tapi, 
                                                               tmxi, 
                                                               tmni, 
                                                               stpi, 
                                                               ici, 
                                                               ifi, 
                                                               iti, 
                                                               idi, 
                                                               xcapi)
                case_tt_dc_lines.append(pd.struct.TwoTerminalDCLine(index, params, rectifier, inverter))

        # PSSE DC VSC-Line
        for vsc_dc_line in net.dc_branches:
            if vsc_dc_line.bus_k.vsc_converters:         
                # index      
                index = vsc_dc_line.index
                # params
                name = vsc_dc_line.name
                mdc = int(vsc_dc_line.is_in_service())
                rdc = vsc_dc_line.r * vsc_dc_line.bus_k.v_base**2 / net.base_power
                o1 = 0 
                f1 = 1 
                o2 = 0 
                f2 = 1 
                o3 = 0 
                f3 = 1 
                o4 = 0 
                f4 = 1
                params = pd.struct.VSCDCLineParameters(name, mdc, rdc, o1, f1, o2, f2, o3, f3, o4, f4)
                # converter 1
                c1 = net.get_vsc_converter_from_name_and_ac_bus_number(vsc_dc_line.name,
                                                                       vsc_dc_line.bus_k.number)
                c2 = net.get_vsc_converter_from_name_and_ac_bus_number(vsc_dc_line.name,
                                                                       vsc_dc_line.bus_m.number)
                converter = []
                for c in [c1, c2]:
                    ibus = c.ac_bus.number
                    type = 0 
                    dcset = 0
                    if vsc_dc_line.is_in_service():
                        if c.is_in_v_dc_mode():
                            type = 1
                            dcset = c.v_dc_st * c.dc_bus.v_base
                        elif c.is_in_P_dc_mode():
                            type = 2
                            dcset = c.P_dc_set * net.base_power
                    mode = 1
                    acset = 1.0 # TODO [DEFAULT VALUE] Voltage set point (c.reg_bus.v_mag ?)
                    if c.is_in_f_ac_mode():
                       mode = 2
                       acset = c.target_power_factor
                    aloss = c.loss_coeff_A * net.base_power
                    bloss = c.loss_coeff_B * net.base_power
                    minloss = 0.0
                    smax = np.sqrt(c.P_max**2 + c.Q_max**2) * net.base_power
                    imax = 0.0
                    pwf = c.target_power_factor
                    maxq = c.Q_max * net.base_power
                    minq = c.Q_min * net.base_power
                    remot = c.reg_bus.number if c.reg_bus else 0
                    rmpct = 100.0
                    converter.append(pd.struct.VSCDCLineConverter(ibus,
                                                                  type,
                                                                  mode,
                                                                  dcset,
                                                                  acset,
                                                                  aloss,
                                                                  bloss,
                                                                  minloss,
                                                                  smax,
                                                                  imax,
                                                                  pwf,
                                                                  maxq,
                                                                  minq,
                                                                  remot,
                                                                  rmpct))          
                case_vsc_dc_lines.append(pd.struct.VSCDCLine(index, params, converter[0] , converter[1]))

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
            vset = 1.0 # voltage set point [PSSE Default]
            shmx = facts.i_max_sh * net.base_power
            trmx = 9999 # maximum bridge active power [PSSE Default]
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
            case_facts.append(pd.struct.FACTSDevice(index, name, i, j, mode, pdes, qdes,
                                                    vset, shmx, trmx, vtmn, vtmx, vsmx,
                                                    imx, linx, rmpct, owner, set1, set2,
                                                    vsref, remot, mname))

        case = pd.struct.Case(ic = 0,
                              sbase = net.base_power,
                              rev = 33,
                              xfrrat = 0,
                              nxfrat = 0,
                              basfrq = self.base_freq,
                              record1 = '',
                              record2 = '',
                              buses = case_buses,
                              loads = case_loads,
                              fixed_shunts = case_fixed_shunts,
                              generators = case_generators,
                              branches = case_branches,
                              transformers = case_transformers,
                              areas = [],
                              tt_dc_lines = case_tt_dc_lines,
                              vsc_dc_lines = case_vsc_dc_lines,
                              transformer_corrections = [],
                              mt_dc_lines = [],
                              line_groupings = [],
                              zones = case_zones,
                              transfers = [],
                              owners = [],
                              facts = case_facts,
                              switched_shunts = case_switched_shunts,
                              gnes = [],
                              induction_machines = [])
        # Write         
        f = open(filename, 'w')
        f.write(case.to_psse())
        f.close()

    def get_tran_series_parameters(self, x, r, cz, tbase, sbase):
        
        den = x**2 + r**2
        g, b = r/den, -x/den 

        if cz == 1:
            return g, b
        else:
            if cz == 3:  # In Pcc: watts and Z: pu 
                g = (1e-6 * r)/(sbase) / x**2		# g = r/z**2
                b = - np.sqrt((1/x)**2 - g**2)
            else:   # Transformer base
                g *= tbase / sbase
                b *= tbase / sbase
        return g, b                  

    def set_tran_mode(self, tr, winding):

        if winding.cod == 0: # No control
            tr.set_as_fixed_tran()
        elif abs(winding.cod) == 1: # Voltage Control
            tr.set_as_tap_changer_v()
            if winding.cont < 0:
                tr.reg_bus = tr.bus_k
            else:
                tr.reg_bus = tr.bus_m
        elif abs(winding.cod) == 2: # Reactive Control
            tr.set_as_tap_changer_Q()
        elif abs(winding.cod) == 3: # Active Control
            tr.set_as_phase_shifter()
            tr.phase_max = np.deg2rad(winding.rmi)
            tr.phase_min = np.deg2rad(winding.rma)
        elif abs(winding.cod) == 4:
            pass # DC-Line Control
            # TODO: en un futuro deberia vincularse al conversor correspondiente
        elif abs(winding.cod) == 5:
            pass # Asymetric PF
