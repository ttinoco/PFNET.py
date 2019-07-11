from __future__ import division
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
            #bus.in_service =raw_bus.ide != self.BUS_TYPE_IS
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
        
        raw_gens = []
        for raw_gen in case.generators:
            if self.keep_all_oos or (raw_gen.stat > 0 and num2rawbus[raw_gen.i].ide != self.BUS_TYPE_IS):
                raw_gens.append(raw_gen)
        net.set_gen_array(len(raw_gens)) # allocate PFNET gen array
        for index, raw_gen in enumerate(reversed(raw_gens)):
            
            gen = net.get_generator(index)           
            
            bus = net.get_bus_from_number(raw_gen.i)
            bus.add_generator(gen)
            gen.bus=bus
            
            gen.name = "%d" %gen.index
            gen.P     = float(raw_gen.pg)/raw_gen.mbase
            gen.P_max = float(raw_gen.pt)/raw_gen.mbase
            gen.P_min = float(raw_gen.pb)/raw_gen.mbase
            gen.Q     = float(raw_gen.qg)/raw_gen.mbase
            gen.Q_max = float(raw_gen.qt)/raw_gen.mbase
            gen.Q_min = float(raw_gen.qb)/raw_gen.mbase   
                        
            
            #El parser de MATPOWER toma una consideracion similar en cuanto al Slack Bus
            if gen.bus.is_slack() or gen.Q_max > gen.Q_min:
                gen.reg_bus = bus
                assert(gen.index in [g.index for g in bus.reg_generators])
                bus.v_set = raw_gen.vs
                

        # PFNET branches
        
        raw_branches = []
        
        #Lines
        
        for raw_line in case.branches:
            if self.keep_all_oos or (raw_line.st > 0):
                raw_branches.append(raw_line)
                
        #2W transformer
    
        for raw_2w_trafo in case.transformers:
            if self.keep_all_oos or (raw_2w_trafo.p1.stat > 0):
                raw_branches.append(raw_2w_trafo)
        
        #3W transformer
        '''Falta pasarlos'''
                
                
        net.set_branch_array(len(raw_branches)) # allocate PFNET branch array
        for index, raw_branch in enumerate(reversed(raw_branches)):
            
         
            
            if type(raw_branch)==pd.struct.Branch:
                
              
                line=net.get_branch(index)
                line.set_as_line()
                line.name="%d" %(raw_line.index)
                
                line.bus_k=net.get_bus_from_number(raw_branch.i)
                line.bus_m=net.get_bus_from_number(raw_branch.j)
                
                line.b_k=raw_branch.bi
                line.b_m=raw_branch.bj
                line.g_k=raw_branch.gi
                line.g_m=raw_branch.gj
                              
                z=raw_branch.r+raw_branch.x*1j
                
                line.b=(1/z).imag
                line.g=(1/z).real
            
                line.ratingA=raw_branch.ratea
                line.ratingB=raw_branch.rateb
                line.ratingC=raw_branch.ratec
                
                
                
            elif type(raw_branch)==pd.struct.TwoWindingTransformer:
                #2w_transformer
                
                trafo_2w=net.get_branch(index)
                trafo_2w.set_as_fixed_tran()
                
                trafo_2w.name="%d" %(raw_branch.index)
                
                trafo_2w.bus_k=net.get_bus_from_number(raw_branch.p1.i)
                trafo_2w.bus_m=net.get_bus_from_number(raw_branch.p1.j)
                
                trafo_2w.ratingA=raw_branch.w1.rata/case.sbase
                trafo_2w.ratingB=raw_branch.w1.ratb/case.sbase
                trafo_2w.ratingC=raw_branch.w1.ratc/case.sbase
                
                trafo_2w.phase=raw_branch.w1.ang*np.pi/180
                #trafo_2w.phase_max=
                #trafo_2w.phase_min=
                
                #Control Modes
                if raw_branch.w1.cod==0:
                    trafo_2w.set_as_fixed_tran
                elif raw_branch.w1.cod==1:
                    trafo_2w.set_as_tap_changer_v()
                elif raw_branch.w1.cod==2:
                    trafo_2w.set_as_tap_changer_Q()
                elif raw_branch.w1.cod==3:
                    trafo_2w.set_as_phase_shifter()
                elif raw_branch.w1.cod==4:
                    pass #DC-Line Control
                elif raw_branch.w1.cod==4:
                    pass #Asymetric PF
                
                
                #Shunt parameters
                
                if raw_branch.p1.cm==2:
                    #No load loss in watts/ Exciting current in P.U. at nominal voltage w1
                    trafo_2w.g_m=raw_branch.p1.mag1*(case.sbase/raw_branch.w1.nomv**2) #ver con taps
                    trafo_2w.b_m=raw_branch.p1.mag2*(raw_branch.p2.sbase12/case.sbase)
                    
                else:
                    #In system base P.U.
                    trafo_2w.g_m=raw_branch.p1.mag1
                    trafo_2w.b_m=raw_branch.p1.mag2
               
                trafo_2w.b_k=0
                trafo_2w.g_k=0 
                
                #Series parameters 
                
                x12=raw_branch.p2.x12
                r12=raw_branch.p2.r12
                
                z=r12+x12*1j
                
                tbase=raw_branch.p2.sbase12
                        
                if raw_branch.p1.cz==1:
                    # In system PU
                    trafo_2w.g= (1/z).real 
                    trafo_2w.b= (1/z).imag 
                     
                    
                elif raw_branch.p1.cz==2:
                    # In transformer PU
                    trafo_2w.g= ((1/z).real)*(tbase/case.sbase)
                    trafo_2w.b= ((1/z).imag)*(tbase/case.sbase)
                     
                                       
                elif raw_branch.p1.cz==3:
                    # r12 in watts & z12 in sbase PU
                    trafo_2w.g=case.sbase/(r12/3)
                    trafo_2w.b=-1/np.sqrt((x12*tbase/case.sbase)**2-(trafo_2w.g)**2)
                    
                
                '''Faltaria poner el trafo t_mk debido a que no hace esa correccion en tension'''
                     
                    
               
                
                

            
      
        
        
        
        

        # PFNET shunts
        
        raw_shunts = []
        
        for raw_shunt in case.fixed_shunts:
            if self.keep_all_oos or (raw_shunt.status > 0 and num2rawbus[raw_shunt.i].ide != self.BUS_TYPE_IS):
                raw_shunts.append(raw_shunt)
                            
        for raw_shunt in case.switched_shunts:
            if self.keep_all_oos or (raw_shunt.status > 0 and num2rawbus[raw_shunt.i].ide != self.BUS_TYPE_IS):
                raw_shunts.append(raw_shunt)
                
        net.set_shunt_array(len(raw_shunts)) # allocate PFNET shunt array
        
   
        
        for index, raw_shunt in enumerate(reversed(raw_shunts)):
            
            if type(raw_shunt)==pd.struct.FixedShunt:
                #Fixed Shunt
                shunt = net.get_shunt(index)           
                shunt.bus=net.get_bus_from_number(raw_shunt.i)
                
                shunt.b    =float(raw_shunt.bl)/net.base_power
                shunt.b_max=float(raw_shunt.bl)/net.base_power
                shunt.b_min=float(raw_shunt.bl)/net.base_power
                shunt.g    =float(raw_shunt.gl)/net.base_power
                shunt.set_as_fixed()
                
            elif type(raw_shunt)==pd.struct.SwitchedShunt:
                #Switched Shunt
                shunt=net.get_shunt(index)
                shunt.bus=net.get_bus_from_number(raw_shunt.i)
                shunt.g=raw_shunt.gl
                
                b_values=[raw_shunt.binit,raw_shunt.b1,raw_shunt.b2,raw_shunt.b3,raw_shunt.b4,raw_shunt.b5,raw_shunt.b6,raw_shunt.b7,raw_shunt.b8]
                shunt.b_values=[B/net.base_power for B in b_values]
                
                shunt.b_max=raw_shunt.binit/net.base_power
                shunt.b_min=raw_shunt.b8/net.base_power
                
                
                if raw_shunt.modsw==0:
                    shunt.set_as_switched()
                    shunt.lock()  #Ver Si El Metodo Es El Adecuado
                    
                elif raw_shunt.modsw==1:
                    shunt.set_as_switched_v()
                    shunt.set_as_discrete()
                    shunt.reg_bus=net.get_bus_from_number(raw_shunt.swrem)
                    
                    bus=shunt.reg_bus().add_reg_shunt(shunt)
                    
                elif raw_shunt.modsw==2:
                    shunt.set_as_switched_v()
                    shunt.set_as_continuous()
                    shunt.reg_bus=net.get_bus_from_number(raw_shunt.swrem)
                    
                    bus=shunt.reg_bus().add_reg_shunt(shunt)
                    
                elif raw_shunt.modsw==3:
                    shunt.set_as_swithed()
                    shunt.set_as_discrete()
                    shunt.reg_bus=net.get_bus_from_number(raw_shunt.swrem)
                    
                    bus=shunt.reg_bus().add_reg_shunt(shunt)
                
                elif raw_shunt.modsw==4:
                    shunt.set_as_discrete()
                
                elif raw_shunt.modsw==5:
                    shunt.set_as_discrete()
                
                elif raw_shunt.modsw==6:
                    shunt.set_as_discrete()
                
                
                #Para Hacer:
                '''Una vez pasadas las VSC-DC y los FACTS se podria terminar mejor 4,6
                   Igualmente hay que terminar MODSW=5
                   discrete adjustment, controlling the admittance setting of the switched shunt at bus 
                   SWREM
                '''
                
            
      
                         



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