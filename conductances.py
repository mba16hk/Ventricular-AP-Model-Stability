import numpy as np

def Select_conduactance(current,model_type,cell_type):
    if current == 'GKr':
        return(GKr_conductance(model_type,cell_type))
    elif current == 'GKs':
        return(GKs_conductance(model_type,cell_type))
    elif current == 'GNa':
        return(GNa_conductance(model_type,cell_type))
    elif current == "GK1":
        return(GK1_conductance(model_type,cell_type))
    elif current == 'Gto':
        return(Gto_conductance(model_type,cell_type))
    elif current == 'GCa':
        return(GCa_conductance(model_type,cell_type)) #need to add
    elif current == 'GNaK':
        return(GNaK_conductance(model_type,cell_type))
    elif current == 'GNCX':
        return(GNCX_conductance(model_type,cell_type))
    elif current == 'GClCa':
        return(GClCa_conductance(model_type, cell_type)) 
    
    ###### Background currents
    elif current == 'GpCa':
        return(GpCa_conductance(model_type,cell_type))
    elif current == 'GKb':
        return(GKb_conductance(model_type, cell_type))
    elif current == 'GNab':
        return(GNab_conductance(model_type, cell_type))
    elif current == 'GCab':
        return(GCab_conductance(model_type, cell_type))
    elif current == 'GClb':
        return(GClb_conductance(model_type, cell_type)) 

def GKr_conductance(model_type,cell_type):
    if model_type == "O'Hara Rudy 2010":
        if cell_type == "EPI":
            Gkr = 0.0598
        elif cell_type == "ENDO":
            Gkr = 0.046
        elif cell_type == "M":
            Gkr = 0.0368
    if model_type == "Grandi Bers 2010":
        Gkr = 0.035 
    if model_type == "Shannon 2004":
        Gkr = 0.03
    if model_type == "Ten Tusscher 2006":
        Gkr = 0.153       
    if model_type == "CiPA 2017":
        if cell_type == "EPI":
            Gkr = 0.06055946
        elif cell_type == "ENDO":
            Gkr = 0.0465842
        elif cell_type == "M":
            Gkr = 0.03726736   
    if model_type == "TOR 2019" or model_type == "TOR-DynCL 2020" or model_type == "BARS 2022":
        if cell_type == "EPI":
            Gkr = 1.3 * 0.0321
        elif cell_type == "ENDO":
            Gkr = 0.0321
        elif cell_type == "M":
            Gkr = 0.8 * 0.0321 
    if model_type == "T-World 2025": 
        if cell_type == "EPI":
            Gkr = 1.25*0.043
        elif cell_type == "ENDO":
            Gkr = 0.043
        elif cell_type == "M":
            Gkr = 0.7*0.043
    if model_type == "Morotti 2021":
        Ko = 5.4   # Extracellular K   [mM]
        Gkr = 1*(3+0.5)*0.035*np.sqrt(Ko/5.4) # added 3*gkr from Pei-Chi
    if model_type == "ANN Mirams 2023":
        #taken from emulator.max_conds_center
        Gkr = 3.209900e-02
    if model_type == "BPS 2020" or model_type == "BPSLand 2022":
        Gkr=0.046*1.2
        if cell_type == "EPI":
            Gkr = Gkr*1.1
        elif cell_type == "M":
            Gkr = Gkr*0.8
            
    return (Gkr)     

def GKs_conductance(model_type,cell_type):
    if model_type == "O'Hara Rudy 2010":
        if cell_type == "EPI":
            Gks = 0.00476
        elif cell_type == "ENDO" or cell_type == "M":
            Gks = 0.0034
    if model_type == "Grandi Bers 2010":
        Gks = 0.0035 
    if model_type == "Shannon 2004":
        Gks = 0.07 # note that this is nt the GKs value, but the scaling factor. GKs is dynamic in this model and depends on Calcium concentrations
    if model_type == "Ten Tusscher 2006":
        if cell_type == "EPI" or cell_type == "M":
            Gks = 0.392
        elif cell_type == "ENDO" :
            Gks = 0.098      
    if model_type == "CiPA 2017":
        if cell_type == "EPI":
            Gks = 0.0089
        elif cell_type == "ENDO" or cell_type == "M":
            Gks = 0.006358  
    if model_type == "TOR 2019" or model_type == "TOR-DynCL 2020":
        if cell_type == "EPI":
            Gks = 1.4 * 0.0011 
        elif cell_type == "ENDO" or cell_type == "M":
            Gks = 0.0011 
    if model_type == "BARS 2022":
        if cell_type == "EPI":
            Gks = 1.4 * 0.0011 * 5
        elif cell_type == "ENDO" or cell_type == "M":
            Gks = 0.0011 * 5
    if model_type == "Morotti 2021":
        Gks = 2.5 # this is the scaling factor
    elif model_type == "T-World 2025":
        Gks = 2.97 # the default multiplier of the current (the variable ‘gks_factor_SA’ in the code) changed from 2.5 to 2.97
        if cell_type == "M":
            Gks = 0.5*Gks
    if model_type == "ANN Mirams 2023":
        #taken from emulator.max_conds_center
        Gks = 1.099935e-03
    if model_type == "BPS 2020" or model_type == "BPSLand 2022":
        Gks = 0.0034*2
        if cell_type == "EPI":
            Gks = Gks*1.4
            
    return (Gks)  

def GNa_conductance(model_type,cell_type):
    if model_type == "O'Hara Rudy 2010":
        GNa_fast = 75
        if cell_type == "EPI":
            GNa_late = 0.0045
        elif cell_type == "ENDO" or cell_type == "M":
            GNa_late = 0.0075
        GNa = GNa_late, GNa_fast
    if model_type == "Grandi Bers 2010":
        GNa = 23.000
    if model_type == "Ten Tusscher 2006":
        GNa = 14.838       
    if model_type == "CiPA 2017":
        GNa_fast = 75
        if cell_type == "EPI":
            GNa_late = 0.0119745
        elif cell_type == "ENDO" or cell_type == "M":
            GNa_late = 0.0199575 
        GNa = GNa_late, GNa_fast
    if model_type == 'Shannon 2004':
        GNa_fast = 16
        GNa_late = 0.0045
        GNa = GNa_late, GNa_fast
    if model_type == 'Morotti 2021':
        GNa_fast = 0.9* 1*16        # [mS/uF]  ###
        GNa_late = 1*2*0.0065*2.5        # [mS/uF]
        GNa = GNa_late, GNa_fast
    if model_type == "TOR 2019" or model_type == "TOR-DynCL 2020" or model_type == "BARS 2022":
        GNa_fast = 11.7802
        if cell_type == "EPI":
            GNa_late = 0.6*0.0279
        elif cell_type == "ENDO" or cell_type == "M":
            GNa_late = 0.0279
        GNa = GNa_late, GNa_fast
    if model_type == "T-World 2025":
        GNa_fast = 22.08788
        if cell_type == "EPI":
            GNa_late = 0.7*0.04229 # Epicardial cells have 70% INaL of the endocardium
        elif cell_type == "ENDO" or cell_type == "M":
            GNa_late = 0.04229
        GNa = GNa_late, GNa_fast
    if model_type == "ANN Mirams 2023":
        #taken from emulator.max_conds_center
        GNa_fast = 1.177950e+01
        GNa_late = 2.789830e-02
        GNa = GNa_late, GNa_fast
    if model_type == "BPS 2020"or model_type == "BPSLand 2022":
        GNa_fast=75*0.27
        GNa_late=0.0075*2.8
        if cell_type== "EPI":
            GNa_late = GNa_late*0.7
        GNa = GNa_late, GNa_fast
        
    return (GNa) 

def GK1_conductance(model_type,cell_type):
    if model_type == "O'Hara Rudy 2010":
        if cell_type == "EPI":
            GK1 = 0.22896
        elif cell_type == "ENDO":
            GK1 = 0.1908
        elif cell_type == "M":
            GK1 = 0.24804
    if model_type == "Grandi Bers 2010":
        GK1 = 0.35
    if model_type == "Shannon 2004":
        GK1 = 0.9
    if model_type == "Ten Tusscher 2006":
        GK1 = 5.405      
    if model_type == "CiPA 2017":
        if cell_type == "EPI":
            GK1 = 0.3887741
        elif cell_type == "ENDO":
            GK1 = 0.3239784
        elif cell_type == "M":
            GK1 = 0.4211719
    if model_type == "TOR 2019" or model_type == "TOR-DynCL 2020" or model_type == "BARS 2022":
        if cell_type == "EPI":
            GK1 = 1.2*0.6992
        elif cell_type == "ENDO":
            GK1 = 0.6992
        elif cell_type == "M":
            GK1 = 1.3*0.6992
    if model_type=='T-World 2025':
        if cell_type == "EPI":
            GK1 = 1.1*0.6992
        elif cell_type == "ENDO":
            GK1 = 0.6992
        elif cell_type == "M":
            GK1 = 1.3*0.6992
    if model_type == "Morotti 2021":
        Ko = 5.4
        GK1 = 1*0.35*np.sqrt(Ko/5.4)
    if model_type == "ANN Mirams 2023":
        #taken from emulator.max_conds_center
        GK1 = 6.991550e-01
    if model_type == "BPS 2020"or model_type == "BPSLand 2022":
        GK1 = 0.1908*0.71
        if cell_type=="EPI":
            GK1 = GK1*1.2
        elif cell_type=="M":
            GK1=GK1*1.3
    return (GK1)   

def Gto_conductance(model_type,cell_type):
    if model_type == "O'Hara Rudy 2010" or model_type == "CiPA 2017":
        if cell_type == "EPI" or cell_type=="M":
            Gto = 0.08
        elif cell_type == "ENDO":
            Gto = 0.02
    if model_type == "TOR 2019" or model_type == "TOR-DynCL 2020" or model_type == "BARS 2022":
        if cell_type == "EPI" or cell_type=="M":
            Gto = 2.0*0.16
        elif cell_type == "ENDO":
            Gto = 0.16
    if model_type == "Grandi Bers 2010" or model_type == "Morotti 2021":
        if cell_type == 'ENDO' or cell_type == 'M':
            Gto_slow = 0.13*0.3*0.964
            Gto_fast = 0.13*0.3*0.036
        elif cell_type == 'EPI':
            Gto_slow = 0.13*0.12
            Gto_fast = 0.13*0.88
        Gto = Gto_slow,Gto_fast
    if model_type == "Shannon 2004":
        Gto_slow = 0.06*1#0.13*0.3*0.964
        Gto_fast = 0.02*1#0.13*0.3*0.036
        Gto = Gto_slow,Gto_fast
    if model_type == "Ten Tusscher 2006":
        if cell_type == "EPI" or cell_type=="M":
            Gto = 0.294
        elif cell_type == "ENDO":
            Gto = 0.073 
    if model_type == "T-World 2025":
        if cell_type == 'ENDO':
            Gto_slow = 0.0721
            Gto_fast = 0.01276
        elif cell_type == 'M':
            Gto_slow = 0.04632
            Gto_fast = 0.14928
        elif cell_type == 'EPI':
            Gto_slow = 0.02036
            Gto_fast = 0.29856
        Gto = Gto_slow,Gto_fast  
    if model_type == "ANN Mirams 2023":
        #taken from emulator.max_conds_center
        Gto = 1.599900e-01
    if model_type == "BPS 2020"or model_type == "BPSLand 2022":
        Gto = 0.02*1
        if cell_type == "EPI" or cell_type== "M":
            Gto=Gto*4.0
    return (Gto) 

def GCa_conductance(model_type,cell_type):
    if model_type == "O'Hara Rudy 2010" or model_type == "CiPA 2017":
        if cell_type == "EPI":
            Gca = 1.2*0.0001
        elif cell_type == "ENDO":
            Gca = 0.0001
        elif cell_type == "M":
            Gca = 2.5*0.0001   
        if model_type == "CiPA 2017":
            pca_scaling_factor = 1.007
            Gca = Gca * pca_scaling_factor
    if model_type == "Grandi Bers 2010":
        pCa = 0.50*5.4e-4
        pNa = 0.50*1.5e-8
        pK =  0.50*2.7e-7 
        Gca = pCa, pNa, pK
    if model_type == "Morotti 2021":
        pCa = 0.47*5.4e-4
        pNa = 0.47*1.5e-8
        pK =  0.47*2.7e-7 
        Gca = pCa, pNa, pK
    if model_type == "Shannon 2004":
        pCa = 5.4e-4
        pNa = 1.5e-8
        pK =  2.7e-7 
        Gca = pCa, pNa, pK
    if model_type == "Ten Tusscher 2006":
        Gca = 0.0000398         
    if model_type == "TOR 2019" or model_type == "TOR-DynCL 2020" or model_type == "BARS 2022":
        if cell_type == "EPI":
            Gca = 1.2* 8.3757e-05 # epicardium includes a 2.5% increase over endocardium (rather than the 20% in ToR-ORd),
        elif cell_type == "ENDO":
            Gca = 8.3757e-05 
        elif cell_type == "M":
            Gca = 1.1*8.3757e-05 # and midmyocardium includes a 10% increase over endocardium (rather than 100% in ToR-ORd)
    if model_type == "T-World 2025":
        if cell_type == "EPI":
            Gca = 1.025* 8.3757e-05 
        elif cell_type == "ENDO":
            Gca = 8.3757e-05 
        elif cell_type == "M":
            Gca = 2*8.3757e-05 
    if model_type == "ANN Mirams 2023":
        #taken from emulator.max_conds_center
        Gca = 8.375450e-05
    if model_type == "BPS 2020" or model_type == "BPSLand 2022":
        Gca=0.0001*0.9
        if cell_type == "EPI":
            Gca = Gca*1.4
        elif cell_type == "M":
            Gca = Gca *2
    return (Gca)  

def GNCX_conductance(model_type,cell_type):
    if model_type == "O'Hara Rudy 2010" or model_type == "CiPA 2017":
        if cell_type == "EPI":
            GNCX = 1.1*0.0008
        elif cell_type == "ENDO":
            GNCX = 0.0008
        elif cell_type == "M":
            GNCX = 1.4*0.0008
    if model_type == "Grandi Bers 2010" or model_type == "Morotti 2021":
        GNCX = 1.0*4.5
    if model_type == "Shannon 2004":
        GNCX = 1.0*9.0
    if model_type == "Ten Tusscher 2006":
        GNCX = 1000  #kNaCa in TP    
    if model_type == "TOR 2019" or model_type == "TOR-DynCL 2020" or model_type == "BARS 2022":
        if cell_type == "EPI":
            GNCX = 1.1*0.0034
        elif cell_type == "ENDO":
            GNCX = 0.0034
        elif cell_type == "M":
            GNCX = 1.4*0.0034 
    if model_type == "ANN Mirams 2023":
        #taken from emulator.max_conds_center
        GNCX = 3.399790e-03
    if model_type == "BPS 2020"or model_type == "BPSLand 2022":
        GNCX = 0.0008*2.4
        if cell_type == "EPI":
            GNCX = GNCX*1.2
        elif cell_type == "M":
            GNCX = GNCX*1.4 
    return (GNCX) 

def GNaK_conductance(model_type,cell_type):
    if model_type == "O'Hara Rudy 2010" or model_type == "CiPA 2017":
        if cell_type == "EPI":
            GNaK = 0.9*30
        elif cell_type == "ENDO":
            GNaK = 30
        elif cell_type == "M":
            GNaK = 0.7*30
    if model_type == "Grandi Bers 2010" or model_type == "Morotti 2021":
        GNaK = 1.0*1.8
    if model_type == "Ten Tusscher 2006":
        GNaK = 2.724  
    if model_type == 'Shannon 2004':
        GNaK = 1.90719       
    if model_type == "TOR 2019" or model_type == "TOR-DynCL 2020" or model_type == "BARS 2022":
        if cell_type == "EPI":
            GNaK = 0.9*15.4509 
        elif cell_type == "ENDO":
            GNaK = 15.4509 
        elif cell_type == "M":
            GNaK =  0.7*15.4509 
    if model_type == "ANN Mirams 2023":
        #taken from emulator.max_conds_center
        GNaK = 1.544995e+01
    if model_type == "BPS 2020"or model_type == "BPSLand 2022":
        GNaK = 30*2
        if cell_type == "EPI":
            GNaK = GNaK*0.9
        elif cell_type == "M":
            GNaK = GNaK*0.7
            
    return (GNaK) 

def GKb_conductance(model_type,cell_type):
    if model_type == "O'Hara Rudy 2010" or model_type == "CiPA 2017" or model_type == "BPS 2020"or model_type == "BPSLand 2022":
        if cell_type == "EPI":
            GKb = 0.6*0.003
        elif cell_type == "ENDO" or cell_type == "M":
            GKb = 0.003
    if model_type == "Grandi Bers 2010" or model_type == "Morotti 2021":
        GKb = 2*0.001 # this is referred to as gkp in the script, so I am not sure if it is the same things
    if model_type == "Shannon 2004":
        GKb = 0.001
    if model_type == "Ten Tusscher 2006":
        GKb = 0.0146 # this is referred to as GpK in the script, not sure if they are the same thing     
    if model_type == "TOR 2019" or model_type == "TOR-DynCL 2020" or model_type == 'BARS 2022':
        if cell_type == "EPI":
            GKb =  0.6 * 0.0189
        elif cell_type == "ENDO" or cell_type == "M":
            GKb = 0.0189
    return (GKb)  

def GNab_conductance(model_type,cell_type):
    if model_type == "O'Hara Rudy 2010" or model_type == "CiPA 2017" or model_type == "BPS 2020"or model_type == "BPSLand 2022":
        GNab = 3.75E-10
    if model_type == "Grandi Bers 2010":
        GNab = 0.597e-3
    if model_type == "Ten Tusscher 2006":
        GNab = 0.00029 
    if model_type == 'Shannon 2004':
        GNab = 0.297e-3  
    if model_type == "TOR 2019" or model_type == "TOR-DynCL 2020" or model_type == "BARS 2022":
        GNab = 1.9239e-09
    if model_type == "Morotti 2021":
        GNab = 0.5*1*0.597e-3 
    return (GNab)

def GCab_conductance(model_type,cell_type):
    if model_type == "O'Hara Rudy 2010" or model_type == "CiPA 2017":
        GCab = 2.5E-8
    if model_type == "Grandi Bers 2010":
        GCab = 5.513e-4
    if model_type == "Shannon 2004":
        GCab = 2.513e-4
    if model_type == "Ten Tusscher 2006":
        GCab = 0.000592   
    if model_type == "TOR 2019" or model_type == "TOR-DynCL 2020" or model_type == "BARS 2022":
        GCab = 5.9194e-08
    if model_type == "Morotti 2021":
        GCab = 0.5*1*5.513e-4    # [uA/uF] ###
    if model_type == "BPS 2020"or model_type == "BPSLand 2022":
        GCab = 2.5e-8*4
    return (GCab)

def GClb_conductance(model_type,cell_type):
    if model_type == "Grandi Bers 2010" or model_type == "Shannon 2004" or model_type=="BPS 2020"or model_type == "BPSLand 2022":
        GClb = 1*9e-3
    if model_type == "TOR 2019" or model_type == "TOR-DynCL 2020" or model_type == "BARS 2022":
        GClb = 1.98e-3
    if model_type == "Morotti 2021":
        GClb = 0.5*1*9e-3
    return (GClb)

def GClCa_conductance(model_type,cell_type):
    if model_type == "Shannon 2004" or model_type == "Morotti 2021":
        GClCa = 0.109625
    if model_type == "Grandi Bers 2010" or model_type== "BPS 2020"or model_type == "BPSLand 2022":
        GClCa = 0.5* 0.109625
    if model_type == "TOR 2019" or model_type == "TOR-DynCL 2020" or model_type == "BARS 2022":
        GClCa = 0.2843
    return (GClCa)

def GpCa_conductance(model_type,cell_type):
    if model_type == "O'Hara Rudy 2010" or model_type == "CiPA 2017" or model_type == "BPS 2020"or model_type == "BPSLand 2022":
        GpCa = 0.0005
    if model_type == "Grandi Bers 2010" or model_type == "Shannon 2004" or model_type == "Morotti 2021":
        GpCa = 0.0673 # this is referred to as IbarSLCaP, not sure if they are the same thing
    if model_type == "Ten Tusscher 2006":
        GpCa = 0.1238   
    if model_type == "TOR 2019" or model_type == "TOR-DynCL 2020" or model_type == "BARS 2022":
        GpCa = 5e-04
    return (GpCa)