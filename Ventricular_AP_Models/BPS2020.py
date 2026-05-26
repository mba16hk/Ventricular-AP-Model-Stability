import numpy as np
import math
from numba import njit
from scipy.integrate import solve_ivp
import pandas as pd
import matplotlib.pyplot as plt
from conductances import *

###########################################################################
# Matlab file Bartolucci-Passini-Severi Model (2020) from:
# https://www.mcbeng.it/en/downloads/software/16-bps2020-model.html
# Related Article
# "Simulation of the effects of extracellular calcium changes leads to a novel 
# computational model of human ventricular action potential with a revised calcium handling", 
# Front. Physiol., 15 April 2020 | https://doi.org/10.3389/fphys.2020.00314
# 
#
# Original Matlab file O'Hara-Rudy Human Ventricular Model (2011) from:
# http://rudylab.wustl.edu/research/cell/code/AllCodes.html
# Related Article
# http://www.ncbi.nlm.nih.gov/pubmed/21637795
#
# * Structure & ICaL modified by Chiara (Last Update: march 3, 2020)
#
###########################################################################
## Optional Inputs:
# (Default values *)
# 1) flag_ode:
#    - flag_ode=0  -> "computed variables" output
#    - flag_ode=1* -> dX output
# 2) celltype:
#    - celltype=0* -> endo
#    - celltype=1  -> epi
#    - celltype=2  -> M
# 3) pstim: stimulation protocol and parameters
#    - pstim=1*             -> I-clamp, single beat
#    - pstim=[2 CL] -> I-clamp, multiple beats with CL as input
#    - pstim=[3 vclamp] -> Ely V-clamp, single V-step
#    - pstim=[4 vclamp] -> Grandi V-clamp, single V-step
# 4) Extracellular Ionic Concentrations [cCao cNao cKo] mM:
#    default values: Cao = 1.8mM* Nao = 144mM* cKo  = 5.4mM*
#    - if length=1 -> [cCao] only
#    - if length=2 -> [cCao cNao]
#    - if length=3 -> [cCao cNao cKo]
# 5) bn: ICaL CDI block (VDI-only ICaL)
#    - bn=1* -> no CDI block
#    - bn=0  -> CDI total block
# 6) Ib: currents block -> [0-1] for each current/flux (11):
#    Ib=[bINa bINaL bIto bICaL bIKr bIKs bIK1 bINaCa bINaK bJup bJrel]
###########################################################################
# @njit
# def IStim(t,cycleLength,Stimdur,amplitude):
#     if np.mod(t,cycleLength) <= Stimdur: #5
#         I_app = amplitude #9.5
#     else:
#         I_app = 0.0
#     return(I_app)


def run_BPS_model(cycles, cycleLength, cell_type, amp=-53):
    model_type  = "BPS 2020"
    GKs         = GKs_conductance(model_type, cell_type)
    GKr         = GKr_conductance(model_type, cell_type)
    GK1         = GK1_conductance(model_type, cell_type)
    Gto         = Gto_conductance(model_type, cell_type)
    GNa_late, GNa_fast = GNa_conductance(model_type, cell_type)
    GCa         = GCa_conductance(model_type, cell_type)
    GNCX        = GNCX_conductance(model_type, cell_type)
    GNaK        = GNaK_conductance(model_type, cell_type)
    GKb         = GKb_conductance(model_type, cell_type)
    GNab        = GNab_conductance(model_type, cell_type)
    GCab        = GCab_conductance(model_type, cell_type)
    GpCa        = GpCa_conductance(model_type, cell_type)
    GClCa_input = GClCa_conductance(model_type, cell_type)
    GClb        = GClb_conductance(model_type, cell_type)

    initial_conds_BPS2020 = [  -87 ,           7,              7,      145,          145 ,      # 1.  v  nai  nass  ki  kss
                1.0e-4 ,        1.0e-4,         0,      1.2,          0 ,        # 2.  cai  cass  0  casr  m 
                1,              1,              1,      1  ,          1 ,        # 3.  hf  hs  j  hsp  jp
                0,              1,              1,      0,            1 ,        # 4.  mL  hL  hLp  a  iF
                1,              0,              1,      1,            0 ,        # 5.  iS  ap  iFp  iSp 0 
                0,              0 ,             0,      0,            0 ,        # 6.  Ok  Okp  0  0  0 
                0,              0,              0,      0,            0 ,        # 7.  nca  0  0  xrf  xrs
                0,              0,              1,      0,            0 ,        # 8.  xs1  xs2  xk1  Jrelnp  Jrelp
                0,              0,              0,      1,            0 ,        # 9.  CaMKt  I1k  I2k  Ck  I1kp
                0,              1,              0,      0,            0 ,        # 10. I2kp  Ckp  I1Cak  I2Cak  CCak
                0,              0,              0,      1,            0 ,        # 11. I1Cakp I2Cakp CCakp jnca EGTA
                0,              0.03,           0,      1,            1 ,        # 12. EGTAi  RyRa  RyRo  RyRc  RyRcp
                1.80145e-8,     8.26619e-5,     0,      0,            0.999637 , # 13. C1_dutta C2_dutta Cb_dutta D_dutta IC1_dutta
                6.83208e-5,     5.67623e-5,     0.0,    0.00015551,   0]         # 14. IC2_dutta IO_dutta IOb_dutta O_dutta Ob_dutta


    # Set default values for optional inputs
    #cEx0 = [1.8,144,5.4] #
    # cell_type = 0
    amp = amp #-53
    duration = 1
    nao = 144
    cao = 1.8
    ko = 5.4
    #constants
    maxEGTA0 = 0
    ### Phosphorilation on/off
    undo_p = 0
    ### CDI on/off
    undo_CDI = 0
    ### ICab on/off
    undo_ICab = 0
    ###########################################################################
    ## Physical Constants:
    R = 8314.0   # J/kmol/K
    T = 310.0    # K
    F = 96485.0  # C/mol
    L = 0.01                         # cm
    rad = 0.0011                     # cm
    vcell = 1000*math.pi*rad**2*L          # 38e-6 uL
    # Geometric Area
    Ageo = 2*math.pi*rad**2 + 2*math.pi*rad*L   # cm**2
    # Capacitive Area
    Acap = 2*Ageo                    # cm**2
    # Compartment Volumes (4)
    vmyo = 0.68*vcell                # uL
    vnsr = 0.0552*vcell              # uL
    vjsr = 0.0048*vcell              # uL
    vsr = 0.95*(vnsr+vjsr)           # uL
    vss  = 1*0.02*vcell              # uL

    ###########################################################################
    ## CaMK Constants
    KmCaMK = 0.15  
    aCaMK  = 0.05  
    bCaMK  = 0.00068
    CaMKo  = 0.05
    KmCaM  = 0.0015

    ###########################################################################
    #,GNab,GCab,GCa,GNCX,GpCa,GKb,GNaK,GClCa_input,GClb,stimDur,amp
    GNa=GNa_fast#75*0.27
    GNaL=GNa_late#0.0075*2.8
    #Gto=0.02*1
    PCa=GCa#0.0001*0.9
    #GKr=0.046*1.2
    #GKs=0.0034*2
    #GK1=0.1908*0.71
    Gncx=GNCX#0.0008*2.4
    GClCa =GClCa_input#0.5* 0.109625   # [mS/uF]
    GClB = GClb#1*9e-3        # [mS/uF]
    PCab = GCab#2.5e-8*4
    #GpCa = 0.0005
    Pnak=GNaK#30*2
    #GKb = 0.003
    PNab = GNab#3.75e-10  
    
    thf_shift = 0.075 # thf
    btj = 1           # tj
    PKNa = 0.01833
    Ahf=0.99
    Ahs=1.0-Ahf
    bthL = 1
    bGCaL =1
    kCDI = 9
    tjnca  = 1
    Kmn  = 0.05
    k2n  = 1000
    ktaup = 2.5

    ### IKr parameters ###
    k1_dutta=1 
    k2_dutta=1
    k3_dutta=1 
    k4_dutta=1
    k11_dutta=1 
    k21_dutta=1
    k31_dutta=1 
    k41_dutta=1
    k51_dutta=1 
    k61_dutta=1
    k52_dutta=1 
    k62_dutta=1
    k53_dutta=1 
    k63_dutta=1
    kD_dutta=1
    ###
    A1_dutta = 0.0264   # per_millisecond (in IKr)
    A11_dutta = 0.0007868   # per_millisecond (in IKr)
    A2_dutta = 4.986e-6   # per_millisecond (in IKr)
    A21_dutta = 5.455e-6   # per_millisecond (in IKr)
    A3_dutta = 0.001214   # per_millisecond (in IKr)
    A31_dutta = 0.005509   # per_millisecond (in IKr)
    A4_dutta = 1.854e-5   # per_millisecond (in IKr)
    A41_dutta = 0.001416   # per_millisecond (in IKr)
    A51_dutta = 0.4492   # per_millisecond (in IKr)
    A52_dutta = 0.3181   # per_millisecond (in IKr)
    A53_dutta = 0.149   # per_millisecond (in IKr)
    A61_dutta = 0.01241   # per_millisecond (in IKr)
    A62_dutta = 0.3226   # per_millisecond (in IKr)
    A63_dutta = 0.008978   # per_millisecond (in IKr)
    B1_dutta = 4.631e-5   # per_millivolt (in IKr)
    B11_dutta = 1.535e-8   # per_millivolt (in IKr)
    B2_dutta = -0.004226   # per_millivolt (in IKr)
    B21_dutta = -0.1688   # per_millivolt (in IKr)
    B3_dutta = 0.008516   # per_millivolt (in IKr)
    B31_dutta = 7.771e-9   # per_millivolt (in IKr)
    B4_dutta = -0.04641   # per_millivolt (in IKr)
    B41_dutta = -0.02877   # per_millivolt (in IKr)
    B51_dutta = 0.008595   # per_millivolt (in IKr)
    B52_dutta = 3.613e-8   # per_millivolt (in IKr)
    B53_dutta = 0.004668   # per_millivolt (in IKr)
    B61_dutta = 0.1725   # per_millivolt (in IKr)
    B62_dutta = -0.0006575   # per_millivolt (in IKr)
    B63_dutta = -0.02215   # per_millivolt (in IKr)
    # GKr_b = 0.046585   # milliS_per_microF (in IKr)
    Kmax_IKr = 0.0   # dimensionless (in IKr)
    Kt_dutta = 3.5e-5   # per_millisecond (in IKr)
    Ku_dutta = 0.0   # per_millisecond (in IKr)
    Temp_dutta = 37.0   # celsius (in IKr)
    halfmax_dutta = 1.0   # dimensionless (in IKr)
    n_dutta = 1.0   # dimensionless (in IKr)
    q1_dutta = 4.843   # dimensionless (in IKr)
    q11_dutta = 4.942   # dimensionless (in IKr)
    q2_dutta = 4.23   # dimensionless (in IKr)
    q21_dutta = 4.156   # dimensionless (in IKr)
    q3_dutta = 4.962   # dimensionless (in IKr)
    q31_dutta = 4.22   # dimensionless (in IKr)
    q4_dutta = 3.769   # dimensionless (in IKr)
    q41_dutta = 1.459   # dimensionless (in IKr)
    q51_dutta = 5.0   # dimensionless (in IKr)
    q52_dutta = 4.663   # dimensionless (in IKr)
    q53_dutta = 2.412   # dimensionless (in IKr)
    q61_dutta = 5.568   # dimensionless (in IKr)
    q62_dutta = 5.0   # dimensionless (in IKr)
    q63_dutta = 5.682   # dimensionless (in IKr)
    vhalf_dutta = 1.0   # millivolt (in IKr)
    kslope_rk1 = 1.09

    ### INaCa parameter ###
    ###
    kna1=15.0      
    kna2=5.0       
    kna3=88.12     
    kasymm=12.5
    wna=6.0e4      
    wca=6.0e4      
    wnaca=5.0e3    
    KmCaAct=150.0e-6
    kcaon = 1.5e6    
    kcaoff=5.0e3   
    qna=0.5224     
    qca=0.1670
    zna=1.0        
    zca=2.0
    ### INaK parameter ###
    ###
    k1p=949.5      
    k1m=182.4      
    k2p=687.2      
    k2m=39.4
    k3p=1899.0     
    k3m=79300.0    
    k4p=639.0      
    k4m=40.0
    Knai0=9.073    
    Knao0=27.78    
    delta2=-0.1550
    Kki=0.5            
    Kko=0.3582     
    MgADP=0.05     
    MgATP=9.8
    Kmgatp=1.698e-7    
    H=1.0e-7       
    eP=4.2         
    Khp=1.698e-7
    Knap=224.0         
    Kxkur=292.0

    zk=1.0   
    IClCa_si = 0
    IClb_si = 0
    ###
    Cli = 15   # Intracellular Cl  [mM]
    Clo = 150  # Extracellular Cl  [mM]
    ecl = (R*T/F)*np.log(Cli/Clo) # [mV]
    KdClCa = 100e-3    # [mM]
    ### ICab parameter ###

    ### Jrel parameters ###
    g_irel_max	= 20*10**-3 # millimolar_per_second (in calcium_dynamics)
    RyRa1		= 0.05  # uM
    RyRa2		= 0.03   # uM
    RyRohalf	= 0.12-(RyRa1-RyRa2/2)	# uM
    RyRchalf	= 0.10-(RyRa1-RyRa2/2) # uM
    bJdiff= 1.7
    RyRtauadapt = 1000 #ms
    RyRtauact = 18.75*10**-1/1.875       #ms
    RyRtauinact = 2*87.5*1/10    #ms
    ### SERCA Jup parameter ###
    cJup = 3.13
    Vmax_SRCaP = 1.0*5.3114e-3  # [mM/msec] (286 umol/L cytosol/sec)
    Kmf = 0.246e-3          # [mM] default
    Kmr = 1.7               # [mM]L cytosol
    hillSRCaP = 1.787       # [mM]
    ### Calcium Buffer Constants ###
    cmdnmax=0.05
    kmcmdn=0.00238     
    trpnmax=0.07   
    kmtrpn=0.0005
    BSRmax=0.047       
    KmBSR=0.00087
    BSLmax=1.124       
    KmBSL=0.0087
    ### buffering scaling 
    csqnmax=10.0*10**-1 # ORd csqnmax=10.0 
    kmcsqn=0.8
    isepi = 0
    Jup_scale = 1
    g_irel_max_p = g_irel_max*1.25
    
    if cell_type=="EPI":
        #GKb = GKb*0.6
        Jup_scale=Jup_scale*1.3
        cmdnmax=cmdnmax*1.2 # modified  respect ORd (*1.3)
        isepi = 1
        #Pnak=Pnak*0.9
        #Gncx=Gncx*1.2
        #GK1=GK1*1.2
        #GKs=GKs*1.4
        #GKr = GKr*1.1 # modified respect ORd (*1.3)
        #PCa=PCa*1.4
        #Gto=Gto*4.0
        #GNaL=GNaL*0.7
    elif cell_type=="M":
        #Pnak=Pnak*0.7
        #Gncx=Gncx*1.4
        #GK1=GK1*1.3
        g_irel_max_p = g_irel_max*1.25*1.7
        #GKr = GKr*0.8 
        #PCa=PCa*2
        #Gto=Gto*4.0 
    
    
    PCap=1.1*PCa
    PCaNa=0.00125*PCa
    PCaK=3.574e-4*PCa
    PCaNap=0.00125*PCap
    PCaKp=3.574e-4*PCap
    reversal_constants = (R*T/F)
    
    
    
    constants = [cycleLength,cell_type,
            amp,duration,nao,cao,ko,maxEGTA0,undo_p,undo_CDI,undo_ICab,R,T,F,Acap,vmyo,
            vsr,vss,KmCaMK,aCaMK,bCaMK,CaMKo,KmCaM,GNa,thf_shift,btj,PKNa,
            Ahf,Ahs,bthL,GNaL,Gto,bGCaL,kCDI,tjnca,Kmn,k2n,ktaup,k1_dutta,k2_dutta,k3_dutta,
            k4_dutta,k11_dutta,k21_dutta,k31_dutta,k41_dutta,k51_dutta,k61_dutta,k52_dutta,k62_dutta,
            k53_dutta,k63_dutta,kD_dutta,A1_dutta,A11_dutta,A2_dutta,A21_dutta,A3_dutta,A31_dutta,A4_dutta,
            A41_dutta,A51_dutta,A52_dutta,A53_dutta,A61_dutta,A62_dutta,A63_dutta,B1_dutta,B11_dutta,B2_dutta,
            B21_dutta,B3_dutta,B31_dutta,B4_dutta,B41_dutta,B51_dutta,B52_dutta,B53_dutta,B61_dutta,B62_dutta,
            B63_dutta,Kmax_IKr,Kt_dutta,Ku_dutta,Temp_dutta,halfmax_dutta,n_dutta,q1_dutta,q11_dutta,q2_dutta,
            q21_dutta,q3_dutta,q31_dutta,q4_dutta,q41_dutta,q51_dutta,q52_dutta,q53_dutta,q61_dutta,q62_dutta,
            q63_dutta,vhalf_dutta,PCa,GKr,GKs,kslope_rk1,GK1,kna1,kna2,kna3,kasymm,
            wna,wca,wnaca,KmCaAct,kcaon,kcaoff,qna,qca,zna,Gncx,zca,k1p,k1m,k2p,k2m,k3p,k3m,k4p,k4m,Knai0,
            Knao0,delta2,Kki,Kko,MgADP,MgATP,Kmgatp,H,eP,Khp,Knap,Kxkur,zk,Pnak,IClCa_si,IClb_si,ecl,GClCa,GClB,
            KdClCa,PCab,GpCa,g_irel_max,RyRa1,RyRa2,RyRohalf,RyRchalf,GKb,PNab,bJdiff,RyRtauadapt,RyRtauact,
            RyRtauinact,cJup,cmdnmax,kmcmdn,trpnmax,kmtrpn,BSRmax,KmBSR,BSLmax,KmBSL,csqnmax,kmcsqn,
            Jup_scale,PCap,PCaNa,PCaK,PCaNap,PCaKp,reversal_constants,isepi,g_irel_max_p]
    
    # constants = [cycleLength,cell_type,amp,duration]
    
    tspan = (0, cycles*cycleLength)
    initial_conds = initial_conds_BPS2020
    sol = solve_ivp(fun = BPS2020, t_span = tspan, y0 = initial_conds, args = constants,method='LSODA',
                    rtol= 1e-8,atol = 1e-8,max_step = 1) #, t_eval=np.linspace(tspan[0], tspan[1], 300000)

    # dydt = [dydt_ecc, dydt_camDyad, dydt_camSL, dydt_camCyt, dydt_CaMKIIDyad, dydt_BAR] # make sure dydt_BAR integrates well in python
    time = sol.t  # Time points
    solutions = sol.y  # Solution vectors, each row corresponds to a variable
    y_names = ["V","Nai",  "Nass",  "Ki", "Kss", "Cai",  "Cass",  "dd","Casr" , "dm", "dhf",  "dhs",  "dj" , "dhsp",  "djp", "dmL",
               "dhL",  "dhLp",
            "da", "diF", "diS","dap", "diFp",  "diSp", "val2","dOk" , "dOkp","dval2","dval3","dval4","dnca",'dval5', 'dval6','dval7','dval8', "dxs1",  "dxs2",
            "dxk1", 'dval9','dval10',  "dCaMKt",  "dI1k" , "dI2k",  "dCk" , "dI1kp","dI2kp",  "dCkp",  "dI1Cak",  "dI2Cak",  "dCCak", "dI1Cakp",
            "dI2Cakp", "dCCakp", "djnca", "dEGTA", "dEGTAi",  "dRyRa",  "dRyRo",  "dRyRc",  "dRyRcp", "dC1_dutta", "dC2_dutta", "dCb_dutta", "dD_dutta",
            "dIC1_dutta","dIC2_dutta", "dIO_dutta", "dIOb_dutta", "dO_dutta", "dOb_dutta"]
    
    df = pd.DataFrame(solutions.T, columns=y_names)
    
    df['time'] = time
    stim_duration = duration
    return df, pd.DataFrame(), stim_duration

@njit
def BPS2020(time,initial_conds,cycleLength,celltype,
            amp,duration,nao,cao,ko,maxEGTA0,undo_p,undo_CDI,undo_ICab,R,T,F,Acap,vmyo,
            vsr,vss,KmCaMK,aCaMK,bCaMK,CaMKo,KmCaM,GNa,thf_shift,btj,PKNa,
            Ahf,Ahs,bthL,GNaL,Gto,bGCaL,kCDI,tjnca,Kmn,k2n,ktaup,k1_dutta,k2_dutta,k3_dutta,
            k4_dutta,k11_dutta,k21_dutta,k31_dutta,k41_dutta,k51_dutta,k61_dutta,k52_dutta,k62_dutta,
            k53_dutta,k63_dutta,kD_dutta,A1_dutta,A11_dutta,A2_dutta,A21_dutta,A3_dutta,A31_dutta,A4_dutta,
            A41_dutta,A51_dutta,A52_dutta,A53_dutta,A61_dutta,A62_dutta,A63_dutta,B1_dutta,B11_dutta,B2_dutta,
            B21_dutta,B3_dutta,B31_dutta,B4_dutta,B41_dutta,B51_dutta,B52_dutta,B53_dutta,B61_dutta,B62_dutta,
            B63_dutta,Kmax_IKr,Kt_dutta,Ku_dutta,Temp_dutta,halfmax_dutta,n_dutta,q1_dutta,q11_dutta,q2_dutta,
            q21_dutta,q3_dutta,q31_dutta,q4_dutta,q41_dutta,q51_dutta,q52_dutta,q53_dutta,q61_dutta,q62_dutta,
            q63_dutta,vhalf_dutta,PCa,GKr,GKs,kslope_rk1,GK1,kna1,kna2,kna3,kasymm,
            wna,wca,wnaca,KmCaAct,kcaon,kcaoff,qna,qca,zna,Gncx,zca,k1p,k1m,k2p,k2m,k3p,k3m,k4p,k4m,Knai0,
            Knao0,delta2,Kki,Kko,MgADP,MgATP,Kmgatp,H,eP,Khp,Knap,Kxkur,zk,Pnak,IClCa_si,IClb_si,ecl,GClCa,GClB,
            KdClCa,PCab,GpCa,g_irel_max,RyRa1,RyRa2,RyRohalf,RyRchalf,GKb,PNab,bJdiff,RyRtauadapt,RyRtauact,
            RyRtauinact,cJup,cmdnmax,kmcmdn,trpnmax,kmtrpn,BSRmax,KmBSR,BSLmax,KmBSL,csqnmax,kmcsqn,
            Jup_scale,PCap,PCaNa,PCaK,PCaNap,PCaKp,reversal_constants,isepi,g_irel_max_p
):
    
    ## Optional Inputs setting:
    
    # skip any new inputs, if empty
    # newVals = cellfun(@(x) ~isempty(x), varargin)
    # # overwrite inputs specified in varargin
    # optargs(newVals) = varargin(newVals)
    # [optargs{1:length(varargin)}] = varargin{:}
    # check_reversibility = []
    
    maxEGTA = maxEGTA0
    bn = 1
    
    ###########################################################################
   
    # Membrane Potential V
    v,nai,  nass,  ki,  kss, cai,  cass,  val1,casr , m, hf,  hs,  j , hsp,  jp, mL,  hL,  hLp,  a,  iF, iS,  ap,  iFp,  iSp, d,Ok , Okp,val2,val3,val4,nca,val5, val6,xrf,xrs, xs1,  xs2,  xk1, Jrelnp,Jrelp,  CaMKt,  I1k , I2k,  Ck , I1kp,I2kp,  Ckp,  I1Cak,  I2Cak,  CCak, I1Cakp, I2Cakp, CCakp, jnca, EGTA, EGTAi,  RyRa,  RyRo,  RyRc,  RyRcp, C1_dutta, C2_dutta, Cb_dutta, D_dutta, IC1_dutta,IC2_dutta, IO_dutta, IOb_dutta, O_dutta, Ob_dutta = initial_conds
   
   
    
    vffrt = v*F*F/(R*T)
    vfrt  = v*F/(R*T)

    # cansr = initial_conds[8] # state variable useful if consider JSR+NSR       
    # ICaL ORd state variables
    # d    = initial_conds[25)     ff   = initial_conds[26)  fs    = initial_conds[27)
    # fcaf = initial_conds[28)     fcas = initial_conds[29)  jca   = initial_conds[30)
    nca  = bn*nca   #ffp  = initial_conds[32)  fcafp = initial_conds[33)
    
    ###########################################################################
    ## update CaMK -> X(41)
    CaMKb = CaMKo*(1.0-CaMKt) / (1.0+KmCaM/cass)
    CaMKa = CaMKb+CaMKt
    dCaMKt = aCaMK*CaMKb*(CaMKb+CaMKt) - bCaMK*CaMKt
    ###########################################################################
    ## Reversal Potentials

    ENa  = reversal_constants*np.log(nao/nai)
    EK   = reversal_constants*np.log(ko/ki)
    EKs  = reversal_constants*np.log((ko+PKNa*nao)/(ki+PKNa*nai))
    EKshift = 8 # shift for compesation in Ito,IKs,IK1 due to LJP
    ###########################################################################
    ### INa current ###
    ### INa parameters ###
    
    ###
    mss=1.0/(1.0+np.exp((-(v+39.57))/9.871))
    tm=1.0/(6.765*np.exp((v+11.64)/34.77)+8.552*np.exp(-(v+77.42)/5.955))
    dm=(mss-m)/tm
    hss=1.0/(1+np.exp((v+78.5)/6.22))
    thf=1.0/(3.6860e-6*np.exp(-(v+3.8875)/7.8579)+16*np.exp((v-0.4963)/9.1843))+thf_shift
    ths=1.0/(0.009794*np.exp(-(v+17.95)/28.05)+0.3343*np.exp((v+5.730)/56.66))
    
    dhf=(hss-hf)/thf
    dhs=(hss-hs)/ths
    h=Ahf*hf+Ahs*hs
    jss=hss
    tj=(4.8590+1.0/(0.8628*np.exp(-(v+116.7258)/7.6005)+1.1096*np.exp((v+6.2719)/9.0358)))*btj
    dj=(jss-j)/tj
    hssp=1.0/(1+np.exp((v+84.7)/6.22))
    thsp=3.0*ths
    dhsp=(hssp-hsp)/thsp
    hp=Ahf*hf+Ahs*hsp
    tjp=1.46*tj
    djp=(jss-jp)/tjp
    
    fINap=(1.0/(1.0+KmCaMK/CaMKa))
    INa=GNa*(v-ENa)*m**3.0*((1.0-fINap)*h*j+fINap*hp*jp)
    ###########################################################################
    ### INaL current ###
    ### INaL parameters ###
    
    ###
    mLss=1.0/(1.0+np.exp((-(v+42.85))/5.264))
    tmL=tm
    dmL=(mLss-mL)/tmL
    hLss=1.0/(1.0+np.exp((v+87.61)/7.488))
    thL=200.0*bthL
    dhL=(hLss-hL)/thL
    hLssp=1.0/(1.0+np.exp((v+93.81)/7.488))
    thLp=3.0*thL
    dhLp=(hLssp-hLp)/thLp
    

    fINaLp=(1.0/(1.0+KmCaMK/CaMKa))
    INaL=GNaL*(v-ENa)*mL*((1.0-fINaLp)*hL+fINaLp*hLp)
    ###########################################################################
    ### Ito current ###
    ### Ito parameter ###
    
    ###
    ass=1.0/(1.0+np.exp((-(v+EKshift-14.34))/14.82))
    ta=1.0515 / (1.0/(1.2089*(1.0+np.exp(-(v+EKshift-18.4099)/29.3814)))+3.5/(1.0+np.exp((v+EKshift+100.0)/29.3814)))
    da=(ass-a)/ta
    iss=1.0/(1.0+np.exp((v+EKshift+43.94)/5.711))
    delta_epi=1.0
    if isepi:
        delta_epi=1.0-(0.95/(1.0+np.exp((v+EKshift+70.0)/5.0)))
        

    tiF=4.562+1/(0.3933*np.exp((-(v+EKshift+100.0))/100.0)+0.08004*np.exp((v+EKshift+50.0)/16.59))
    tiS=23.62+1/(0.001416*np.exp((-(v+EKshift+96.52))/59.05)+1.780e-8*np.exp((v+EKshift+114.1)/8.079))
    tiF=tiF*delta_epi
    tiS=tiS*delta_epi
    AiF=1.0/(1.0+np.exp((v-213.6+EKshift)/151.2))
    AiS=1.0-AiF
    diF=(iss-iF)/tiF
    diS=(iss-iS)/tiS
    i=AiF*iF+AiS*iS
    assp=1.0/(1.0+np.exp((-(v+EKshift-24.34))/14.82))
    dap=(assp-ap)/ta
    dti_develop=1.354+1.0e-4/(np.exp((v+EKshift-167.4)/15.89)+np.exp(-(v+EKshift-12.23)/0.2154))
    dti_recover=1.0-0.5/(1.0+np.exp((v+EKshift+70.0)/20.0))
    tiFp=dti_develop*dti_recover*tiF
    tiSp=dti_develop*dti_recover*tiS
    diFp=(iss-iFp)/tiFp
    diSp=(iss-iSp)/tiSp
    ip=AiF*iFp+AiS*iSp   

    fItop=(1.0/(1.0+KmCaMK/CaMKa))
    Ito=Gto*(v-EK)*((1.0-fItop)*a*i+fItop*ap*ip)
    ###########################################################################
    ### ICaL, ICaNa, ICaK current ###
    ### ICaL parameters ###
    
    ###
    # p vs np ICaL
    fICaLp=(1.0/(1.0+KmCaMK/CaMKa))*(1-undo_p)
    ###########################################################################
    ## up/down rates
    r_down = bn*(1e-1)*(1-undo_CDI)
    r_up = bn*(r_down*nca/(1-nca))*(1-undo_CDI)
    ###########################################################################
    ## n gate -> used for compute nca
    jncass = 1.0/(1.0+np.exp((v+19.58+25)/3.696))
    # fss=1.0/(1.0+np.exp((v+19.58)/3.696)) # ORd formulation
    
    djnca  =(jncass-jnca)/tjnca
    
    km2n = 150*jnca
    # anca=1/(k2n/km2n+(1.0+Kmn/cass)**4.0) # ORd formulation
    anca = (1-nca)/(1+Kmn/cass)**4.0
    dnca=bn*(anca*k2n-nca*km2n)
    ###########################################################################
    # Activation (d)
    dss = 1.0/(1.0+np.exp((-(v+3.940))/4.230))
    td  = (0.6+1.0/( np.exp( -0.05*(v+6))+np.exp(0.09*(v+14))) )
    alpha = dss/td
    beta = (1-dss) / td
    ###########################################################################
    # Recovery (jca)
    jcass_new= 1.0/(1.0+np.exp((v+19.58)/3.696))
    jcass_VD = jcass_new
    jcass_CD = jcass_new
    jcass_VDp = jcass_new
    jcass_CDp = jcass_new
    tjca_new = 35 + 350*np.exp(-(v-(-20))**2/(2*10**2))
    tjca_VD = tjca_new
    tjca_VDp = tjca_new
    tjca_CD = tjca_new
    tjca_CDp = tjca_new
    # psi and omega rates
    psi_VD=jcass_VD/tjca_VD
    psi_VDp=jcass_VDp/tjca_VDp
    psi_CD=jcass_CD/tjca_CD
    psi_CDp=jcass_CDp/tjca_CDp
    omega_VD=(1-jcass_VD)/tjca_VD
    omega_VDp=(1-jcass_VDp)/tjca_VDp
    omega_CD=(1-jcass_CD)/tjca_CD
    omega_CDp=(1-jcass_CDp)/tjca_CDp
    ###########################################################################
    # Fact Inactivation (f1)
    # ORd formulation #
    # f1ss  = 1.0 / (1.0+np.exp((v+19.58)/3.696))  
    # tff   = 7.0 + 1.0/(0.0045*np.exp(-(v+20.0)/10.0)+0.0045*np.exp((v+20.0)/10.0))
    # tfcaf = 7.0 + 1.0/(0.04*np.exp(-(v-4.0)/7.0)+0.04*np.exp((v-4.0)/7.0))
    #
    f1ss_0 = 0.8 / (1.0+np.exp((v+19.58)/3.696)) + 0.2
    tf1_0   = 1*(70 + 1.2/ (0.0045*np.exp((v+20)/(-50))+0.0045*np.exp((v+30)/10)))
    
    gamma_VD  = (1-f1ss_0)/ tf1_0
    delta_VD  = f1ss_0  / tf1_0
    gamma_VDp  = gamma_VD/ktaup
    delta_VDp  = delta_VD/ktaup
    gamma_CD=gamma_VD*kCDI
    delta_CD=delta_VD*kCDI
    gamma_CDp=gamma_VDp*kCDI
    delta_CDp=delta_VDp*kCDI
    # tf1_VD = 1/(gamma_VD+delta_VD)
    # tf1_CD = 1/(gamma_CD+delta_CD)
    # f1ss_VD = gamma_VD / (gamma_VD+delta_VD)
    # f1ss_CD = gamma_CD / (gamma_CD+delta_CD)
    ###########################################################################
    # Slow Inactivation (f2)
    # ORd formulation #
    # fss=1.0/(1.0+np.exp((v+19.58)/3.696))
    # tfs=1000.0+1.0/(0.000035*np.exp(-(v+5.0)/4.0)+0.000035*np.exp((v+5.0)/6.0))
    # tfcas=100.0+1.0/(0.00012*np.exp(-v/3.0)+0.00012*np.exp(v/7.0))
    #
    tf2_new   = 1*(100 + 0./ (0.0035*np.exp((v+5)/(-84))+0.0035*np.exp((v+5)/4)))
    tf2_VD = tf2_new
    tf2_CD = tf2_VD/kCDI
    tf2_VDp = tf2_new*ktaup
    tf2_CDp = tf2_VD/kCDI*ktaup
    # Reversibility
    theta_VD = alpha*gamma_VD*psi_VD/tf2_VD/(alpha*gamma_VD*psi_VD+beta*delta_VD*omega_VD)
    theta_CD = alpha*gamma_CD*psi_CD/tf2_CD/(alpha*gamma_CD*psi_CD+beta*delta_CD*omega_CD)
    theta_VDp = alpha*gamma_VDp*psi_VDp/tf2_VDp/(alpha*gamma_VDp*psi_VDp+beta*delta_VDp*omega_VDp)
    theta_CDp = alpha*gamma_CDp*psi_CDp/tf2_CDp/(alpha*gamma_CDp*psi_CDp+beta*delta_CDp*omega_CDp)
    eta_VD=1/tf2_VD - theta_VD
    eta_VDp=1/tf2_VDp - theta_VDp
    eta_CD=1/tf2_CD - theta_CD
    eta_CDp=1/tf2_CDp - theta_CDp

    tf2_VD = 1/(eta_VD+theta_VD)
    tf2_CD = 1/(eta_CD+theta_CD)
    # f2ss_VD=eta_VD/(eta_VD+theta_VD)
    # f2ss_CD=eta_CD/(eta_CD+theta_CD)
    ###########################################################################
    # Driving Forces
    PhiCaL=4.0*vffrt*(1.2*cass*np.exp(2.0*vfrt)-0.341*cao)/(np.exp(2.0*vfrt)-1.0)
    PhiCaNa=1.0*vffrt*(0.75*nass*np.exp(1.0*vfrt)-0.75*nao)/(np.exp(1.0*vfrt)-1.0)
    PhiCaK=1.0*vffrt*(0.75*kss*np.exp(1.0*vfrt)-0.75*ko)/(np.exp(1.0*vfrt)-1.0)

    ###########################################################################
    # Markov Model: VDI states
    OCak = 1-CCak-I1Cak-I2Cak-Ck-I1k-I2k-Ok
    OCakp = 1-CCakp-I1Cakp-I2Cakp-Ckp-I1kp-I2kp-Okp
    dOk =  alpha*Ck        + delta_VD*I1k    - (beta+gamma_VD)*Ok      - r_up*Ok + r_down*OCak
    dI2k = eta_VD*I1k      + omega_VD*Ck     - (theta_VD+psi_VD)*I2k   - r_up*I2k + r_down*I2Cak
    dI1k = theta_VD*I2k    + gamma_VD*Ok     - (eta_VD+delta_VD)*I1k   - r_up*I1k + r_down*I1Cak
    dCk  = beta*Ok         + psi_VD*I2k      - (omega_VD+alpha)*Ck     - r_up*Ck + r_down*CCak
    dOkp = alpha*Ckp       + delta_VDp*I1kp  - (beta+gamma_VDp)*Okp    - r_up*Okp + r_down*OCakp
    dI2kp = eta_VDp*I1kp    + omega_VDp*Ckp   - (theta_VDp+psi_VDp)*I2kp - r_up*I2kp + r_down*I2Cakp
    dI1kp = theta_VDp*I2kp  + gamma_VDp*Okp   - (eta_VDp+delta_VDp)*I1kp - r_up*I1kp + r_down*I1Cakp
    dCkp  = beta*Okp       + psi_VDp*I2kp    - (omega_VDp+alpha)*Ckp   - r_up*Ckp + r_down*CCakp
    ###########################################################################
    # Markov Model: CDI states
    dI2Cak = eta_CD*I1Cak     + omega_CD*CCak   - (theta_CD+psi_CD)*I2Cak + r_up*I2k - r_down*I2Cak
    dI1Cak = theta_CD*I2Cak   + gamma_CD*OCak   - (eta_CD+delta_CD)*I1Cak + r_up*I1k - r_down*I1Cak
    dCCak  = beta*OCak     + psi_CD*I2Cak    - (omega_CD+alpha)*CCak + r_up*Ck - r_down*CCak
    dI2Cakp = eta_CDp*I1Cakp   + omega_CDp*CCakp - (theta_CDp+psi_CDp)*I2Cakp + r_up*I2kp - r_down*I2Cakp
    dI1Cakp = theta_CDp*I2Cakp + gamma_CDp*OCakp - (eta_CDp+delta_CDp)*I1Cakp + r_up*I1kp - r_down*I1Cakp
    dCCakp  = beta*OCakp   + psi_CDp*I2Cakp  - (omega_CDp+alpha)*CCakp + r_up*Ckp - r_down*CCakp
    ###########################################################################
    # Reversibility
    # if check_reversibility > 0:
    #     revTol = check_reversibility
    #     rev1=abs(alpha*gamma_VD*eta_VD*psi_VD-beta*delta_VD*theta_VD*omega_VD)
    #     rev2=abs(alpha*gamma_VDp*eta_VDp*psi_VDp-beta*delta_VDp*theta_VDp*omega_VDp)
    #     rev3=abs(alpha*gamma_CD*eta_CD*psi_CD-beta*delta_CD*theta_CD*omega_CD)
    #     rev4=abs(alpha*gamma_CDp*eta_CDp*psi_CDp-beta*delta_CDp*theta_CDp*omega_CDp)
    #     if rev1>revTol or rev2>revTol or rev3>revTol or rev4>revTol:
    #         print('REVERSIBILITY FAILED')


    ###########################################################################
    # ICaL ICaNa ICaK currents
    ICaL_VD   = PCa    *PhiCaL *Ok
    ICaL_VDp  = PCap   *PhiCaL *Okp
    ICaL_CD   = PCa    *PhiCaL *OCak
    ICaL_CDp  = PCap   *PhiCaL *OCakp
    ICaNa_VD  = PCaNa  *PhiCaNa *Ok
    ICaNa_VDp = PCaNap *PhiCaNa *Okp
    ICaNa_CD  = PCaNa  *PhiCaNa *OCak
    ICaNa_CDp = PCaNap *PhiCaNa *OCakp
    ICaK_VD   = PCaK   *PhiCaK *Ok
    ICaK_VDp  = PCaKp  *PhiCaK *Okp
    ICaK_CD   = PCaK   *PhiCaK *OCak
    ICaK_CDp  = PCaKp  *PhiCaK *OCakp
    # ICaL VD vs CD & ICaL p vs np
    ICaLnp = ICaL_VD  + ICaL_CD
    ICaLp  = ICaL_VDp + ICaL_CDp
    ICaLVD = ICaL_VD*(1-fICaLp)  + ICaL_VDp*fICaLp
    ICaLCD = ICaL_CD*(1-fICaLp)  + ICaL_CDp*fICaLp

    ICaNanp = ICaNa_VD  + ICaNa_CD
    ICaNap  = ICaNa_VDp + ICaNa_CDp
    ICaKnp = ICaK_VD  + ICaK_CD
    ICaKp  = ICaK_VDp + ICaK_CDp

    ICaL  = (ICaLp*fICaLp + ICaLnp*(1-fICaLp))*bGCaL
    ICaNa = (ICaNap*fICaLp + ICaNanp*(1-fICaLp))*bGCaL
    ICaK  = (ICaKp*fICaLp + ICaKnp*(1-fICaLp))*bGCaL

    # ICaL conductance*
    gICaL = ICaL/PhiCaL
    ###########################################################################
    ### IKr current from DUTTA - Markovian formulation ###
    
    const_IKr = (Temp_dutta-20.0)
    rate1_dutta = A1_dutta*np.exp(B1_dutta*v)*np.exp(const_IKr*np.log(q1_dutta)/10.0)*k1_dutta
    rate2_dutta = A2_dutta*np.exp(B2_dutta*v)*np.exp(const_IKr*np.log(q2_dutta)/10.0)*k2_dutta
    rate3_dutta = A3_dutta*np.exp(B3_dutta*v)*np.exp(const_IKr*np.log(q3_dutta)/10.0)*k3_dutta
    rate4_dutta = A4_dutta*np.exp(B4_dutta*v)*np.exp(const_IKr*np.log(q4_dutta)/10.0)*k4_dutta
    rate11_dutta = A11_dutta*np.exp(B11_dutta*v)*np.exp(const_IKr*np.log(q11_dutta)/10.0)*k11_dutta
    rate21_dutta = A21_dutta*np.exp(B21_dutta*v)*np.exp(const_IKr*np.log(q21_dutta)/10.0)*k21_dutta
    rate31_dutta = (A31_dutta*np.exp(B31_dutta*v)*np.exp(const_IKr*np.log(q31_dutta)/10.0))*k31_dutta
    rate41_dutta = (A41_dutta*np.exp(B41_dutta*v)*np.exp(const_IKr*np.log(q41_dutta)/10.0))*k41_dutta
    rate51_dutta = A51_dutta*np.exp(B51_dutta*v)*np.exp(const_IKr*np.log(q51_dutta)/10.0)*k51_dutta
    rate52_dutta = A52_dutta*np.exp(B52_dutta*v)*np.exp(const_IKr*np.log(q52_dutta)/10.0)*k52_dutta
    rate53_dutta = A53_dutta*np.exp(B53_dutta*v)*np.exp(const_IKr*np.log(q53_dutta)/10.0)*k53_dutta
    rate61_dutta = A61_dutta*np.exp(B61_dutta*v)*np.exp(const_IKr*np.log(q61_dutta)/10.0)*k61_dutta
    rate62_dutta = A62_dutta*np.exp(B62_dutta*v)*np.exp(const_IKr*np.log(q62_dutta)/10.0)*k62_dutta
    rate63_dutta = A63_dutta*np.exp(B63_dutta*v)*np.exp(const_IKr*np.log(q63_dutta)/10.0)*k63_dutta
    rateD_dutta = Ku_dutta*np.exp(n_dutta*np.log(D_dutta))/(np.exp(n_dutta*np.log(D_dutta))+halfmax_dutta)*kD_dutta
    vrect_dutta = 1.0/(1.0+np.exp(-(v-vhalf_dutta)/6.789))

    dIC1_dutta = -(rate11_dutta*IC1_dutta-rate21_dutta*IC2_dutta)+rate51_dutta*C1_dutta-rate61_dutta*IC1_dutta
    dIC2_dutta = rate11_dutta*IC1_dutta-rate21_dutta*IC2_dutta-(rate3_dutta*IC2_dutta-rate4_dutta*IO_dutta)+rate52_dutta*C2_dutta-rate62_dutta*IC2_dutta
    dC1_dutta = -(rate1_dutta*C1_dutta-rate2_dutta*C2_dutta)-(rate51_dutta*C1_dutta-rate61_dutta*IC1_dutta)
    dC2_dutta = rate1_dutta*C1_dutta-rate2_dutta*C2_dutta-(rate31_dutta*C2_dutta-rate41_dutta*O_dutta)-(rate52_dutta*C2_dutta-rate62_dutta*IC2_dutta)
    dO_dutta = rate31_dutta*C2_dutta-rate41_dutta*O_dutta-(rate53_dutta*O_dutta-rate63_dutta*IO_dutta)-(Kmax_IKr*rateD_dutta*O_dutta-Ku_dutta*Ob_dutta)
    dIO_dutta = rate3_dutta*IC2_dutta-rate4_dutta*IO_dutta+rate53_dutta*O_dutta-rate63_dutta*IO_dutta-(Kmax_IKr*rateD_dutta*IO_dutta-Ku_dutta*rate53_dutta/rate63_dutta*IOb_dutta)
    dIOb_dutta = Kmax_IKr*rateD_dutta*IO_dutta-Ku_dutta*rate53_dutta/rate63_dutta*IOb_dutta+Kt_dutta*vrect_dutta*Cb_dutta-Kt_dutta*IOb_dutta
    dOb_dutta = Kmax_IKr*rateD_dutta*O_dutta-Ku_dutta*Ob_dutta+Kt_dutta*vrect_dutta*Cb_dutta-Kt_dutta*Ob_dutta
    dCb_dutta = -(Kt_dutta*vrect_dutta*Cb_dutta-Kt_dutta*Ob_dutta)-(Kt_dutta*vrect_dutta*Cb_dutta-Kt_dutta*IOb_dutta)
    dD_dutta = 0.0
    
    IKr = GKr*np.sqrt(ko/5.4)*O_dutta*(v-EK)
    ###########################################################################
    ### IKs current ###
    ### IKs parameter ###
    
    ###
    xs1ss=1.0/(1.0+np.exp((-(v+11.60 + EKshift))/8.932))
    txs1=817.3+1.0/(2.326e-4*np.exp((v+48.28+ EKshift)/17.80)+0.001292*np.exp((-(v+210.0+ EKshift))/230.0))
    dxs1=(xs1ss-xs1)/txs1
    xs2ss=xs1ss
    txs2=1.0/(0.01*np.exp((v-50.0+ EKshift)/20.0)+0.0193*np.exp((-(v+66.54+ EKshift))/31.0))
    dxs2=(xs2ss-xs2)/txs2
    KsCa=1.0+0.6/(1.0+(3.8e-5/cai)**1.4)
    
    IKs=GKs*KsCa*xs1*xs2*(v-EKs)
    ###########################################################################
    ### IK1 current ###
    ### IK1 parameters ###
    
    ###
    xk1ss=1.0/(1.0+np.exp(-(v+2.5538*ko+144.59+EKshift)/(1.5692*ko+3.8115)))
    txk1=122.2/(np.exp((-(v+EKshift+127.2))/20.36)+np.exp((v+EKshift+236.8)/69.33))
    dxk1=(xk1ss-xk1)/txk1
    rk1=1.0/(1.0+np.exp((v+105.8-2.6*ko+EKshift)/(kslope_rk1*9.493)))    

    IK1=GK1*np.sqrt(ko)*rk1*xk1*(v-EK)
    ###########################################################################
    ### INaCa current ###

    ###########################################################################
    ### INaCa_i current ###
    hca=np.exp((qca*v*F)/(R*T))       
    hna=np.exp((qna*v*F)/(R*T))
    h1=1+nai/kna3*(1+hna)          
    h2=(nai*hna)/(kna3*h1)
    h3=1.0/h1                      
    h4=1.0+nai/kna1*(1+nai/kna2)
    h5=nai*nai/(h4*kna1*kna2)      
    h6=1.0/h4
    h7=1.0+nao/kna3*(1.0+1.0/hna)  
    h8=nao/(kna3*hna*h7)
    h9=1.0/h7                      
    h10=kasymm+1.0+nao/kna1*(1.0+nao/kna2)
    h11=nao*nao/(h10*kna1*kna2)    
    h12=1.0/h10

    k1=h12*cao*kcaon   
    k2=kcaoff        
    k3p=h9*wca     
    k3pp=h8*wnaca
    k3=k3p+k3pp        
    k4p=h3*wca/hca   
    k4pp=h2*wnaca  
    k4=k4p+k4pp
    k5=kcaoff          
    k6=h6*cai*kcaon  
    k7=h5*h2*wna   
    k8=h8*h11*wna

    x1=k2*k4*(k7+k6)+k5*k7*(k2+k3) 
    x2=k1*k7*(k4+k5)+k4*k6*(k1+k8)
    x3=k1*k3*(k7+k6)+k8*k6*(k2+k3) 
    x4=k2*k8*(k4+k5)+k3*k5*(k1+k8)
    x_sum = (x1+x2+x3+x4)
    E1=x1/x_sum    
    E2=x2/x_sum
    E3=x3/x_sum    
    E4=x4/x_sum

    allo=1.0/(1.0+(KmCaAct/cai)**2.0)
    JncxNa=3.0*(E4*k7-E1*k8)+E3*k4pp-E2*k3pp
    JncxCa=E2*k2-E1*k1

    INaCa_i=0.8*Gncx*allo*(zna*JncxNa+zca*JncxCa)
    ###########################################################################
    ### INaCa_ss current ###
    h1=1+nass/kna3*(1+hna)         
    h2=(nass*hna)/(kna3*h1)
    h3=1.0/h1                      
    h4=1.0+nass/kna1*(1+nass/kna2)
    h5=nass*nass/(h4*kna1*kna2)    
    h6=1.0/h4
    h7=1.0+nao/kna3*(1.0+1.0/hna)  
    h8=nao/(kna3*hna*h7)
    h9=1.0/h7                      
    h10=kasymm+1.0+nao/kna1*(1+nao/kna2)
    h11=nao*nao/(h10*kna1*kna2)    
    h12=1.0/h10

    k1=h12*cao*kcaon   
    k2=kcaoff      
    k3p=h9*wca     
    k3pp=h8*wnaca
    k3=k3p+k3pp        
    k4p=h3*wca/hca 
    k4pp=h2*wnaca  
    k4=k4p+k4pp
    k5=kcaoff          
    k6=h6*cass*kcaon   
    k7=h5*h2*wna   
    k8=h8*h11*wna

    x1=k2*k4*(k7+k6)+k5*k7*(k2+k3)     
    x2=k1*k7*(k4+k5)+k4*k6*(k1+k8)
    x3=k1*k3*(k7+k6)+k8*k6*(k2+k3)     
    x4=k2*k8*(k4+k5)+k3*k5*(k1+k8)
    x_sum = (x1+x2+x3+x4)
    E1=x1/x_sum    
    E2=x2/x_sum
    E3=x3/x_sum    
    E4=x4/x_sum

    allo=1.0/(1.0+(KmCaAct/cass)**2.0)
    JncxNa=3.0*(E4*k7-E1*k8)+E3*k4pp-E2*k3pp
    JncxCa=E2*k2-E1*k1

    INaCa_ss=0.2*Gncx*allo*(zna*JncxNa+zca*JncxCa)
    ###########################################################################
    ### INaK current ###
    
    Knai=Knai0*np.exp((delta2*v*F)/(3.0*R*T))
    Knao=Knao0*np.exp(((1.0-delta2)*v*F)/(3.0*R*T))
    
    P=eP/(1.0+H/Khp+nai/Knap+ki/Kxkur)

    a1=(k1p*(nai/Knai)**3.0)/((1.0+nai/Knai)**3.0+(1.0+ki/Kki)**2.0-1.0)
    b1=k1m*MgADP
    a2=k2p
    b2=(k2m*(nao/Knao)**3.0)/((1.0+nao/Knao)**3.0+(1.0+ko/Kko)**2.0-1.0)
    a3=(k3p*(ko/Kko)**2.0)/((1.0+nao/Knao)**3.0+(1.0+ko/Kko)**2.0-1.0)
    b3=(k3m*P*H)/(1.0+MgATP/Kmgatp)
    a4=(k4p*MgATP/Kmgatp)/(1.0+MgATP/Kmgatp)
    b4=(k4m*(ki/Kki)**2.0)/((1.0+nai/Knai)**3.0+(1.0+ki/Kki)**2.0-1.0)

    x1=a4*a1*a2+b2*b4*b3+a2*b4*b3+b3*a1*a2
    x2=b2*b1*b4+a1*a2*a3+a3*b1*b4+a2*a3*b4
    x3=a2*a3*a4+b3*b2*b1+b2*b1*a4+a3*a4*b1
    x4=b4*b3*b2+a3*a4*a1+b2*a4*a1+b3*b2*a1

    E1=x1/(x1+x2+x3+x4)    
    E2=x2/(x1+x2+x3+x4)
    E3=x3/(x1+x2+x3+x4)    
    E4=x4/(x1+x2+x3+x4)
   
    
    JnakNa=3.0*(E1*a3-E2*b3)   
    JnakK=2.0*(E4*b1-E3*a1)    
    

    INaK=Pnak*(zna*JnakNa+zk*JnakK)
    ###########################################################################
    ### CaCl current (set to 0)###
    ### ICl parameters###
    
    IClCa = IClCa_si*GClCa/(1+KdClCa/cass)*(v-ecl)
    IClbk = IClb_si*GClB*(v-ecl)
    ###########################################################################
    ### Background currents: IKb, INab, ICab ###
    ### IKb current ###
    xkb = 1.0 / (1.0+np.exp(-(v-14.48)/18.34))
    IKb = GKb*xkb*(v-EK)
    ###########################################################################
    ### INab current ###
    
    INab = PNab*vffrt*(nai*np.exp(vfrt)-nao)/(np.exp(vfrt)-1.0)
    ###########################################################################
    ### ICab current ###
    
    ICab = (1-undo_ICab)*PCab*4.0*vffrt*(1.2*cai*np.exp(2.0*vfrt)-0.341*cao)/(np.exp(2.0*vfrt)-1.0)
    ###########################################################################
    ### IpCa current ###
    
    IpCa = GpCa*cai/(0.0005+cai)
    ###########################################################################
    ### Simulation Procotols ###
    
    # I_stim = IStim(time,cycleLength,amp,duration)
    
    if np.mod(time,cycleLength) <= duration: #5
        I_stim = amp #9.5
    else:
        I_stim = 0.0

    dv = - (INa+INaL+Ito+ICaL+ICaNa+ICaK+IKr+IKs+IK1+INaCa_i+INaCa_ss+INaK+INab+IKb+IpCa+ICab+I_stim+IClCa+IClbk)
    ###########################################################################
    ### Diffusion Fluxes ###
    ### Jdiff,Ca  parameter ###
    
    ###
    JdiffNa = (nass-nai) /2.0
    JdiffK  = (kss-ki)   /2.0
    Jdiff   = (cass-cai) *bJdiff/0.2
    ###########################################################################
    ### RyRs CICR from SR ###
    
    ###
    fJrelp=(1.0/(1.0+KmCaMK/CaMKa))
    RyRSRCass = (1 - 1/(1 +  np.exp((casr-0.3)/0.1)))

    RyRainfss = RyRa1-RyRa2/(1 + np.exp((1000*cass-(0.043))/0.0082))

    dRyRa = (RyRainfss- RyRa)/RyRtauadapt

    RyRoinfss = (1 - 1/(1 +  np.exp((1000*cass-(RyRa+ RyRohalf))/0.003)))
    
    dRyRo = (RyRoinfss- RyRo)/RyRtauact

    RyRcinfss = (1/(1 + np.exp((1000*cass-(RyRa+RyRchalf))/0.001)))
    
    dRyRc = (RyRcinfss- RyRc)/RyRtauinact

    if celltype=="M":
        #g_irel_max_M = g_irel_max*1.7
        Jrelnp = 1.7*g_irel_max*RyRSRCass*RyRo*RyRc*(casr-cass)
    else:
        Jrelnp = g_irel_max*RyRSRCass*RyRo*RyRc*(casr-cass)


    RyRtauinactp = RyRtauinact*1.25
    dRyRcp = (RyRcinfss- RyRcp)/RyRtauinactp
    Jrelp = g_irel_max_p*RyRSRCass*RyRo*RyRcp*(casr-cass)

    Jrel=((1.0-fJrelp)*Jrelnp+fJrelp*Jrelp)
    ###########################################################################
    ### Ca2+ Uptake Flux ###
    
    ###
    Jupnp=Jup_scale*0.004375*cai/(cai+0.00092)
    Jupp=Jup_scale*2.75*0.004375*cai/(cai+0.00092-0.00017) 


    fJupp=(1.0/(1.0+KmCaMK/CaMKa))
    Jleak=0.0123*casr/15.0
    Jup = cJup*((1.0-fJupp)*Jupnp+fJupp*Jupp)
    
    # Jup2=Vmax_SRCaP*((cai/Kmf)**hillSRCaP-(casr/Kmr)**hillSRCaP)/(1+(cai/Kmf)**hillSRCaP+(casr/Kmr)**hillSRCaP)
    ###########################################################################
    ### Tranlocation Flux (usefull if considering JSR+NSR) ###
    # Jtr=0
    ###########################################################################
    


    ###########################################################################
    ### EGTA ###
    if maxEGTA==0:
        dEGTA=0
        dEGTAi=0
    else:
        # Hellam & Podolsky Values
        kon=2 #mM**-1 ms**-1
        koff=4.0e-4  #ms**-1
        dEGTA=kon*cass*(maxEGTA-EGTA)-koff*EGTA
        dEGTAi=0

    ###########################################################################
    ### update intracellular [Na], [K] and [Ca] ###
    # [Na]
    dnai=-(INa+INaL+3.0*INaCa_i+3.0*INaK+INab)*Acap/(F*vmyo)+JdiffNa*vss/vmyo
    dnass=-(ICaNa+3.0*INaCa_ss)*Acap/(F*vss)-JdiffNa
    # [K]
    # if ki_cost==0:
    dki=-(Ito+IKr+IKs+IK1+IKb+I_stim-2.0*INaK)*Acap/(F*vmyo)+JdiffK*vss/vmyo
    dkss=-(ICaK)*Acap/(F*vss)-JdiffK
    # else:
    #     dki=0
    #     dkss=0

    # [Ca]
    Bcai   = 1.0 / (1.0+cmdnmax*kmcmdn/(kmcmdn+cai)**2.0 +trpnmax*kmtrpn/(kmtrpn+cai)**2.0)
    dcai   = Bcai*(-(IpCa+ICab-2.0*INaCa_i)*Acap/(2.0*F*vmyo) -Jup*vsr/vmyo+Jleak*vsr/vmyo+Jdiff*vss/vmyo-dEGTAi)
    Bcass  = 1.0/(1.0+BSRmax*KmBSR/(KmBSR+cass)**2.0 +BSLmax*KmBSL/(KmBSL+cass)**2.0)
    dcass  =  Bcass*(-(ICaL-2.0*INaCa_ss)*Acap/(2.0*F*vss) +Jrel*vsr/vss-Jdiff-dEGTA)
    # dcansr = Jup-Jtr*vjsr/vnsr (usefull if considering JSR+NSR)
    Bcasr = 1.0/(1.0+csqnmax*kmcsqn/(kmcsqn+casr)**2.0)
    dcasr = Bcasr*(Jup-Jleak-Jrel)
    ###########################################################################
    ## Output Computation
    #dd = dval1=dval2=dval3=dval4=dval5 =dval6= dxrf=dxrs=dJrelnp=dJrelp =0
    # When flag==1 -> dX
    # output=[dv,dnai,  dnass,  dki, dkss, dcai,  dcass,  0,dcasr , dm, dhf,  dhs,  dj , dhsp,  djp, dmL,  dhL,  dhLp,
    #         da, diF, diS,dap, diFp,  diSp, 0,dOk , dOkp,0,0,0,dnca,0, 0,0,0, dxs1,  dxs2,
    #         dxk1, 0,0,  dCaMKt,  dI1k , dI2k,  dCk , dI1kp,dI2kp,  dCkp,  dI1Cak,  dI2Cak,  dCCak, dI1Cakp,
    #         dI2Cakp, dCCakp, djnca, dEGTA, dEGTAi,  dRyRa,  dRyRo,  dRyRc,  dRyRcp, dC1_dutta, dC2_dutta, dCb_dutta, dD_dutta,
    #         dIC1_dutta,dIC2_dutta, dIO_dutta, dIOb_dutta, dO_dutta, dOb_dutta]

    output=[dv,      dnai,    dnass,   dki,     dkss,     #   1
        dcai,    dcass,   0,       dcasr,   dm,       #   2
        dhf,     dhs,     dj,      dhsp,    djp,      #   3
        dmL,     dhL,     dhLp,    da,      diF,      #   4
        diS,     dap,     diFp,    diSp,    0 ,       #   5
        dOk,     dOkp,    0,       0,       0 ,       #   6
        dnca,    0,       0,       0,       0  ,     #   7
        dxs1,    dxs2,    dxk1,    0,       0 ,   #   8
        dCaMKt,  dI1k,    dI2k,    dCk,     dI1kp,    #   9
        dI2kp,   dCkp,    dI1Cak,  dI2Cak,  dCCak,    #  10
        dI1Cakp,dI2Cakp, dCCakp,  djnca,   dEGTA,    #  11
        dEGTAi,  dRyRa,   dRyRo,   dRyRc,   dRyRcp    ,    #  12
        dC1_dutta,     dC2_dutta,     dCb_dutta,     dD_dutta,      dIC1_dutta,    # 13
        dIC2_dutta,    dIO_dutta,     dIOb_dutta,    dO_dutta,      dOb_dutta ]
    
    return(output)

# @njit
# def BPS2020_model2(time,initial_conds,cycleLength,celltype,amp,dur):

#     v,nai,  nass,  ki,  kss, cai,  cass,  val1,casr , m, hf,  hs,  j , hsp,  jp, mL,  hL,  hLp,  a,  iF, iS,  ap,  iFp,  iSp, d,Ok , Okp,val2,val3,val4,nca,val5, val6,xrf,xrs, xs1,  xs2,  xk1, Jrelnp,Jrelp,  CaMKt,  I1k , I2k,  Ck , I1kp,I2kp,  Ckp,  I1Cak,  I2Cak,  CCak, I1Cakp, I2Cakp, CCakp, jnca, EGTA, EGTAi,  RyRa,  RyRo,  RyRc,  RyRcp, C1_dutta, C2_dutta, Cb_dutta, D_dutta, IC1_dutta,IC2_dutta, IO_dutta, IOb_dutta, O_dutta, Ob_dutta = initial_conds

#     undo_p = 0
#     ki_cost = 0
#     maxEGTA = 0
#     bn = 1
#     #%% CDI on/off
#     undo_CDI = 0
#     #%% ICab on/off
#     undo_ICab = 0

#     cao = 1.8#mM*; 
#     nao = 144#mM*; 
#     ko  = 5.4#mM*
#     ## Physical Constants:
#     R = 8314.0   # J/kmol/K
#     T = 310.0    # K
#     F = 96485.0  # C/mol
#     vffrt = v*F*F/(R*T)
#     vfrt  = v*F/(R*T)
#     ###########################################################################
#     ## Cell Geometry
#     # Cell geometry was approxymate by a cylinder of length L and radius r
#     L = 0.01                         # cm
#     rad = 0.0011                     # cm
#     vcell = 1000*math.pi*rad**2*L          # 38e-6 uL
#     # Geometric Area
#     Ageo = 2*math.pi*rad**2 + 2*math.pi*rad*L   # cm**2
#     # Capacitive Area
#     Acap = 2*Ageo                    # cm**2
#     # Compartment Volumes (4)
#     vmyo = 0.68*vcell                # uL
#     vnsr = 0.0552*vcell              # uL
#     vjsr = 0.0048*vcell              # uL
#     vsr = 0.95*(vnsr+vjsr)           # uL
#     vss  = 1*0.02*vcell              # uL

#     ###########################################################################
#     ## CaMK Constants
#     KmCaMK = 0.15  
#     aCaMK  = 0.05  
#     bCaMK  = 0.00068
#     CaMKo  = 0.05
#     KmCaM  = 0.0015
#     ###########################################################################
#     ## update CaMK -> X(41)
#     CaMKb = CaMKo*(1.0-CaMKt) / (1.0+KmCaM/cass)
#     CaMKa = CaMKb+CaMKt
#     dCaMKt = aCaMK*CaMKb*(CaMKb+CaMKt) - bCaMK*CaMKt
#     ###########################################################################
#     ## Reversal Potentials
#     ENa  = (R*T/F)*np.log(nao/nai)
#     EK   = (R*T/F)*np.log(ko/ki)
#     PKNa = 0.01833
#     EKs  = (R*T/F)*np.log((ko+PKNa*nao)/(ki+PKNa*nai))
#     EKshift = 8 # shift for compesation in Ito,IKs,IK1 due to LJP
#     ###########################################################################
#     ### INa current ###
#     ### INa parameters ###
#     bGNa = 0.27
#     thf_shift = 0.075 # thf
#     btj = 1           # tj
#     ###
#     mss=1.0/(1.0+np.exp((-(v+39.57))/9.871))
#     tm=1.0/(6.765*np.exp((v+11.64)/34.77)+8.552*np.exp(-(v+77.42)/5.955))
#     dm=(mss-m)/tm
#     hss=1.0/(1+np.exp((v+78.5)/6.22))
#     thf=1.0/(3.6860e-6*np.exp(-(v+3.8875)/7.8579)+16*np.exp((v-0.4963)/9.1843))+thf_shift
#     ths=1.0/(0.009794*np.exp(-(v+17.95)/28.05)+0.3343*np.exp((v+5.730)/56.66))
#     Ahf=0.99
#     Ahs=1.0-Ahf
#     dhf=(hss-hf)/thf
#     dhs=(hss-hs)/ths
#     h=Ahf*hf+Ahs*hs
#     jss=hss
#     tj=(4.8590+1.0/(0.8628*np.exp(-(v+116.7258)/7.6005)+1.1096*np.exp((v+6.2719)/9.0358)))*btj
#     dj=(jss-j)/tj
#     hssp=1.0/(1+np.exp((v+84.7)/6.22))
#     thsp=3.0*ths
#     dhsp=(hssp-hsp)/thsp
#     hp=Ahf*hf+Ahs*hsp
#     tjp=1.46*tj
#     djp=(jss-jp)/tjp
#     GNa=75*bGNa
#     fINap=(1.0/(1.0+KmCaMK/CaMKa))
#     INa=GNa*(v-ENa)*m**3.0*((1.0-fINap)*h*j+fINap*hp*jp)
#     ###########################################################################
#     ### INaL current ###
#     ### INaL parameters ###
#     bGnal = 2.8
#     bthL = 1
#     ###
#     mLss=1.0/(1.0+np.exp((-(v+42.85))/5.264))
#     tmL=tm
#     dmL=(mLss-mL)/tmL
#     hLss=1.0/(1.0+np.exp((v+87.61)/7.488))
#     thL=200.0*bthL
#     dhL=(hLss-hL)/thL
#     hLssp=1.0/(1.0+np.exp((v+93.81)/7.488))
#     thLp=3.0*thL
#     dhLp=(hLssp-hLp)/thLp
#     GNaL=0.0075*bGnal
#     if celltype==1:
#         GNaL=GNaL*0.7

#     fINaLp=(1.0/(1.0+KmCaMK/CaMKa))
#     INaL=GNaL*(v-ENa)*mL*((1.0-fINaLp)*hL+fINaLp*hLp)
#     ###########################################################################
#     ### Ito current ###
#     ### Ito parameter ###
#     bGto = 1
#     ###
#     ass=1.0/(1.0+np.exp((-(v+EKshift-14.34))/14.82))
#     ta=1.0515 / (1.0/(1.2089*(1.0+np.exp(-(v+EKshift-18.4099)/29.3814)))+3.5/(1.0+np.exp((v+EKshift+100.0)/29.3814)))
#     da=(ass-a)/ta
#     iss=1.0/(1.0+np.exp((v+EKshift+43.94)/5.711))
#     if celltype==1:
#         delta_epi=1.0-(0.95/(1.0+np.exp((v+EKshift+70.0)/5.0)))
#     else:
#         delta_epi=1.0

#     tiF=4.562+1/(0.3933*np.exp((-(v+EKshift+100.0))/100.0)+0.08004*np.exp((v+EKshift+50.0)/16.59))
#     tiS=23.62+1/(0.001416*np.exp((-(v+EKshift+96.52))/59.05)+1.780e-8*np.exp((v+EKshift+114.1)/8.079))
#     tiF=tiF*delta_epi
#     tiS=tiS*delta_epi
#     AiF=1.0/(1.0+np.exp((v-213.6+EKshift)/151.2))
#     AiS=1.0-AiF
#     diF=(iss-iF)/tiF
#     diS=(iss-iS)/tiS
#     i=AiF*iF+AiS*iS
#     assp=1.0/(1.0+np.exp((-(v+EKshift-24.34))/14.82))
#     dap=(assp-ap)/ta
#     dti_develop=1.354+1.0e-4/(np.exp((v+EKshift-167.4)/15.89)+np.exp(-(v+EKshift-12.23)/0.2154))
#     dti_recover=1.0-0.5/(1.0+np.exp((v+EKshift+70.0)/20.0))
#     tiFp=dti_develop*dti_recover*tiF
#     tiSp=dti_develop*dti_recover*tiS
#     diFp=(iss-iFp)/tiFp
#     diSp=(iss-iSp)/tiSp
#     ip=AiF*iFp+AiS*iSp
#     Gto=0.02*bGto
#     if celltype==1:
#         Gto=Gto*4.0
#     elif celltype==2:
#         Gto=Gto*4.0

#     fItop=(1.0/(1.0+KmCaMK/CaMKa))
#     Ito=Gto*(v-EK)*((1.0-fItop)*a*i+fItop*ap*ip)
#     ###########################################################################
#     ### ICaL, ICaNa, ICaK current ###
#     ### ICaL parameters ###
#     bGCaL =1
#     cPCa = 0.9
#     kCDI = 9
#     ###
#     # p vs np ICaL
#     fICaLp=(1.0/(1.0+KmCaMK/CaMKa))*(1-undo_p)
#     ###########################################################################
#     ## up/down rates
#     r_down = bn*(1e-1)*(1-undo_CDI)
#     r_up = bn*(r_down*nca/(1-nca))*(1-undo_CDI)
#     ###########################################################################
#     ## n gate -> used for compute nca
#     jncass = 1.0/(1.0+np.exp((v+19.58+25)/3.696))
#     # fss=1.0/(1.0+np.exp((v+19.58)/3.696)) # ORd formulation
#     tjnca  = 1
#     djnca  =(jncass-jnca)/tjnca
#     Kmn  = 0.05
#     k2n  = 1000
#     km2n = 150*jnca
#     # anca=1/(k2n/km2n+(1.0+Kmn/cass)**4.0) # ORd formulation
#     anca = (1-nca)/(1+Kmn/cass)**4.0
#     dnca=bn*(anca*k2n-nca*km2n)
#     ###########################################################################
#     # Activation (d)
#     dss = 1.0/(1.0+np.exp((-(v+3.940))/4.230))
#     td  = (0.6+1.0/( np.exp( -0.05*(v+6))+np.exp(0.09*(v+14))) )
#     alpha = dss/td
#     beta = (1-dss) / td
#     ###########################################################################
#     # Recovery (jca)
#     jcass_new= 1.0/(1.0+np.exp((v+19.58)/3.696))
#     jcass_VD = jcass_new
#     jcass_CD = jcass_new
#     jcass_VDp = jcass_new
#     jcass_CDp = jcass_new
#     tjca_new = 35 + 350*np.exp(-(v-(-20))**2/(2*10**2))
#     tjca_VD = tjca_new
#     tjca_VDp = tjca_new
#     tjca_CD = tjca_new
#     tjca_CDp = tjca_new
#     # psi and omega rates
#     psi_VD=jcass_VD/tjca_VD
#     psi_VDp=jcass_VDp/tjca_VDp
#     psi_CD=jcass_CD/tjca_CD
#     psi_CDp=jcass_CDp/tjca_CDp
#     omega_VD=(1-jcass_VD)/tjca_VD
#     omega_VDp=(1-jcass_VDp)/tjca_VDp
#     omega_CD=(1-jcass_CD)/tjca_CD
#     omega_CDp=(1-jcass_CDp)/tjca_CDp
#     ###########################################################################
#     # Fact Inactivation (f1)
#     # ORd formulation #
#     # f1ss  = 1.0 / (1.0+np.exp((v+19.58)/3.696))  
#     # tff   = 7.0 + 1.0/(0.0045*np.exp(-(v+20.0)/10.0)+0.0045*np.exp((v+20.0)/10.0))
#     # tfcaf = 7.0 + 1.0/(0.04*np.exp(-(v-4.0)/7.0)+0.04*np.exp((v-4.0)/7.0))
#     #
#     f1ss_0 = 0.8 / (1.0+np.exp((v+19.58)/3.696)) + 0.2
#     tf1_0   = 1*(70 + 1.2/ (0.0045*np.exp((v+20)/(-50))+0.0045*np.exp((v+30)/10)))
#     ktaup = 2.5
#     gamma_VD  = (1-f1ss_0)/ tf1_0
#     delta_VD  = f1ss_0  / tf1_0
#     gamma_VDp  = gamma_VD/ktaup
#     delta_VDp  = delta_VD/ktaup
#     gamma_CD=gamma_VD*kCDI
#     delta_CD=delta_VD*kCDI
#     gamma_CDp=gamma_VDp*kCDI
#     delta_CDp=delta_VDp*kCDI
#     tf1_VD = 1/(gamma_VD+delta_VD)
#     tf1_CD = 1/(gamma_CD+delta_CD)
#     f1ss_VD = gamma_VD / (gamma_VD+delta_VD)
#     f1ss_CD = gamma_CD / (gamma_CD+delta_CD)
#     ###########################################################################
#     # Slow Inactivation (f2)
#     # ORd formulation #
#     # fss=1.0/(1.0+np.exp((v+19.58)/3.696))
#     # tfs=1000.0+1.0/(0.000035*np.exp(-(v+5.0)/4.0)+0.000035*np.exp((v+5.0)/6.0))
#     # tfcas=100.0+1.0/(0.00012*np.exp(-v/3.0)+0.00012*np.exp(v/7.0))
#     #
#     tf2_new   = 1*(100 + 0./ (0.0035*np.exp((v+5)/(-84))+0.0035*np.exp((v+5)/4)))
#     tf2_VD = tf2_new
#     tf2_CD = tf2_VD/kCDI
#     tf2_VDp = tf2_new*ktaup
#     tf2_CDp = tf2_VD/kCDI*ktaup
#     # Reversibility
#     theta_VD = alpha*gamma_VD*psi_VD/tf2_VD/(alpha*gamma_VD*psi_VD+beta*delta_VD*omega_VD)
#     theta_CD = alpha*gamma_CD*psi_CD/tf2_CD/(alpha*gamma_CD*psi_CD+beta*delta_CD*omega_CD)
#     theta_VDp = alpha*gamma_VDp*psi_VDp/tf2_VDp/(alpha*gamma_VDp*psi_VDp+beta*delta_VDp*omega_VDp)
#     theta_CDp = alpha*gamma_CDp*psi_CDp/tf2_CDp/(alpha*gamma_CDp*psi_CDp+beta*delta_CDp*omega_CDp)
#     eta_VD=1/tf2_VD - theta_VD
#     eta_VDp=1/tf2_VDp - theta_VDp
#     eta_CD=1/tf2_CD - theta_CD
#     eta_CDp=1/tf2_CDp - theta_CDp

#     tf2_VD = 1/(eta_VD+theta_VD)
#     tf2_CD = 1/(eta_CD+theta_CD)
#     f2ss_VD=eta_VD/(eta_VD+theta_VD)
#     f2ss_CD=eta_CD/(eta_CD+theta_CD)
#     ###########################################################################
#     # Driving Forces
#     PhiCaL=4.0*vffrt*(1.2*cass*np.exp(2.0*vfrt)-0.341*cao)/(np.exp(2.0*vfrt)-1.0)
#     PhiCaNa=1.0*vffrt*(0.75*nass*np.exp(1.0*vfrt)-0.75*nao)/(np.exp(1.0*vfrt)-1.0)
#     PhiCaK=1.0*vffrt*(0.75*kss*np.exp(1.0*vfrt)-0.75*ko)/(np.exp(1.0*vfrt)-1.0)
#     PCa=0.0001*cPCa
#     if celltype==1:
#         PCa=PCa*1.4
#     elif celltype==2:
#         PCa=PCa*2

#     PCap=1.1*PCa
#     PCaNa=0.00125*PCa
#     PCaK=3.574e-4*PCa
#     PCaNap=0.00125*PCap
#     PCaKp=3.574e-4*PCap
#     ###########################################################################
#     # Markov Model: VDI states
#     OCak = 1-CCak-I1Cak-I2Cak-Ck-I1k-I2k-Ok
#     OCakp = 1-CCakp-I1Cakp-I2Cakp-Ckp-I1kp-I2kp-Okp
#     dOk =  alpha*Ck        + delta_VD*I1k    - (beta+gamma_VD)*Ok      - r_up*Ok + r_down*OCak
#     dI2k = eta_VD*I1k      + omega_VD*Ck     - (theta_VD+psi_VD)*I2k   - r_up*I2k + r_down*I2Cak
#     dI1k = theta_VD*I2k    + gamma_VD*Ok     - (eta_VD+delta_VD)*I1k   - r_up*I1k + r_down*I1Cak
#     dCk  = beta*Ok         + psi_VD*I2k      - (omega_VD+alpha)*Ck     - r_up*Ck + r_down*CCak
#     dOkp = alpha*Ckp       + delta_VDp*I1kp  - (beta+gamma_VDp)*Okp    - r_up*Okp + r_down*OCakp
#     dI2kp = eta_VDp*I1kp    + omega_VDp*Ckp   - (theta_VDp+psi_VDp)*I2kp - r_up*I2kp + r_down*I2Cakp
#     dI1kp = theta_VDp*I2kp  + gamma_VDp*Okp   - (eta_VDp+delta_VDp)*I1kp - r_up*I1kp + r_down*I1Cakp
#     dCkp  = beta*Okp       + psi_VDp*I2kp    - (omega_VDp+alpha)*Ckp   - r_up*Ckp + r_down*CCakp
#     ###########################################################################
#     # Markov Model: CDI states
#     dI2Cak = eta_CD*I1Cak     + omega_CD*CCak   - (theta_CD+psi_CD)*I2Cak + r_up*I2k - r_down*I2Cak
#     dI1Cak = theta_CD*I2Cak   + gamma_CD*OCak   - (eta_CD+delta_CD)*I1Cak + r_up*I1k - r_down*I1Cak
#     dCCak  = beta*OCak     + psi_CD*I2Cak    - (omega_CD+alpha)*CCak + r_up*Ck - r_down*CCak
#     dI2Cakp = eta_CDp*I1Cakp   + omega_CDp*CCakp - (theta_CDp+psi_CDp)*I2Cakp + r_up*I2kp - r_down*I2Cakp
#     dI1Cakp = theta_CDp*I2Cakp + gamma_CDp*OCakp - (eta_CDp+delta_CDp)*I1Cakp + r_up*I1kp - r_down*I1Cakp
#     dCCakp  = beta*OCakp   + psi_CDp*I2Cakp  - (omega_CDp+alpha)*CCakp + r_up*Ckp - r_down*CCakp
#     ###########################################################################
#     # Reversibility
#     # if check_reversibility > 0
#     #     revTol = check_reversibility
#     #     rev1=abs(alpha*gamma_VD*eta_VD*psi_VD-beta*delta_VD*theta_VD*omega_VD)
#     #     rev2=abs(alpha*gamma_VDp*eta_VDp*psi_VDp-beta*delta_VDp*theta_VDp*omega_VDp)
#     #     rev3=abs(alpha*gamma_CD*eta_CD*psi_CD-beta*delta_CD*theta_CD*omega_CD)
#     #     rev4=abs(alpha*gamma_CDp*eta_CDp*psi_CDp-beta*delta_CDp*theta_CDp*omega_CDp)
#     #     if rev1>revTol or rev2>revTol || rev3>revTol || rev4>revTol
#     #         disp('REVERSIBILITY FAILED')
        

#     ###########################################################################
#     # ICaL ICaNa ICaK currents
#     ICaL_VD   = PCa    *PhiCaL *Ok
#     ICaL_VDp  = PCap   *PhiCaL *Okp
#     ICaL_CD   = PCa    *PhiCaL *OCak
#     ICaL_CDp  = PCap   *PhiCaL *OCakp
#     ICaNa_VD  = PCaNa  *PhiCaNa *Ok
#     ICaNa_VDp = PCaNap *PhiCaNa *Okp
#     ICaNa_CD  = PCaNa  *PhiCaNa *OCak
#     ICaNa_CDp = PCaNap *PhiCaNa *OCakp
#     ICaK_VD   = PCaK   *PhiCaK *Ok
#     ICaK_VDp  = PCaKp  *PhiCaK *Okp
#     ICaK_CD   = PCaK   *PhiCaK *OCak
#     ICaK_CDp  = PCaKp  *PhiCaK *OCakp
#     # ICaL VD vs CD & ICaL p vs np
#     ICaLnp = ICaL_VD  + ICaL_CD
#     ICaLp  = ICaL_VDp + ICaL_CDp
#     ICaLVD = ICaL_VD*(1-fICaLp)  + ICaL_VDp*fICaLp
#     ICaLCD = ICaL_CD*(1-fICaLp)  + ICaL_CDp*fICaLp

#     ICaNanp = ICaNa_VD  + ICaNa_CD
#     ICaNap  = ICaNa_VDp + ICaNa_CDp
#     ICaKnp = ICaK_VD  + ICaK_CD
#     ICaKp  = ICaK_VDp + ICaK_CDp

#     ICaL  = (ICaLp*fICaLp + ICaLnp*(1-fICaLp))*bGCaL
#     ICaNa = (ICaNap*fICaLp + ICaNanp*(1-fICaLp))*bGCaL
#     ICaK  = (ICaKp*fICaLp + ICaKnp*(1-fICaLp))*bGCaL

#     # ICaL conductance*
#     gICaL = ICaL/PhiCaL
#     ###########################################################################
#     ### IKr current from DUTTA - Markovian formulation ###
#     ### IKr parameters ###
#     bGKr = 1.2
#     k1_dutta=k2_dutta=k3_dutta= k4_dutta=k11_dutta=k21_dutta=k31_dutta=k41_dutta=k51_dutta=k61_dutta=k52_dutta=k62_dutta=k53_dutta=k63_dutta=kD_dutta=1
#     ###
#     A1_dutta = 0.0264   # per_millisecond (in IKr)
#     A11_dutta = 0.0007868   # per_millisecond (in IKr)
#     A2_dutta = 4.986e-6   # per_millisecond (in IKr)
#     A21_dutta = 5.455e-6   # per_millisecond (in IKr)
#     A3_dutta = 0.001214   # per_millisecond (in IKr)
#     A31_dutta = 0.005509   # per_millisecond (in IKr)
#     A4_dutta = 1.854e-5   # per_millisecond (in IKr)
#     A41_dutta = 0.001416   # per_millisecond (in IKr)
#     A51_dutta = 0.4492   # per_millisecond (in IKr)
#     A52_dutta = 0.3181   # per_millisecond (in IKr)
#     A53_dutta = 0.149   # per_millisecond (in IKr)
#     A61_dutta = 0.01241   # per_millisecond (in IKr)
#     A62_dutta = 0.3226   # per_millisecond (in IKr)
#     A63_dutta = 0.008978   # per_millisecond (in IKr)
#     B1_dutta = 4.631e-5   # per_millivolt (in IKr)
#     B11_dutta = 1.535e-8   # per_millivolt (in IKr)
#     B2_dutta = -0.004226   # per_millivolt (in IKr)
#     B21_dutta = -0.1688   # per_millivolt (in IKr)
#     B3_dutta = 0.008516   # per_millivolt (in IKr)
#     B31_dutta = 7.771e-9   # per_millivolt (in IKr)
#     B4_dutta = -0.04641   # per_millivolt (in IKr)
#     B41_dutta = -0.02877   # per_millivolt (in IKr)
#     B51_dutta = 0.008595   # per_millivolt (in IKr)
#     B52_dutta = 3.613e-8   # per_millivolt (in IKr)
#     B53_dutta = 0.004668   # per_millivolt (in IKr)
#     B61_dutta = 0.1725   # per_millivolt (in IKr)
#     B62_dutta = -0.0006575   # per_millivolt (in IKr)
#     B63_dutta = -0.02215   # per_millivolt (in IKr)
#     # GKr_b = 0.046585   # milliS_per_microF (in IKr)
#     Kmax_IKr = 0.0   # dimensionless (in IKr)
#     Kt_dutta = 3.5e-5   # per_millisecond (in IKr)
#     Ku_dutta = 0.0   # per_millisecond (in IKr)
#     Temp_dutta = 37.0   # celsius (in IKr)
#     halfmax_dutta = 1.0   # dimensionless (in IKr)
#     n_dutta = 1.0   # dimensionless (in IKr)
#     q1_dutta = 4.843   # dimensionless (in IKr)
#     q11_dutta = 4.942   # dimensionless (in IKr)
#     q2_dutta = 4.23   # dimensionless (in IKr)
#     q21_dutta = 4.156   # dimensionless (in IKr)
#     q3_dutta = 4.962   # dimensionless (in IKr)
#     q31_dutta = 4.22   # dimensionless (in IKr)
#     q4_dutta = 3.769   # dimensionless (in IKr)
#     q41_dutta = 1.459   # dimensionless (in IKr)
#     q51_dutta = 5.0   # dimensionless (in IKr)
#     q52_dutta = 4.663   # dimensionless (in IKr)
#     q53_dutta = 2.412   # dimensionless (in IKr)
#     q61_dutta = 5.568   # dimensionless (in IKr)
#     q62_dutta = 5.0   # dimensionless (in IKr)
#     q63_dutta = 5.682   # dimensionless (in IKr)
#     vhalf_dutta = 1.0   # millivolt (in IKr)

#     rate1_dutta = A1_dutta*np.exp(B1_dutta*v)*np.exp((Temp_dutta-20.0)*np.log(q1_dutta)/10.0)*k1_dutta
#     rate2_dutta = A2_dutta*np.exp(B2_dutta*v)*np.exp((Temp_dutta-20.0)*np.log(q2_dutta)/10.0)*k2_dutta
#     rate3_dutta = A3_dutta*np.exp(B3_dutta*v)*np.exp((Temp_dutta-20.0)*np.log(q3_dutta)/10.0)*k3_dutta
#     rate4_dutta = A4_dutta*np.exp(B4_dutta*v)*np.exp((Temp_dutta-20.0)*np.log(q4_dutta)/10.0)*k4_dutta
#     rate11_dutta = A11_dutta*np.exp(B11_dutta*v)*np.exp((Temp_dutta-20.0)*np.log(q11_dutta)/10.0)*k11_dutta
#     rate21_dutta = A21_dutta*np.exp(B21_dutta*v)*np.exp((Temp_dutta-20.0)*np.log(q21_dutta)/10.0)*k21_dutta
#     rate31_dutta = (A31_dutta*np.exp(B31_dutta*v)*np.exp((Temp_dutta-20.0)*np.log(q31_dutta)/10.0))*k31_dutta
#     rate41_dutta = (A41_dutta*np.exp(B41_dutta*v)*np.exp((Temp_dutta-20.0)*np.log(q41_dutta)/10.0))*k41_dutta
#     rate51_dutta = A51_dutta*np.exp(B51_dutta*v)*np.exp((Temp_dutta-20.0)*np.log(q51_dutta)/10.0)*k51_dutta
#     rate52_dutta = A52_dutta*np.exp(B52_dutta*v)*np.exp((Temp_dutta-20.0)*np.log(q52_dutta)/10.0)*k52_dutta
#     rate53_dutta = A53_dutta*np.exp(B53_dutta*v)*np.exp((Temp_dutta-20.0)*np.log(q53_dutta)/10.0)*k53_dutta
#     rate61_dutta = A61_dutta*np.exp(B61_dutta*v)*np.exp((Temp_dutta-20.0)*np.log(q61_dutta)/10.0)*k61_dutta
#     rate62_dutta = A62_dutta*np.exp(B62_dutta*v)*np.exp((Temp_dutta-20.0)*np.log(q62_dutta)/10.0)*k62_dutta
#     rate63_dutta = A63_dutta*np.exp(B63_dutta*v)*np.exp((Temp_dutta-20.0)*np.log(q63_dutta)/10.0)*k63_dutta
#     rateD_dutta = Ku_dutta*np.exp(n_dutta*np.log(D_dutta))/(np.exp(n_dutta*np.log(D_dutta))+halfmax_dutta)*kD_dutta
#     vrect_dutta = 1.0/(1.0+np.exp(-(v-vhalf_dutta)/6.789))

#     dIC1_dutta = -(rate11_dutta*IC1_dutta-rate21_dutta*IC2_dutta)+rate51_dutta*C1_dutta-rate61_dutta*IC1_dutta
#     dIC2_dutta = rate11_dutta*IC1_dutta-rate21_dutta*IC2_dutta-(rate3_dutta*IC2_dutta-rate4_dutta*IO_dutta)+rate52_dutta*C2_dutta-rate62_dutta*IC2_dutta
#     dC1_dutta = -(rate1_dutta*C1_dutta-rate2_dutta*C2_dutta)-(rate51_dutta*C1_dutta-rate61_dutta*IC1_dutta)
#     dC2_dutta = rate1_dutta*C1_dutta-rate2_dutta*C2_dutta-(rate31_dutta*C2_dutta-rate41_dutta*O_dutta)-(rate52_dutta*C2_dutta-rate62_dutta*IC2_dutta)
#     dO_dutta = rate31_dutta*C2_dutta-rate41_dutta*O_dutta-(rate53_dutta*O_dutta-rate63_dutta*IO_dutta)-(Kmax_IKr*rateD_dutta*O_dutta-Ku_dutta*Ob_dutta)
#     dIO_dutta = rate3_dutta*IC2_dutta-rate4_dutta*IO_dutta+rate53_dutta*O_dutta-rate63_dutta*IO_dutta-(Kmax_IKr*rateD_dutta*IO_dutta-Ku_dutta*rate53_dutta/rate63_dutta*IOb_dutta)
#     dIOb_dutta = Kmax_IKr*rateD_dutta*IO_dutta-Ku_dutta*rate53_dutta/rate63_dutta*IOb_dutta+Kt_dutta*vrect_dutta*Cb_dutta-Kt_dutta*IOb_dutta
#     dOb_dutta = Kmax_IKr*rateD_dutta*O_dutta-Ku_dutta*Ob_dutta+Kt_dutta*vrect_dutta*Cb_dutta-Kt_dutta*Ob_dutta
#     dCb_dutta = -(Kt_dutta*vrect_dutta*Cb_dutta-Kt_dutta*Ob_dutta)-(Kt_dutta*vrect_dutta*Cb_dutta-Kt_dutta*IOb_dutta)
#     dD_dutta = 0.0

#     GKr=0.046*bGKr
#     if (celltype == 1.0):
#         GKr = GKr*1.1 # modified respect ORd (*1.3)
#     elif (celltype == 2.0):
#         GKr = GKr*0.8

#     IKr = GKr*np.sqrt(ko/5.4)*O_dutta*(v-EK)
#     ###########################################################################
#     ### IKs current ###
#     ### IKs parameter ###
#     bGKs = 2
#     ###
#     xs1ss=1.0/(1.0+np.exp((-(v+11.60 + EKshift))/8.932))
#     txs1=817.3+1.0/(2.326e-4*np.exp((v+48.28+ EKshift)/17.80)+0.001292*np.exp((-(v+210.0+ EKshift))/230.0))
#     dxs1=(xs1ss-xs1)/txs1
#     xs2ss=xs1ss
#     txs2=1.0/(0.01*np.exp((v-50.0+ EKshift)/20.0)+0.0193*np.exp((-(v+66.54+ EKshift))/31.0))
#     dxs2=(xs2ss-xs2)/txs2
#     KsCa=1.0+0.6/(1.0+(3.8e-5/cai)**1.4)
#     GKs=0.0034*bGKs
#     if celltype==1:
#         GKs=GKs*1.4

#     IKs=GKs*KsCa*xs1*xs2*(v-EKs)
#     ###########################################################################
#     ### IK1 current ###
#     ### IK1 parameters ###
#     kslope_rk1 = 1.09
#     bGK1 = .71
#     ###
#     xk1ss=1.0/(1.0+np.exp(-(v+2.5538*ko+144.59+EKshift)/(1.5692*ko+3.8115)))
#     txk1=122.2/(np.exp((-(v+EKshift+127.2))/20.36)+np.exp((v+EKshift+236.8)/69.33))
#     dxk1=(xk1ss-xk1)/txk1
#     rk1=1.0/(1.0+np.exp((v+105.8-2.6*ko+EKshift)/(kslope_rk1*9.493)))
#     GK1=0.1908*bGK1
#     if celltype==1:
#         GK1=GK1*1.2
#     elif celltype==2:
#         GK1=GK1*1.3

#     IK1=GK1*np.sqrt(ko)*rk1*xk1*(v-EK)
#     ###########################################################################
#     ### INaCa current ###
#     ### INaCa parameter ###
#     bGncx = 2.4
#     ###
#     kna1=15.0      
#     kna2=5.0       
#     kna3=88.12     
#     kasymm=12.5
#     wna=6.0e4      
#     wca=6.0e4      
#     wnaca=5.0e3    
#     KmCaAct=150.0e-6
#     kcaon = 1.5e6    
#     kcaoff=5.0e3   
#     qna=0.5224     
#     qca=0.1670
#     zna=1.0        
#     Gncx=0.0008*bGncx    
#     zca=2.0
#     if celltype==1:
#         Gncx=Gncx*1.2
#     elif celltype==2:
#         Gncx=Gncx*1.4

#     ###########################################################################
#     ### INaCa_i current ###
#     hca=np.exp((qca*v*F)/(R*T))       
#     hna=np.exp((qna*v*F)/(R*T))
#     h1=1+nai/kna3*(1+hna)          
#     h2=(nai*hna)/(kna3*h1)
#     h3=1.0/h1                      
#     h4=1.0+nai/kna1*(1+nai/kna2)
#     h5=nai*nai/(h4*kna1*kna2)      
#     h6=1.0/h4
#     h7=1.0+nao/kna3*(1.0+1.0/hna)  
#     h8=nao/(kna3*hna*h7)
#     h9=1.0/h7                      
#     h10=kasymm+1.0+nao/kna1*(1.0+nao/kna2)
#     h11=nao*nao/(h10*kna1*kna2)    
#     h12=1.0/h10

#     k1=h12*cao*kcaon   
#     k2=kcaoff        
#     k3p=h9*wca     
#     k3pp=h8*wnaca
#     k3=k3p+k3pp        
#     k4p=h3*wca/hca   
#     k4pp=h2*wnaca  
#     k4=k4p+k4pp
#     k5=kcaoff          
#     k6=h6*cai*kcaon  

#     k7=h5*h2*wna   
#     k8=h8*h11*wna

#     x1=k2*k4*(k7+k6)+k5*k7*(k2+k3) 
#     x2=k1*k7*(k4+k5)+k4*k6*(k1+k8)
#     x3=k1*k3*(k7+k6)+k8*k6*(k2+k3) 
#     x4=k2*k8*(k4+k5)+k3*k5*(k1+k8)

#     E1=x1/(x1+x2+x3+x4)    
#     E2=x2/(x1+x2+x3+x4)
#     E3=x3/(x1+x2+x3+x4)    
#     E4=x4/(x1+x2+x3+x4)

#     allo=1.0/(1.0+(KmCaAct/cai)**2.0)
#     JncxNa=3.0*(E4*k7-E1*k8)+E3*k4pp-E2*k3pp
#     JncxCa=E2*k2-E1*k1

#     INaCa_i=0.8*Gncx*allo*(zna*JncxNa+zca*JncxCa)
#     ###########################################################################
#     ### INaCa_ss current ###
#     h1=1+nass/kna3*(1+hna)         
#     h2=(nass*hna)/(kna3*h1)
#     h3=1.0/h1                      
#     h4=1.0+nass/kna1*(1+nass/kna2)
#     h5=nass*nass/(h4*kna1*kna2)    
#     h6=1.0/h4
#     h7=1.0+nao/kna3*(1.0+1.0/hna)  
#     h8=nao/(kna3*hna*h7)
#     h9=1.0/h7                      
#     h10=kasymm+1.0+nao/kna1*(1+nao/kna2)
#     h11=nao*nao/(h10*kna1*kna2)    
#     h12=1.0/h10

#     k1=h12*cao*kcaon   
#     k2=kcaoff      
#     k3p=h9*wca     
#     k3pp=h8*wnaca
#     k3=k3p+k3pp        
#     k4p=h3*wca/hca 
#     k4pp=h2*wnaca  
#     k4=k4p+k4pp
#     k5=kcaoff          
#     k6=h6*cass*kcaon   
#     k7=h5*h2*wna   
#     k8=h8*h11*wna

#     x1=k2*k4*(k7+k6)+k5*k7*(k2+k3)     
#     x2=k1*k7*(k4+k5)+k4*k6*(k1+k8)
#     x3=k1*k3*(k7+k6)+k8*k6*(k2+k3)     
#     x4=k2*k8*(k4+k5)+k3*k5*(k1+k8)

#     E1=x1/(x1+x2+x3+x4)    
#     E2=x2/(x1+x2+x3+x4)
#     E3=x3/(x1+x2+x3+x4)    
#     E4=x4/(x1+x2+x3+x4)

#     allo=1.0/(1.0+(KmCaAct/cass)**2.0)
#     JncxNa=3.0*(E4*k7-E1*k8)+E3*k4pp-E2*k3pp
#     JncxCa=E2*k2-E1*k1

#     INaCa_ss=0.2*Gncx*allo*(zna*JncxNa+zca*JncxCa)
#     ###########################################################################
#     ### INaK current ###
#     ### INaK parameter ###
#     bGnak = 2
#     ###
#     k1p=949.5      
#     k1m=182.4      
#     k2p=687.2      
#     k2m=39.4
#     k3p=1899.0     
#     k3m=79300.0    
#     k4p=639.0      
#     k4m=40.0
#     Knai0=9.073    
#     Knao0=27.78    
#     delta2=-0.1550
#     Knai=Knai0*np.exp((delta2*v*F)/(3.0*R*T))
#     Knao=Knao0*np.exp(((1.0-delta2)*v*F)/(3.0*R*T))
#     Kki=0.5            
#     Kko=0.3582     
#     MgADP=0.05     
#     MgATP=9.8
#     Kmgatp=1.698e-7    
#     H=1.0e-7       
#     eP=4.2         
#     Khp=1.698e-7
#     Knap=224.0         
#     Kxkur=292.0
#     P=eP/(1.0+H/Khp+nai/Knap+ki/Kxkur)

#     a1=(k1p*(nai/Knai)**3.0)/((1.0+nai/Knai)**3.0+(1.0+ki/Kki)**2.0-1.0)
#     b1=k1m*MgADP
#     a2=k2p
#     b2=(k2m*(nao/Knao)**3.0)/((1.0+nao/Knao)**3.0+(1.0+ko/Kko)**2.0-1.0)
#     a3=(k3p*(ko/Kko)**2.0)/((1.0+nao/Knao)**3.0+(1.0+ko/Kko)**2.0-1.0)
#     b3=(k3m*P*H)/(1.0+MgATP/Kmgatp)
#     a4=(k4p*MgATP/Kmgatp)/(1.0+MgATP/Kmgatp)
#     b4=(k4m*(ki/Kki)**2.0)/((1.0+nai/Knai)**3.0+(1.0+ki/Kki)**2.0-1.0)

#     x1=a4*a1*a2+b2*b4*b3+a2*b4*b3+b3*a1*a2
#     x2=b2*b1*b4+a1*a2*a3+a3*b1*b4+a2*a3*b4
#     x3=a2*a3*a4+b3*b2*b1+b2*b1*a4+a3*a4*b1
#     x4=b4*b3*b2+a3*a4*a1+b2*a4*a1+b3*b2*a1

#     E1=x1/(x1+x2+x3+x4)    
#     E2=x2/(x1+x2+x3+x4)
#     E3=x3/(x1+x2+x3+x4)    
#     E4=x4/(x1+x2+x3+x4)
#     zk=1.0   
#     JnakNa=3.0*(E1*a3-E2*b3)   
#     JnakK=2.0*(E4*b1-E3*a1)    
#     Pnak=30*bGnak
#     if celltype==1:
#         Pnak=Pnak*0.9
#     elif celltype==2:
#         Pnak=Pnak*0.7

#     INaK=Pnak*(zna*JnakNa+zk*JnakK)
#     ###########################################################################
#     ### CaCl current (set to 0)###
#     ### ICl parameters###
#     IClCa_si = 0
#     IClb_si = 0
#     ###
#     Cli = 15   # Intracellular Cl  [mM]
#     Clo = 150  # Extracellular Cl  [mM]
#     ecl = (R*T/F)*np.log(Cli/Clo) # [mV]
#     GClCa =0.5* 0.109625   # [mS/uF]
#     GClB = 1*9e-3        # [mS/uF]
#     KdClCa = 100e-3    # [mM]
#     IClCa = IClCa_si*GClCa/(1+KdClCa/cass)*(v-ecl)
#     IClbk = IClb_si*GClB*(v-ecl)
#     ###########################################################################
#     ### Background currents: IKb, INab, ICab ###
#     ### IKb current ###
#     xkb = 1.0 / (1.0+np.exp(-(v-14.48)/18.34))
#     GKb = 0.003
#     if celltype==1:
#         GKb = GKb*0.6

#     IKb = GKb*xkb*(v-EK)
#     ###########################################################################
#     ### INab current ###
#     PNab = 3.75e-10
#     INab = PNab*vffrt*(nai*np.exp(vfrt)-nao)/(np.exp(vfrt)-1.0)
#     ###########################################################################
#     ### ICab current ###
#     ### ICab parameter ###
#     bPcab = 4
#     ###
#     PCab = 2.5e-8*bPcab
#     ICab = (1-undo_ICab)*PCab*4.0*vffrt*(1.2*cai*np.exp(2.0*vfrt)-0.341*cao)/(np.exp(2.0*vfrt)-1.0)
#     ###########################################################################
#     ### IpCa current ###
#     GpCa = 0.0005
#     IpCa = GpCa*cai/(0.0005+cai)
#     ###########################################################################
#     ### Simulation Procotols ###
#     #I_stim = IStim(time,cycleLength,amp,dur)
    
#     if np.mod(time,cycleLength) <= dur: #5
#         I_stim = amp
#     else:
#         I_stim = 0.0

#     dv = - (INa+INaL+Ito+ICaL+ICaNa+ICaK+IKr+IKs+IK1+INaCa_i+INaCa_ss+INaK+INab+IKb+IpCa+ICab+I_stim+IClCa+IClbk)

#     ###########################################################################
#     ### Diffusion Fluxes ###
#     ### Jdiff,Ca  parameter ###
#     bJdiff= 1.7
#     ###
#     JdiffNa = (nass-nai) /2.0
#     JdiffK  = (kss-ki)   /2.0
#     Jdiff   = (cass-cai) *bJdiff/0.2
#     ###########################################################################
#     ### RyRs CICR from SR ###
#     ### Jrel parameters ###
#     g_irel_max	= 20*10**-3 # millimolar_per_second (in calcium_dynamics)
#     RyRa1		= 0.05  # uM
#     RyRa2		= 0.03   # uM
#     RyRohalf	= 0.12-(RyRa1-RyRa2/2)	# uM
#     RyRchalf	= 0.10-(RyRa1-RyRa2/2) # uM
#     ###
#     fJrelp=(1.0/(1.0+KmCaMK/CaMKa))
#     RyRSRCass = (1 - 1/(1 +  np.exp((casr-0.3)/0.1)))

#     RyRainfss = RyRa1-RyRa2/(1 + np.exp((1000*cass-(0.043))/0.0082))
#     RyRtauadapt = 1000 #ms
#     dRyRa = (RyRainfss- RyRa)/RyRtauadapt

#     RyRoinfss = (1 - 1/(1 +  np.exp((1000*cass-(RyRa+ RyRohalf))/0.003)))
#     RyRtauact = 18.75*10**-1/1.875       #ms
#     dRyRo = (RyRoinfss- RyRo)/RyRtauact

#     RyRcinfss = (1/(1 + np.exp((1000*cass-(RyRa+RyRchalf))/0.001)))
#     RyRtauinact = 2*87.5*1/10    #ms
#     dRyRc = (RyRcinfss- RyRc)/RyRtauinact

#     if celltype==2:
#         g_irel_max_M = g_irel_max*1.7
#         Jrelnp = g_irel_max_M*RyRSRCass*RyRo*RyRc*(casr-cass)
#     else:
#         Jrelnp = g_irel_max*RyRSRCass*RyRo*RyRc*(casr-cass)


#     RyRtauinactp = RyRtauinact*1.25
#     dRyRcp = (RyRcinfss- RyRcp)/RyRtauinactp
#     g_irel_max_p = g_irel_max*1.25
#     if celltype==2:
#         g_irel_max_p = g_irel_max*1.25*1.7

#     Jrelp = g_irel_max_p*RyRSRCass*RyRo*RyRcp*(casr-cass)

#     Jrel=((1.0-fJrelp)*Jrelnp+fJrelp*Jrelp)
#     ###########################################################################
#     ### Ca2+ Uptake Flux ###
#     ### SERCA Jup parameter ###
#     cJup = 3.13
#     ###
#     Jupnp=0.004375*cai/(cai+0.00092)
#     Jupp=2.75*0.004375*cai/(cai+0.00092-0.00017) 
#     if celltype==1:                               
#         Jupnp=Jupnp*1.3
#         Jupp=Jupp*1.3

#     fJupp=(1.0/(1.0+KmCaMK/CaMKa))
#     Jleak=0.0123*casr/15.0
#     Jup = cJup*((1.0-fJupp)*Jupnp+fJupp*Jupp)
#     Vmax_SRCaP = 1.0*5.3114e-3  # [mM/msec] (286 umol/L cytosol/sec)
#     Kmf = 0.246e-3          # [mM] default
#     Kmr = 1.7               # [mM]L cytosol
#     hillSRCaP = 1.787       # [mM]
#     Jup2=Vmax_SRCaP*((cai/Kmf)**hillSRCaP-(casr/Kmr)**hillSRCaP)/(1+(cai/Kmf)**hillSRCaP+(casr/Kmr)**hillSRCaP)
#     ###########################################################################
#     ### Tranlocation Flux (usefull if considering JSR+NSR) ###
#     Jtr=0
#     ###########################################################################
#     ### Calcium Buffer Constants ###
#     cmdnmax=0.05
#     if celltype==1:
#         cmdnmax=cmdnmax*1.2 # modified  respect ORd (*1.3)

#     kmcmdn=0.00238     
#     trpnmax=0.07   
#     kmtrpn=0.0005
#     BSRmax=0.047       
#     KmBSR=0.00087
#     BSLmax=1.124       
#     KmBSL=0.0087
#     ### buffering scaling 
#     csqnmax=10.0*10**-1 # ORd csqnmax=10.0 
#     kmcsqn=0.8
#     ###########################################################################
#     ### EGTA ###
#     if maxEGTA==0:
#         dEGTA=0
#         dEGTAi=0
#     else:
#         # Hellam & Podolsky Values
#         kon=2 #mM**-1 ms**-1
#         koff=4.0e-4  #ms**-1
#         dEGTA=kon*cass*(maxEGTA-EGTA)-koff*EGTA
#         dEGTAi=0

#     ###########################################################################
#     ### update intracellular [Na], [K] and [Ca] ###
#     # [Na]
#     dnai=-(INa+INaL+3.0*INaCa_i+3.0*INaK+INab)*Acap/(F*vmyo)+JdiffNa*vss/vmyo
#     dnass=-(ICaNa+3.0*INaCa_ss)*Acap/(F*vss)-JdiffNa
#     # [K]
#     if ki_cost==0:
#         dki=-(Ito+IKr+IKs+IK1+IKb+I_stim-2.0*INaK)*Acap/(F*vmyo)+JdiffK*vss/vmyo
#         dkss=-(ICaK)*Acap/(F*vss)-JdiffK
#     else:
#         dki=0
#         dkss=0

#     # [Ca]
#     Bcai   = 1.0 / (1.0+cmdnmax*kmcmdn/(kmcmdn+cai)**2.0 +trpnmax*kmtrpn/(kmtrpn+cai)**2.0)
#     dcai   = Bcai*(-(IpCa+ICab-2.0*INaCa_i)*Acap/(2.0*F*vmyo) -Jup*vsr/vmyo+Jleak*vsr/vmyo+Jdiff*vss/vmyo-dEGTAi)
#     Bcass  = 1.0/(1.0+BSRmax*KmBSR/(KmBSR+cass)**2.0 +BSLmax*KmBSL/(KmBSL+cass)**2.0)
#     dcass  =  Bcass*(-(ICaL-2.0*INaCa_ss)*Acap/(2.0*F*vss) +Jrel*vsr/vss-Jdiff-dEGTA)
#     # dcansr = Jup-Jtr*vjsr/vnsr (usefull if considering JSR+NSR)
#     Bcasr = 1.0/(1.0+csqnmax*kmcsqn/(kmcsqn+casr)**2.0)
#     dcasr = Bcasr*(Jup-Jleak-Jrel)
#     ###########################################################################
#     ## Output Computation
#     # When flag==1 -> dX
#     # output=[dv,dnai,  dnass,  dki, dkss, dcai,  dcass,  0,dcasr , dm, dhf,  dhs,  dj , dhsp,  djp, dmL,  dhL,  dhLp,
#     #             da, diF, diS,dap, diFp,  diSp, 0,dOk , dOkp,0,0,0,dnca,0, 0,0,0, dxs1,  dxs2,
#     #             dxk1, 0,0,  dCaMKt,  dI1k , dI2k,  dCk , dI1kp,dI2kp,  dCkp,  dI1Cak,  dI2Cak,  dCCak, dI1Cakp,
#     #             dI2Cakp, dCCakp, djnca, dEGTA, dEGTAi,  dRyRa,  dRyRo,  dRyRc,  dRyRcp, dC1_dutta, dC2_dutta, dCb_dutta, dD_dutta,
#     #             dIC1_dutta,dIC2_dutta, dIO_dutta, dIOb_dutta, dO_dutta, dOb_dutta]
    
#     output=[dv,      dnai,    dnass,   dki,     dkss,     #   1
#         dcai,    dcass,   0,       dcasr,   dm,       #   2
#         dhf,     dhs,     dj,      dhsp,    djp,      #   3
#         dmL,     dhL,     dhLp,    da,      diF,      #   4
#         diS,     dap,     diFp,    diSp,    0 ,       #   5
#         dOk,     dOkp,    0,       0,       0 ,       #   6
#         dnca,    0,       0,       0,       0  ,     #   7
#         dxs1,    dxs2,    dxk1,    0,       0 ,   #   8
#         dCaMKt,  dI1k,    dI2k,    dCk,     dI1kp,    #   9
#         dI2kp,   dCkp,    dI1Cak,  dI2Cak,  dCCak,    #  10
#         dI1Cakp,dI2Cakp, dCCakp,  djnca,   dEGTA,    #  11
#         dEGTAi,  dRyRa,   dRyRo,   dRyRc,   dRyRcp    ,    #  12
#         dC1_dutta,     dC2_dutta,     dCb_dutta,     dD_dutta,      dIC1_dutta,    # 13
#         dIC2_dutta,    dIO_dutta,     dIOb_dutta,    dO_dutta,      dOb_dutta ]
    
#     return(output)

# from APD import*
# from conductances import *
# cycle_length= 1000
# numCLs = 2
# cycles = 100

# cell_type = "ENDO"
# option = "BPS 2020"

# GKs_val = GKs_conductance(option,cell_type)  
# GKr_val =GKr_conductance(option,cell_type) 
# GK1_val = GK1_conductance(option,cell_type)
# GNaK_val = GNaK_conductance(option,cell_type)  
# GNCX_val =GNCX_conductance(option,cell_type) 
# GpCa_val = GpCa_conductance(option,cell_type)
# Gto = Gto_conductance(option,cell_type)
# GNa_late_val , GNa_fast_val = GNa_conductance(option,cell_type)
# GNab_val = GNab_conductance(option,cell_type)  
# GCab_val =GCab_conductance(option,cell_type) 
# GKb_val = GKb_conductance(option,cell_type)
# GClb_val =GClb_conductance(option,cell_type) 
# GClCa_val = GClCa_conductance(option,cell_type)
# GCa = GCa_conductance(option,cell_type)

# df, currents_df, artefact_duration = run_BPS_model(cycles,cycle_length,numCLs,cell_type,
#                                                    GKs_val,GKr_val,GK1_val,Gto,GNa_late_val,GNa_fast_val,GNab_val,GCab_val,GCa,GNCX_val,GpCa_val,GKb_val,GNaK_val,GClCa_val,GClb_val)
# df_chunks,start_APD_time,target_voltage,max_APD_voltage,state_chunks = APD_df(df,cycle_length)
# # APD_dfs, combined_figure= plot_APD(df_chunks,start_APD_time,target_voltage,max_APD_voltage,artefact_duration,plot_fig=True)