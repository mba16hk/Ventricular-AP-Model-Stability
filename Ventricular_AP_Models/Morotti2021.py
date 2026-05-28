# Morotti code
# # https://github.com/drgrandilab/Morotti-et-al-2021-Cross-species-translators-of-electrophysiological-response
# this only allows for modelling EPI and ENDO
import numpy as np
import math
from numba import njit
from scipy.integrate import solve_ivp
import pandas as pd
import matplotlib.pyplot as plt
from conductances import *


def flatten(seq):
    flat = []
    for v in seq:
        if isinstance(v, (list, tuple, np.ndarray)):
            flat.extend(flatten(v))   # recursive flatten
        else:
            flat.append(float(v))
    return flat

@njit
def I_stimulus(t,cycleLength,Stimdur,amplitude):
    if np.mod(t,cycleLength) <= Stimdur: #5
        I_app = amplitude #9.5
    else:
        I_app = 0.0
    return(I_app)


# master ODE file
def run_Morotti_model(cycles, cycleLength, cell_type,
                      flag_BARS, camkii_exp, amp=9.5):
    # stimDur: stimulus pulse duration in ms. Hardcoded to 5 ms (standard value used for
    # ventricular pacing protocols); not exposed because the upstroke is short relative to
    # the cycle length, so this is rarely varied between experiments.
    stimDur = 5
    # flag_cam and flag_CaMKII are locked ON to match the C# reference (MasterOde.cs:42-67).
    # Both modules must be active by default; previously these were exposed as parameters.
    flag_cam = 1     # CaM module active
    flag_CaMKII = 1  # CaMKII module active
    # Ligtot (ISO concentration): C# default is 0.0 uM (no ISO). When flag_BARS is enabled,
    # default ISO concentration is 0.1 uM (matches MorottiModel Program.cs sample value).
    Ligtot = 0.1 if flag_BARS else 0.0
    # This function calls the ode files for EC coupling, CaM reactions, CaMKII
    # phosphorylation module, and PKA phosphorylation module
    model_type = "Morotti 2021"
    GKs                          = GKs_conductance(model_type, cell_type)
    GKr                          = GKr_conductance(model_type, cell_type)
    GK1                          = GK1_conductance(model_type, cell_type)
    Gto_slow, Gto_fast           = Gto_conductance(model_type, cell_type)
    GNa_late, GNa_fast           = GNa_conductance(model_type, cell_type)
    pCa_input, pNa_input, pK_input = GCa_conductance(model_type, cell_type)
    GNCX                         = GNCX_conductance(model_type, cell_type)
    GNaK                         = GNaK_conductance(model_type, cell_type)
    GKb                          = GKb_conductance(model_type, cell_type)
    GNab                         = GNab_conductance(model_type, cell_type)
    GCab                         = GCab_conductance(model_type, cell_type)
    GpCa                         = GpCa_conductance(model_type, cell_type)
    GClCa_input                  = GClCa_conductance(model_type, cell_type)
    GClb                         = GClb_conductance(model_type, cell_type)

    ## Collect params and ICs for each module
    
    #### Starting variables from yf_hvm_1Hz_NEW.mat
    y = [
    #ecc conditions
    0.002395370924782,0.972900457609240, 0.982937732524635,0.037537355638997,0.995148893960571,0.025561364688608,0.015035426126818,4.109046687846818e-04, #8
    0.753963674787844,4.108957753253549e-04,0.999996530696865,0.026852387145408,0.038699630005395,0.891449425814954,8.924931884392747e-07,1.086766595461244e-07, #8
    3.388079539664790,0.739245812195284,0.009408959341664,0.119700181505944,0.009517255004738,2.902163683341909e-04,0.002156008859874,0.137324360960861,0.002302413651648, #9
    0.007826115587695,0.009847847549085,0.076292934976944,0.114170292079697,1.238677768359822,0.591440205388164,8.112760693294007,8.112379967834052,8.112600963211435,120, #10
    1.860085927360863e-04,1.061732687568628e-04,9.265398453208212e-05,-82.355812930887012,0.994600000000000,0.164919523639099,0.672116674747578,0.024823559212822,3.743018764790034e-04,#9
    1.092767907666780e-06,6.622277013764972e-07,2.445823203636432e-08,3.687597980497831e-10,1.074945747079239e-12,0.099863433656389,0.003711398645852,7.395416328923418e-05,0.198248145419195, #9
    2.087633329357285e-05,2.959152839152456e-07,6.965501104018141e-07,1.189825000552348e-06,1.029347666224818,1.035247627394144,0.931430269412762,3.152528512379242e-05,3.340785017955655e-04, #9
    0.002287417148897,5.770919706511836e-05,0.065855847130062,0.931438489380696,3.154709753100310e-05,3.305402531059830e-04,0.002263594152655,5.770291640282686e-05,0.065846498448716, #9
    0.931905327586486,3.153182263258687e-05,2.363267274509064e-04,0.001617775935312,5.796979643729833e-05,0.066147914376066,0.931955290202505,3.154801438372434e-05,2.280479439154208e-04, #9
    0.001561383536989,5.795855188398627e-05,0.066134167059486,#3
    
    #y_camDyad
    3.575678244822496e+02,6.425193745248360,0.002560149767141,0,0,0,0.542531390122104,0.036017602373236,8.892053960258144e-06, #9
    3.555478858577988e-09,2.757378229410981e-09,1.283857636116461e-04,0.003911611295999,0.012587339797252,3.600841846112160, #6
    
    #y_camSL
    0.038262808137495,6.665651876059739e-05,1.615476478982412e-08, #3
    1.884620552018856,14.464420238220532,3.821893183298638e-04,1.356665593984852e-05,2.097580407707189e-05,6.447563492233093e-07,7.131500687082256e-12,4.148623670244480e-08, #8
    2.998336959424326e-04,9.831501810743470e-07,5.182335614689293e-06,0.002196396792865, #4
    
    # y_camCyt
    0.038004587707597,5.134025932153523e-05,1.088072437795273e-09,3.625174855897097,1.654631450816830, #5
    3.328462251330528e-05,1.028103148155894e-05,1.008929317387250e-07,1.663137117241292e-12,1.810658360769449e-17,1.108302069893868e-13,1.416060462647712e-04,4.607451914606630e-07, #8
    2.873942502356324e-07,4.462242271376960e-06, #2
    
    #y_CaMKII
    16.454392570076561,16.014188504168150,2.973572313530868e+02,75.488671120894125,0.475430409772086,2.648481214137712e-05, #6
    
    #y_BAR
    0,4.940656458412465e-324, #2
    0.002787194409188,-2.130515531454168e-43,8.090625461215628e-04,0.055743888183766,6.534959700062312e-04,0.056397384151646,0.005266496098760,6.493415652004783e-04,6.493415652004783e-04, #9
    0.569723858593256,0.961031412390131,0.059415669444682,6.591163099993006e-04,0.158584968334472,0.018183525756143,0.140401332055390,0.070599803808102,0.004364825698558,4.842035501934286e-05, #10
    0.042982760025185,0.004928465577490,0.038054398324534,0.017447158763135,0.017467154295713,0.648149274513712,0.706550421700272,0.001064311291498,0.001252650118185,0.005069209158498, #10
    0.514141026958008,0.514141026958008,0.004021159536356,0.004021159536354,0.005798151105184,0.004021159536354,0.004021159536354,0.004021159536354] #8

    # CaMKII
    #camkii_exp = p[0] # 0, 1 or 6
    # ISO
    #Ligtot = p[1] # [uM] ISO
    # Protocol
    #prot_index = p[2]
    #cycleLength = p[3]     # [ms]
    #prot_input_par = p[4] # Input parameter for stimulation protocols
    # Ca clamp
    Ca_clamp = 0#p[5] # 0 Ca-free, 1 Ca-clamp to initial value
    # Na clamp
    Na_clamp = 0#p[6] # 0 Na-free, 1 Na-clamp to initial value
    # Modules to use
    # flag_ECC = 1#p[7]    # if 0, module clamped
    # flag_cam = 1#p[8]    # if 0, module clamped
    # flag_CaMKII = 1#p[9] # if 0, module clamped
    #flag_BAR = 1#p[10]   # if 0, module clamped

    # Ca_j is y[35], Ca_sl is y[36], Ca_cytosol is y[37]
    ny_ECC = 83
    ny_cam = 15
    ny_CaMKII = 6
    ny_BAR = 39
    y_ecc = y[0:ny_ECC] #this should be of length 83
    y_camDyad = y[ny_ECC:ny_ECC+ny_cam] # this should be of length 15, starts at 83 end at 97
    y_camSL = y[ny_ECC+ny_cam:ny_ECC+2*ny_cam] #this should length 15, starts at 99 and end at 113
    y_camCyt = y[ny_ECC+2*ny_cam:ny_ECC+3*ny_cam] #this should length 15, starts at 114 and ends at 128
    y_CaMKII = y[ny_ECC+3*ny_cam:ny_ECC+3*ny_cam+ny_CaMKII] #this should be 0-6, starts at 129 and ends at 134
    y_BAR = y[ny_ECC+3*ny_cam+ny_CaMKII:ny_ECC+3*ny_cam+ny_CaMKII+ny_BAR] #this should be 0-38, starts at 135 and ends at 173
    ## Parameters

    K = y[34] #135 # [mM]
    Mg = 1  # [mM]

    CKIIOE = 0+int(camkii_exp>1) # if camkii_exp>1, return true which is logical 1

    CaMtotDyad = 418           # [uM]
    BtotDyad = 1.54/8.293e-4   # [uM]
    CaMKIItotDyad = 120*camkii_exp        # [uM] 
    CaNtotDyad = 3e-3/8.293e-4 # [uM] 
    PP1totDyad = 96.5          # [uM]
    CaMtotSL = 5.65            # [uM]
    BtotSL = 24.2              # [uM]
    CaMKIItotSL = 120*8.293e-4*camkii_exp # [uM]
    CaNtotSL = 3e-3            # [uM]
    PP1totSL = 0.57            # [uM]
    CaMtotCyt = 5.65           # [uM]
    BtotCyt = 24.2             # [uM]
    CaMKIItotCyt = 120*8.293e-4*camkii_exp# [uM]
    CaNtotCyt = 3e-3           # [uM] 
    PP1totCyt = 0.57           # [uM]

    LCCtotDyad = 31.4*.9       # [uM] - Total Dyadic [LCC] - (umol/l dyad)
    LCCtotSL = 0.0846          # [uM] - Total Subsarcolemmal [LCC] (umol/l sl)
    RyRtot = 382.6             # [uM] - Total RyR (in Dyad)
    PP1_dyad = 95.7            # [uM] - Total dyadic [PP1]
    PP1_SL = 0.57              # [uM] - Total Subsarcolemmal [PP1]
    PP2A_dyad = 95.76          # [uM] - Total dyadic PP2A
    OA = 0                     # [uM] - PP1/PP2A inhibitor Okadaic Acid
    PLBtot = 38                # [uM] - Total [PLB] in cytosolic units

    # Parameters for BAR module
    LCCtotBA = 0.025           # [uM] - [umol/L cytosol]
    RyRtotBA = 0.135           # [uM] - [umol/L cytosol]
    PLBtotBA = PLBtot          # [uM] - [umol/L cytosol]
    TnItotBA = 70              # [uM] - [umol/L cytosol]
    IKstotBA = 0.025           # [uM] - [umol/L cytosol]
    PP1_PLBtot = 0.89          # [uM] - [umol/L cytosol]
    PLMtotBA = 48              # [uM] - [umol/L cytosol] as in Yang & Saucerman (mouse) model
    MyototBA = 70              # [uM] - [umol/L cytosol] as TnI
    IKrtotBA = 0.025           # [uM] - [umol/L cytosol] as IKs
    IClCatotBA = 0.025         # [uM] - [umol/L cytosol] as ICFTR
    ItototBA = 0.025         # [uM] - [umol/L cytosol] as IKr
    IK1totBA = 0.025         # [uM] - [umol/L cytosol] as IKr
    INatotBA = 0.025         # [uM] - [umol/L cytosol] as IKr
    
    ########### FOR CAMKII ################
    # L-Type Ca Channel (LCC) parameters
    k_ckLCC = 0.4                  # [s**-1]
    k_pp1LCC = 0.1103              # [s**-1] 
    k_pkaLCC = 13.5                # [s**-1] 
    k_pp2aLCC = 10.1               # [s**-1] 

    KmCK_LCC = 12                  # [uM] 
    KmPKA_LCC = 21                 # [uM] 
    KmPP2A_LCC = 47                # [uM] 
    KmPP1_LCC = 9                  # [uM] 

    # Ryanodine Receptor (RyR) parameters
    k_ckRyR = 0.4                  # [s**-1] 
    k_pkaRyR = 1.35                # [s**-1] 
    k_pp1RyR = 1.07                # [s**-1] 
    k_pp2aRyR = 0.481              # [s**-1] 

    # Basal RyR phosphorylation (numbers based on param estimation)
    kb_2809 = 0.51                 # [uM/s] - PKA site
    kb_2815 = 0.35                 # [uM/s] - CaMKII site

    KmCK_RyR = 12                  # [uM] 
    KmPKA_RyR = 21                 # [uM] 
    KmPP1_RyR = 9                  # [uM] 
    KmPP2A_RyR = 47                # [uM] 

    # Phospholamban (PLB) parameters
    k_ckPLB = 8e-3                 # [s**-1]
    k_pp1PLB = .0428               # [s**-1]

    KmCK_PLB = 12
    KmPP1_PLB = 9
    
    # Okadaic Acid inhibition params (based on Huke/Bers [2008])
    # Want to treat OA as non-competitive inhibitor of PP1 and PP2A
    Ki_OA_PP1 = 0.78        # [uM] - Values from fit
    Ki_OA_PP2A = 0.037      # [uM] - Values from fit

    # Default PKA level
    PKAc = 95.6*.54
    
    #BARS constants
    FSK = 0 # (uM) forskolin concentration
    IBMX = 0 # (uM) IBMX concentration
    ## b-AR module
    #b1ARtot = 0.00528        # (uM) total b1-AR protein # MOUSE
    b1ARtot = 0.028 # RABBIT

    kf_LR           = 1              # (1/[uM ms]) forward rate for ISO binding to b1AR
    kr_LR           = 0.285          # (1/ms) reverse rate for ISO binding to b1AR
    kf_LRG          = 1              # (1/[uM ms]) forward rate for ISO:b1AR association with Gs
    kr_LRG          = 0.062          # (1/ms) reverse rate for ISO:b1AR association with Gs
    kf_RG           = 1              # (1/[uM ms]) forward rate for b1AR association with Gs
    kr_RG           = 33             # (1/ms) reverse rate for b1AR association with Gs

    Gstot           = 3.83           # (uM) total Gs protein
    k_G_act         = 16e-3          # (1/ms) rate constant for Gs activation
    k_G_hyd         = 0.8e-3         # (1/ms) rate constant for G-protein hydrolysis
    k_G_reassoc     = 1.21           # (1/[uM ms]) rate constant for G-protein reassociation

    kf_bARK         = 1.1e-6         # (1/[uM ms]) forward rate for b1AR phosphorylation by b1ARK
    kr_bARK         = 2.2e-6         # (1/ms) reverse rate for b1AR phosphorylation by b1ARK
    kf_PKA          = 3.6e-6         # (1/[uM ms]) forward rate for b1AR phosphorylation by PKA
    kr_PKA          = 2.2e-6         # (1/ms) reverse rate for b1AR phosphorylation by PKA
    
    #cAMP module
    #ACtot = 70.57e-3       # (uM) total adenylyl cyclase # MOUSE
    ACtot = 47e-3 # RABBIT

    ATP             = 5e3            # (uM) total ATP
    k_AC_basal      = 0.2e-3         # (1/ms) basal cAMP generation rate by AC
    Km_AC_basal     = 1.03e3         # (uM) basal AC affinity for ATP

    Kd_AC_Gsa       = 0.4            # (uM) Kd for AC association with Gsa
    kf_AC_Gsa       = 1              # (1/[uM ms]) forward rate for AC association with Gsa
    kr_AC_Gsa       = Kd_AC_Gsa      # (1/ms) reverse rate for AC association with Gsa

    k_AC_Gsa        = 8.5e-3         # (1/ms) basal cAMP generation rate by AC:Gsa
    Km_AC_Gsa       = 315.0          # (uM) AC:Gsa affinity for ATP

    Kd_AC_FSK       = 44.0           # (uM) Kd for FSK binding to AC
    k_AC_FSK        = 7.3e-3         # (1/ms) basal cAMP generation rate by AC:FSK
    Km_AC_FSK       = 860.0          # (uM) AC:FSK affinity for ATP

    # MOUSE
    #PDEtot          = 22.85e-3       # (uM) total phosphodiesterase
    #k_cAMP_PDE      = 5e-3           # (1/ms) cAMP hydrolysis rate by PDE
    #k_cAMP_PDEp     = 2*k_cAMP_PDE   # (1/ms) cAMP hydrolysis rate by phosphorylated PDE
    #Km_PDE_cAMP     = 1.3            # (uM) PDE affinity for cAMP

    # RABBIT
    #PDE3tot = 0.036       # (uM) total phosphodiesterase 
    #PDE4tot = 0.036       # (uM) total phosphodiesterase
    PDE3tot = 0.75*0.036 # PROVA
    PDE4tot = 0.75*0.036 # PROVA
    k_cAMP_PDE3 = 3.5e-3               # k_pde3        [1/ms]
    k_cAMP_PDE3p = 2*k_cAMP_PDE3   # (1/ms) cAMP hydrolysis rate by phosphorylated PDE
    Km_PDE3_cAMP = 0.15             # Km_pde3       [uM]
    k_cAMP_PDE4 = 5.0e-3               # k_pde4        [1/ms]
    k_cAMP_PDE4p = 2*k_cAMP_PDE4   # (1/ms) cAMP hydrolysis rate by phosphorylated PDE
    Km_PDE4_cAMP = 1.3              # Km_pde4       [uM]

    Kd_PDE_IBMX     = 30.0           # (uM) Kd_R2cAMP_C for IBMX binding to PDE
    k_PKA_PDE       = 7.5e-3         # (1/ms) rate constant for PDE phosphorylation by type 1 PKA
    k_PP_PDE        = 1.5e-3         # (1/ms) rate constant for PDE dephosphorylation by phosphatases
    
    #pKa module
    #PKAIItot = 0.059          # (uM) total type 2 PKA # MOUSE
    PKItot          = 0.18           # (uM) total PKI
    kf_RC_cAMP      = 1              # (1/[uM ms]) Kd for PKA RC binding to cAMP
    kf_RCcAMP_cAMP  = 1              # (1/[uM ms]) Kd for PKA RC:cAMP binding to cAMP
    kf_RcAMPcAMP_C  = 4.375          # (1/[uM ms]) Kd for PKA R:cAMPcAMP binding to C
    kf_PKA_PKI      = 1              # (1/[uM ms]) Ki for PKA inhibition by PKI
    kr_RC_cAMP      = 1.64           # (1/ms) Kd for PKA RC binding to cAMP
    kr_RCcAMP_cAMP  = 9.14           # (1/ms) Kd for PKA RC:cAMP binding to cAMP
    kr_RcAMPcAMP_C  = 1              # (1/ms) Kd for PKA R:cAMPcAMP binding to C
    kr_PKA_PKI      = 2e-4           # (1/ms) Ki for PKA inhibition by PKI

    epsilon         = 10             # (-) AKAP-mediated scaling factor

    PKAIItot = 0.084 # RABBIT
    I1tot           = 0.3            # (uM) total inhibitor 1
    k_PKA_I1        = 60e-3          # (1/ms) rate constant for I-1 phosphorylation by type 1 PKA
    Km_PKA_I1       = 1.0            # (uM) Km for I-1 phosphorylation by type 1 PKA
    Vmax_PP2A_I1    = 14.0e-3        # (uM/ms) Vmax for I-1 dephosphorylation by PP2A
    Km_PP2A_I1      = 1.0            # (uM) Km for I-1 dephosphorylation by PP2A

    Ki_PP1_I1       = 1.0e-3         # (uM) Ki for PP1 inhibition by I-1
    kf_PP1_I1       = 1              # (uM) Ki for PP1 inhibition by I-1
    kr_PP1_I1       = Ki_PP1_I1      # (uM) Ki for PP1 inhibition by I-1
    k_PKA_PLB = 54e-3     #p[43] = 54     # k_pka_plb     [1/ms]
    Km_PKA_PLB = 21     #p[44] = 21     # Km_pka_plb    [uM]
    k_PP1_PLB = 8.5e-3    #p[45] = 8.5    # k_pp1_plb     [1/ms]
    Km_PP1_PLB = 7.0    #p[46] = 7.0    # Km_pp1_plb    [uM]
    
    k_PKA_PLM = 54e-3 # p(103) = 54     # k_pka_plb     [1/ms]
    Km_PKA_PLM = 21 # p(104) = 21     # Km_pka_plb    [uM]
    k_PP1_PLM = 8.5e-3 # p(105) = 8.5    # k_pp1_plb     [1/ms]
    Km_PP1_PLM = 7.0 # p(106) = 7.0    # Km_pp1_plb    [uM]
    PKACII_LCCtot = 0.025  #p[53] = 0.025  # PKAIIlcctot   [uM]
    PP1_LCC = 0.025  #p[54] = 0.025  # PP1lcctot     [uM]
    PP2A_LCC = 0.025  #p[55] = 0.025  # PP2Alcctot    [uM]
    k_PKA_LCC = 54e-3     #p[56] = 54     # k_pka_lcc     [1/ms]
    Km_PKA_LCC = 21     #p[57] = 21#*1.6     # Km_pka_lcc    [uM]
    k_PP1_LCC = 8.52e-3   #p[58] = 8.52   # k_pp1_lcc     [1/ms] RABBIT, MOUSE #p[58] = 8.5   # k_pp1_lcc     [1/sec] RAT
    Km_PP1_LCC = 3      #p[59] = 3      # Km_pp1_lcc    [uM]
    k_PP2A_LCC = 10.1e-3   #p[60] = 10.1   # k_pp2a_lcc    [1/ms]
    Km_PP2A_LCC = 3      #p[61] = 3      # Km_pp2a_lcc   [uM]
    PKAIIryrtot = 0.034  #p[63] = 0.034  # PKAIIryrtot   [uM]
    PP1ryr = 0.034  #p[64] = 0.034  # PP1ryr        [uM]
    PP2Aryr = 0.034  #p[65] = 0.034  # PP2Aryr       [uM]
    kcat_pka_ryr = 54e-3     #p[66] = 54     # kcat_pka_ryr  [1/ms]
    Km_pka_ryr = 21     #p[67] = 21     # Km_pka_ryr    [uM]
    kcat_pp1_ryr = 8.52e-3   #p[68] = 8.52   # kcat_pp1_ryr  [1/ms]
    Km_pp1_ryr = 7      #p[69] = 7      # Km_pp1_ryr    [uM]
    kcat_pp2a_ryr = 10.1e-3   #p[70] = 10.1   # kcat_pp2a_ryr [1/ms]
    Km_pp2a_ryr = 4.1    #p[71] = 4.1    # Km_pp2a_ryr   [uM]
    PP2A_TnI = 0.67   # PP2Atni       [uM]
    k_PKA_TnI = 54e-3     # kcat_pka_tni  [1/ms]
    Km_PKA_TnI = 21     # Km_pka_tni    [uM]
    k_PP2A_TnI = 10.1e-3   # kcat_pp2a_tni [1/ms]
    Km_PP2A_TnI = 4.1    # Km_pp2a_tni   [uM]
    PP2A_myo = 0.67             # PP2Amyo       [uM]
    kcat_pka_myo = 54e-3          # kcat_pka_myo  [1/ms]
    Km_pka_myo = 21            # Km_pka_myo    [uM]
    kcat_pp2a_myo = 10.1e-3       # kcat_pp2a_myo [1/ms]
    Km_pp2a_myo = 4.1          # Km_pp2a_myo   [uM]
    Yotiao_tot = 0.025         # Yotiao_tot    [uM]
    K_yotiao = 0.1e-3          # K_yotiao      [uM] ** apply G589D mutation here: 1e2 **
    PKAII_ikstot = 0.025       # PKAII_ikstot  [uM]
    PP1_ikstot = 0.025         # PP1_ikstot    [uM]
    k_pka_iks = 1.87e-3 #54      # k_pka_iks     [1/ms] # adjusted as in Xie et al 2013
    Km_pka_iks = 21            # Km_pka_iks    [uM]
    k_pp1_iks = 0.19e-3 #8.52    # k_pp1_iks     [1/ms] # adjusted as in Xie et al 2013
    Km_pp1_iks = 7             # Km_pp1_iks    [uM]
    PKAII_ikrtot = PKAII_ClCatot = PP1_ClCatot = PP1_ikrtot = 0.025       # PKAII_ikrtot  [uM]
    #PP1_ikrtot = 0.025         # PP1_ikrtot    [uM]
    k_pka_ikr = 1.87e-3 #54      # k_pka_ikr     [1/ms] # adjusted as in Xie et al 2013
    Km_pka_ikr = 21            # Km_pka_ikr    [uM]
    k_pp1_ikr = 0.19e-3 #8.52    # k_pp1_ikr     [1/ms] # adjusted as in Xie et al 2013
    Km_pp1_ikr = 7             # Km_pp1_ikr    [uM]
    #PKAII_ClCatot = 0.025      # PKAII_ClCatot [uM]
    #PP1_ClCatot = 0.025        # PP1_ClCatot   [uM]
    k_pka_ClCa = 54e-3            # k_pka_ClCa    [1/ms]
    Km_pka_ClCa = 8.5          # Km_pka_ClCa   [uM]
    k_pp1_ClCa = 8.52e-3          # k_pp1_ClCa    [1/ms]
    Km_pp1_ClCa = 7            # Km_pp1_ClCa   [uM]
    PKAII_itotot = 0.025       # PKAII_itotot  [uM]
    PP1_itotot = 0.025         # PP1_itotot    [uM]
    k_pka_ito = 1.87e-3 #54      # k_pka_ito     [1/ms] # adjusted as in Xie et al 2013
    Km_pka_ito = 21            # Km_pka_ito    [uM]
    k_pp1_ito = 0.19e-3 #8.52    # k_pp1_ito     [1/ms] # adjusted as in Xie et al 2013
    Km_pp1_ito = 7   
    PKAII_ik1tot = 0.025       # PKAII_ik1tot  [uM]
    PP1_ik1tot = 0.025         # PP1_ik1tot    [uM]
    k_pka_ik1 = 1.87e-3 #54      # k_pka_ik1     [1/ms] # adjusted as in Xie et al 2013
    Km_pka_ik1 = 21            # Km_pka_ik1    [uM]
    k_pp1_ik1 = 0.19e-3 #8.52    # k_pp1_ik1     [1/ms] # adjusted as in Xie et al 2013
    Km_pp1_ik1 = 7             # Km_pp1_ik1    [uM]
    PKAII_iNatot = 0.025       # PKAII_iNatot  [uM]
    PP1_iNatot = 0.025         # PP1_iNatot    [uM]
    k_pka_iNa = 1.87e-3 #54      # k_pka_iNa     [1/ms] # adjusted as in Xie et al 2013
    Km_pka_iNa = 21            # Km_pka_iNa    [uM]
    k_pp1_iNa = 0.19e-3 #8.52    # k_pp1_iNa     [1/ms] # adjusted as in Xie et al 2013
    Km_pp1_iNa = 7             # Km_pp1_iNa    [uM]
    
    #######################################
    
    # Additional params
    # Cell type
    #epi = 1 # EPI or ENDO?
    # Model for INa and ICa
    flagMina = 0 # 1 select Markov model for INa (0 HH for fast and late)
    flagMica = 1 # 1 select Markov model for ICa (0 for HH formulation)
    # myoFlag = 1: myofilament/contraction module active (matches C# default in MasterOde.cs;
    # original .m reference used 0). Required to integrate the myofilament states at y[53..58].
    myoFlag = 1
    # Set mechFlag to 0 for isometric or 1 for isotonic contraction
    mechFlag = 1
    # Set CKIIflag to 1 for CKII OE, 0 otherwise
    CKIIflag = CKIIOE
    # Max INa alterations with CKII hyperactivity as in Hund & Rudy 2008
    if CKIIflag == 1:    
        inashift = -3.25
        alphaCKII = -.18
        #deltGbarNal_CKII = 2  
    else:
        inashift = 0
        alphaCKII = 0
        #deltGbarNal_CKII = 0
    # Set IKs_flag to 0 for original IKs model, 1 for Bartos model
    #IKs_flag = True
    ## Model Parameters

    # Constants
    R = 8314       # [J/kmol*K]  
    Frdy = 96485   # [C/mol]  
    Temp = 310     # [K]
    FoRT = Frdy/R/Temp
    Cmem = 1.3810e-10   # [F] membrane capacitance
    Qpow = (Temp-310)/10

    # Cell geometry
    cellLength = 100     # cell length [um]
    cellRadius = 10.25   # cell radius [um]
    junctionLength = 160e-3  # junc length [um]
    junctionRadius = 15e-3   # junc radius [um]
    distSLcyto = 0.45    # dist. SL to cytosol [um]
    distJuncSL = 0.5  # dist. junc to SL [um]
    DcaJuncSL = 1.64e-6  # Dca junc to SL [cm**2/sec]
    DcaSLcyto = 1.22e-6 # Dca SL to cyto [cm**2/sec]
    DnaJuncSL = 1.09e-5  # Dna junc to SL [cm**2/sec]
    DnaSLcyto = 1.79e-5  # Dna SL to cyto [cm**2/sec] 
    Vcell = math.pi*cellRadius**2*cellLength*1e-15    # [L]
    Vmyo = 0.65*Vcell 
    Vsr = 0.035*Vcell 
    Vsl = 0.02*Vcell 
    Vjunc = 0.0539*.01*Vcell #same as Vdyad
    SAjunc = 20150*math.pi*2*junctionLength*junctionRadius  # [um**2]
    SAsl = math.pi*2*cellRadius*cellLength          # [um**2]
    J_ca_juncsl =1/1.2134e12 # [L/msec] = 8.2413e-13
    J_ca_slmyo = 1/2.68510e11 # [L/msec] = 3.2743e-12
    J_na_juncsl = 1/(1.6382e12/3*100) # [L/msec] = 6.1043e-13
    J_na_slmyo = 1/(1.8308e10/3*100)  # [L/msec] = 5.4621e-11

    # Fractional currents in compartments
    Fjunc = 0.11    
    Fsl = 1-Fjunc
    Fjunc_CaL = 0.9 
    Fsl_CaL = 1-Fjunc_CaL

    # Fixed ion concentrations     
    Cli = 15   # Intracellular Cl  [mM]
    Clo = 150  # Extracellular Cl  [mM]
    
    ### NOTE THAT Ko is passed into conductances, make sure ot change that if you ever want to change Ko
    Ko = 5.4   # Extracellular K   [mM]
    Nao = 140  # Extracellular Na  [mM]
    Nao3 = Nao**3
    #Nao = 300  # Extracellular Na  [mM]
    #Nao = 70  # Extracellular Na  [mM]
    Cao = 1.8  # Extracellular Ca  [mM]
    Mgi = 1    # Intracellular Mg  [mM]
    # OA inhibition term (non-competitive) for PP1 and PP2A
    OA_PP1 = 1/(1 + (OA/Ki_OA_PP1)**3)
    OA_PP2A = 1/(1 + (OA/Ki_OA_PP2A)**3)
    
    # Nernst Potentials
    FoRT_reciprocal = (1/FoRT)
    ## Na transport parameters

    GNa = GNa_fast#0.9* 1*16        # [mS/uF]  ###
    GNaL = GNa_late#1*2*0.0065*2.5        # [mS/uF]
    GNaB = GNab#0.5*1*0.597e-3    # [mS/uF] 0.897e-3 ###

    IbarNaK = GNaK #1*1.8#1.90719     # [uA/uF]
    KmNaip = 11         # [mM]
    KmKo = 1.5         # [mM]
    Q10NaK = 1.63  
    Q10KmNai = 1.39
    
    # modified for human myocytes
    GtoFast = Gto_fast
    GtoSlow = Gto_slow
    # if cell_type == "EPI":
    #     GtoFast = 1*0.13*0.88 # epi 88# 0.1144 [mS/uF]
    #     GtoSlow = 1*0.13*0.12 # epi 12# 0.0156 [mS/uF]
    # else:
    #     GtoFast = 1*0.13*0.3*0.036 # endo 4# 0.0014 [mS/uF]
    #     GtoSlow = 1*0.13*0.3*0.964 # endo 96# 0.0376 [mS/uF]
    pNaK = 0.01833 
    gks_factor_SA = GKs#2.5*1  ###
    gks_factor = 0.05/5
    #kPKA_Iks,gks_factor,P_g_0,P_g_max,P_vh_0,P_vh_max,P_tau_0,P_tau_max
    
    gkr = GKr #1*(3+0.5)*0.035*np.sqrt(Ko/5.4) # added 3*gkr from Pei-Chi
    gkp = GKb #1*2*0.001
    gki = GK1 #1*0.35*np.sqrt(Ko/5.4)
    ## Cl current parameters

    GClCa = GClCa_input#2*1*(0.5*0.109625)   # [mS/uF]  ###
    GClB = GClb #0.5*1*9e-3        # [mS/uF]  ###
    KdClCa = 100e-3    # [mM]
    ## Ca transport parameters

    #K_Ica = 1*(0.47) # added 0.47*GCaL from Pei-Chi ###
    pCa = pCa_input #K_Ica*5.4e-4       # [cm/sec] # Markov model
    pNa = pNa_input #K_Ica*1.5e-8       # [cm/sec]
    pK = pK_input #K_Ica*2.7e-7        # [cm/sec]
    Q10CaL = 1.8       

    GCaB = GCab#0.5*1*5.513e-4    # [uA/uF] ###

    IbarNCX = GNCX#1*4.5      # [uA/uF]5.5 before - 9 in rabbit
    KmCai = 3.59e-3    # [mM]
    KmCao = 1.3        # [mM]
    KmNai = 12.29      # [mM]
    KmNao = 87.5       # [mM]
    ksat = 0.32        # [none]  
    nu = 0.27          # [none]
    Kdact = 0.150e-3   # [mM] 
    Q10NCX = 1.57      # [none]

    IbarSLCaP = GpCa #1*0.0673 # IbarSLCaP FEI changed [uA/uF](2.2 umol/L cytosol/sec) jeff 0.093 [uA/uF]
    KmPCa = 0.5e-3     # [mM] 
    Q10SLCaP = 2.35    # [none]

    # SR flux parameters
    Q10SRCaP = 2.6          # [none]
    Vmax_SRCaP = 1*5.3114e-3  # [mM/msec] (286 umol/L cytosol/sec)
    Kmf = 0.246e-3          # [mM] default
    #Kmf = 0.175e-3          # [mM]
    Kmr = 1.7               # [mM]L cytosol
    hillSRCaP = 1.787       # [mM]

    ks = 1*25                 # [1/ms]      
    koCa = 10               # [mM**-2 1/ms]   #default 10   modified 20
    kom = 0.06              # [1/ms]     
    kiCa = 0.5              # [1/mM/ms]
    kim = 0.005             # [1/ms]
    ec50SR = 0.45           # [mM]

    kleak = 1*5.348e-6
    ## Buffering parameters

    # koff: [1/s] = 1e-3*[1/ms]  kon: [1/uM/s] = [1/mM/ms]
    Bmax_Naj = 7.561       # [mM] # Bmax_Naj = 3.7 (c-code difference?)  # Na buffering
    Bmax_Nasl = 1.65       # [mM]
    koff_na = 1e-3         # [1/ms]
    kon_na = 0.1e-3        # [1/mM/ms]
    Bmax_TnClow = 70e-3    # [mM]                      # TnC low affinity
    koff_tncl = 19.6e-3    # [1/ms] 
    kon_tncl = 32.7        # [1/mM/ms]
    Bmax_TnChigh = 140e-3  # [mM]                      # TnC high affinity 
    koff_tnchca = 0.032e-3 # [1/ms] 
    kon_tnchca = 2.37      # [1/mM/ms]
    koff_tnchmg = 3.33e-3  # [1/ms] 
    kon_tnchmg = 3e-3      # [1/mM/ms]
    Bmax_CaM = 24e-3       # [mM] **? about setting to 0 in c-code**   # CaM buffering
    koff_cam = 238e-3      # [1/ms] 
    kon_cam = 34           # [1/mM/ms]
    Bmax_myosin = 140e-3   # [mM]                      # Myosin buffering
    koff_myoca = 0.46e-3   # [1/ms]
    kon_myoca = 13.8       # [1/mM/ms]
    koff_myomg = 0.057e-3  # [1/ms]
    kon_myomg = 0.0157     # [1/mM/ms]
    Bmax_SR = 19*.9e-3     # [mM] (Bers text says 47e-3) 19e-3
    koff_sr = 60e-3        # [1/ms]
    kon_sr = 100           # [1/mM/ms]
    Bmax_SLlowsl = 37.4e-3*Vmyo/Vsl        # [mM] # SL buffering
    Bmax_SLlowj = 4.6e-3*Vmyo/Vjunc*0.1    # [mM] #Fei *0.1!!! junction reduction factor
    koff_sll = 1300e-3     # [1/ms]
    kon_sll = 100          # [1/mM/ms]
    Bmax_SLhighsl = 13.4e-3*Vmyo/Vsl       # [mM] 
    Bmax_SLhighj = 1.65e-3*Vmyo/Vjunc*0.1  # [mM] #Fei *0.1!!! junction reduction factor
    koff_slh = 30e-3       # [1/ms]
    kon_slh = 100          # [1/mM/ms]
    Bmax_Csqn = 140e-3*Vmyo/Vsr            # [mM] # Bmax_Csqn = 2.6      # Csqn buffering
    koff_csqn = 65         # [1/ms] 
    kon_csqn = 100         # [1/mM/ms] 
    ## Myofilament parameters

    nc=3                 #Addim.(=1,2,3)
    Ap=1008e+04          #[mN/mm2/um/mM]
    Aw=Ap/5              #[mN/mm2/um/mM]
    alfa=0.5             #[mN/mm2]
    bet=80               #[1/um]
    Bp=0.5               #[1/ms]
    Bw=0.35              #[1/ms]
    f=0.0023             #[1/ms]
    gama=28000           #[1/um2]
    hpr=0.006            #[um]
    hwr=0.0001           #[um]
    Ke=105000            #[mN/mm2/um**5] 
    Lz=0.97              #[um] (0.97)
    La=1.15              #[um]
    Lc=1.05              #[um]
    Le=10                #[mN/mm2/um][9]
    RLa=20               #[1/um2]
    TSt=0.07/nc          #[mM]
    Yb=181.6e+06
    Yc=4                 #[1/um]
    Yd=0.028             #[1/ms]
    Yp=0.1397            #[1/ms]  
    Yr=0.1397            #[1/ms]
    Yv=0.9               #[1/ms]
    Za=0.0023            #[1/ms]
    Zb=0.1397            #[1/ms]
    Zp=0.2095            #[1/ms]
    Zr=7262.6e+06        #[1/mM**nc/ms]
    Fh=0.1               #Addim. 
    ## Derived parameteres for PKA-dependent phosphoregulation

    fracLCCapo = 0.04267 # Derived quantity - (LCCap(baseline)/LCCatot)

    fracLCCbpo = 0.05022 # Derived quantity - (LCCbp(baseline)/LCCbtot)

    fracRyRpo = 0.03764 # Derived quantity - (RyRp(baseline)/RyRtot)

    fracPKA_PLBo = 1-0.0171   # Derived quantity - ((PLBtot - PLBp(baseline))/PLBtot)

    fracPKA_PLMo = 0.01476 # Derived quantity (PLM_PKAp(baseline)/PLMtot)
        #fracPKA_PLMiso = 0.6369 # Derived quantity 0.1 ISO - steady-state
    fracPKA_PLMiso = 0.8204 # Derived quantity 0.1 ISO - MAX

    fracTnIpo = 0.007365  # Derived quantity (TnI_PKAp(baseline)/TnItot)

    fracPKA_Myoo = 0.007365 # Derived quantity (Myo_PKAp(baseline)/Myotot)
    #fracPKA_Myoiso = 0.6433 # Derived quantity 0.1 ISO - steady-state
    fracPKA_Myoiso = 0.8325 # Derived quantity 0.1 ISO - MAX

    fracPKA_Ikso = 0.1613 # Derived quantity (IKs_PKAp(baseline)/Ikrtot)
        #fracPKA_Iksiso = 0.6487 # Derived quantity 0.1 ISO - steady-state
    fracPKA_Iksiso = 0.6933 # Derived quantity 0.1 ISO - MAX

    fracPKA_Ikro = 0.1613 # Derived quantity (IKr_PKAp(baseline)/Ikrtot)
    #fracPKA_Ikriso = 0.6486 # Derived quantity 0.1 ISO - steady-state
    fracPKA_Ikriso = 0.6951 # Derived quantity 0.1 ISO - MAX

    fracPKA_IClCao = 0.2324 # Derived quantity (IClCa_PKAp(baseline)/Ikrtot)
    #fracPKA_IClCaiso = 0.7457 # Derived quantity 0.1 ISO - steady-state
    fracPKA_IClCaiso = 0.7999 # Derived quantity 0.1 ISO - MAX   

    fracPKA_Itoo = 0.1613 # Derived quantity (Ito_PKAp(baseline)/Itotot)
    #fracPKA_Itoiso = 0.6486 # Derived quantity 0.1 ISO - steady-state
    fracPKA_Itoiso = 0.6951 # Derived quantity 0.1 ISO - MAX

    fracPKA_Ik1o = 0.1613 # Derived quantity (IK1_PKAp(baseline)/Ik1tot)
    #fracPKA_Ik1iso = 0.6486 # Derived quantity 0.1 ISO - steady-state
    fracPKA_Ik1iso = 0.6951 # Derived quantity 0.1 ISO - MAX 

    fracPKA_Inao = 0.1613 # Derived quantity (Ina_PKAp(baseline)/Inatot)
    #fracPKA_Inaiso = 0.6486 # Derived quantity 0.1 ISO - steady-state
    fracPKA_Inaiso = 0.6951 # Derived quantity 0.1 ISO - MAX
    
    r1 = 0.3                               # [1/ms] - Opening rate
    r2 = 3                                 # [1/ms] - closing rate
    # LCC Current Fixed Parameters
    taupo = 1          # [ms] - Time constant of activation
    TBa = 450          # [ms] - Time constant
    s1o = 0.0221
    k1o = 0.03
    kop = 2.5e-3       # [mM]
    cpbar = 8e-3       # [mM]
    tca = 78.0312
    ICa_scale = 5.25
    recoveryReduc = 3
    k1p = .00413                           # [ms] - Inactivation rate
    k2 = 1e-4                              # [ms] - Inactivation rate
    k2p = 0.00224                           # [ms] - Inactivation rate
    s1p = 0.00195                           # [ms] - Inactivation rate
    ### Re-define all parameters as mode 2 specific parameters ###
    s1om2 = .0221
    k1om2 = .03
    kopm2 = 2.5e-3
    cpbarm2 = 8e-3
    tcam2 = 78.0312
    r1m2 = 0.3                               # [1/ms] - Opening rate
    r2m2 = 3/10                              # [1/ms] - closing rate
    s1pm2 = .00195                           # [ms] - Inactivation rate
    k1pm2 = .00413                           # [ms] - Inactivation rate
    k2m2 = 1e-4                              # [ms] - Inactivation rate
    k2pm2 = .00224                           # [ms] - Inactivation rate
    Icftr = 0 #gCFTR*(Vm - ecl)
    
    ############################################################
    
    ## Distribute parameters by module

    # CaM module
    CaDyad = y[35]*1e3 # from ECC model, *** Converting from [mM] to [uM] ***
    #compart_dyad = 2
    # ** NOTE: Btotdyad being sent to the dyad camODEfile is set to zero, but is used below for transfer between SL and dyad
    #BtotDyad can also be set to 0
    #pCaMDyad = [K, Mg, CaMtotDyad, BtotDyad, CaMKIItotDyad, CaNtotDyad, PP1totDyad, CaDyad, cycleLength]
    CaSL = y[36]*1e3 # from ECC model, *** Converting from [mM] to [uM] ***
    #compartSL = 1
    #pCaMSL = [K, Mg, CaMtotSL, BtotSL, CaMKIItotSL, CaNtotSL, PP1totSL, CaSL, cycleLength]
    CaCyt = y[37]*1e3 # from ECC model, *** Converting from [mM] to [uM] ***
    #compartCyt = 0
    #pCaMCyt = [K, Mg, CaMtotCyt, BtotCyt, CaMKIItotCyt, CaNtotCyt, PP1totCyt, CaCyt, cycleLength]

    # Phosphorylation feedback (CaMKII fractions, PKA fractions, PP1, favail, kPKA, myofilament scaling) is now computed INSIDE Morotti_model from live state every ODE step (fix for frozen-feedback bug H1).

    # mechFlag defined for isometric (0) or isotonic [0] contraction
    uMyo = 1
    uXBCa = 1
    uXBcy = 1
    Liso = 1.05
    sigma = (np.exp(Nao/67.3)-1)/7

    # ECC module
    ECC_constants = [
        
        ## flags
        flag_cam,flag_CaMKII,flag_BARS,
        
        ## ECC Constants
        cycleLength,stimDur,amp,CKIIOE,Ca_clamp,Na_clamp,
        # (H1 fix) CaMKII fractions (LCC_CKdyadp, RyR_CKp, PLB_CKp), PKA fractions, CaMKIIact_*, PP1_PLB_avail,
        # kPKA_Myo/Iks, and P_g_0..P_tau_max are now computed INSIDE Morotti_model from live state.

        # Camkii constants
        LCCtotDyad,RyRtot,PP1_dyad,PP2A_dyad,OA,PLBtot,LCCtotSL,PP1_SL,
        ## RATE CONSTANTS and KM VALUES for Camkii
        k_ckLCC, k_pp1LCC, k_pkaLCC, k_pp2aLCC, KmCK_LCC, KmPKA_LCC, KmPP2A_LCC, KmPP1_LCC, k_ckRyR, k_pkaRyR, 
        k_pp1RyR, k_pp2aRyR, kb_2809, kb_2815, KmCK_RyR, KmPKA_RyR, KmPP1_RyR, KmPP2A_RyR, k_ckPLB, k_pp1PLB, KmCK_PLB, KmPP1_PLB,
        Ki_OA_PP1,Ki_OA_PP2A,PKAc,
        
        #camDyad constants
        K, Mg, CaMtotDyad, 0, CaMKIItotDyad, CaNtotDyad, PP1totDyad, CaDyad, 
        
        #camSL constants
        CaMtotSL, BtotSL, CaMKIItotSL, CaNtotSL, PP1totSL, CaSL, 
        
        #camcyt constants
        CaMtotCyt, BtotCyt, CaMKIItotCyt, CaNtotCyt, PP1totCyt, CaCyt, 
        
        #BARS parameters
        FSK, IBMX, b1ARtot, kf_LR, kr_LR, kf_LRG, kr_LRG, kf_RG, kr_RG, Gstot, k_G_act, k_G_hyd, k_G_reassoc,
        kf_bARK, kr_bARK, kf_PKA, kr_PKA, ACtot, ATP, k_AC_basal, Km_AC_basal, Kd_AC_Gsa, kf_AC_Gsa, kr_AC_Gsa,
        k_AC_Gsa, Km_AC_Gsa, Kd_AC_FSK, k_AC_FSK, Km_AC_FSK, PDE3tot, PDE4tot, k_cAMP_PDE3, k_cAMP_PDE3p, Km_PDE3_cAMP,
        k_cAMP_PDE4, k_cAMP_PDE4p, Km_PDE4_cAMP, Kd_PDE_IBMX, k_PKA_PDE, k_PP_PDE, PKItot, kf_RC_cAMP, kf_RCcAMP_cAMP, 
        kf_RcAMPcAMP_C, kf_PKA_PKI, kr_RC_cAMP, kr_RCcAMP_cAMP, kr_RcAMPcAMP_C, kr_PKA_PKI, epsilon, PKAIItot, I1tot, 
        k_PKA_I1, Km_PKA_I1, Vmax_PP2A_I1, Km_PP2A_I1, Ki_PP1_I1, kf_PP1_I1, kr_PP1_I1, k_PKA_PLB, Km_PKA_PLB, k_PP1_PLB, Km_PP1_PLB,
        k_PKA_PLM, Km_PKA_PLM, k_PP1_PLM, Km_PP1_PLM, PKACII_LCCtot, PP1_LCC, PP2A_LCC, k_PKA_LCC, Km_PKA_LCC, k_PP1_LCC, Km_PP1_LCC, 
        k_PP2A_LCC, Km_PP2A_LCC, PKAIIryrtot, PP1ryr, PP2Aryr, kcat_pka_ryr, Km_pka_ryr, kcat_pp1_ryr, Km_pp1_ryr, kcat_pp2a_ryr, Km_pp2a_ryr,
        PP2A_TnI, k_PKA_TnI, Km_PKA_TnI, k_PP2A_TnI, Km_PP2A_TnI, PP2A_myo, kcat_pka_myo, Km_pka_myo, kcat_pp2a_myo, Km_pp2a_myo,
        Yotiao_tot, K_yotiao, PKAII_ikstot, PP1_ikstot, k_pka_iks, Km_pka_iks, k_pp1_iks, Km_pp1_iks, PKAII_ikrtot, PP1_ikrtot, 
        k_pka_ikr, Km_pka_ikr, k_pp1_ikr, Km_pp1_ikr, PKAII_ClCatot, PP1_ClCatot, k_pka_ClCa, Km_pka_ClCa, k_pp1_ClCa, Km_pp1_ClCa,
        PKAII_itotot, PP1_itotot, k_pka_ito, Km_pka_ito, k_pp1_ito, Km_pp1_ito, PKAII_ik1tot, PP1_ik1tot, k_pka_ik1, Km_pka_ik1, 
        k_pp1_ik1, Km_pp1_ik1, PKAII_iNatot, PP1_iNatot, k_pka_iNa, Km_pka_iNa, k_pp1_iNa, Km_pp1_iNa,
        Ligtot,PP1_PLBtot,PLMtotBA,LCCtotBA,RyRtotBA,TnItotBA,MyototBA,IKstotBA,IKrtotBA,IClCatotBA,ItototBA,IK1totBA,INatotBA,
        
        #Additional params for full ecc model
        flagMina, flagMica, myoFlag, mechFlag, CKIIflag, 
        R, Frdy, Temp, FoRT, Cmem, Qpow, cellLength, cellRadius, junctionLength, junctionRadius, 
        Vmyo, Vsr, Vsl, Vjunc, 
        SAjunc, SAsl, J_ca_juncsl, J_ca_slmyo, J_na_juncsl, J_na_slmyo, Fjunc, Fsl, Fjunc_CaL, Fsl_CaL, 
        Cli, Clo, Ko, Nao, Cao, Mgi, OA_PP1, OA_PP2A, FoRT_reciprocal, 
        GNa, GNaL, GNaB, IbarNaK, KmNaip, KmKo, Q10NaK, Q10KmNai, 
        GtoFast, GtoSlow, pNaK, gks_factor_SA, gkr, gkp, gki, GClCa, GClB, KdClCa, 
        pCa, pNa, pK, Q10CaL, GCaB, IbarNCX, KmCai, KmCao, KmNai, KmNao, ksat, nu, Kdact, Q10NCX, 
        IbarSLCaP, KmPCa, Q10SLCaP, Q10SRCaP, Vmax_SRCaP, Kmf, Kmr, hillSRCaP, ks, koCa, kom, kiCa, kim, ec50SR, kleak, 
        Bmax_Naj, Bmax_Nasl, koff_na, kon_na, Bmax_TnClow, koff_tncl, kon_tncl, Bmax_TnChigh, koff_tnchca, kon_tnchca, 
        koff_tnchmg, kon_tnchmg, Bmax_CaM, koff_cam, kon_cam, Bmax_myosin, koff_myoca, kon_myoca, koff_myomg, kon_myomg, 
        Bmax_SR, koff_sr, kon_sr, Bmax_SLlowsl, Bmax_SLlowj, koff_sll, kon_sll, Bmax_SLhighsl, Bmax_SLhighj, koff_slh, kon_slh, 
        Bmax_Csqn, koff_csqn, kon_csqn, nc, Ap, Aw, alfa, bet, Bp, Bw, f, gama, hpr, hwr, Ke, Lz, La, Lc, Le, RLa, 
        TSt, Yb, Yc, Yd, Yp, Yr, Yv, Za, Zb, Zp, Zr, Fh, 
        fracLCCapo, fracLCCbpo, fracRyRpo, fracPKA_PLBo, fracPKA_PLMo, fracPKA_PLMiso, fracTnIpo, fracPKA_Myoo, fracPKA_Myoiso, 
        fracPKA_Ikso, fracPKA_Iksiso, fracPKA_Ikro, fracPKA_Ikriso, fracPKA_IClCao, fracPKA_IClCaiso, fracPKA_Itoo, fracPKA_Itoiso, 
        fracPKA_Ik1o, fracPKA_Ik1iso, fracPKA_Inao, fracPKA_Inaiso, 
        r1, r2, taupo, TBa, s1o, k1o, kop, cpbar, tca, ICa_scale, recoveryReduc, k1p, k2, k2p, s1p, 
        s1om2, k1om2, kopm2, cpbarm2, tcam2, r1m2, r2m2, s1pm2, k1pm2, k2m2, k2pm2, Icftr,Nao3,inashift, alphaCKII,
        uMyo, uXBCa, uXBcy, Liso, gks_factor,
        sigma

        ]
    
    tspan = (0, cycles*cycleLength)
    # State vector layout follows the canonical C# (MasterOde.cs) layout:
    # [0:83] ECC, [83:98] camDyad, [98:113] camSL, [113:128] camCyt, [128:134] CaMKII, [134:173] BAR
    initial_conds = np.concatenate((y_ecc, y_camDyad, y_camSL, y_camCyt, y_CaMKII, y_BAR))
    sol = solve_ivp(fun = Morotti_model, t_span = tspan, y0 = initial_conds, args = ECC_constants,method='LSODA',
                    rtol= 1e-6,atol = 1e-6,max_step = 1) #, t_eval=np.linspace(tspan[0], tspan[1], 300000)

    # dydt = [dydt_ecc, dydt_camDyad, dydt_camSL, dydt_camCyt, dydt_CaMKIIDyad, dydt_BAR] # make sure dydt_BAR integrates well in python
    time = sol.t  # Time points
    solutions = sol.y  # Solution vectors, each row corresponds to a variable
    # Column names follow canonical layout: ECC, camDyad, camSL, camCyt, CaMKII, BAR
    y_names = ["m","h","j","xks_sl","f","fcaBj","fcaBsl","xtos","ytos","xtof","ytof","xkr","xks_j","RyRr","RyRo","RyRi","NaBj","NaBsl","TnCL","TnCHc","TnCHm","CaM", "Myoc", "Myom","SRB","SLLj","SLLsl","SLHj","SLHsl",
               "Csqnb","Ca_sr","Naj","Nasl","Nai","Ki","Caj","Casl","Cai","V","rtos","h_L","PH1","PH2","PH3","PH4","PH5","PH6","PH7","PH8","PH9","PH10","PH11","PH12","TSCa","TSCa_star","TSCa_tilde","TS_star","L_p",
               "L_w","Pc2_LCCj_m1","Pc1_LCCj_m1","Pi1Ca_LCCj_m1","Pi2Ca_LCCj_m1","Pi1Ba_LCCj_m1","Pi2Ba_LCCj_m1","Pc2_LCCj_m2","Pc1_LCCj_m2","Pi1Ca_LCCj_m2","Pi2Ca_LCCj_m2","Pi1Ba_LCCj_m2",
               "Pi2Ba_LCCj_m2","Pc2_LCCsl_m1","Pc1_LCCsl_m1","Pi1Ca_LCCsl_m1","Pi2Ca_LCCsl_m1","Pi1Ba_LCCsl_m1","Pi2Ba_LCCsl_m1","Pc2_LCCsl_m2","Pc1_LCCsl_m2","Pi1Ca_LCCsl_m2","Pi2Ca_LCCsl_m2",
               "Pi1Ba_LCCsl_m2","Pi2Ba_LCCsl_m2",
               "CaM_dyad","Ca2CaM_dyad","Ca4CaM_dyad","CaMB_dyad","Ca2CaMB_dyad","Ca4CaMB_dyad","Pb2_dyad","Pb_dyad","Pt_dyad","Pt2_dyad","Pa_dyad","Ca4CaN_dyad","CaMCa4CaN_dyad","Ca2CaMCa4CaN_dyad","Ca4CaMCa4CaN_dyad",
               "CaM_sl","Ca2CaM_sl","Ca4CaM_sl","CaMB_sl","Ca2CaMB_sl","Ca4CaMB_sl","Pb2_sl","Pb_sl","Pt_sl","Pt2_sl","Pa_sl","Ca4CaN_sl","CaMCa4CaN_sl","Ca2CaMCa4CaN_sl","Ca4CaMCa4CaN_sl",
               "CaM_cyt","Ca2CaM_cyt","Ca4CaM_cyt","CaMB_cyt","Ca2CaMB_cyt","Ca4CaMB_cyt","Pb2_cyt","Pb_cyt","Pt_cyt","Pt2_cyt","Pa_cyt","Ca4CaN_cyt","CaMCa4CaN_cyt","Ca2CaMCa4CaN_cyt","Ca4CaMCa4CaN_cyt",
               "LCC_PKAp","LCC_CKdyadp","RyR2809p","RyR2815p","PLBT17p","LCC_CKslp",
               "LR","LRG","RG","b1AR_S464","b1AR_S301","GsaGTPtot","GsaGDP","Gsby","AC_GsaGTP","PDE3p","PDE4p","cAMPtot","RC_I","RCcAMP_I","RCcAMPcAMP_I","RcAMPcAMP_I","PKACI","PKACI_PKI","RC_II","RCcAMP_II",
               "RCcAMPcAMP_II","RcAMPcAMP_II","PKACII","PKACII_PKI","I1p_PP1","I1ptot","PLBp","PLMp","LCCap","LCCbp","RyRp","TnIp","Myop","KSp","KRp","ClCap","Top","K1p","Nap"
]
    
    df = pd.DataFrame(solutions.T, columns=y_names)
    
    df['time'] = time
    stim_duration = stimDur
    return df, pd.DataFrame(), stim_duration


# ecc ODE file
@njit
def Morotti_model(t,initial_conds,
                  
                ## flags
                flag_cam,flag_CaMKII,flag_BARS,
                
                ## ECC Constants
                cycleLength,stimDur,amp,CKIIOE,Ca_clamp,Na_clamp,
                # (H1 fix) CaMKII fractions, PKA fractions, CaMKIIact_*, PP1_PLB_avail,
                # kPKA_Myo/Iks, P_g_0..P_tau_max are computed live below from state.

                # Camkii constants
                LCCtotDyad,RyRtot,PP1_dyad,PP2A_dyad,OA,PLBtot,LCCtotSL,PP1_SL,
                ## RATE CONSTANTS and KM VALUES for Camkii
                k_ckLCC, k_pp1LCC, k_pkaLCC, k_pp2aLCC, KmCK_LCC, KmPKA_LCC, KmPP2A_LCC, KmPP1_LCC, k_ckRyR, k_pkaRyR, 
                k_pp1RyR, k_pp2aRyR, kb_2809, kb_2815, KmCK_RyR, KmPKA_RyR, KmPP1_RyR, KmPP2A_RyR, k_ckPLB, k_pp1PLB, KmCK_PLB, KmPP1_PLB,
                Ki_OA_PP1,Ki_OA_PP2A,PKAc,
                
                #camDyad constants
                K, Mg, CaMtotDyad, BtotDyad, CaMKIItotDyad, CaNtotDyad, PP1totDyad, CaDyad, 
                
                #camSL constants
                CaMtotSL, BtotSL, CaMKIItotSL, CaNtotSL, PP1totSL, CaSL, 
                
                #camcyt constants
                CaMtotCyt, BtotCyt, CaMKIItotCyt, CaNtotCyt, PP1totCyt, CaCyt,
                
                #BARS constants
                FSK, IBMX, b1ARtot, kf_LR, kr_LR, kf_LRG, kr_LRG, kf_RG, kr_RG, Gstot, k_G_act, k_G_hyd, k_G_reassoc,
                kf_bARK, kr_bARK, kf_PKA, kr_PKA, ACtot, ATP, k_AC_basal, Km_AC_basal, Kd_AC_Gsa, kf_AC_Gsa, kr_AC_Gsa,
                k_AC_Gsa, Km_AC_Gsa, Kd_AC_FSK, k_AC_FSK, Km_AC_FSK, PDE3tot, PDE4tot, k_cAMP_PDE3, k_cAMP_PDE3p, Km_PDE3_cAMP,
                k_cAMP_PDE4, k_cAMP_PDE4p, Km_PDE4_cAMP, Kd_PDE_IBMX, k_PKA_PDE, k_PP_PDE, PKItot, kf_RC_cAMP, kf_RCcAMP_cAMP, 
                kf_RcAMPcAMP_C, kf_PKA_PKI, kr_RC_cAMP, kr_RCcAMP_cAMP, kr_RcAMPcAMP_C, kr_PKA_PKI, epsilon, PKAIItot, I1tot, 
                k_PKA_I1, Km_PKA_I1, Vmax_PP2A_I1, Km_PP2A_I1, Ki_PP1_I1, kf_PP1_I1, kr_PP1_I1, k_PKA_PLB, Km_PKA_PLB, k_PP1_PLB, Km_PP1_PLB,
                k_PKA_PLM, Km_PKA_PLM, k_PP1_PLM, Km_PP1_PLM, PKACII_LCCtot, PP1_LCC, PP2A_LCC, k_PKA_LCC, Km_PKA_LCC, k_PP1_LCC, Km_PP1_LCC, 
                k_PP2A_LCC, Km_PP2A_LCC, PKAIIryrtot, PP1ryr, PP2Aryr, kcat_pka_ryr, Km_pka_ryr, kcat_pp1_ryr, Km_pp1_ryr, kcat_pp2a_ryr, Km_pp2a_ryr,
                PP2A_TnI, k_PKA_TnI, Km_PKA_TnI, k_PP2A_TnI, Km_PP2A_TnI, PP2A_myo, kcat_pka_myo, Km_pka_myo, kcat_pp2a_myo, Km_pp2a_myo,
                Yotiao_tot, K_yotiao, PKAII_ikstot, PP1_ikstot, k_pka_iks, Km_pka_iks, k_pp1_iks, Km_pp1_iks, PKAII_ikrtot, PP1_ikrtot, 
                k_pka_ikr, Km_pka_ikr, k_pp1_ikr, Km_pp1_ikr, PKAII_ClCatot, PP1_ClCatot, k_pka_ClCa, Km_pka_ClCa, k_pp1_ClCa, Km_pp1_ClCa,
                PKAII_itotot, PP1_itotot, k_pka_ito, Km_pka_ito, k_pp1_ito, Km_pp1_ito, PKAII_ik1tot, PP1_ik1tot, k_pka_ik1, Km_pka_ik1, 
                k_pp1_ik1, Km_pp1_ik1, PKAII_iNatot, PP1_iNatot, k_pka_iNa, Km_pka_iNa, k_pp1_iNa, Km_pp1_iNa,
                ISO,PP1tot,PLMtot,LCCtot,RyRtotBA,TnItot,Myotot,Kstot,Krtot,ClCatot,Totot,K1tot,Natot,
                
                #Additional ecc params
                flagMina, flagMica, myoFlag, mechFlag, CKIIflag,  
                R, Frdy, Temp, FoRT, Cmem, Qpow, cellLength, cellRadius, junctionLength, junctionRadius, 
                Vmyo, Vsr, Vsl, Vjunc, 
                SAjunc, SAsl, J_ca_juncsl, J_ca_slmyo, J_na_juncsl, J_na_slmyo, Fjunc, Fsl, Fjunc_CaL, Fsl_CaL, 
                Cli, Clo, Ko, Nao, Cao, Mgi, OA_PP1, OA_PP2A, FoRT_reciprocal, 
                GNa, GNaL, GNaB, IbarNaK, KmNaip, KmKo, Q10NaK, Q10KmNai, 
                GtoFast, GtoSlow, pNaK, gks_factor_SA, gkr, gkp, gki, GClCa, GClB, KdClCa, 
                pCa, pNa, pK, Q10CaL, GCaB, IbarNCX, KmCai, KmCao, KmNai, KmNao, ksat, nu, Kdact, Q10NCX, 
                IbarSLCaP, KmPCa, Q10SLCaP, Q10SRCaP, Vmax_SRCaP, Kmf, Kmr, hillSRCaP, ks, koCa, kom, kiCa, kim, ec50SR, kleak, 
                Bmax_Naj, Bmax_Nasl, koff_na, kon_na, Bmax_TnClow, koff_tncl, kon_tncl, Bmax_TnChigh, koff_tnchca, kon_tnchca, 
                koff_tnchmg, kon_tnchmg, Bmax_CaM, koff_cam, kon_cam, Bmax_myosin, koff_myoca, kon_myoca, koff_myomg, kon_myomg, 
                Bmax_SR, koff_sr, kon_sr, Bmax_SLlowsl, Bmax_SLlowj, koff_sll, kon_sll, Bmax_SLhighsl, Bmax_SLhighj, koff_slh, kon_slh, 
                Bmax_Csqn, koff_csqn, kon_csqn, nc, Ap, Aw, alfa, bet, Bp, Bw, f, gama, hpr, hwr, Ke, Lz, La, Lc, Le, RLa, 
                TSt, Yb, Yc, Yd, Yp, Yr, Yv, Za, Zb, Zp, Zr, Fh, 
                fracLCCapo, fracLCCbpo, fracRyRpo, fracPKA_PLBo, fracPKA_PLMo, fracPKA_PLMiso, fracTnIpo, fracPKA_Myoo, fracPKA_Myoiso, 
                fracPKA_Ikso, fracPKA_Iksiso, fracPKA_Ikro, fracPKA_Ikriso, fracPKA_IClCao, fracPKA_IClCaiso, fracPKA_Itoo, fracPKA_Itoiso, 
                fracPKA_Ik1o, fracPKA_Ik1iso, fracPKA_Inao, fracPKA_Inaiso, 
                r1, r2, taupo, TBa, s1o, k1o, kop, cpbar, tca, ICa_scale, recoveryReduc, k1p, k2, k2p, s1p, 
                s1om2, k1om2, kopm2, cpbarm2, tcam2, r1m2, r2m2, s1pm2, k1pm2, k2m2, k2pm2, Icftr,Nao3,inashift, alphaCKII,
                uMyo, uXBCa, uXBcy, Liso, gks_factor,
                sigma
                ):
    # This function describes the ODE's for ECC - human ventricular myocyte

    ## State variables
    # 1       2       3       4       5       6       7       8      9       10  
    # m       h       j       xks_sl  f       fcaBj   fcaBsl  xtos   ytos    xtof  
    # 11      12      13      14      15      16      17      18     19      20   
    # ytof    xkr     xks_j   RyRr    RyRo    RyRi    NaBj    NaBsl  TnCL    TnCHc
    # 21      22      23      24      25      26      27      28     29      30
    # TnCHm   CaM     Myoc    Myom    SRB     SLLj    SLLsl   SLHj   SLHsl   Csqnb
    # 31      32      33      34      35      36      37      38     39      40
    # Ca_sr   Naj     Nasl    Nai     Ki      Caj     Casl    Cai    Vm      rtos
    # 41: INaL h gate
    # 42-53: placeholder for INa Markov model
    # 54-59: myofilament (TSCa TSCa_star TSCa_tilde TS_star L_p L_w)
    # 60-83: ICaL Markov model
    #print('entered ODE')
    # Unpacking matches canonical layout: ECC (0..82), camDyad (83..97), camSL (98..112), camCyt (113..127), CaMKII (128..133), BAR (134..172)
    m,h,j,xks_sl,f,fcaBj,fcaBsl,xtos,ytos,xtof,ytof,xkr,xks_j,RyRr,RyRo,RyRi,NaBj,NaBsl,TnCL,TnCHc,TnCHm,CaM, Myoc, Myom,SRB,SLLj,SLLsl,SLHj,SLHsl,Csqnb,Ca_sr,Naj,Nasl,Nai,Ki,Caj,Casl,Cai,Vm,rtos,h_L,PH1,PH2,PH3,PH4,PH5,PH6,PH7,PH8,PH9,PH10,PH11,PH12,TSCa,TSCa_star,TSCa_tilde,TS_star,L_p,L_w,Pc2_LCCj_m1,Pc1_LCCj_m1,Pi1Ca_LCCj_m1,Pi2Ca_LCCj_m1,Pi1Ba_LCCj_m1,Pi2Ba_LCCj_m1,Pc2_LCCj_m2,Pc1_LCCj_m2,Pi1Ca_LCCj_m2,Pi2Ca_LCCj_m2,Pi1Ba_LCCj_m2,Pi2Ba_LCCj_m2,Pc2_LCCsl_m1,Pc1_LCCsl_m1,Pi1Ca_LCCsl_m1,Pi2Ca_LCCsl_m1,Pi1Ba_LCCsl_m1,Pi2Ba_LCCsl_m1,Pc2_LCCsl_m2,Pc1_LCCsl_m2,Pi1Ca_LCCsl_m2,Pi2Ca_LCCsl_m2,Pi1Ba_LCCsl_m2,Pi2Ba_LCCsl_m2,CaM_dyad,Ca2CaM_dyad,Ca4CaM_dyad,CaMB_dyad,Ca2CaMB_dyad,Ca4CaMB_dyad,Pb2_dyad,Pb_dyad,Pt_dyad,Pt2_dyad,Pa_dyad,Ca4CaN_dyad,CaMCa4CaN_dyad,Ca2CaMCa4CaN_dyad,Ca4CaMCa4CaN_dyad,CaM_sl,Ca2CaM_sl,Ca4CaM_sl,CaMB_sl,Ca2CaMB_sl,Ca4CaMB_sl,Pb2_sl,Pb_sl,Pt_sl,Pt2_sl,Pa_sl,Ca4CaN_sl,CaMCa4CaN_sl,Ca2CaMCa4CaN_sl,Ca4CaMCa4CaN_sl,CaM_cyt,Ca2CaM_cyt,Ca4CaM_cyt,CaMB_cyt,Ca2CaMB_cyt,Ca4CaMB_cyt,Pb2_cyt,Pb_cyt,Pt_cyt,Pt2_cyt,Pa_cyt,Ca4CaN_cyt,CaMCa4CaN_cyt,Ca2CaMCa4CaN_cyt,Ca4CaMCa4CaN_cyt,LCC_PKAp,LCC_CKdyadp,RyR2809p,RyR2815p,PLBT17p,LCC_CKslp,LR,LRG,RG,b1AR_S464,b1AR_S301,GsaGTPtot,GsaGDP,Gsby,AC_GsaGTP,PDE3p,PDE4p,cAMPtot,RC_I,RCcAMP_I,RCcAMPcAMP_I,RcAMPcAMP_I,PKACI,PKACI_PKI,RC_II,RCcAMP_II,RCcAMPcAMP_II,RcAMPcAMP_II,PKACII,PKACII_PKI,I1p_PP1,I1ptot,PLBp,PLMp,LCCap,LCCbp,RyRp,TnIp,Myop,KSp,KRp,ClCap,Top,K1p,Nap = initial_conds
    
    ydot = np.zeros(len(initial_conds))

    # === H1 fix: live phosphorylation feedback (recomputed every ODE step) ===
    # CaMKII active concentration per compartment (matches C# MasterOde.cs:165-166)
    CaMKIIact_Dyad = CaMKIItotDyad * (Pb_dyad + Pt_dyad + Pt2_dyad + Pa_dyad)
    CaMKIIact_SL   = CaMKIItotSL   * (Pb_sl   + Pt_sl   + Pt2_sl   + Pa_sl)

    # PP1 available near PLB (matches C# MasterOde.cs:182).
    # PP1tot here is the BAR-passed PP1_PLBtot from the outer wrapper.
    PP1_PLB_avail = 1 - I1p_PP1/PP1tot + 0.0196035491719399  # PROVA basal

    # CaMKII target fractions (state concentration / total). Use *_frac to avoid
    # collision with the unpacked state-concentration names LCC_CKdyadp / LCC_CKslp.
    LCC_CKdyadp_frac = LCC_CKdyadp / LCCtotDyad
    LCC_CKslp_frac   = LCC_CKslp   / LCCtotSL
    RyR_CKp = RyR2815p / RyRtot
    PLB_CKp = PLBT17p  / PLBtot

    # PKA target fractions (matches C# MasterOde.cs:221-233).
    # Inner-ODE arg names: PLMtot=PLMtotBA, LCCtot=LCCtotBA, TnItot=TnItotBA, etc.
    # PLBtotBA equals PLBtot in the outer wrapper, so we use PLBtot here.
    PLB_PKAn   = (PLBtot - PLBp) / PLBtot   # inverted (fraction NOT phos.)
    PLM_PKAp   = PLMp   / PLMtot
    LCCa_PKAp  = LCCap  / LCCtot
    LCCb_PKAp  = LCCbp  / LCCtot
    RyR_PKAp   = RyRp   / RyRtotBA
    TnI_PKAp   = TnIp   / TnItot
    Myo_PKAp   = Myop   / Myotot
    IKs_PKAp   = KSp    / Kstot
    IKr_PKAp   = KRp    / Krtot
    IClCa_PKAp = ClCap  / ClCatot
    Ito_PKAp   = Top    / Totot
    IK1_PKAp   = K1p    / K1tot
    INa_PKAp   = Nap    / Natot

    # PKA-derived ECC modulators
    favail = 1 + 1.25*(LCCb_PKAp - fracLCCbpo)/(1 - fracLCCbpo)  # PROVA
    ICa_scale = ICa_scale * favail

    # PKA-dependent myofilament phosphoregulation (100 nM ISO)
    kPKA_Myo = (Myo_PKAp - fracPKA_Myoo)/(fracPKA_Myoiso - fracPKA_Myoo)
    kPKA_Iks = (IKs_PKAp - fracPKA_Ikso)/(fracPKA_Iksiso - fracPKA_Ikso)
    P_g_0    = gks_factor*(0.2 + 0.2*kPKA_Iks)
    P_g_max  = gks_factor*(0.8 + 0.5*kPKA_Iks)
    P_vh_0   = -1  - 10*kPKA_Iks
    P_vh_max = -12 - 9*kPKA_Iks
    P_tau_0  = 26  + 9*kPKA_Iks
    P_tau_max = 40 + 4*kPKA_Iks

    # Myofilament constants modulated by PKA (uMyo/uXBCa/uXBcy come in as args)
    Ke  = (1 + uMyo *(0.5  - 1)*kPKA_Myo) * Ke
    Zb  = (1 + uXBCa*(4.2  - 1)*kPKA_Myo) * Zb
    Zr  = (1 + uXBCa*(1.8  - 1)*kPKA_Myo) * Zr
    Yr  = (1 + uXBCa*(2.2  - 1)*kPKA_Myo) * Yr
    Za  = (1 + uXBcy*(1.24 - 1)*kPKA_Myo) * Za
    f   = (1 + uXBcy*(1.24 - 1)*kPKA_Myo) * f
    RLa = (1 + uXBcy*(0.4  - 1)*kPKA_Myo) * RLa
    Zp  = (1 + uXBcy*(2.2  - 1)*kPKA_Myo) * Zp
    Yp  = (1 + uXBcy*(2.2  - 1)*kPKA_Myo) * Yp
    Bp  = (1 + uXBcy*(3.4  - 1)*kPKA_Myo) * Bp
    Bw  = (1 + uXBcy*(3.4  - 1)*kPKA_Myo) * Bw
    Yc  = (1 + uXBcy*(0.4  - 1)*kPKA_Myo) * Yc
    Yd  = (1 + uXBcy*(2.2  - 1)*kPKA_Myo) * Yd
    Yv  = (1 + uXBcy*(1.6  - 1)*kPKA_Myo) * Yv
    # === end H1 fix ===

    Nav_CKp = RyR_CKp
    VFoRT = Vm*FoRT
    VFoRT_Frdy = VFoRT*Frdy
    
    if flag_CaMKII:
        
        ## ODEs
        ## LCC states (note: PP2A is acting on PKA site and PP1 on CKII site)

        # CaMKII phosphorylation of Dyadic LCCs
        LCC_CKdyadn = LCCtotDyad - LCC_CKdyadp
        LCCDyad_PHOS = (k_ckLCC*CaMKIIact_Dyad*LCC_CKdyadn)/(KmCK_LCC+LCC_CKdyadn)
        LCCDyad_DEPHOS = (k_pp1LCC*PP1_dyad*LCC_CKdyadp)/(KmPP1_LCC+LCC_CKdyadp)*OA_PP1
        # CaMKII state indices (canonical layout): 128=LCC_PKAp, 129=LCC_CKdyadp, 130=RyR2809p, 131=RyR2815p, 132=PLBT17p, 133=LCC_CKslp
        ydot[129] = (LCCDyad_PHOS - LCCDyad_DEPHOS)*1e-3  # LCC_CKdyadp

        # CaMKII phosphorylation of Sub-sarcolemmal LCCs
        LCC_CKsln = LCCtotSL - LCC_CKslp
        LCCSL_PHOS = (k_ckLCC*CaMKIIact_SL*LCC_CKsln)/(KmCK_LCC+LCC_CKsln)
        LCCSL_DEPHOS = (k_pp1LCC*PP1_SL*LCC_CKslp)/(KmPP1_LCC+LCC_CKslp)*OA_PP1
        ydot[133] = (LCCSL_PHOS - LCCSL_DEPHOS) *1e-3  # LCC_CKslp

        # PKA phosphorylation (currently unused elsewhere)
        LCC_PKAn = LCCtotDyad - LCC_PKAp
        ydot[128] = ((k_pkaLCC*PKAc*LCC_PKAn)/(KmPKA_LCC+LCC_PKAn) -  (k_pp2aLCC*PP2A_dyad*LCC_PKAp)/(KmPP2A_LCC+LCC_PKAp)*OA_PP2A)*1e-3  # LCC_PKAp
        ## RyR states

        RyR2815n = RyRtot - RyR2815p
        RyR_BASAL = kb_2815*RyR2815n
        RyR_PHOS = (k_ckRyR*CaMKIIact_Dyad*RyR2815n)/(KmCK_RyR+RyR2815n)
        RyR_PP1_DEPHOS = (k_pp1RyR*PP1_dyad*RyR2815p)/(KmPP1_RyR+RyR2815p)*OA_PP1
        RyR_PP2A_DEPHOS = (k_pp2aRyR*PP2A_dyad*RyR2815p)/(KmPP2A_RyR+RyR2815p)*OA_PP2A
        ydot[131] = (RyR_BASAL + RyR_PHOS - RyR_PP1_DEPHOS - RyR_PP2A_DEPHOS)*1e-3  # RyR2815p

        # PKA phosphorylation of Ser 2809 on RyR (currently unused elsewhere)
        RyR2809n = RyRtot - RyR2809p
        ydot[130] = (kb_2809*RyR2809n + (k_pkaRyR*PKAc*RyR2809n)/(KmPKA_RyR+RyR2809n) - (k_pp1RyR*PP1_dyad*RyR2809p)/(KmPP1_RyR+RyR2809p)*OA_PP1)*1e-3  # RyR2809p
        ## PLB states

        PP1_PLB = PP1_dyad*PP1_PLB_avail    # Inhibitor-1 regulation of PP1_dyad included here
        PLBT17n = PLBtot - PLBT17p
        PLB_PHOS = (k_ckPLB*PLBT17n*CaMKIIact_Dyad)/(KmCK_PLB+PLBT17n)
        PLB_DEPHOS = (k_pp1PLB*PP1_PLB*PLBT17p)/(KmPP1_PLB+PLBT17p)*OA_PP1
        ydot[132] = (PLB_PHOS - PLB_DEPHOS)*1e-3  # PLBT17p
    else:
        ydot[133] = ydot[132] = ydot[131] = ydot[130] = ydot[129] = ydot[128] = 0
        
        
    if flag_cam:
        
        pCaMDyad = [CaMtotDyad, BtotDyad, CaMKIItotDyad, CaNtotDyad, PP1totDyad, CaDyad]
        pCaMSL = [CaMtotSL, BtotSL, CaMKIItotSL, CaNtotSL, PP1totSL, CaSL]
        pCaMCyt = [CaMtotCyt, BtotCyt, CaMKIItotCyt, CaNtotCyt, PP1totCyt, CaCyt]
        
        dCaM_dyad,dCa2CaM_dyad,dCa4CaM_dyad,dCaMB_dyad,dCa2CaMB_dyad,dCa4CaMB_dyad,dPb2_dyad,dPb_dyad,dPt_dyad,dPt2_dyad,dPa_dyad,dCa4CaN_dyad,dCaMCa4CaN_dyad,dCa2CaMCa4CaN_dyad,dCa4CaMCa4CaN_dyad,JCa_Dyad = HumanVentricularMyocyte_camODEfile(t,CaM_dyad,Ca2CaM_dyad,Ca4CaM_dyad,CaMB_dyad,Ca2CaMB_dyad,Ca4CaMB_dyad,Pb2_dyad,Pb_dyad,Pt_dyad,Pt2_dyad,Pa_dyad,Ca4CaN_dyad,CaMCa4CaN_dyad,Ca2CaMCa4CaN_dyad,Ca4CaMCa4CaN_dyad,pCaMDyad,K,Mg)
        dCaM_sl,dCa2CaM_sl,dCa4CaM_sl,dCaMB_sl,dCa2CaMB_sl,dCa4CaMB_sl,dPb2_sl,dPb_sl,dPt_sl,dPt2_sl,dPa_sl,dCa4CaN_sl,dCaMCa4CaN_sl,dCa2CaMCa4CaN_sl,dCa4CaMCa4CaN_sl,JCa_SL = HumanVentricularMyocyte_camODEfile(t,CaM_sl,Ca2CaM_sl,Ca4CaM_sl,CaMB_sl,Ca2CaMB_sl,Ca4CaMB_sl,Pb2_sl,Pb_sl,Pt_sl,Pt2_sl,Pa_sl,Ca4CaN_sl,CaMCa4CaN_sl,Ca2CaMCa4CaN_sl,Ca4CaMCa4CaN_sl,pCaMSL,K,Mg)
        dCaM_cyt,dCa2CaM_cyt,dCa4CaM_cyt,dCaMB_cyt,dCa2CaMB_cyt,dCa4CaMB_cyt,dPb2_cyt,dPb_cyt,dPt_cyt,dPt2_cyt,dPa_cyt,dCa4CaN_cyt,dCaMCa4CaN_cyt,dCa2CaMCa4CaN_cyt,dCa4CaMCa4CaN_cyt,JCa_Cyt = HumanVentricularMyocyte_camODEfile(t,CaM_cyt,Ca2CaM_cyt,Ca4CaM_cyt,CaMB_cyt,Ca2CaMB_cyt,Ca4CaMB_cyt,Pb2_cyt,Pb_cyt,Pt_cyt,Pt2_cyt,Pa_cyt,Ca4CaN_cyt,CaMCa4CaN_cyt,Ca2CaMCa4CaN_cyt,Ca4CaMCa4CaN_cyt,pCaMCyt,K,Mg)
        
        #derivatives: dCaM_dyad,dCa2CaM_dyad,dCa4CaM_dyad,dCaMB_dyad,dCa2CaMB_dyad,dCa4CaMB_dyad,dPb2_dyad,dPb_dyad,dPt_dyad,dPt2_dyad,dPa_dyad,dCa4CaN_dyad,dCaMCa4CaN_dyad,dCa2CaMCa4CaN_dyad,dCa4CaMCa4CaN_dyad,
        
        JCaCyt = JCa_Cyt
        JCaSL = JCa_SL
        JCaDyad = JCa_Dyad
        
        # Canonical layout: camDyad [83:98], camSL [98:113], camCyt [113:128]
        ydot[83:98] = [dCaM_dyad,dCa2CaM_dyad,dCa4CaM_dyad,dCaMB_dyad,dCa2CaMB_dyad,dCa4CaMB_dyad,dPb2_dyad,dPb_dyad,dPt_dyad,dPt2_dyad,dPa_dyad,dCa4CaN_dyad,dCaMCa4CaN_dyad,dCa2CaMCa4CaN_dyad,dCa4CaMCa4CaN_dyad]
        ydot[98:113] = [dCaM_sl,dCa2CaM_sl,dCa4CaM_sl,dCaMB_sl,dCa2CaMB_sl,dCa4CaMB_sl,dPb2_sl,dPb_sl,dPt_sl,dPt2_sl,dPa_sl,dCa4CaN_sl,dCaMCa4CaN_sl,dCa2CaMCa4CaN_sl,dCa4CaMCa4CaN_sl]
        ydot[113:128] = [dCaM_cyt,dCa2CaM_cyt,dCa4CaM_cyt,dCaMB_cyt,dCa2CaMB_cyt,dCa4CaMB_cyt,dPb2_cyt,dPb_cyt,dPt_cyt,dPt2_cyt,dPa_cyt,dCa4CaN_cyt,dCaMCa4CaN_cyt,dCa2CaMCa4CaN_cyt,dCa4CaMCa4CaN_cyt]
        
    else:
        
        ydot[83:98] = np.zeros(15)
        ydot[98:113] = np.zeros(15)
        ydot[113:128] = np.zeros(15)
        JCaDyad = JCaSL = JCaCyt = 0 
        
        
    if flag_BARS:
        # Truthy check: accepts boolean True, integer 1, the string "True", etc.
        # Matches the corresponding Ligtot truthy check at the top of run_Morotti_model.
        ## State variables
        #y[0]-y[38]
        
        ## Drug Concentrations    
        b1ARact = b1ARtot - b1AR_S464 - b1AR_S301
        b1AR = b1ARact - LR - LRG - RG
        Gs = Gstot - LRG - RG - Gsby

        dLR = kf_LR*ISO*b1AR - kr_LR*LR + kr_LRG*LRG - kf_LRG*LR*Gs
        dLRG = kf_LRG*LR*Gs - kr_LRG*LRG - k_G_act*LRG
        dRG = kf_RG*b1AR*Gs - kr_RG*RG - k_G_act*RG

        bARK_desens = kf_bARK*(LR+LRG)
        bARK_resens = kr_bARK*b1AR_S464
        PKA_desens = kf_PKA*PKACI*b1ARact
        PKA_resens = kr_PKA*b1AR_S301
        db1AR_S464 = bARK_desens - bARK_resens # ydot[4]
        db1AR_S301 = PKA_desens - PKA_resens # ydot[5]

        G_act = k_G_act*(RG+LRG)
        G_hyd = k_G_hyd*GsaGTPtot
        G_reassoc = k_G_reassoc*GsaGDP*Gsby
        dGsaGTPtot = G_act - G_hyd # ydot[6]
        dGsaGDP = G_hyd - G_reassoc # ydot[7]
        dGsby = G_act - G_reassoc # ydot[8]
        # end b-AR module
        ## cAMP module

        cAMP = cAMPtot - (RCcAMP_I+2*RCcAMPcAMP_I+2*RcAMPcAMP_I) - (RCcAMP_II+2*RCcAMPcAMP_II+2*RcAMPcAMP_II)
        AC = ACtot - AC_GsaGTP
        GsaGTP = GsaGTPtot - AC_GsaGTP
        dAC_GsaGTP = kf_AC_Gsa*GsaGTP*AC - kr_AC_Gsa*AC_GsaGTP

        AC_FSK = FSK*AC/Kd_AC_FSK
        AC_ACT_BASAL = k_AC_basal*AC*ATP/(Km_AC_basal+ATP)
        AC_ACT_GSA = k_AC_Gsa*AC_GsaGTP*ATP/(Km_AC_Gsa+ATP)
        AC_ACT_FSK = k_AC_FSK*AC_FSK*ATP/(Km_AC_FSK+ATP)

        # RABBIT                                # Add constrain on total IBMX?
        #PDE3_IBMX = PDE3tot*IBMX/Kd_PDE_IBMX
        PDE3_IBMX = PDE3tot*IBMX/(Kd_PDE_IBMX+IBMX)
        PDE3 = PDE3tot - PDE3_IBMX - PDE3p
        dPDE3p = k_PKA_PDE*PKACII*PDE3 - k_PP_PDE*PDE3p # ydot[9]
        PDE3_ACT = k_cAMP_PDE3*PDE3*cAMP/(Km_PDE3_cAMP+cAMP) + k_cAMP_PDE3p*PDE3p*cAMP/(Km_PDE3_cAMP+cAMP)

        #PDE4_IBMX = PDE4tot*IBMX/Kd_PDE_IBMX
        PDE4_IBMX = PDE4tot*IBMX/(Kd_PDE_IBMX+IBMX)
        PDE4 = PDE4tot - PDE4_IBMX - PDE4p
        dPDE4p = k_PKA_PDE*PKACII*PDE4 - k_PP_PDE*PDE4p  # ydot() - NEW STATE VARIABLE NEEDED
        PDE4_ACT = k_cAMP_PDE4*PDE4*cAMP/(Km_PDE4_cAMP+cAMP) + k_cAMP_PDE4p*PDE4p*cAMP/(Km_PDE4_cAMP+cAMP)

        dcAMPtot = AC_ACT_BASAL + AC_ACT_GSA + AC_ACT_FSK - PDE3_ACT - PDE4_ACT # ydot[10]
        # end cAMP module
        ## PKA module

        PKI = PKItot - PKACI_PKI - PKACII_PKI

        dRC_I = - kf_RC_cAMP*RC_I*cAMP + kr_RC_cAMP*RCcAMP_I
        dRCcAMP_I = - kr_RC_cAMP*RCcAMP_I + kf_RC_cAMP*RC_I*cAMP - kf_RCcAMP_cAMP*RCcAMP_I*cAMP + kr_RCcAMP_cAMP*RCcAMPcAMP_I
        dRCcAMPcAMP_I = - kr_RCcAMP_cAMP*RCcAMPcAMP_I + kf_RCcAMP_cAMP*RCcAMP_I*cAMP - kf_RcAMPcAMP_C*RCcAMPcAMP_I + kr_RcAMPcAMP_C*RcAMPcAMP_I*PKACI
        dRcAMPcAMP_I = - kr_RcAMPcAMP_C*RcAMPcAMP_I*PKACI + kf_RcAMPcAMP_C*RCcAMPcAMP_I
        dPKACI = - kr_RcAMPcAMP_C*RcAMPcAMP_I*PKACI + kf_RcAMPcAMP_C*RCcAMPcAMP_I - kf_PKA_PKI*PKACI*PKI + kr_PKA_PKI*PKACI_PKI # ydot[16]
        dPKACI_PKI = - kr_PKA_PKI*PKACI_PKI + kf_PKA_PKI*PKACI*PKI

        dRC_II = - kf_RC_cAMP*RC_II*cAMP + kr_RC_cAMP*RCcAMP_II
        dRCcAMP_II = - kr_RC_cAMP*RCcAMP_II + kf_RC_cAMP*RC_II*cAMP - kf_RCcAMP_cAMP*RCcAMP_II*cAMP + kr_RCcAMP_cAMP*RCcAMPcAMP_II
        dRCcAMPcAMP_II = - kr_RCcAMP_cAMP*RCcAMPcAMP_II + kf_RCcAMP_cAMP*RCcAMP_II*cAMP - kf_RcAMPcAMP_C*RCcAMPcAMP_II + kr_RcAMPcAMP_C*RcAMPcAMP_II*PKACII
        dRcAMPcAMP_II = - kr_RcAMPcAMP_C*RcAMPcAMP_II*PKACII + kf_RcAMPcAMP_C*RCcAMPcAMP_II
        dPKACII = - kr_RcAMPcAMP_C*RcAMPcAMP_II*PKACII + kf_RcAMPcAMP_C*RCcAMPcAMP_II - kf_PKA_PKI*PKACII*PKI + kr_PKA_PKI*PKACII_PKI # ydot[17]
        dPKACII_PKI = - kr_PKA_PKI*PKACII_PKI + kf_PKA_PKI*PKACII*PKI
        # end PKA module
        ## I-1/PP1 module
        I1 = I1tot - I1ptot
        PP1 = PP1tot - I1p_PP1
        I1p = I1ptot - I1p_PP1
        I1_phosph = k_PKA_I1*PKACI*I1/(Km_PKA_I1+I1)
        I1_dephosph = Vmax_PP2A_I1*I1ptot/(Km_PP2A_I1+I1ptot)

        dI1p_PP1 = kf_PP1_I1*PP1*I1p - kr_PP1_I1*I1p_PP1
        dI1ptot = I1_phosph - I1_dephosph # ydot
        # end I-1/PP1 module
        ## PLB module
        PLB = PLBtot - PLBp
        PLB_phosph = k_PKA_PLB*PKACI*PLB/(Km_PKA_PLB+PLB)
        PLB_dephosph = k_PP1_PLB*PP1*PLBp/(Km_PP1_PLB+PLBp)
        dPLBp = PLB_phosph - PLB_dephosph # ydot
        # end PLB module
        ## PLM module (from PLB, different total concentration)

        PLM = PLMtot - PLMp
        PLM_phosph = k_PKA_PLM*PKACI*PLM/(Km_PKA_PLM+PLM)
        PLM_dephosph = k_PP1_PLM*PP1*PLMp/(Km_PP1_PLM+PLMp)
        dPLMp = PLM_phosph - PLM_dephosph # ydot
        # end PLM module
        ## LCC module

        PKACII_LCC = (PKACII_LCCtot/PKAIItot)*PKACII
        LCCa = LCCtot - LCCap
        LCCa_phosph = epsilon*k_PKA_LCC*PKACII_LCC*LCCa/(Km_PKA_LCC+epsilon*LCCa)
        LCCa_dephosph = epsilon*k_PP2A_LCC*PP2A_LCC*LCCap/(Km_PP2A_LCC+epsilon*LCCap)
        dLCCap = LCCa_phosph - LCCa_dephosph # ydot
        LCCb = LCCtot - LCCbp
        LCCb_phosph = epsilon*k_PKA_LCC*PKACII_LCC*LCCb/(Km_PKA_LCC+epsilon*LCCb)
        LCCb_dephosph = epsilon*k_PP1_LCC*PP1_LCC*LCCbp/(Km_PP1_LCC+epsilon*LCCbp)
        dLCCbp = LCCb_phosph - LCCb_dephosph # ydot
        # end LCC module
        ## RyR module
        PKACryr = (PKAIIryrtot/PKAIItot)*PKACII
        RyR = RyRtot-RyRp
        RyRPHOSPH = epsilon*kcat_pka_ryr*PKACryr*RyR/(Km_pka_ryr+epsilon*RyR)
        RyRDEPHOSPH1 = epsilon*kcat_pp1_ryr*PP1ryr*RyRp/(Km_pp1_ryr+epsilon*RyRp)
        RyRDEPHOSPH2A = epsilon*kcat_pp2a_ryr*PP2Aryr*RyRp/(Km_pp2a_ryr+epsilon*RyRp)
        dRyRp = RyRPHOSPH-RyRDEPHOSPH1-RyRDEPHOSPH2A # ydot
        # end RyR module
        ## TnI module

        TnIn = TnItot - TnIp
        TnI_phosph = k_PKA_TnI*PKACI*TnIn/(Km_PKA_TnI+TnIn)
        TnI_dephosph = k_PP2A_TnI*PP2A_TnI*TnIp/(Km_PP2A_TnI+TnIp)
        dTnIp = TnI_phosph - TnI_dephosph # ydot[25]
        # end TnI module
        ## Myofilament module (from TnI)

        Myon = Myotot-Myop  # Non-phos = tot - phos
        MyoPHOSPH = kcat_pka_myo*PKACI*Myon/(Km_pka_myo+Myon)
        MyoDEPHOSPH = kcat_pp2a_myo*PP2A_myo*Myop/(Km_pp2a_myo+Myop)
        dMyop = MyoPHOSPH-MyoDEPHOSPH # ydot
        # end Myo module
        ## IKs module

        # Effect of G589D mutation (IKs-Yotiao)
        y1 = (-(K_yotiao+Kstot-Yotiao_tot)+np.sqrt((K_yotiao+Kstot-Yotiao_tot)**2+4*K_yotiao*Yotiao_tot))/2
        x1 = Kstot/(1+y1/K_yotiao)
        y2 = (-(K_yotiao+Kstot-Yotiao_tot)-np.sqrt((K_yotiao+Kstot-Yotiao_tot)**2+4*K_yotiao*Yotiao_tot))/2
        x2 = Kstot/(1+y2/K_yotiao)

        free_IKs = x1*(y1 > 0) + x2*(y1 <= 0)
        free_Yotiao = y1*(y1 > 0) + y2*(y1 <= 0)
        IksYot = free_IKs*free_Yotiao/K_yotiao # [uM] # IKs-Yotiao

        PKACiks = IksYot/Kstot*(PKAII_ikstot/PKAIItot)*PKACII
        PP1iks = IksYot/Kstot*PP1_ikstot

        KSn = Kstot-KSp  # Non-phos = tot - phos
        IKS_PHOSPH = epsilon*k_pka_iks*PKACiks*KSn/(Km_pka_iks+epsilon*KSn)
        IKS_DEPHOSPH = epsilon*k_pp1_iks*PP1iks*KSp/(Km_pp1_iks+epsilon*KSp)
        dKSp = IKS_PHOSPH-IKS_DEPHOSPH # ydot
        # end Iks module
        ## IKr module (from IKs, without mutation)

        KRn = Krtot-KRp  # Non-phos = tot - phos
        PKACikr = (PKAII_ikrtot/PKAIItot)*PKACII
        IKR_PHOSPH = epsilon*k_pka_ikr*PKACikr*KRn/(Km_pka_ikr+epsilon*KRn)
        IKR_DEPHOSPH = epsilon*k_pp1_ikr*PP1_ikrtot*KRp/(Km_pp1_ikr+epsilon*KRp)
        dKRp = IKR_PHOSPH-IKR_DEPHOSPH
        # end IKr module
        ## IClCa module (from CFTR)
      
        ClCan = ClCatot - ClCap  # Non-phos = tot - phos
        PKAC_ClCa = (PKAII_ClCatot/PKAIItot)*PKACII    # (PKACFTRtot/PKAIItot)*PKAIIact
        ClCaphos = epsilon*ClCan*PKAC_ClCa*k_pka_ClCa/(Km_pka_ClCa+epsilon*ClCan)
        ClCadephos = PP1_ClCatot*k_pp1_ClCa*epsilon*ClCap/(Km_pp1_ClCa+epsilon*ClCap)
        dClCap = ClCaphos - ClCadephos
        # end ICl(Ca) module
        ## Ito module (from IKr)

        Ton = Totot-Top  # Non-phos = tot - phos
        PKACito = (PKAII_itotot/PKAIItot)*PKACII
        Ito_PHOSPH = epsilon*k_pka_ito*PKACito*Ton/(Km_pka_ito+epsilon*Ton)
        Ito_DEPHOSPH = epsilon*k_pp1_ito*PP1_itotot*Top/(Km_pp1_ito+epsilon*Top)
        dTop = Ito_PHOSPH-Ito_DEPHOSPH
        # end Ito module
        ## IK1 module (from IKr)

        K1n = K1tot-K1p  # Non-phos = tot - phos
        PKACik1 = (PKAII_ik1tot/PKAIItot)*PKACII
        IK1_PHOSPH = epsilon*k_pka_ik1*PKACik1*K1n/(Km_pka_ik1+epsilon*K1n)
        IK1_DEPHOSPH = epsilon*k_pp1_ik1*PP1_ik1tot*K1p/(Km_pp1_ik1+epsilon*K1p)
        dK1p = IK1_PHOSPH-IK1_DEPHOSPH
        # end IK1 module
        ## INa module (from IKr)
        Nan = Natot-Nap  # Non-phos = tot - phos
        PKACiNa = (PKAII_iNatot/PKAIItot)*PKACII
        INa_PHOSPH = epsilon*k_pka_iNa*PKACiNa*Nan/(Km_pka_iNa+epsilon*Nan)
        INa_DEPHOSPH = epsilon*k_pp1_iNa*PP1_iNatot*Nap/(Km_pp1_iNa+epsilon*Nap)
        dNap = INa_PHOSPH-INa_DEPHOSPH
        # end INa module
        ## ydot

        ydot[134:173]=[dLR,dLRG,dRG,db1AR_S464,db1AR_S301,dGsaGTPtot,dGsaGDP,dGsby,dAC_GsaGTP,dPDE3p,dPDE4p,dcAMPtot,dRC_I,dRCcAMP_I,
            dRCcAMPcAMP_I,dRcAMPcAMP_I,dPKACI,dPKACI_PKI,dRC_II,dRCcAMP_II,dRCcAMPcAMP_II,dRcAMPcAMP_II,dPKACII,dPKACII_PKI,
            dI1p_PP1,dI1ptot,dPLBp,dPLMp,dLCCap,dLCCbp,dRyRp,dTnIp,dMyop,dKSp,dKRp,dClCap,dTop,dK1p,dNap] # output
    else:
        ydot[134:173] = np.zeros(39)

    
    ena_junc = FoRT_reciprocal*np.log(Nao/Naj)     # [mV]
    ena_sl = FoRT_reciprocal*np.log(Nao/Nasl)       # [mV]
    ek = FoRT_reciprocal*np.log(Ko/Ki)	        # [mV]
    eca_junc = (FoRT_reciprocal/2)*np.log(Cao/Caj)   # [mV]
    eca_sl = (FoRT_reciprocal/2)*np.log(Cao/Casl)     # [mV]
    ecl = FoRT_reciprocal*np.log(Cli/Clo)            # [mV]
    eks = FoRT_reciprocal*np.log((Ko+pNaK*Nao)/(Ki+pNaK*Nai))

    ## I_Na: Fast Na Current - original

    # GNa =  23.0        # [mS/uF]
    # 
    # mss = 1 / ((1 + np.exp( -(56.86 + Vm) / 9.03 ))**2)
    # taum = 0.1292 * np.exp(-((Vm+45.79)/15.54)**2) + 0.06487 * np.exp(-((Vm-4.823)/51.12)**2)                 
    #  
    # ah = (Vm >= -40) * (0) 
    #    + (Vm < -40) * (0.057 * np.exp( -(Vm + 80) / 6.8 )) 
    # bh = (Vm >= -40) * (0.77 / (0.13*(1 + np.exp( -(Vm + 10.66) / 11.1 )))) 
    #    + (Vm < -40) * ((2.7 * np.exp( 0.079 * Vm) + 3.1*10**5 * np.exp(0.3485 * Vm))) 
    # tauh = 1 / (ah + bh) 
    # hss = 1 / ((1 + np.exp( (Vm + 71.55)/7.43 ))**2)
    #  
    # aj = (Vm >= -40) * (0) 
    #     +(Vm < -40) * (((-2.5428 * 10**4*np.exp(0.2444*Vm) - 6.948*10**-6 * np.exp(-0.04391*Vm)) * (Vm + 37.78)) / 
    #                      (1 + np.exp( 0.311 * (Vm + 79.23) )))
    # bj = (Vm >= -40) * ((0.6 * np.exp( 0.057 * Vm)) / (1 + np.exp( -0.1 * (Vm + 32) ))) 
    #    + (Vm < -40) * ((0.02424 * np.exp( -0.01052 * Vm )) / (1 + np.exp( -0.1378 * (Vm + 40.14) ))) 
    # tauj = 1 / (aj + bj)
    # jss = 1 / ((1 + np.exp( (Vm + 71.55)/7.43 ))**2)         
    #  
    # ydot[0] = (mss - m) / taum
    # ydot[1] = (hss - h) / tauh
    # ydot[2] = (jss - j) / tauj
    #     
    # I_Na_junc1 = Fjunc*GNa*m**3*h*j*(Vm-ena_junc)
    # I_Na_sl1 = Fsl*GNa*m**3*h*j*(Vm-ena_sl)
    # #I_Na = I_Na_junc+I_Na_sl
    # 
    # I_NaL_junc = 0
    # I_NaL_sl = 0
    # I_NaL = I_NaL_junc + I_NaL_sl
    ## I_Na: Fast Na Current - from rabbit

    # PKA-dependent INa phosphoregulation
    kPKA_Ina = (INa_PKAp-fracPKA_Inao)/(fracPKA_Inaiso-fracPKA_Inao)
    GNa = GNa*(1+0.25*kPKA_Ina)

    am = 0.32*(Vm+47.13)/(1-np.exp(-0.1*(Vm+47.13)))
    bm = 0.08*np.exp(-Vm/11)
    ah = np.where((Vm-inashift) >= -40,0,0.135*np.exp((80+(Vm-inashift))/-6.8))
    # Including alteration to aj as in Hund and Rudy 2008
    aj = np.where((Vm-inashift) >= -40,0,(1+alphaCKII)*((-1.2714e5*np.exp(0.2444*(Vm-inashift))-3.474e-5*np.exp(-0.04391*(Vm-inashift)))*((Vm-inashift)+37.78)/(1+np.exp(0.311*((Vm-inashift)+79.23)))))
    bh = np.where((Vm-inashift) >= -40,1/(0.13*(1+np.exp(-((Vm - inashift)+10.66)/11.1))),3.56*np.exp(0.079*(Vm-inashift))+3.1e5*np.exp(0.35*(Vm-inashift)))
    bj = np.where((Vm-inashift) >= -40,0.3*np.exp(-2.535e-7*(Vm-inashift))/(1+np.exp(-0.1*((Vm-inashift)+32))),0.1212*np.exp(-0.01052*(Vm-inashift))/(1+np.exp(-0.1378*((Vm-inashift)+40.14))))

    ydot[0] = am*(1-m)-bm*m
    ydot[1] = ah*(1-h)-bh*h
    ydot[2] = aj*(1-j)-bj*j

    m_cubed = m**3
    INa_constant = GNa*m_cubed*h*j
    I_Na_junc1 = Fjunc*INa_constant*(Vm-ena_junc)
    I_Na_sl1 = Fsl*INa_constant*(Vm-ena_sl)
    I_NaF = I_Na_junc1+I_Na_sl1
    ## I_Na,L: Late Na current (as in Hund & Rudy 2008)
    # modifications? tauhl = 100 ms (and 2*GNaL?)

    # if CKIIflag == 1    
    #     deltGbarNal_CKII = 2  
    # else
    #     deltGbarNal_CKII = 0
    # end
    # GbarNal = GNaL * (1+deltGbarNal_CKII)   # deltGbar assigned in 'Fast INa' section

    # CaMKII-dependent INa phosphoregulation (late component only)
    GbarK = (1+0.5*CKIIflag)*((14/11)/(1+np.exp(-(Nav_CKp-0.12)/0.1))) #(current) includes dynamnic CMKII phosphorylation in control and HF
    GbarNal = GbarK * GNaL # *2

    # h-gate (note - m gate is same as INa m gate - using m for this)
    hlss = 1/(1+np.exp((Vm+91)/6.1))
    #tauhl = 600#/6 # ms
    ydot[40] = (hlss-h_L)/600

    INaL_constant = GbarNal*m_cubed*h_L
    I_NaL_junc = Fjunc*INaL_constant*(Vm-ena_junc)
    I_NaL_sl = Fsl*INaL_constant*(Vm-ena_sl)
    I_NaL = I_NaL_junc+I_NaL_sl
    ## Placeholders for Markov INa 

    # If flagMina = 1, INa currents will be dictated by Markov scheme. If
    # flag = 0, the original H-H scheme is used to compute the current.

    # Parameters
    #GNa2 = 23      # [mS/uF]
    GNa2 = 23*(1+0.25*kPKA_Ina)
    # If not using INa Markov, set odes to zero to speed simulations
    # ydot(42:53) = np.zeros(1,12) # already set to 0, see line 45
    INa2_constant = GNa2*(TSCa_star+L_w)
    I_Na_junc2 = Fjunc*INa2_constant*(Vm-ena_junc)     # junctional current
    I_Na_sl2 = Fsl*INa2_constant*(Vm-ena_sl)           # sl current
    ## Compute total INa (fast and late components, HH or Markov)

    I_Na_junc = (I_Na_junc1+I_NaL_junc)*(1-flagMina)+I_Na_junc2*flagMina
    I_Na_sl = (I_Na_sl1+I_NaL_sl)*(1-flagMina)+I_Na_sl2*flagMina

    I_Na = I_Na_junc + I_Na_sl
    ## I_nabk: Na Background Current

    I_nabk_junc = Fjunc*GNaB*(Vm-ena_junc)
    I_nabk_sl = Fsl*GNaB*(Vm-ena_sl)
    I_nabk = I_nabk_junc+I_nabk_sl
    ## I_nak: Na/K Pump Current

    # PKA-dependent PLM phosphoregulation
    kPKA_PLM = KmNaip*(1-13.6/18.8)/(fracPKA_PLMiso/fracPKA_PLMo-1) # PLM_PKAp ISO
    #kPKA_PLM=KmNaip*(0.5)*(1-13.6/18.8)/(fracPKA_PLMiso/fracPKA_PLMo-1) # PLM_PKAp ISO (50#)
    KmNaip_PKA = -kPKA_PLM+kPKA_PLM*(PLM_PKAp/fracPKA_PLMo)
    KmNaip = KmNaip-KmNaip_PKA # 27.66# reduction w/ ISO

    fnak = 1/(1+0.1245*np.exp(-0.1*VFoRT)+0.0365*sigma*np.exp(-VFoRT))
    INaK_constant_1 = IbarNaK*fnak*Ko
    INak_constant_2 = (Ko+KmKo)
    I_nak_junc = Fjunc*INaK_constant_1 /(1+(KmNaip/Naj)**4) /INak_constant_2#(Ko+KmKo)
    I_nak_sl = Fsl*INaK_constant_1 /(1+(KmNaip/Nasl)**4) /INak_constant_2#(Ko+KmKo)
    I_nak = I_nak_junc+I_nak_sl
    ## I_kr: Rapidly Activating K Current

    # PKA-dependent IKr phosphoregulation
    kPKA_Ikr = (IKr_PKAp-fracPKA_Ikro)/(fracPKA_Ikriso-fracPKA_Ikro)
    Vkr_PKA = 10*kPKA_Ikr# 10 mV shift w/ ISO
    dGkr_PKA = 0.3*kPKA_Ikr# 30# increase w/ ISO 
    # total effect: SSA shifted and peak increased by 37#
    #Vkr_PKA = 0 dGkr_PKA = 0

    xrss = 1/(1+np.exp(-(Vm+10+Vkr_PKA)/5))
    tauxr = 550/(1+np.exp((-22-Vm)/9))*6/(1+np.exp((Vm-(-11))/9))+230/(1+np.exp((Vm-(-40))/20))
    ydot[11] = (xrss-xkr)/tauxr
    rkr = 1/(1+np.exp((Vm+74)/24))

    I_kr = (1+dGkr_PKA)*gkr*xkr*rkr*(Vm-ek)
    ## I_ks: Slowly Activating K Current

    # PKA-dependent IKs phosphoregulation # 100 nM ISO: IKs_PKAp=0.8380
    
    V_EKs = (Vm-eks) 
    # if IKs_flag: # OLD IKS ################################################
    #     fracIKsavail = 1+(2.8-1)*kPKA_Iks # 2.8-fold increase w/ ISO
    #     Xs05 = 1.5-(13.5+1.5)*kPKA_Iks # 15 mV shift w/ ISO
    #     gks = gks_factor_SA*0.0035
    #     gks_junc = fracIKsavail*gks
    #     gks_sl = fracIKsavail*gks #FRA
    #     xsss = 1 / (1+np.exp(-(Vm + 3.8 - Xs05)/14.25)) # fitting Fra
    #     tauxs = 990.1/(1+np.exp(-(Vm+2.436)/14.12))
    #     ydot[12] = (xsss-xks_j)/tauxs

    #     I_ks_junc = Fjunc*gks_junc*xks_j**2*V_EKs
    #     I_ks_sl = Fsl*gks_sl*xks_j**2*V_EKs
    # else: # NEW IKS - Bartos ###################################################
    caks_junc = Caj 
    caks_sl = Casl # normal simulation
        
    gks_junc = P_g_0 + (P_g_max-P_g_0)/(1 + (150e-6/caks_junc)**1.3) # Regulated by PKA
    gks_sl = P_g_0 + (P_g_max-P_g_0)/(1 + (150e-6/caks_sl)**1.3) # Regulated by PKA
    VsXs_Ca_junc = P_vh_0 + (P_vh_max-P_vh_0)/(1 + (350e-6/caks_junc)**4) # Regulated by PKA
    xsss_junc = 1/(1+np.exp(-(Vm-VsXs_Ca_junc)/25))
    VsTs_Ca_junc = P_tau_0 + (P_tau_max-P_tau_0)/(1 + (150e-6/caks_junc)**3) # Regulated by PKA
    tau_IKs_const = (50+350*np.exp(-((Vm+30)**2)/4000))*1
    tauxs_junc = 2*(50+tau_IKs_const/(1+np.exp(-(Vm+VsTs_Ca_junc)/10)))
    VsXs_Ca_sl = P_vh_0 + (P_vh_max-P_vh_0)/(1 + (350e-6/caks_sl)**4) # Regulated by PKA
    xsss_sl = 1/(1+np.exp(-(Vm-VsXs_Ca_sl)/25))
    VsTs_Ca_sl = P_tau_0 + (P_tau_max-P_tau_0)/(1 + (150e-6/caks_sl)**3) # Regulated by PKA
    tauxs_sl = 2*(50+tau_IKs_const/(1+np.exp(-(Vm+VsTs_Ca_sl)/10))) 
    ydot[12] = (xsss_junc-xks_j)/tauxs_junc
    ydot[3] = (xsss_sl-xks_sl)/tauxs_sl
    
    I_ks_junc = Fjunc*gks_factor_SA*gks_junc*xks_j**2*V_EKs
    I_ks_sl = Fsl*gks_factor_SA*gks_sl*xks_sl**2*V_EKs   

    I_ks = I_ks_junc+I_ks_sl
    ## I_kp: Plateau K current

    kp_kp = 1/(1+np.exp(7.488-Vm/5.98))
    kp_const = gkp*kp_kp*(Vm-ek)
    I_kp_junc = Fjunc*kp_const #gkp*kp_kp*(Vm-ek)
    I_kp_sl = Fsl*kp_const #gkp*kp_kp*(Vm-ek)
    I_kp = I_kp_junc+I_kp_sl
    ## I_to: Transient Outward K Current (slow and fast components)

    # PKA-dependent Itof phosphoregulation
    kPKA_Ito = (Ito_PKAp-fracPKA_Itoo)/(fracPKA_Itoiso-fracPKA_Itoo)
    GtoFast = GtoFast*(1-0.4*kPKA_Ito)

    xtoss = 1/(1+np.exp(-(Vm-19.0)/13))
    ytoss = 1/(1+np.exp((Vm+19.5)/5))
    # rtoss = 1/(1+np.exp((Vm+33.5)/10))
    tauxtos = 9/(1+np.exp((Vm+3.0)/15))+0.5
    tauytos = 800/(1+np.exp((Vm+60.0)/10))+30
    # taurtos = 2.8e3/(1+np.exp((Vm+60.0)/10))+220 #Fei changed here!! time-dependent gating variable
    ydot[7] = (xtoss-xtos)/tauxtos
    ydot[8] = (ytoss-ytos)/tauytos
    # ydot[39] = 0
    I_tos = GtoSlow*xtos*ytos*(Vm-ek)    # [uA/uF]

    tauxtof = 8.5*np.exp(-((Vm+45)/50)**2)+0.5
    #tauxtof = 3.5*np.exp(-((Vm+3)/30)**2)+1.5
    tauytof = 85*np.exp((-(Vm+40)**2/220))+7
    #tauytof = 20.0/(1+np.exp((Vm+33.5)/10))+20.0
    ydot[9] = (xtoss-xtof)/tauxtof
    ydot[10] = (ytoss-ytof)/tauytof
    I_tof = GtoFast*xtof*ytof*(Vm-ek)
    I_to = I_tos + I_tof
    ## I_ki: Time-Independent K Current

    # PKA-dependent IK1 phosphoregulation
    kPKA_IK1 = (IK1_PKAp-fracPKA_Ik1o)/(fracPKA_Ik1iso-fracPKA_Ik1o)
    gki = gki*(1-0.45*kPKA_IK1)

    aki = 1.02/(1+np.exp(0.2385*(Vm-ek-59.215)))
    bki =(0.49124*np.exp(0.08032*(Vm+5.476-ek)) + np.exp(0.06175*(Vm-ek-594.31))) /(1 + np.exp(-0.5143*(Vm-ek+4.753)))
    kiss = aki/(aki+bki)
    I_ki = gki*kiss*(Vm-ek)
    ## I_Ca (H&H) L-type Calcium Current

    # If not using H&H model, set odes to zero to speed simulations
    # dss = 1/(1+np.exp(-(Vm+5)/6.0))
    # taud = dss*(1-np.exp(-(Vm+5)/6.0))/(0.035*(Vm+5))
    # # fss = 1/(1+np.exp((Vm+35.06)/3.6))+0.6/(1+np.exp((50-Vm)/20))
    # fss = 1/(1+np.exp((Vm+35)/9))+0.6/(1+np.exp((50-Vm)/20))
    # tauf = 1/(0.0197*np.exp( -(0.0337*(Vm+14.5))**2 )+0.02)
    # ydot[3] = (dss-xks_sl)/taud
    # ydot[4] = (fss-f)/tauf
    # ydot[5] = 1.7*Caj*(1-fcaBj)-11.9e-3*fcaBj # fCa_junc   koff!!!!!!!!
    # ydot[6] = 1.7*Casl*(1-fcaBsl)-11.9e-3*fcaBsl # fCa_sl
    # ydot(4:7) = np.zeros(1,4)
    # NOW xks_sl is used for IKs !!!
    #fcaCaMSL = 0.1/(1+(0.01/Casl))
    #fcaCaj = 0.1/(1+(0.01/Caj))
    fcaCaMSL = 0
    fcaCaj = 0
    VFoRT_exp = np.exp(VFoRT)
    VFoRT_2exp = np.exp(2*VFoRT)
    Na_K_denom = (VFoRT_exp-1)
    Ca_denom = (VFoRT_2exp-1)
    const_exponent = Q10CaL**Qpow
    ibarca_j = pCa*4*(VFoRT_Frdy) * (0.341*Caj*VFoRT_2exp-0.341*Cao) /Ca_denom
    ibarca_sl = pCa*4*(VFoRT_Frdy) * (0.341*Casl*VFoRT_2exp-0.341*Cao) /Ca_denom
    ibark = pK*(VFoRT_Frdy) * (0.75*Ki*VFoRT_exp-0.75*Ko) /Na_K_denom
    ibarna_j = pNa*(VFoRT_Frdy) *(0.75*Naj*VFoRT_exp-0.75*Nao)  /Na_K_denom
    ibarna_sl = pNa*(VFoRT_Frdy) *(0.75*Nasl*VFoRT_exp-0.75*Nao)  /Na_K_denom

    I_Ca_junc1 = (Fjunc_CaL*ibarca_j*xks_sl*f*((1-fcaBj)+fcaCaj)*const_exponent)*0.45
    I_Ca_sl1 = (Fsl_CaL*ibarca_sl*xks_sl*f*((1-fcaBsl)+fcaCaMSL)*const_exponent)*0.45
    I_CaK1 = (ibark*xks_sl*f*(Fjunc_CaL*(fcaCaj+(1-fcaBj))+Fsl_CaL*(fcaCaMSL+(1-fcaBsl)))*const_exponent)*0.45
    I_CaNa_junc1 = (Fjunc_CaL*ibarna_j*xks_sl*f*((1-fcaBj)+fcaCaj)*const_exponent)*0.45
    I_CaNa_sl1 = (Fsl_CaL*ibarna_sl*xks_sl*f*((1-fcaBsl)+fcaCaMSL)*const_exponent)*0.45
    ## I_Ca (MARKOV): L-type Calcium Current # flagMica=1

    # input LTCC module ODE (24 state vars)
    # To allow for CDI KO
    cajLCC = Caj
    caslLCC = Casl
    #y[59] to y[82]
    #Pc2_LCCj_m1,Pc1_LCCj_m1,Pi1Ca_LCCj_m1,Pi2Ca_LCCj_m1,Pi1Ba_LCCj_m1,Pi2Ba_LCCj_m1,Pc2_LCCj_m2,Pc1_LCCj_m2,Pi1Ca_LCCj_m2,Pi2Ca_LCCj_m2,Pi1Ba_LCCj_m2,Pi2Ba_LCCj_m2,Pc2_LCCsl_m1,Pc1_LCCsl_m1,Pi1Ca_LCCsl_m1,Pi2Ca_LCCsl_m1,Pi1Ba_LCCsl_m1,Pi2Ba_LCCsl_m1,Pc2_LCCsl_m2,Pc1_LCCsl_m2,Pi1Ca_LCCsl_m2,Pi2Ca_LCCsl_m2,Pi1Ba_LCCsl_m2,Pi2Ba_LCCsl_m2

    Po_LCCj_m1 =  1-Pc2_LCCj_m1-Pc1_LCCj_m1-Pi1Ca_LCCj_m1-Pi2Ca_LCCj_m1-Pi1Ba_LCCj_m1-Pi2Ba_LCCj_m1
    Po_LCCj_m2 =  1-Pc2_LCCj_m2-Pc1_LCCj_m2-Pi1Ca_LCCj_m2-Pi2Ca_LCCj_m2-Pi1Ba_LCCj_m2-Pi2Ba_LCCj_m2
    Po_LCCsl_m1 = 1-Pc2_LCCsl_m1-Pc1_LCCsl_m1-Pi1Ca_LCCsl_m1-Pi2Ca_LCCsl_m1-Pi1Ba_LCCsl_m1-Pi2Ba_LCCsl_m1
    Po_LCCsl_m2 = 1-Pc2_LCCsl_m2-Pc1_LCCsl_m2-Pi1Ca_LCCsl_m2-Pi2Ca_LCCsl_m2-Pi1Ba_LCCsl_m2-Pi2Ba_LCCsl_m2

    ### PKA PHOSPHOREGULATION OF LCC AVAILABLILITY (beta subunit phosph) ######
    #favail = 1*(.017*LCCb_PKAp/fracLCCbpo + 0.983) # Test (max x1.5 pCa)
    #favail = 1+0.5*(LCCb_PKAp - fracLCCbpo)/(1 - fracLCCbpo) # 1.5 with max phosphorylation

    # Voltage- and Ca-dependent Parameters
    Vm_40 = (Vm+40)
    poss = 1/(1+np.exp(-Vm/8))
    fcaj = 1/(1+(kop/cajLCC)**3)            
    Rv = 10 + 4954*np.exp(Vm/15.6)
    PrLCC = 1-1/(1+np.exp(-Vm_40/4))     
    PsLCC = 1/(1+np.exp(-Vm_40/11.32))
    TCaj = (tca + 0.1*(1+(cajLCC/cpbar)**2))/(1+(cajLCC/cpbar)**2) 
    tauCaj = (Rv-TCaj)*PrLCC + TCaj     
    tauBa = (Rv-TBa)*PrLCC + TBa

    # Transition Rates (20 rates)
    alphaLCC = poss/taupo
    betaLCC = (1-poss)/taupo
    
    s1 = s1o*fcaj 
    k1 = k1o*fcaj  
    s2 = s1*(k2/k1)*(r1/r2)
    s2p = s1p*(k2p/k1p)*(r1/r2)
    k3 = np.exp(-Vm_40/3)/(3*(1+np.exp(-Vm_40/3)))
    k3p = k3
    k5 = (1-PsLCC)/tauCaj
    k6 = (fcaj*PsLCC)/tauCaj
    k5p = (1-PsLCC)/tauBa

    # Recovery terms
    k5 = k5/recoveryReduc
    k5p = k5p/recoveryReduc

    k6p = PsLCC/tauBa
    k4 = k3*(alphaLCC/betaLCC)*(k1/k2)*(k5/k6)
    k4p = k3p*(alphaLCC/betaLCC)*(k1p/k2p)*(k5p/k6p)

    # State transitions for MODE 1 junctional LCCs ###
    # O = no differential C2 = 60 C1 = 61 I1Ca = 62 I2Ca = 63 I1Ba = 64 I2Ba = 65
    dPc2_LCCj_m1 = betaLCC*Pc1_LCCj_m1 + k5*Pi2Ca_LCCj_m1 + k5p*Pi2Ba_LCCj_m1 - (k6+k6p+alphaLCC)*Pc2_LCCj_m1                      # C2_m1j
    dPc1_LCCj_m1 = alphaLCC*Pc2_LCCj_m1 + k2*Pi1Ca_LCCj_m1 + k2p*Pi1Ba_LCCj_m1 + r2*Po_LCCj_m1 - (r1+betaLCC+k1+k1p)*Pc1_LCCj_m1   # C1_m1j
    dPi1Ca_LCCj_m1 = k1*Pc1_LCCj_m1 + k4*Pi2Ca_LCCj_m1 + s1*Po_LCCj_m1 - (k2+k3+s2)*Pi1Ca_LCCj_m1                              # I1Ca_m1j
    dPi2Ca_LCCj_m1 = k3*Pi1Ca_LCCj_m1 + k6*Pc2_LCCj_m1 - (k4+k5)*Pi2Ca_LCCj_m1                                                 # I2Ca_m1j
    dPi1Ba_LCCj_m1 = k1p*Pc1_LCCj_m1 + k4p*Pi2Ba_LCCj_m1 + s1p*Po_LCCj_m1 - (k2p+k3p+s2p)*Pi1Ba_LCCj_m1                        # I1Ba_m1j
    dPi2Ba_LCCj_m1 = k3p*Pi1Ba_LCCj_m1 + k6p*Pc2_LCCj_m1 - (k5p+k4p)*Pi2Ba_LCCj_m1                                             # I2Ba_m1j

    ibarca_jm1 = (4*pCa*VFoRT_Frdy)*(.001*VFoRT_2exp-0.341*Cao)/Ca_denom
    I_Ca_junc_m1 = (Fjunc_CaL*ibarca_jm1*Po_LCCj_m1*const_exponent)*ICa_scale

    possm2 = 1/(1+np.exp(-Vm/8))
    fcajm2 = 1/(1+(kopm2/cajLCC)**3)    # Depends on junctional Ca
    Rvm2 = 10 + 4954*np.exp(Vm/15.6)
    PrLCCm2 = 1-1/(1+np.exp(-Vm_40/4))     # Correct version I believe
    PsLCCm2 = 1/(1+np.exp(-Vm_40/11.32))
    TCajm2 = (tcam2 + 0.1*(1+(cajLCC/cpbarm2)**2))/(1+(cajLCC/cpbarm2)**2) # Caj dependent
    tauCajm2 = (Rvm2-TCajm2)*PrLCCm2 + TCajm2     # Caj dependence
    tauBam2 = (Rvm2-TBa)*PrLCCm2 + TBa

    alphaLCCm2 = possm2/taupo
    betaLCCm2 = (1-possm2)/taupo
    s1m2 = s1om2*fcajm2 
    k1m2 = k1om2*fcajm2 
    s2m2 = s1m2*(k2m2/k1m2)*(r1m2/r2m2)
    s2pm2 = s1pm2*(k2pm2/k1pm2)*(r1m2/r2m2)
    k3m2 = np.exp(-Vm_40/3)/(3*(1+np.exp(-Vm_40/3)))
    k3pm2 = k3m2
    k5m2 = (1-PsLCCm2)/tauCajm2
    k6m2 = (fcajm2*PsLCCm2)/tauCajm2
    k5pm2 = (1-PsLCCm2)/tauBam2
    k5m2 = k5m2/recoveryReduc      # reduced for recovery
    k5pm2 = k5pm2/recoveryReduc    # reduced for recovery    
    k6pm2 = PsLCCm2/tauBam2
    k4m2 = k3m2*(alphaLCCm2/betaLCCm2)*(k1m2/k2m2)*(k5m2/k6m2)
    k4pm2 = k3pm2*(alphaLCCm2/betaLCCm2)*(k1pm2/k2pm2)*(k5pm2/k6pm2)

    ### State transitions for MODE 2 junctional LCCs ###
    # O = no differential C2 = 66 C1 = 67 I1Ca = 68 I2Ca = 69 I1Ba = 70 I2Ba = 71
    dPc2_LCCj_m2 = betaLCCm2*Pc1_LCCj_m2 + k5m2*Pi2Ca_LCCj_m2 + k5pm2*Pi2Ba_LCCj_m2 - (k6m2+k6pm2+alphaLCCm2)*Pc2_LCCj_m2                          # C2_m2j
    dPc1_LCCj_m2 = alphaLCCm2*Pc2_LCCj_m2 + k2m2*Pi1Ca_LCCj_m2 + k2pm2*Pi1Ba_LCCj_m2 + r2m2*Po_LCCj_m2 - (r1m2+betaLCCm2+k1m2+k1pm2)*Pc1_LCCj_m2   # C1_m2j
    dPi1Ca_LCCj_m2 = k1m2*Pc1_LCCj_m2 + k4m2*Pi2Ca_LCCj_m2 + s1m2*Po_LCCj_m2 - (k2m2+k3m2+s2m2)*Pi1Ca_LCCj_m2                                  # I1Ca_m2j
    dPi2Ca_LCCj_m2 = k3m2*Pi1Ca_LCCj_m2 + k6m2*Pc2_LCCj_m2 - (k4m2+k5m2)*Pi2Ca_LCCj_m2                                                         # I2Ca_m2j
    dPi1Ba_LCCj_m2 = k1pm2*Pc1_LCCj_m2 + k4pm2*Pi2Ba_LCCj_m2 + s1pm2*Po_LCCj_m2 - (k2pm2+k3pm2+s2pm2)*Pi1Ba_LCCj_m2                            # I1Ba_m2j
    dPi2Ba_LCCj_m2 = k3pm2*Pi1Ba_LCCj_m2 + k6pm2*Pc2_LCCj_m2 - (k5pm2+k4pm2)*Pi2Ba_LCCj_m2                                                     # I2Ba_m2j

    ibarca_jm2 = (4*pCa*VFoRT_Frdy)*(.001*VFoRT_2exp-0.341*Cao)/Ca_denom
    I_Ca_junc_m2 = (Fjunc_CaL*ibarca_jm2*(Po_LCCj_m2)*const_exponent)*ICa_scale

    ### CaMKII AND PKA-DEPENDENT SHIFTING OF DYADIC LCCS TO MODE 2 ####
    #fpkam2 = 0.1543*LCCa_PKAp - .0043  # Assumes max phosphorylation results in 15# mode 2 channels
    fpkam2 = 0.15*(LCCa_PKAp - fracLCCapo)/(1 - fracLCCapo) # Assumes max phosphorylation results in 15# mode 2 channels
    if fpkam2 < 0:
        fpkam2 = 0
    fckiim2 = LCC_CKdyadp_frac*0.1 # Assumes max phosphorylation results in 10# mode 2 channels
    # Sum up total fraction of CKII and PKA-shifted mode 2 channels
    junc_mode2 = fckiim2 + fpkam2
    # Total junctional ICa
    I_Ca_junc2 = (1-junc_mode2)*I_Ca_junc_m1 + junc_mode2*I_Ca_junc_m2

    ### SUB-SARCOLEMMAL LCCs ###

    # Re-assign necessary params to be Casl sensitive
    fcasl = 1/(1+(kop/caslLCC)**3)    # Depends on sl Ca
    #TCasl = (tca + 0.1*(1+(caslLCC/cpbar))**2)/(1+(caslLCC/cpbar)**2) # Error
    TCasl = (tca + 0.1*(1+(caslLCC/cpbar)**2))/(1+(caslLCC/cpbar)**2)
    tauCasl = (Rv-TCasl)*PrLCC + TCasl

    # Re-assign necessary rates to be Casl sensitive
    s1sl = s1o*fcasl
    k1sl = k1o*fcasl
    s2sl = s1sl*(k2/k1sl)*(r1/r2)
    s2psl = s1p*(k2p/k1p)*(r1/r2)
    k5sl = (1-PsLCC)/tauCasl
    k5sl = k5sl/recoveryReduc  # Reduced for recovery
    k6sl = (fcasl*PsLCC)/tauCasl
    k4sl = k3*(alphaLCC/betaLCC)*(k1sl/k2)*(k5sl/k6sl)
    k4psl = k3p*(alphaLCC/betaLCC)*(k1p/k2p)*(k5p/k6p)

    # State transitions for 'mode 1' sarcolemmal LCCs
    # O = no differential C2 = 72 C1 = 73 I1Ca = 74 I2Ca = 75 I1Ba = 76 I2Ba = 77
    dPc2_LCCsl_m1 = betaLCC*Pc1_LCCsl_m1 + k5sl*Pi2Ca_LCCsl_m1 + k5p*Pi2Ba_LCCsl_m1 - (k6sl+k6p+alphaLCC)*Pc2_LCCsl_m1                      # C2_m1sl
    dPc1_LCCsl_m1 = alphaLCC*Pc2_LCCsl_m1 + k2*Pi1Ca_LCCsl_m1 + k2p*Pi1Ba_LCCsl_m1 + r2*Po_LCCsl_m1 - (r1+betaLCC+k1sl+k1p)*Pc1_LCCsl_m1    # C1_m1sl
    dPi1Ca_LCCsl_m1 = k1sl*Pc1_LCCsl_m1 + k4sl*Pi2Ca_LCCsl_m1 + s1sl*Po_LCCsl_m1 - (k2+k3+s2sl)*Pi1Ca_LCCsl_m1                         # I1Ca_m1sl
    dPi2Ca_LCCsl_m1 = k3*Pi1Ca_LCCsl_m1 + k6sl*Pc2_LCCsl_m1 - (k4sl+k5sl)*Pi2Ca_LCCsl_m1                                               # I2Ca_m1sl
    dPi1Ba_LCCsl_m1 = k1p*Pc1_LCCsl_m1 + k4psl*Pi2Ba_LCCsl_m1 + s1p*Po_LCCsl_m1 - (k2p+k3p+s2psl)*Pi1Ba_LCCsl_m1                       # I1Ba_m1sl
    dPi2Ba_LCCsl_m1 = k3p*Pi1Ba_LCCsl_m1 + k6p*Pc2_LCCsl_m1 - (k5p+k4psl)*Pi2Ba_LCCsl_m1                                               # I2Ba_m1sl

    ibarca_slm1 = (4*pCa*VFoRT_Frdy)*(.001*VFoRT_2exp-0.341*Cao)/Ca_denom
    I_Casl_m1 = (Fsl_CaL*ibarca_slm1*Po_LCCsl_m1*const_exponent)*ICa_scale

    # Adjust closing rate for 'mode 2' sarcolemmal LCCs
    r2slm2 = r2m2
    s2slm2 = s1sl*(k2/k1sl)*(r1/r2slm2)
    s2pslm2 = s1p*(k2p/k1p)*(r1/r2slm2)

    ### State transitions for mode 2 sarcolemmal LCCs
    # O = no differential C2 = 78 C1 = 79 I1Ca = 80 I2Ca = 81 I1Ba = 82 I2Ba = 83
    dPc2_LCCsl_m2 = betaLCC*Pc1_LCCsl_m2 + k5sl*Pi2Ca_LCCsl_m2 + k5p*Pi2Ba_LCCsl_m2 - (k6sl+k6p+alphaLCC)*Pc2_LCCsl_m2                      # C2_m2sl
    dPc1_LCCsl_m2 = alphaLCC*Pc2_LCCsl_m2 + k2*Pi1Ca_LCCsl_m2 + k2p*Pi1Ba_LCCsl_m2 + r2slm2*Po_LCCsl_m2 - (r1+betaLCC+k1sl+k1p)*Pc1_LCCsl_m2# C1_m2sl
    dPi1Ca_LCCsl_m2 = k1sl*Pc1_LCCsl_m2 + k4sl*Pi2Ca_LCCsl_m2 + s1sl*Po_LCCsl_m2 - (k2+k3+s2slm2)*Pi1Ca_LCCsl_m2                       # I1Ca_m2sl
    dPi2Ca_LCCsl_m2 = k3*Pi1Ca_LCCsl_m2 + k6sl*Pc2_LCCsl_m2 - (k4sl+k5sl)*Pi2Ca_LCCsl_m2                                               # I2Ca_m2sl
    dPi1Ba_LCCsl_m2 = k1p*Pc1_LCCsl_m2 + k4psl*Pi2Ba_LCCsl_m2 + s1p*Po_LCCsl_m2 - (k2p+k3p+s2pslm2)*Pi1Ba_LCCsl_m2                     # I1Ba_m2sl
    dPi2Ba_LCCsl_m2 = k3p*Pi1Ba_LCCsl_m2 + k6p*Pc2_LCCsl_m2 - (k5p+k4psl)*Pi2Ba_LCCsl_m2                                               # I2Ba_m2sl

    ibarca_slm2 = (4*pCa*VFoRT_Frdy)*(.001*VFoRT_2exp-0.341*Cao)/Ca_denom
    I_Casl_m2 = (Fsl_CaL*ibarca_slm2*Po_LCCsl_m2*const_exponent)*ICa_scale

    # Sum mode 1 and mode 2 sl channels for total sl current
    fckiim2_sl = 0 # Set to zero since SL LCCp by CaMKII is negligible
    sl_mode2 = fckiim2_sl + fpkam2
    I_Ca_sl2 = (1-sl_mode2)*I_Casl_m1 + sl_mode2*I_Casl_m2 

    # Na and K currents through LCC
    I_CaKj2 = ibark*Fjunc_CaL*((1-junc_mode2)*Po_LCCj_m1 + junc_mode2*Po_LCCj_m2)*const_exponent*ICa_scale 
    I_CaKsl2 = ibark*Fsl_CaL*((1-sl_mode2)*Po_LCCsl_m1 + sl_mode2*Po_LCCsl_m2)*const_exponent*ICa_scale
    I_CaK2 = I_CaKj2+I_CaKsl2
    I_CaNa_junc2 = (Fjunc_CaL*ibarna_j*((1-junc_mode2)*Po_LCCj_m1+junc_mode2*Po_LCCj_m2)*const_exponent)*ICa_scale
    I_CaNa_sl2 = Fsl_CaL*ibarna_sl*((1-sl_mode2)*Po_LCCsl_m1 + sl_mode2*Po_LCCsl_m2)*const_exponent*ICa_scale

    # These are now able to switch depending on whether or not the flag to
    # switch to Markov model of ICa is ON
    I_Ca_junc = (1-flagMica)*I_Ca_junc1 + flagMica*I_Ca_junc2
    I_Ca_sl = (1-flagMica)*I_Ca_sl1 + flagMica*I_Ca_sl2
    I_Ca = I_Ca_junc+I_Ca_sl   # Total Ca curren throuhgh LCC
    I_CaNa_junc = (1-flagMica)*(I_CaNa_junc1) + (flagMica)*(I_CaNa_junc2)
    I_CaNa_sl = (1-flagMica)*(I_CaNa_sl1) + (flagMica)*(I_CaNa_sl2)
    I_CaNa = I_CaNa_junc + I_CaNa_sl   # Total Na current through LCC
    I_CaK = (1-flagMica)*(I_CaK1) + flagMica*(I_CaK2)  # Total K current through LCC

    # Collect all currents through LCC
    I_Catot = I_Ca+I_CaK+I_CaNa

    # output LTCC module ODE
    ydot[59:65]=[dPc2_LCCj_m1, dPc1_LCCj_m1, dPi1Ca_LCCj_m1, dPi2Ca_LCCj_m1, dPi1Ba_LCCj_m1, dPi2Ba_LCCj_m1] #6
    ydot[65:71]=[dPc2_LCCj_m2, dPc1_LCCj_m2, dPi1Ca_LCCj_m2, dPi2Ca_LCCj_m2, dPi1Ba_LCCj_m2, dPi2Ba_LCCj_m2] #6
    ydot[71:77]=[dPc2_LCCsl_m1, dPc1_LCCsl_m1, dPi1Ca_LCCsl_m1, dPi2Ca_LCCsl_m1, dPi1Ba_LCCsl_m1, dPi2Ba_LCCsl_m1] #6
    ydot[77:83]=[dPc2_LCCsl_m2, dPc1_LCCsl_m2, dPi1Ca_LCCsl_m2, dPi2Ca_LCCsl_m2, dPi1Ba_LCCsl_m2, dPi2Ba_LCCsl_m2] #6
    
    ## I_ncx: Na/Ca Exchanger flux

    Ka_junc = 1/(1+(Kdact/Caj)**2)
    Ka_sl = 1/(1+(Kdact/Casl)**2)
    s1_junc = np.exp(nu*VFoRT)*Naj**3*Cao
    s1_sl = np.exp(nu*VFoRT)*Nasl**3*Cao
    s2_junc = np.exp((nu-1)*VFoRT)*Nao3*Caj
    s2_sl = np.exp((nu-1)*VFoRT)*Nao3*Casl
    # Shannon
    #s3_junc = KmCai*Nao3*(1+(Naj/KmNai)**3) + KmNao**3*Caj*(1+Caj/KmCai)               + KmCao*Naj**3 + Naj**3*Cao + Nao3*Caj
    #s3_sl =   KmCai*Nao3*(1+(Nasl/KmNai)**3) + KmNao**3*Casl*(1+Casl/KmCai)               + KmCao*Nasl**3 + Nasl**3*Cao + Nao3*Casl
    # Soltis (Weber 2001) - version used in Rabbit and Mouse
    s3_junc = KmCai*Nao3*(1+(Naj/KmNai)**3) + KmNao**3*Caj + KmNai**3*Cao*(1+Caj/KmCai) + KmCao*Naj**3 + Naj**3*Cao + Nao3*Caj
    s3_sl =   KmCai*Nao3*(1+(Nasl/KmNai)**3) + KmNao**3*Casl + KmNai**3*Cao*(1+Casl/KmCai) + KmCao*Nasl**3 + Nasl**3*Cao + Nao3*Casl

    I_ncx_junc = Fjunc*IbarNCX*Q10NCX**Qpow*Ka_junc*(s1_junc-s2_junc)/s3_junc/(1+ksat*np.exp((nu-1)*VFoRT))
    I_ncx_sl = Fsl*IbarNCX*Q10NCX**Qpow*Ka_sl*(s1_sl-s2_sl)/s3_sl/(1+ksat*np.exp((nu-1)*VFoRT))
    I_ncx = I_ncx_junc+I_ncx_sl
    ## I_pca: Sarcolemmal Ca Pump Current

    I_pca_junc = Fjunc*Q10SLCaP**Qpow*IbarSLCaP*Caj**1.6/(KmPCa**1.6+Caj**1.6)
    I_pca_sl = Fsl*Q10SLCaP**Qpow*IbarSLCaP*Casl**1.6/(KmPCa**1.6+Casl**1.6)
    I_pca = I_pca_junc+I_pca_sl
    ## I_cabk: Ca Background Current

    I_cabk_junc = Fjunc*GCaB*(Vm-eca_junc)
    I_cabk_sl = Fsl*GCaB*(Vm-eca_sl)
    I_cabk = I_cabk_junc+I_cabk_sl
    ## I_ClCa: Ca-activated Cl Current, I_Clbk: background Cl Current

    # PKA-dependent IClCa phosphoregulation
    kPKA_IClCa = (IClCa_PKAp-fracPKA_IClCao)/(fracPKA_IClCaiso-fracPKA_IClCao)
    KdClCa = (1+(0.704-1)*kPKA_IClCa)*KdClCa # -29.6# w/ 100 nM ISO

    I_ClCa_junc = Fjunc*GClCa/(1+KdClCa/Caj)*(Vm-ecl)
    I_ClCa_sl = Fsl*GClCa/(1+KdClCa/Casl)*(Vm-ecl)
    I_ClCa = I_ClCa_junc+I_ClCa_sl

    I_Clbk = GClB*(Vm-ecl)
    ## I_CFTR or I_cl_(cAMP): Cystic Fibrosis Transmembrane Conductance Reg.

    # no ICFTR in human ventricular myocytes
    #gCFTR = 4.9e-3 # [A/F] - Max value as in Shannon et al. (2005) with max phosphorylation
    
    ## SR fluxes: Calcium Release, SR Ca pump, SR Ca leak
    # MaxSR = 15
    # MinSR = 1
    kCaSR = 15 - (15-1)/(1+(ec50SR/Ca_sr)**2.5)
    koSRCa = koCa/kCaSR
    kiSRCa = kiCa*kCaSR

    ### CaMKII and PKA-dependent phosphoregulation of RyR Po ###
    fCKII_RyR = (20*RyR_CKp/3 - 1/3)
    #fPKA_RyR = RyR_PKAp*1.025 + 0.9750
    fPKA_RyR = 1 + (RyR_PKAp-fracRyRpo) / (1-fracRyRpo) # 2 with max phosphorylation
    koSRCa = (fCKII_RyR + fPKA_RyR - 1)*koSRCa

    # ODEs for RyR states and SR release through open RyRs
    RI = 1-RyRr-RyRo-RyRi
    ydot[13] = (kim*RI-kiSRCa*Caj*RyRr)-(koSRCa*Caj**2*RyRr-kom*RyRo)   # R
    ydot[14] = (koSRCa*Caj**2*RyRr-kom*RyRo)-(kiSRCa*Caj*RyRo-kim*RyRi)# O
    ydot[15] = (kiSRCa*Caj*RyRo-kim*RyRi)-(kom*RyRi-koSRCa*Caj**2*RI)   # I
    J_SRCarel = ks*RyRo*(Ca_sr-Caj)          # [mM/ms]

    # Passive RyR leak - includes CaMKII regulation of leak flux
    kleak = (1/3 + 10*RyR_CKp/3)*kleak
    J_SRleak = kleak*(Ca_sr-Caj)           #   [mM/ms] 
    ## SERCA model - SR Ca uptake fluxes

    # CaMKII and PKA-dependent phosphoregulation of PLB (changes to SERCA flux)
    fCKII_PLB = (1-.5*PLB_CKp)
    #fPKA_PLB = (PLB_PKAn/fracPKA_PLBo)*3/4 + 1/4 # 0.25 with max PKA phosphorylation
    fPKA_PLB = (PLB_PKAn/fracPKA_PLBo)*(100-55.31)/100 + 55.31/100 # Max effect: fPKA_PLB=0.45 PROVA

    # Select smaller value (resulting in max reduction of Kmf)
    Kmf = np.where((fCKII_PLB < fPKA_PLB),Kmf*fCKII_PLB,Kmf*fPKA_PLB)
    
    # if fCKII_PLB < fPKA_PLB:
    #     Kmf = Kmf*fCKII_PLB
    # else:
    #     Kmf = Kmf*fPKA_PLB

    J_serca = 1*Q10SRCaP**Qpow*Vmax_SRCaP*((Cai/Kmf)**hillSRCaP-(Ca_sr/Kmr)**hillSRCaP)/(1+(Cai/Kmf)**hillSRCaP+(Ca_sr/Kmr)**hillSRCaP)
    ## Myofilament

    if myoFlag == 1:
        # input contractile module ODE (6 state vars)
        # 53    54       55         56     57   58
        #TSCa,TSCa_star,TSCa_tilde,TS_star,L_p,L_w  

        # ###################################
        # if t>150 && t<=170,
        #     Liso = Liso-(t-150)*(0.05/20)
        # elseif t>170,
        #     Liso = Liso-(170-150)*(0.05/20)
        # end
        # ###################################
        Lm=1.05*mechFlag+Liso*(1-mechFlag)
        Fm=0.87*mechFlag+0.87*(1-mechFlag)  
        con=0
        corL=10
        L=1.03
        while abs(corL)>.00001:
            if mechFlag==0:
                Fm=alfa*(np.exp(bet*(Lm-L))-1)
            FB=Ap*(TSCa_star+TS_star)*(L-L_p)+Aw*TSCa_tilde*(L-L_w) 
            w=FB+Ke*(L-Lz)**5+Le*(L-Lz)-Fm
            w1=Ap*(TSCa_star+TS_star)+Aw*TSCa_tilde+5*Ke*(L-Lz)**4+Le+bet*(Fm+alfa)
            corL=-w/w1
            L=L+.1*corL 
            con=con+1
            if con>200:
                break
        TS=TSt-TSCa-TSCa_star-TSCa_tilde-TS_star
        ER=np.exp(-RLa*(L-La)**2) 
        Yh=Yv*(1-np.exp(-gama*(L-L_w-hwr)**2))
        if (L-L_w)>hwr:
            Yh=Fh*Yh
        fa=f*ER
        ga=Za+Yh
        gd=Yd*np.exp(-Yc*(L-Lc)) 
        dTSCa=ga*TSCa_tilde-fa*TSCa+Yb*TS*Cai**nc-Zb*TSCa
        dTSCa_star=Zr*TS_star*Cai**nc-Yr*TSCa_star+Yp*TSCa_tilde-Zp*TSCa_star
        dTSCa_tilde=Zp*TSCa_star-Yp*TSCa_tilde+fa*TSCa-ga*TSCa_tilde
        dTS_star=-gd*TS_star+Yr*TSCa_star-Zr*TS_star*Cai**nc
        dL_p=Bp*(L-L_p-hpr)
        dL_w=Bw*(L-L_w-hwr)
        if mechFlag==1:
            Lm=L+np.log((Fm+alfa)/alfa)/bet

        # output contractile module ODE (dTSCa dTSCa_star dTSCa_tilde -> Ca buffer)
        ydot[53:59] = [dTSCa, dTSCa_star, dTSCa_tilde, dTS_star, dL_p, dL_w]
        ydot[18] = nc*(dTSCa+dTSCa_star+dTSCa_tilde) # myofilament
    else:
        Lm = 0
        Fm = 0
        ydot[18] = kon_tncl*Cai*(Bmax_TnClow-TnCL)-koff_tncl*TnCL # TnCL      [mM/ms]

    ## Sodium and Calcium Buffering

    # PKA-dependent phosphoregulation of TnI (increases Kd of TnC)
    fPKA_TnI = (1.45-0.45*(1-TnI_PKAp)/(1-fracTnIpo)) # 1.45 with maximal phosphorylation
    koff_tncl = koff_tncl*fPKA_TnI

    # PKA/CaMKII-dependent phosphoregulation of SERCA (NEW)
    # Select smaller value (as for Kmf, see SERCA module)
    koff_sr = np.where((fCKII_PLB < fPKA_PLB),koff_sr*fCKII_PLB,koff_sr*fPKA_PLB)
    
    # if fCKII_PLB < fPKA_PLB:
    #     koff_sr = koff_sr*fCKII_PLB
    # else:
    #     koff_sr = koff_sr*fPKA_PLB

    # Na Buffers
    ydot[16] = kon_na*Naj*(Bmax_Naj-NaBj)-koff_na*NaBj        # NaBj      [mM/ms]
    ydot[17] = kon_na*Nasl*(Bmax_Nasl-NaBsl)-koff_na*NaBsl       # NaBsl     [mM/ms]

    # Cytosolic Ca Buffers
    # if myoFlag == 1:
    #     ydot[18] = nc*(dTSCa+dTSCa_star+dTSCa_tilde) # myofilament
    # else:
    #     ydot[18] = kon_tncl*Cai*(Bmax_TnClow-TnCL)-koff_tncl*TnCL # TnCL      [mM/ms]
    ydot[19] = kon_tnchca*Cai*(Bmax_TnChigh-TnCHc-TnCHm)-koff_tnchca*TnCHc # TnCHc     [mM/ms]
    ydot[20] = kon_tnchmg*Mgi*(Bmax_TnChigh-TnCHc-TnCHm)-koff_tnchmg*TnCHm   # TnCHm     [mM/ms]
    ydot[21] = 0# *** commented b/c buffering done by CaM module 
    #ydot[21] = kon_cam*Cai*(Bmax_CaM-CaM)-koff_cam*CaM                 # CaM       [mM/ms]
    ydot[22] = kon_myoca*Cai*(Bmax_myosin-Myoc-Myom)-koff_myoca*Myoc    # Myosin_ca [mM/ms]
    ydot[23] = kon_myomg*Mgi*(Bmax_myosin-Myoc-Myom)-koff_myomg*Myom      # Myosin_mg [mM/ms]
    ydot[24] = kon_sr*Cai*(Bmax_SR-SRB)-koff_sr*SRB                    # SRB       [mM/ms]
    #J_CaB_cytosol = sum(ydot(19:25)) # wrong formulation
    J_CaB_cytosol = ydot[18]+ydot[19]+ydot[21]+ydot[22]+ydot[24]

    # Junctional and SL Ca Buffers
    ydot[25] = kon_sll*Caj*(Bmax_SLlowj-SLLj)-koff_sll*SLLj       # SLLj      [mM/ms]
    ydot[26] = kon_sll*Casl*(Bmax_SLlowsl-SLLsl)-koff_sll*SLLsl      # SLLsl     [mM/ms]
    ydot[27] = kon_slh*Caj*(Bmax_SLhighj-SLHj)-koff_slh*SLHj      # SLHj      [mM/ms]
    ydot[28] = kon_slh*Casl*(Bmax_SLhighsl-SLHsl)-koff_slh*SLHsl     # SLHsl     [mM/ms]
    J_CaB_junction = ydot[25]+ydot[27]
    J_CaB_sl = ydot[26]+ydot[28]
    ## Ion concentrations

    # SR Ca Concentrations
    ydot[29] = kon_csqn*Ca_sr*(Bmax_Csqn-Csqnb)-koff_csqn*Csqnb       # Csqn      [mM/ms]
    ydot[30] = J_serca-(J_SRleak*Vmyo/Vsr+J_SRCarel)-ydot[29]         # Ca_sr     [mM/ms] #Ratio 3 leak current

    # Sodium Concentrations
    I_Na_tot_junc = I_Na_junc+I_nabk_junc+3*I_ncx_junc+3*I_nak_junc+I_CaNa_junc   # [uA/uF]
    I_Na_tot_sl = I_Na_sl+I_nabk_sl+3*I_ncx_sl+3*I_nak_sl+I_CaNa_sl   # [uA/uF]
    ydot[31] = -I_Na_tot_junc*Cmem/(Vjunc*Frdy)+J_na_juncsl/Vjunc*(Nasl-Naj)-ydot[16]
    ydot[32] = -I_Na_tot_sl*Cmem/(Vsl*Frdy)+J_na_juncsl/Vsl*(Naj-Nasl)+J_na_slmyo/Vsl*(Nai-Nasl)-ydot[17]
    ydot[33] = np.where(Na_clamp == 1,0,J_na_slmyo/Vmyo*(Nasl-Nai) )            # [mM/msec] 

    # if Na_clamp == 1:
    #     ydot[33] = 0

    # Potassium Concentration
    I_K_tot = I_to+I_kr+I_ks+I_ki-2*I_nak+I_CaK+I_kp     # [uA/uF]
    ydot[34] = 0 #-I_K_tot*Cmem/(Vmyo*Frdy)           # [mM/msec]

    # Calcium Concentrations
    I_Ca_tot_junc = I_Ca_junc+I_cabk_junc+I_pca_junc-2*I_ncx_junc                   # [uA/uF]
    I_Ca_tot_sl = I_Ca_sl+I_cabk_sl+I_pca_sl-2*I_ncx_sl            # [uA/uF]
    ydot[35] = np.where(Ca_clamp == 1,0,(-I_Ca_tot_junc*Cmem/(Vjunc*2*Frdy)+J_ca_juncsl/Vjunc*(Casl-Caj)-J_CaB_junction+(J_SRCarel)*Vsr/Vjunc+J_SRleak*Vmyo/Vjunc)+ 1e-3*JCaDyad ) # Ca_j
    ydot[36] = np.where(Ca_clamp == 1,0,(-I_Ca_tot_sl*Cmem/(Vsl*2*Frdy)+J_ca_juncsl/Vsl*(Caj-Casl)+ J_ca_slmyo/Vsl*(Cai-Casl)-J_CaB_sl)+ 1e-3*JCaSL  )  # Ca_sl
    ydot[37] = np.where(Ca_clamp == 1,0, (-J_serca*Vsr/Vmyo-J_CaB_cytosol +J_ca_slmyo/Vmyo*(Casl-Cai))+ 1e-3*JCaCyt)

    # if Ca_clamp == 1:
    #     ydot[35] = 0 
    #     ydot[36] = 0 
    #     ydot[37] = 0 
    ## Simulation type

    #switch prot_index
    I_app = I_stimulus(t,cycleLength,stimDur,amp)
            
    ## Membrane Potential

    I_Na_tot = I_Na_tot_junc + I_Na_tot_sl          # [uA/uF]
    I_Cl_tot = I_ClCa + I_Clbk + Icftr              # [uA/uF]
    I_Ca_tot = I_Ca_tot_junc + I_Ca_tot_sl
    I_tot = I_Na_tot + I_Cl_tot + I_Ca_tot + I_K_tot

    ydot[38] = -(I_tot-I_app)
    

    # incorporate CaM diffusion between compartments
    kSLmyo = 8.587e-15     # [L/msec]
    k0Boff = 0.0014        # [s**-1] 
    k0Bon = k0Boff/0.2     # [uM**-1 s**-1] kon = koff/Kd
    k2Boff = k0Boff/100    # [s**-1] 
    k2Bon = k0Bon          # [uM**-1 s**-1]
    k4Bon = k0Bon          # [uM**-1 s**-1]
    # Live total of all bound CaM species in dyad (matches C# MasterOde.cs:338-342).
    # CaMKII subunit forms (Pb2, Pb, Pt, Pt2) are weighted by CaMKIItotDyad together.
    CaMtotDyad_live = (CaM_dyad + Ca2CaM_dyad + Ca4CaM_dyad
                       + CaMB_dyad + Ca2CaMB_dyad + Ca4CaMB_dyad
                       + CaMKIItotDyad*(Pb2_dyad + Pb_dyad + Pt_dyad + Pt2_dyad)
                       + CaMCa4CaN_dyad + Ca2CaMCa4CaN_dyad + Ca4CaMCa4CaN_dyad)
    Bdyad = BtotDyad - CaMtotDyad_live  # [uM dyad] free buffer
    J_cam_dyadSL = 1e-3*(k0Boff*CaM_dyad - k0Bon*Bdyad*CaM_sl) # [uM/msec dyad]
    J_ca2cam_dyadSL = 1e-3*(k2Boff*Ca2CaM_dyad - k2Bon*Bdyad*Ca2CaM_sl) # [uM/msec dyad]
    J_ca4cam_dyadSL = 1e-3*(k2Boff*Ca4CaM_dyad - k4Bon*Bdyad*Ca4CaM_sl) # [uM/msec dyad]
    J_cam_SLmyo = kSLmyo*(CaM_sl-CaM_cyt) # [umol/msec]
    J_ca2cam_SLmyo = kSLmyo*(Ca2CaM_sl-Ca2CaM_cyt) # [umol/msec]
    J_ca4cam_SLmyo = kSLmyo*(Ca4CaM_sl-Ca4CaM_cyt) # [umol/msec]
    # CaM diffusion between compartments (canonical layout indices)
    ydot[83] = ydot[83] - J_cam_dyadSL                                              # CaM_dyad
    ydot[84] = ydot[84] - J_ca2cam_dyadSL                                           # Ca2CaM_dyad
    ydot[85] = ydot[85] - J_ca4cam_dyadSL                                           # Ca4CaM_dyad
    ydot[98] = ydot[98] + J_cam_dyadSL*Vjunc/Vsl - J_cam_SLmyo/Vsl                  # CaM_sl
    ydot[99] = ydot[99] + J_ca2cam_dyadSL*Vjunc/Vsl - J_ca2cam_SLmyo/Vsl            # Ca2CaM_sl
    ydot[100] = ydot[100] + J_ca4cam_dyadSL*Vjunc/Vsl - J_ca4cam_SLmyo/Vsl          # Ca4CaM_sl
    ydot[113] = ydot[113] + J_cam_SLmyo/Vmyo                                        # CaM_cyt
    ydot[114] = ydot[114] + J_ca2cam_SLmyo/Vmyo                                     # Ca2CaM_cyt
    ydot[115] = ydot[115] + J_ca4cam_SLmyo/Vmyo                                     # Ca4CaM_cyt

    return(ydot)

##### cam ode file, no errors here
@njit
def HumanVentricularMyocyte_camODEfile(t,CaM,Ca2CaM,Ca4CaM,CaMB,Ca2CaMB,Ca4CaMB,Pb2,Pb,Pt,Pt2,Pa,Ca4CaN,CaMCa4CaN,Ca2CaMCa4CaN,Ca4CaMCa4CaN,params,K,Mg):
    # This function describes the ODE's for CaM, CaMKII, and CaN

    ## State variables

    #CaM,Ca2CaM,Ca4CaM,CaMB,Ca2CaMB,Ca4CaMB,Pb2,Pb,Pt,Pt2,Pa,Ca4CaN,CaMCa4CaN,Ca2CaMCa4CaN,Ca4CaMCa4CaN = y
    
    # CaM = y[0] # Ca-free CaM
    # Ca2CaM = y[1] # 2 Ca bound to C terminal sites
    # Ca4CaM = y[2] # 4 Ca bound
    # CaMB = y[3]
    # Ca2CaMB = y[4]
    # Ca4CaMB = y[5]
    # Pb2 = y[6] # probability of a Ca2CaM bound CaMKII subunit
    # Pb = y[7] # probability of a Ca4CaM bound CaMKII subunit
    # Pt = y[8] # probability of a Ca4CaM bound autophosphorylated CaMKII subunit
    # Pt2 = y[9] # probability of a Ca2CaM bound autophosphorylated CaMKII subunit
    # Pa = y[10] # probability of an autonomous autophosphorylated CaMKII subunit
    # Ca4CaN = y[11]
    # CaMCa4CaN = y[12]
    # Ca2CaMCa4CaN = y[13]
    # Ca4CaMCa4CaN = y[14] # active calcineurin
    
    ## Parameters

    CaMtot = params[0]
    Btot = params[1]
    CaMKIItot = params[2]
    CaNtot = params[3]
    PP1tot = params[4]
    Ca = params[5]

    # Ca/CaM parameters
    if Mg <= 1:
        Kd02 = 0.0025*(1+K/0.94-Mg/0.012)*(1+K/8.1+Mg/0.022)  # [uM**2]
        Kd24 = 0.128*(1+K/0.64+Mg/0.0014)*(1+K/13.0-Mg/0.153) # [uM**2]
    else:
        Kd02 = 0.0025*(1+K/0.94-1/0.012+(Mg-1)/0.060)*(1+K/8.1+1/0.022+(Mg-1)/0.068)   # [uM**2]
        Kd24 = 0.128*(1+K/0.64+1/0.0014+(Mg-1)/0.005)*(1+K/13.0-1/0.153+(Mg-1)/0.150)  # [uM**2]
    k20 = 10               # [s**-1]      
    k02 = k20/Kd02         # [uM**-2 s**-1]
    k42 = 500              # [s**-1]      
    k24 = k42/Kd24         # [uM**-2 s**-1]

    # CaM buffering (B) parameters
    k0Boff = 0.0014        # [s**-1] 
    k0Bon = k0Boff/0.2   # [uM**-1 s**-1] kon = koff/Kd
    k2Boff = k0Boff/100    # [s**-1] 
    k2Bon = k0Bon          # [uM**-1 s**-1]
    k4Boff = k2Boff        # [s**-1]
    k4Bon = k0Bon          # [uM**-1 s**-1]
    # using thermodynamic constraints
    k20B = k20/100 # [s**-1] thermo constraint on loop 1
    k02B = k02     # [uM**-2 s**-1] 
    k42B = k42     # [s**-1] thermo constraint on loop 2
    k24B = k24     # [uM**-2 s**-1]

    # CaMKII parameters
    # Wi Wa Wt Wp
    kbi = 2.2      # [s**-1] (Ca4CaM dissocation from Wb)
    kib = kbi/33.5e-3 # [uM**-1 s**-1]
    kib2 = kib
    kb2i = kib2*5
    kpp1 = 1.72    # [s**-1] (PP1-dep dephosphorylation rates)
    Kmpp1 = 11.5   # [uM]
    kta = kbi/1000 # [s**-1] (Ca4CaM dissociation from Wt)
    kat = kib      # [uM**-1 s**-1] (Ca4CaM reassociation with Wa)
    kt42 = k42*33.5e-6/5
    kt24 = k24
    kat2 = kib
    kt2a = kib*5
    
    
    kb24 = k24
    kb42 = k42*33.5e-3/5
    

    # CaN parameters
    kcanCaoff = 1              # [s**-1] 
    kcanCaon = kcanCaoff/0.5   # [uM**-1 s**-1] 
    kcanCaM4on = 46            # [uM**-1 s**-1]
    kcanCaM4off = 1.3e-3       # [s**-1]
    kcanCaM2on = kcanCaM4on
    kcanCaM2off = 2508*kcanCaM4off
    kcanCaM0on = kcanCaM4on
    kcanCaM0off = 165*kcanCaM2off
    k02can = k02
    k20can = k20/165
    k24can = k24
    k42can = k20/2508

    # CaM Reaction fluxes
    rcn02 = k02*Ca**2*CaM - k20*Ca2CaM
    rcn24 = k24*Ca**2*Ca2CaM - k42*Ca4CaM
    # CaM buffer fluxes
    B = Btot - CaMB - Ca2CaMB - Ca4CaMB
    rcn02B = k02B*Ca**2*CaMB - k20B*Ca2CaMB
    rcn24B = k24B*Ca**2*Ca2CaMB - k42B*Ca4CaMB
    rcn0B = k0Bon*CaM*B - k0Boff*CaMB
    rcn2B = k2Bon*Ca2CaM*B - k2Boff*Ca2CaMB
    rcn4B = k4Bon*Ca4CaM*B - k4Boff*Ca4CaMB
    # CaN reaction fluxes 
    Ca2CaN = CaNtot - Ca4CaN - CaMCa4CaN - Ca2CaMCa4CaN - Ca4CaMCa4CaN
    rcnCa4CaN = kcanCaon*Ca**2*Ca2CaN - kcanCaoff*Ca4CaN
    rcn02CaN = k02can*Ca**2*CaMCa4CaN - k20can*Ca2CaMCa4CaN 
    rcn24CaN = k24can*Ca**2*Ca2CaMCa4CaN - k42can*Ca4CaMCa4CaN
    rcn0CaN = kcanCaM0on*CaM*Ca4CaN - kcanCaM0off*CaMCa4CaN
    rcn2CaN = kcanCaM2on*Ca2CaM*Ca4CaN - kcanCaM2off*Ca2CaMCa4CaN
    rcn4CaN = kcanCaM4on*Ca4CaM*Ca4CaN - kcanCaM4off*Ca4CaMCa4CaN
    # CaMKII reaction fluxes
    Pi = 1 - Pb2 - Pb - Pt - Pt2 - Pa
    rcnCKib2 = kib2*Ca2CaM*Pi - kb2i*Pb2
    rcnCKb2b = kb24*Ca**2*Pb2 - kb42*Pb
    rcnCKib = kib*Ca4CaM*Pi - kbi*Pb
    T = Pb + Pt + Pt2 + Pa
    kbt = 0.055*T + .0074*T**2 + 0.015*T**3
    rcnCKbt = kbt*Pb - kpp1*PP1tot*Pt/(Kmpp1+CaMKIItot*Pt)
    rcnCKtt2 = kt42*Pt - kt24*Ca**2*Pt2
    rcnCKta = kta*Pt - kat*Ca4CaM*Pa
    rcnCKt2a = kt2a*Pt2 - kat2*Ca2CaM*Pa
    rcnCKt2b2 = kpp1*PP1tot*Pt2/(Kmpp1+CaMKIItot*Pt2)
    rcnCKai = kpp1*PP1tot*Pa/(Kmpp1+CaMKIItot*Pa)
    ## ODEs

    # CaM equations
    dCaM = 1e-3*(-rcn02 - rcn0B - rcn0CaN)
    dCa2CaM = 1e-3*(rcn02 - rcn24 - rcn2B - rcn2CaN + CaMKIItot*(-rcnCKib2 + rcnCKt2a) )
    dCa4CaM = 1e-3*(rcn24 - rcn4B - rcn4CaN + CaMKIItot*(-rcnCKib+rcnCKta) )
    dCaMB = 1e-3*(rcn0B-rcn02B)
    dCa2CaMB = 1e-3*(rcn02B + rcn2B - rcn24B)
    dCa4CaMB = 1e-3*(rcn24B + rcn4B)

    # CaMKII equations
    dPb2 = 1e-3*(rcnCKib2 - rcnCKb2b + rcnCKt2b2) # Pb2
    dPb = 1e-3*(rcnCKib + rcnCKb2b - rcnCKbt)    # Pb
    dPt = 1e-3*(rcnCKbt-rcnCKta-rcnCKtt2)        # Pt
    dPt2 = 1e-3*(rcnCKtt2-rcnCKt2a-rcnCKt2b2)     # Pt2
    dPa = 1e-3*(rcnCKta+rcnCKt2a-rcnCKai)       # Pa

    # CaN equations
    dCa4CaN = 1e-3*(rcnCa4CaN - rcn0CaN - rcn2CaN - rcn4CaN)                       # Ca4CaN
    dCaMCa4CaN = 1e-3*(rcn0CaN - rcn02CaN)           # CaMCa4CaN
    dCa2CaMCa4CaN = 1e-3*(rcn2CaN+rcn02CaN-rcn24CaN)    # Ca2CaMCa4CaN
    dCa4CaMCa4CaN = 1e-3*(rcn4CaN+rcn24CaN)             # Ca4CaMCa4CaN
    
    JCa = 1e-3*(2*CaMKIItot*(rcnCKtt2-rcnCKb2b) - 2*(rcn02+rcn24+rcn02B+rcn24B+rcnCa4CaN+rcn02CaN+rcn24CaN)) # [uM/msec]

    dydt = np.array([dCaM,dCa2CaM,dCa4CaM,dCaMB,dCa2CaMB,dCa4CaMB,dPb2,dPb,dPt,dPt2,dPa,dCa4CaN,dCaMCa4CaN,dCa2CaMCa4CaN,dCa4CaMCa4CaN,JCa])

    return dydt



# intracellular_concs.savefig(os.path.join(final_dir,'intracellular currents.png') )
# current_plots.savefig(os.path.join(final_dir,'currents.png') )
# #Voltage_fig.savefig(os.path.join(final_dir,'voltage.png') )