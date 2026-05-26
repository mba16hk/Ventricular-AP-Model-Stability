# Ten Tusscher-Noble-Noble-Panfilov 2006 (TNNP06) Model
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import ode
import time as tm
import pandas as pd
import os
import math
from numba import njit
from conductances import *


### Different code sources for 2006 model
## C code from https://www-binf.bio.uu.nl/khwjtuss/SourceCodes/HVM2/
## Python code and mathematics from CellML https://models.cellml.org/exposure/de5058f16f829f91a1e4e5990a10ed71/tentusscher_panfilov_2006_a.cellml/@@cellml_codegen/Python


# --- Ito / IKs / IKr gating (inlined from former IK_2006.py) ---
@njit
def r_inf(V):
    return 1 / (1 + np.exp((20 - V) / 6))

@njit
def tau_r(V):
    return 9.5 * np.exp(-((V + 40) ** 2) / 1800) + 0.8

@njit
def s_inf(V, cell_type):
    if cell_type == 'EPI' or cell_type == 'M':
        return 1 / (1 + np.exp((V + 20) / 5))
    elif cell_type == 'ENDO':
        return 1 / (1 + np.exp((V + 28) / 5))

@njit
def tau_s(V, cell_type):
    if cell_type == 'EPI' or cell_type == 'M':
        return 85 * np.exp(-((V + 45) ** 2) / 320) + (5 / (1 + np.exp((V - 20) / 5))) + 3
    elif cell_type == 'ENDO':
        return 1000 * np.exp(-((V + 67) ** 2) / 1000) + 8

@njit
def Xs_inf(V):
    return 1 / (1 + np.exp((-5 - V) / 14))

@njit
def alpha_Xs(V):
    return 1400 / np.sqrt(1 + np.exp((5 - V) / 6))

@njit
def beta_Xs(V):
    return 1 / (1 + np.exp((V - 35) / 15))

@njit
def xr1_inf(V, mutation_K897T):
    if mutation_K897T:
        return 1 / (1 + np.exp((-33 - V) / 6))
    else:
        return 1 / (1 + np.exp((-26 - V) / 7))

@njit
def alpha_xr1(V, mutation_K897T):
    if mutation_K897T:
        return 657.25 / (1.0 + (np.exp(((-34.37 - V) / 8.93))))
    else:
        return 450 / (1 + np.exp((-45 - V) / 10))

@njit
def beta_xr1(V, mutation_K897T):
    if mutation_K897T:
        return 11.04 / (1.0 + (np.exp(((V + 55.11) / 12.8))))
    else:
        return 6 / (1 + np.exp((30 + V) / 11.5))

@njit
def xr2_inf(V, mutation_K897T):
    if mutation_K897T:
        return 1.0 / (1.0 + (np.exp(((V + 90.0) / 23.0))))
    else:
        return 1 / (1 + np.exp((88 + V) / 24))

@njit
def alpha_xr2(V, mutation_K897T):
    if mutation_K897T:
        return 3.0 / (1.0 + (np.exp(((-58.0 - V) / 20.0))))
    else:
        return 3 / (1 + np.exp((-60 - V) / 20))

@njit
def beta_xr2(V, mutation_K897T):
    if mutation_K897T:
        return 1.05 / (1.0 + (np.exp(((V - 59.0) / 22.0))))
    else:
        return 1.12 / (1 + np.exp((V - 60) / 20))


# --- ICaL gating (inlined from former ICa_2006.py) ---
@njit
def alpha_d(V):
    return 1.4 / (1 + np.exp((-35 - V) / 13)) + 0.25

@njit
def beta_d(V):
    return 1.4 / (1 + np.exp((V + 5) / 5))

@njit
def gamma_d(V):
    return 1 / (1 + np.exp((50 - V) / 20))

@njit
def alpha_f(V):
    return 1102.5 * np.exp(-((V + 27) ** 2) / 225)

@njit
def beta_f(V):
    return 200 / (1 + np.exp((13 - V) / 10))

@njit
def gamma_f(V):
    return (180 / (1 + np.exp((V + 30) / 10))) + 20

@njit
def alpha_f2(V):
    return 562 * np.exp(-((V + 27) ** 2) / 240)

@njit
def beta_f2(V):
    return 31 / (1 + np.exp((25 - V) / 10))

@njit
def gamma_f2(V):
    return 80 / (1 + np.exp((V + 30) / 10))

@njit
def tau_fcass(Ca_ss):
    return (80 / (1 + (Ca_ss / 0.05) ** 2)) + 2

@njit
def g_inf(Cai):
    return np.where(Cai <= 0.00035, 1 / (1 + (Cai / 0.00035) ** 6), 1 / (1 + (Cai / 0.00035) ** 16))


@njit
def Istim_TNNP06(time, cycle_length, amplitude, duration):
    if (time % cycle_length <= duration):
        return amplitude
    else:
        return 0


def run_TNNP06_model(cycles, cycle_length, cell_type, user_K_conc, amp=-52):
    model_type = "Ten Tusscher 2006"
    G_Ks       = GKs_conductance(model_type, cell_type)
    G_Kr       = GKr_conductance(model_type, cell_type)
    G_K1       = GK1_conductance(model_type, cell_type)
    G_to       = Gto_conductance(model_type, cell_type)
    G_Na       = GNa_conductance(model_type, cell_type)
    GCa        = GCa_conductance(model_type, cell_type)
    GNCX       = GNCX_conductance(model_type, cell_type)
    GNaK       = GNaK_conductance(model_type, cell_type)
    GKb        = GKb_conductance(model_type, cell_type)
    GNab       = GNab_conductance(model_type, cell_type)
    GCab       = GCab_conductance(model_type, cell_type)
    GpCa_input = GpCa_conductance(model_type, cell_type)
    #Physical constants
    #cell_type = 'epi' # could be 'endo' or 'M'

    # Parameters in Appendix of 2004 paper
    # Cross-checked against a reference C++ implementation
    R=8314.472#units have been changed from publications
    Temp =310#K
    Frdy = 96485.3415#units have been changed from publications
    Cm = 0.185#2#.μFcm−2
    capacitance = 0.185#162#0.185 # This is Cm in the CeLLML model
    VC=0.016404
    VSS=0.00005468
    VSR=0.001094
    amplitude = amp#-52 # this is the value for 2D string of cells, but use -252 for 1D string
    duration = 1 #ms
    pKNa = 0.03
    GNa = G_Na#14.838#nSpF−1 -CC and CellML
    GK1 = G_K1#5.405#nSpF−1 -CC and CellML
    GbNa = GNab#0.00029#nSpF−1 -CC and CellML
    GbCa = GCab#0.000592#nSpF−1 -CC and CellML
    GKr = G_Kr#0.153#nSpF−1 -CC and CellML
    GCaL= GCa#0.0000398#3.98**-5#cm.ms−1 ?????????? are we sure about this param, 0.0398 from CellML - cc and paper Gcal seem to work better than cellML
    PNaK = GNaK#2.724#pApF−1 # this is referred to as knak in CC -CC and CellML
    KmK = 1#mM -CC and CellML
    KmNa = 40#mM -CC and CellML
    kNaCa = GNCX#1000#pApF−1
    ksat = 0.1 #-CC and CellML
    alpha = 2.5 #-CC and CellML
    gamma = 0.35 # this is referred to as n in CC -CC and CellML
    KmCa = 1.38#mM -CC and CellML
    KmNai = 87.5#mM -CC and CellML
    GpCa = GpCa_input#0.1238 #nS/pF -CC and CellML
    GpK = GKb #0.0146 #nS/pF -CC and CellML
    k1_prime = 0.15 #-CC and CellML
    k2_prime = 0.045 #-CC and CellML
    k3 = 0.06 #-CC and CellML
    k4 = 0.005 #0.000015 from publication, correted by CC to 0.005 -CC and CellML
    EC = 1.5 #-CC and CellML
    max_SR = 2.5 #-CC and CellML
    min_SR = 1 #-CC and CellML
    Vrel = 0.102 #40.8 corrected by CC -CC and CellML
    Vxfer = 0.0038 #-CC and CellML
    Vleak = 0.00036 #ms-1 -CC and CellML
    Vmaxup = 0.006375#mMms−1 -CC and CellML
    Bu_fc = 0.2#mM -CC and CellML
    Kbu_fc = 0.001#mM -CC and CellML
    Bu_fsr= 10#mM -CC and CellML
    Kbu_fsr = 0.3#mM -CC and CellML
    Bu_fss = 0.4 #-CC and CellML
    Kup = 0.00025#mM -CC and CellML
    Kbu_fss = 0.00025 #-CC and CellML
    KpCa = 0.0005 #mM
    #External Concentrations - same in CellML and CC
    Ko= user_K_conc #baseline K conc in TTP is 5.4#mM
    Nao = 140#mM
    Cao = 2#mM

    FoRT = Frdy/R/Temp
    RxT = R*Temp
    RxTdivF = RxT/Frdy
    FoRT_reciprocal = (1/FoRT)
    #S = 0.2#μm−1
    #cellular_resistivity = 162#Ωcm


    #Intracellular volumes
    #VC= 16404#μm3 ### seem to have issues, are they decimal points or commas, corrected by CC
    #VSR = 1094#μm3 ### seem to have issues, are they decimal points or commas, corrected by CC
    #VSS = 54.68 # from cellML model ### seem to have issues, are they decimal points or commas, corrected by CC


    #### Initial conditions from 2006 TP paper - initial conditions from CellML or CC don't make a difference
    #Parameters for currents
    if cell_type == 'EPI' :
        V = -85.23
        Ki = 136.89
        Nai = 8.604
        Cai = 0.000126
        xr1 = 0.00621
        xr2 = 0.4712
        Xs = 0.0095
        m = 0.00172
        h = 0.7444
        j = 0.7045
        Ca_ss = 0.00036
        d = 3.373e-5
        f = 0.7888
        f2 = 0.9755
        fcass = 0.9953
        s = 0.999998
        r = 2.42e-8
        Ca_SR = 3.64
        R_prime = 0.9073   
        Gto = G_to#0.294#nSpF−1 -CC and CellML
        GKs = G_Ks#0.392#nSpF−1
    elif cell_type == 'ENDO' :
        V = -86.709
        Ki = 138.4 
        Nai = 10.355
        Cai = 0.00013  #3 CC says it should be 0.00007
        xr1 = 0.00448  #4
        xr2 = 0.476  #5
        Xs = 0.0087  #6
        m = 0.00155 #7     
        h = 0.7573  #8
        j = 0.7225  #9
        Ca_ss = 0.00036  #10 CC says it should be 0.00007
        d = 3.164e-5  #11
        f = 0.8009 #12
        f2 = 0.9778 # 13 from CellML model
        fcass = 0.9953 # 14 from CellML model
        s = 0.3212 #15
        r = 2.235e-8 #16
        Ca_SR = 3.715 #17 CC says it should be 1.3
        R_prime = 0.9068 # 18 from CellML model
        Gto = G_to#was0.073#nSpF−1
        GKs = G_Ks#0.392#nSpF−1
    elif cell_type == 'M' :
        Gto = G_to#0.294#nSpF−1 -CC and CellML
        GKs = G_Ks#0.098#nSpF−1 -CC and CellML
        V = -85.423
        Ki = 138.52 
        Nai = 10.132
        Cai = 0.000153  #3 CC says it should be 0.00007
        xr1 = 0.0165  #4
        xr2 = 0.473  #5
        Xs = 0.0174  #6
        m = 0.00165 #7     
        h = 0.749  #8
        j = 0.6788  #9
        Ca_ss = 0.00042  #10 CC says it should be 0.00007
        d = 3.288e-5  #11
        f = 0.7026 #12
        f2 = 0.9526 # 13 from CellML model
        fcass = 0.9942 # 14 from CellML model
        s = 0.999998 #15
        r = 2.347e-8 #16
        Ca_SR = 4.272 #17 CC says it should be 1.3
        R_prime = 0.8978 # 18 from CellML model
        

    # Assemble initial conditions
    y6 = [V,Ki,Nai,Cai,xr1,xr2,Xs]
    y12 = [m,h,j,Ca_ss,d,f]
    y18 = [f2,fcass,s,r,Ca_SR,R_prime]

    # Combine all initial conditions
    y0 = np.concatenate([y6, y12, y18])

    mutation_K897T = False; #// MUTATION K897T
    
    initial_conds = y0
    constants = [cell_type,mutation_K897T,Gto,GKs,cycle_length, cycles, R, Temp, Frdy, Cm, capacitance, VC, VSS, VSR, amplitude, duration, pKNa, GNa, GK1, GbNa, GbCa, GKr, GCaL, PNaK, KmK, KmNa, kNaCa, ksat, alpha, gamma, KmCa, KmNai, GpCa, GpK, k1_prime, k2_prime, k3, k4, EC, max_SR, min_SR, Vrel, Vxfer, Vleak, Vmaxup, Bu_fc, Kbu_fc, Bu_fsr, Kbu_fsr, Bu_fss, Kup, Kbu_fss, KpCa, Ko, Nao, Cao, FoRT, RxT, RxTdivF, FoRT_reciprocal]
    # Solve the ODE
    tspan = (0, cycles*cycle_length)
    start_time = tm.time()
    sol = solve_ivp(fun = TNNP06_model, t_span = tspan, y0 = initial_conds, args = constants,method='BDF',rtol= 1e-5,max_step = 1)
    end_time = tm.time()
    elapsed_time = end_time - start_time

    time = sol.t  # Time points
    solutions = sol.y  # Solution vectors, each row corresponds to a variable
    y6_n = ["V","Ki","Nai","Cai","xr1","xr2","Xs"]
    y12_n = ["m","h","j","Ca_ss","d","f"]
    y18_n = ["f2","fcass","s","r","Ca_SR","R_prime"]

    y_names = np.concatenate([y6_n, y12_n, y18_n])
    df = pd.DataFrame(solutions.T, columns=y_names)
    df['time'] = time
    df_temp = df[df['time'] >= ((cycles*cycle_length)-(cycle_length*1))]

    return df, pd.DataFrame(), duration


# Define the ODE system (replace 'f' with the actual system of equations)
@njit
def TNNP06_model(t, y,cell_type,mutation_K897T,Gto,GKs,cycle_length, cycles, R, Temp, Frdy, Cm, capacitance, VC, VSS, VSR, amplitude, duration, pKNa, GNa, GK1, GbNa, GbCa, GKr, GCaL, PNaK, KmK, KmNa, kNaCa, ksat, alpha, gamma, KmCa, KmNai, GpCa, GpK, k1_prime, k2_prime, k3, k4, EC, max_SR, min_SR, Vrel, Vxfer, Vleak, Vmaxup, Bu_fc, Kbu_fc, Bu_fsr, Kbu_fsr, Bu_fss, Kup, Kbu_fss, KpCa, Ko, Nao, Cao, FoRT, RxT, RxTdivF, FoRT_reciprocal):
    
    #initialise output code
    V,Ki,Nai,Cai,xr1,xr2,Xs,m,h,j,Ca_ss,d,f,f2,fcass,s,r,Ca_SR,R_prime = y
    ydot= np.zeros_like(y)
    
    ENa = FoRT_reciprocal*np.log(Nao/Nai) #correct
    EK = FoRT_reciprocal*np.log(Ko/Ki) # correct
    ECa = (FoRT_reciprocal/2)*np.log(Cao/Cai) #correct
    EKs = FoRT_reciprocal*np.log((Ko+(pKNa*Nao))/(Ki+(pKNa*Nai))) #correct
    
    #Pre-calculate recurring equations
    V_ENa = (V - ENa)
    V_EK = (V - EK)
    V_EKs = (V - EKs)
    V_FoRT = V*FoRT
    #V_FoRT_Frdy = V_FoRT * Frdy
    
    # Fast Na+ Current - No change from 2004 - same as CellML
    INa = GNa*(m**3)*h*j*V_ENa #correct
    tau_m = 1/(1+np.exp((-60-V)/5))*(0.1/(1+np.exp((V+35)/5))+0.1/(1+np.exp((V-50)/200))) # correct
    betah = np.where(V >= -40, (0.77/(0.13*(1+np.exp(-(V+10.66)/11.1)))),(2.7*np.exp(0.079*V)+3.1E5*np.exp(0.3485*V)))
    alphah = np.where(V >= -40, 0,(0.057*np.exp(-(V+80)/6.8)))
    tau_h = 1/(alphah+betah) # correct
    betaj = np.where(V >= -40, ((0.6*np.exp(0.057*V))/(1+np.exp(-0.1*(V+32)))), ((0.02424*np.exp(-0.01052*V))/(1+np.exp(-0.1378*(V+40.14)))))
    alphaj = np.where(V >= -40, 0, (((-25428*np.exp(0.2444*V)-6.948E-6*np.exp(-0.04391*V))*(V+37.78))/1/(1+np.exp(0.311*(V+79.23)))))
    tau_j = 1/(alphaj+betaj) # correct
    hinf = 1/(1+np.exp((V+71.55)/7.43))**2
    jinf = hinf
    minf = 1/(1+np.exp((-56.86-V)/9.03))**2
    ydot[7] = (minf - m)/tau_m # correct
    ydot[8] = (hinf-h)/tau_h # correct
    ydot[9] = (jinf-j)/tau_j # correct
    

    #I_CaL - corrected to 2006 - differences in alpha and gamma f2 between CellML and CC (however, differences have no effect on shape of the curve)
    CaL_exp = np.exp(2*(V-15)*FoRT)#correct
    ICaL = (GCaL * d*f*f2*fcass*4*((V-15)*(Frdy**2)/RxT))*(0.25*Ca_ss*CaL_exp-Cao)/(CaL_exp-1) # correct changed from 2004 CellML different from paper
    tau_d = alpha_d(V)*beta_d(V)+gamma_d(V) # correct # gamma_d subtracted in 2004, but added in 2006
    tau_f = alpha_f(V)+beta_f(V)+gamma_f(V) # tau_f was not a calculated parameter in 2004 
    tau_f2 = alpha_f2(V)+beta_f2(V)+gamma_f2(V) # tau_f2 was not a calculated parameter in 2004 
    dinf = 1/(1+np.exp((-8-V)/7.5))
    finf = 1/(1+np.exp((V+20)/7))
    f2inf = (0.67/(1+np.exp((V+35)/7)))+0.33
    fcassinf = (0.6/(1+(Ca_ss/0.05)**2))+0.4
    ydot[11] = (dinf-d)/tau_d
    ydot[12] = (finf-f)/tau_f # correct
    ydot[13] = (f2inf-f2)/tau_f2 #correct
    ydot[14] = (fcassinf-fcass)/tau_fcass(Ca_ss) # correct
    
    
    # I_ks: Slowly Activating K Current - corrected to 2006 - same as CellMLinvest
    IKs = GKs*(Xs**2)*V_EKs #correct
    tau_Xs = alpha_Xs(V)*beta_Xs(V) + 80 # correct # the +80 was not present in the 2004 version
    ydot[6] = (Xs_inf(V)-Xs)/tau_Xs # correct
    
    
    # I_to: Transient outward K current - No change from 2004 #- same as CellML
    Ito = Gto *s *r* V_EK #correct
    ydot[15] = (s_inf(V,cell_type)-s)/tau_s(V,cell_type) #correct
    ydot[16] = (r_inf(V)-r)/tau_r(V) # correct
    
    
    # I_kr: Rapidly Activating K Current - No change from 2004- same as CellML
    IKr = GKr * np.sqrt(Ko/5.4)* xr1 * xr2 * V_EK # correct
    tau_xr1 = alpha_xr1(V,mutation_K897T)*beta_xr1(V,mutation_K897T) #correct
    tau_xr2 = alpha_xr2(V,mutation_K897T)*beta_xr2(V,mutation_K897T) #correct
    ydot[4] = (xr1_inf(V,mutation_K897T)-xr1)/tau_xr1 #correct
    ydot[5] = (xr2_inf(V,mutation_K897T)-xr2)/tau_xr2 #correct
    
    
    # I_k1: Time-independent K current - No change from 2004 - same as CellML
    alpha_K1 = 0.1 / (1 + np.exp(0.06 * (V_EK - 200))) #correct
    beta_K1 = (3 * np.exp(0.0002 * (V_EK + 100)) + np.exp(0.1 * (V_EK - 10))) / (1 + np.exp(-0.5*V_EK)) # correct
    xk1_inf = alpha_K1 / (alpha_K1 + beta_K1) # correct
    IK1 =  GK1* xk1_inf * V_EK *np.sqrt(Ko/5.4) #correct CC does not multiply with np.sqrt(Ko/5.4)
   
    
    # I_nak: Na/K Pump Current
    #correct - Calculate shared constants - No change from 2004 #- same as CellML
    INaK = PNaK*(Ko/(Ko+KmK))*(Nai/(Nai+KmNa))*(1/(1+0.1245*np.exp(-0.1*V_FoRT)+0.0353*np.exp(-V_FoRT)))
    #INaCa - No change from 2004 #- same as CellML
    Nao3 = Nao**3
    INaCa = kNaCa*(np.exp(gamma*V_FoRT)*(Nai**3)*Cao - np.exp((gamma-1)*V_FoRT)*(Nao3)*Cai*alpha)/((KmNai**3+Nao3)*(KmCa+Cao)*(1+ksat*np.exp((gamma-1)*V_FoRT)))
    #Other currents - No change from 2004 #- same as CellML
    IpCa = GpCa*(Cai/(Cai+KpCa)) #correct
    IpK = GpK*(V_EK/(1+np.exp((25-V)/5.98))) #correct

    # Background currents - No change from 2004
    IbNa = GbNa * V_ENa #correct - same as CellML
    IbCa = GbCa * (V-ECa) #correct - same as CellML
    
    # Calcium Dynamics #- same as CellML
    Ileak = Vleak*(Ca_SR-Cai)#correct
    Iup = Vmaxup/(1+(Kup**2/Cai**2))#correct
    k_casr = max_SR - ((max_SR-min_SR)/(1+(EC/Ca_SR)**2))
    k1 = k1_prime/k_casr
    k2 = k2_prime*k_casr
    ydot[18] = -k2*Ca_ss*R_prime+(k4*(1-R_prime))
    Ca_ss2 = (Ca_ss)**2
    O = (k1*(Ca_ss2)*R_prime)/(k3+k1*(Ca_ss2)) # this is referred to as sOO in CC
    Irel = Vrel*O*(Ca_SR-Ca_ss)
    Ixfer = Vxfer*(Ca_ss-Cai)#correct
    
    #Ca_ibu_fc = (Cai*Bu_fc)/(Cai+Kbu_fc)
    VC_Frdy = (VC*Frdy) # correct
    Ca_ibu_fc = 1.00000 / (1.00000 + Bu_fc * Kbu_fc / ((Cai + Kbu_fc)**2))
    ydot[3] = Ca_ibu_fc * ((Ileak - Iup) * VSR / VC + Ixfer - (IbCa + IpCa - 2.0 * INaCa) * capacitance / (2.0 * VC_Frdy)) #- (otherRates[11] + otherRates[12]) * 0.07)
    
    Casrbu_fsr = (Ca_SR*Bu_fsr)/(Ca_SR+Kbu_fsr) # this is referred to as CaCSQN in CC
    ydot[17] = Casrbu_fsr* (Iup-Ileak-Irel) # this is referred to as dCaSR in CC
    
    Ca_ssbu_fss = 1/(1+((Bu_fss*Kbu_fss)/(Ca_ss+Kbu_fss)**2))
    ydot[10] = Ca_ssbu_fss* (-1.0 * ICaL * capacitance / (2.0 * VSS * Frdy) + Irel * VSR / VSS - Ixfer * VC / VSS)
    
    # Calculate stimulus current #- same as CellML
    I_stim = Istim_TNNP06(t, cycle_length,amplitude,duration) #same as CellML and CC
    
    # Sodium and Potassium Dynamics #- same as CellML
    
    ydot[2] = (-(INa+IbNa+3*INaK+3*INaCa)/VC_Frdy)*capacitance # correct #multiplication by capacitance in CC and CellML
    ydot[1] = (-(IK1+Ito+IKr+IKs - 2*INaK+IpK+I_stim)/VC_Frdy)*capacitance #correct #multiplication by capacitance in CC and CellML

    # Membrane Potential
    Iion = INa+IK1+Ito+IKr+IKs+ICaL+INaCa+INaK +IpCa+IpK +IbCa+IbNa
    
    #Calculate Voltage
    ydot[0] = -(Iion + I_stim)#/Cm # there is no division by Cm in CellML or CC
    
    return ydot




# def TP_currents(states_df):
    

# Solve the ODE
# tspan = (0, cycles*cycle_length)
# start_time = tm.time()
# sol = solve_ivp(TNNP06_model, tspan, y0, method='BDF',rtol= 1e-5,max_step = 1) #modifying the max_step to 0.02 does not affect not performance
# end_time = tm.time()
# elapsed_time = end_time - start_time


# time = sol.t  # Time points
# solutions = sol.y  # Solution vectors, each row corresponds to a variable

# # Plot each solution vector against time
# num_variables = solutions.shape[0]  # Number of variables in the system

# y6_n = ["V","Ki","Nai","Cai","xr1","xr2","Xs"]
# y12_n = ["m","h","j","Ca_ss","d","f"]
# y18_n = ["f2","fcass","s","r","Ca_SR","R_prime"]

# y_names = np.concatenate([y6_n, y12_n, y18_n])

# i=0
# name = y_names[i]
# plt.figure()
# plt.plot(time, solutions[i], label=name)
# plt.xlabel('Time')
# plt.ylabel(f'{name}')
# plt.title(f'Plot of {name} vs Time')
# plt.legend()
# plt.grid()
# plt.show()

# import numpy as np
# def test_func(y,z):
#     x = 2*y
#     w = 3*z
#     return [x,w]

# test1 = test_func(2,2)
# print(test1)

# test2 = test_func(np.array([1,2,3,4]),np.array([1,2,3,4]))
# print(test2)
