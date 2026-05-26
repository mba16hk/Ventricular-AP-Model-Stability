# Dutta 2017 (CiPA) Model

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import ode
from .ORd import Istim
import time as tm
import pandas as pd
import os
from numba import njit
from conductances import *


# --- NCX (Na-Ca exchanger) kinetic helpers (inlined from former INaCa_final.py) ---
@njit
def h1(Na_ion_con, h_Na_val):
    k_Na3 = 88.12  # mM
    return 1 + (Na_ion_con / k_Na3) * (1 + h_Na_val)

@njit
def h2(Na_ion_con, h_Na_val):
    k_Na3 = 88.12
    return (Na_ion_con * h_Na_val) / (k_Na3 * h1(Na_ion_con, h_Na_val))

@njit
def h3(Na_ion_con, h_Na_val):
    return 1 / h1(Na_ion_con, h_Na_val)

@njit
def h4(Na_ion_con):
    k_Na1 = 15
    k_Na2 = 5
    return 1 + (Na_ion_con / k_Na1) * (1 + (Na_ion_con / k_Na2))

@njit
def h5(Na_ion_con):
    return (Na_ion_con ** 2) / (h4(Na_ion_con) * 75)

@njit
def h6(Na_ion_con):
    return 1 / h4(Na_ion_con)

@njit
def k4_prime(Na_ion_con, h_Na_val, h_Ca_val):
    omega_Ca = 6E4
    return (h3(Na_ion_con, h_Na_val) * omega_Ca) / h_Ca_val

@njit
def k4_double_prime(Na_ion_con, h_Na_val):
    omega_NaCa = 5E3
    return h2(Na_ion_con, h_Na_val) * omega_NaCa

@njit
def k4(Na_ion_con, h_Na_val, h_Ca_val):
    return k4_prime(Na_ion_con, h_Na_val, h_Ca_val) + k4_double_prime(Na_ion_con, h_Na_val)

@njit
def k6(Na_ion_con, Ca_ion_con):
    k_Ca_on = 1.5E6
    return h6(Na_ion_con) * Ca_ion_con * k_Ca_on

@njit
def k7(Na_ion_con, h_Na_val):
    omega_Na = 6E4
    return h5(Na_ion_con) * h2(Na_ion_con, h_Na_val) * omega_Na

@njit
def x1(Na_ion_con, h_Na_val, h_Ca_val, Ca_ion_con, k3):
    k_Ca_off = 5E3
    k2 = k_Ca_off
    k5 = k_Ca_off
    k7_val = k7(Na_ion_con, h_Na_val)
    return k2 * k4(Na_ion_con, h_Na_val, h_Ca_val) * (k7_val + k6(Na_ion_con, Ca_ion_con)) + k5 * k7_val * (k2 + k3)

@njit
def x2(Na_ion_con, h_Na_val, h_Ca_val, Ca_ion_con, k1, k8):
    k_Ca_off = 5E3
    k5 = k_Ca_off
    k4_val = k4(Na_ion_con, h_Na_val, h_Ca_val)
    return k1 * k7(Na_ion_con, h_Na_val) * (k4_val + k5) + k4_val * k6(Na_ion_con, Ca_ion_con) * (k1 + k8)

@njit
def x3(Na_ion_con, h_Na_val, Ca_ion_con, k1, k3, k8):
    k_Ca_off = 5E3
    k2 = k_Ca_off
    k6_val = k6(Na_ion_con, Ca_ion_con)
    return k1 * k3 * (k7(Na_ion_con, h_Na_val) + k6_val) + k8 * k6_val * (k2 + k3)

@njit
def x4(Na_ion_con, h_Na_val, h_Ca_val, k1, k3, k8):
    k_Ca_off = 5E3
    k2 = k_Ca_off
    k5 = k_Ca_off
    return k2 * k8 * (k4(Na_ion_con, h_Na_val, h_Ca_val) + k5) + k3 * k5 * (k1 + k8)

@njit
def E_function(x_val, sum_of_xs):
    return x_val / sum_of_xs

@njit
def allo_Y(Ca_ion_conc, K_mCaAct2):
    return 1 / (1 + K_mCaAct2 / (Ca_ion_conc ** 2))

@njit
def J_NaCa_Na_Y(E1_val, E2_val, E3_val, E4_val, Na_ion_con, h_Na_val, k3_double_prime, k8):
    return 3 * (E4_val * k7(Na_ion_con, h_Na_val) - E1_val * k8) + E3_val * k4_double_prime(Na_ion_con, h_Na_val) - E2_val * k3_double_prime

@njit
def J_NaCa_Ca_Y(E1_val, E2_val, k1):
    k_Ca_off = 5E3
    k2 = k_Ca_off
    return (E2_val * k2) - (E1_val * k1)


def run_Dutta_Model(cycles, cycle_length, cell_type, amp=-80):
  model_type = "CiPA 2017"
  G_Ks       = GKs_conductance(model_type, cell_type)
  G_Kr       = GKr_conductance(model_type, cell_type)
  G_K1       = GK1_conductance(model_type, cell_type)
  G_to       = Gto_conductance(model_type, cell_type)
  G_Na_late, G_Na_fast = GNa_conductance(model_type, cell_type)
  GCa        = GCa_conductance(model_type, cell_type)
  GNCX       = GNCX_conductance(model_type, cell_type)
  GNaK       = GNaK_conductance(model_type, cell_type)
  GKb        = GKb_conductance(model_type, cell_type)
  GNab       = GNab_conductance(model_type, cell_type)
  GCab       = GCab_conductance(model_type, cell_type)
  GpCa_input = GpCa_conductance(model_type, cell_type)

  #GKs = 0.0034 default value
  GKs = G_Ks
  GKr = G_Kr
  GK1 = G_K1
  Gto = G_to
  GNa_fast = G_Na_fast# was 75 # correct
  GNa_late = G_Na_late# was 0.0075 * GNaL_scaling_factor
  # Conductance scaling factors in Chang 2017 model
  Gkr_scaling_factor = 1.0127
  Gks_scaling_factor = 1.87
  GK1_scaling_factor = 1.698
  #PCa_scaling_factor = 1.007
  GNaL_scaling_factor = 2.661
  J_rel_scaling_factor = 1
  
  amplitude = amp#-80 #uA/uF correct
  duration = 0.5 #ms correct

  #External concentration correct
  Na_ion_conc_o = 140 #mM
  Ca_ion_conc_o = 1.8 #mM
  K_ion_conc_o = 5.4 #mM

  # Cell Geometry
  L=0.01 #cm correct
  radius=0.0011 #cm correct
  vcell= 1000*np.pi*(radius**2)*L#38E-6 #μL # π*r**2*L correct
  Ageo=2*np.pi*(radius**2)+ 2.00000*np.pi*radius*L #0.767E-4 #cm2 #=2π*r**2+2π*r*L correct
  Acap=2*Ageo#1.534E-4 #cm2 #2*Ageo correct
  vmyo=0.68*vcell#25.84E-6 #μL #0.68*vcell correct
  vnsr=0.0552*vcell#2.098E-6 #μL #=0.0552*vcell correct
  vjsr=0.0048*vcell#0.182E-6 #μL #=0.0048*vcell correct
  vss=0.02*vcell#0.76E-6 #μL #=0.02*vcell correct

  ####################################
  #Consants correct
  R = 8314  # J/(kmol*K)
  Temp = 310    # K, body temperature
  Faradays_constant = 96485  # C/mol

  # Valence values correct
  z_Na = 1
  z_Ca = 2
  z_K = 1

  #reversal potentials
  PR_Na_K = 0.01833 # correct

  #CaMK
  alpha_CaMK = 0.05 #correct
  beta_CaMK = 0.00068 #correct
  CaMK_0 = 0.05 #correct
  K_mCaM = 0.0015 #correct

  # Phi params
  K_m_CaMK=0.15 #correct

  #I_Na
  A_h_fast = 0.99 # correct
  A_h_slow = 0.01 # correct
  Ah_CaMK_fast=A_h_fast # correct
  Ah_CaMK_slow=A_h_slow # correct
  # GNa_fast = 75 # correct
  # GNa_late = 0.0075 * GNaL_scaling_factor #Value in OrD = 0.0075, from CeLLML, need t check from publication
  tau_h_L = 200 #ms
  tau_h_L_CaMK = 3*tau_h_L # correct

  #Ito, IKr, IK1 and IKs
  #Gto = 0.02 # correct
  #GKr = 0.046* Gkr_scaling_factor # correct
  #GKs = 0.0034, value in ORd, but in the CiPA model it would be 0.0034 * Gks_scaling_factor
  #GK1 = 0.1908  * GK1_scaling_factor # correct

  #ICaL
  K_m_n = 0.002 #correct
  k_plus2_n = 1000 #correct
  tau_j_Ca = 75 #correct, called tjCa
  A_f_fast = 0.6 # correct, called Aff
  A_f_slow = (1-A_f_fast) #correct, called Afs
  P_Ca = GCa#0.0001 *PCa_scaling_factor #cm/s #correct
  
  gamma_Cai = 1 #correct
  gamma_Cao = 0.341 #correct
  gamma_Nai=0.75 #correct
  gamma_Nao=0.75 #correct
  gamma_Ki=0.75 #correct
  gamma_Ko=0.75 #correct

  #Background Currents
  P_Nab = GNab#3.75E-10 # correct
  P_Cab = GCab#2.5E-8 # correct
  Gkb = GKb#0.003 # correct
  GpCa = GpCa_input#0.0005  #correct
  Cm = 1 # correct

  #INaCa
  q_Na = 0.5224 #correct
  q_Ca = 0.1670 #correct
  G_NaCa = GNCX#0.0008 # correct
  k_Na3 = 88.12 #mM, # correct
  k_Na1 = 15 #mM, correct
  k_Na2 = 5 #mM, # correct
  k_asymm = 12.5 #correct
  k_Ca_on = 1.5E6 #mM/ms correct
  omega_Ca= 6E4 #Hz, correct
  omega_NaCa= 5E3 #Hz correct
  omega_Na = 6E4 #Hz, correct
  K_mCaAct = 150E-6 #mM # correct
  K_mCaAct2 = (150E-6)**2 #mM # correct

  #I_NaK
  K0_Nai = 9.073 #mM correct
  K0_Nao = 27.78 #mM correct
  k1_pos = 949.5 #Hz correct
  k1_neg = 182.4 #mM-1 correct
  k2_pos = 687.2 # correct
  k2_neg = 39.4 #Hz correct
  k3_pos = 1899 #Hz correct
  k3_neg = 79300 #Hz*mM-2 correct
  k4_pos = 639# correct
  k4_neg = 40 #Hz correct
  delta = -0.155 #correct
  K_Ki = 0.5 #mM correct
  K_Ko = 0.3582 #mM correct
  MgADP = 0.05 #correct
  MgATP = 9.8 # correct
  K_MgATP = 1.698E-7 #mM correct
  H_conc = 1E-7 #mM correct
  alpha2 = k2_pos # correct
  beta1 = k1_neg*MgADP # correct
  beta3_alpha4_denominator = (1+(MgATP/K_MgATP)) # correct
  alpha4 = (k4_pos*(MgATP/K_MgATP))/beta3_alpha4_denominator # correct
  sigmaP = 4.2 #mM correct
  K_H_P = 1.698E-7 #mM correct
  K_Na_P = 224 #mM correct
  K_K_P = 292 #mM correct

  #Diffusion Flxes in ms
  tau_diff_Na = 2 # correct
  tau_diff_K = 2 # correct
  tau_diff_Ca = 0.2 # correct

  #SR Calciu Release Flux
  beta_tau = 4.75 # correct
  alpha_rel = 0.5*beta_tau # correct
  beta_tau_CaMK = 1.25*beta_tau # correct
  alpha_rel_CaMK = 0.5*beta_tau_CaMK # correct
  tau_tr = 100 # correct
  deltaK_m_PLB = 0.00017 # correct
  deltaJ_up_CaMK = 1.75 # correct


  #Model Concentrations
  CMDN = 0.05 #mM, # correct, but will be scaled according to celltype
  K_m_CMDN= 0.00238 #mM correct
  TRPN = 0.07 #mM correct
  K_m_TRPN= 0.0005 #mM correct
  BSR =0.047 #mM correct
  K_m_BSR= 0.00087 #mM correct
  BSL = 1.124 #mM correct
  K_m_BSL= 0.0087 #mM correct
  CSQN = 10.0 #mM correct
  K_m_CSQN= 0.8 #mM correct
  
  
  ### IKr Markov Model Parameters taken from CiPA supplementary section (not in original ORd) - Markov Model from Li 2016
  A1 = 0.0264
  B1 = 4.631E-05
  q1 = 4.843
  A2 = 4.986E-06
  B2 = -0.004226
  q2 = 4.23
  A3 = 0.001214
  B3 = 0.008516
  q3 = 4.962
  A4 = 1.854E-05
  B4 = -0.04641
  q4 = 3.769
  A11 = 0.0007868
  B11 = 1.535E-08
  q11 = 4.942
  A21 = 5.46E-06
  B21 = -0.1688
  q21 = 4.156
  A31 = 0.005509
  B31 = 7.77E-09
  q31 = 4.22
  A41 = 0.001416
  B41 = -0.02877
  q41 = 1.459
  A51 = 0.4492
  B51 = 0.008595
  q51 = 5
  A52 = 0.3181
  B52 = 3.61E-08
  q52 = 4.663
  A53 = 0.149
  B53 = 0.004668
  q53 = 2.412
  A61 = 0.01241
  B61 = 0.1725
  q61 = 5.568
  A62 = 0.3226
  B62 = -0.00066
  q62 = 5
  A63 = 0.008978
  B63 = -0.02215
  q63 = 5.682
  temperature = 37
  shift_INa_inact = 0
  Kmax = 0
  Ku = 0
  halfmax = 1
  Kt = 0
  Vhalf = 1
  PNaK = GNaK#30 # NOTE THAT THIS HAS BEEN ADDED IN CiPA ONLY, but it is the same as the original ORd model, except the ORd model does not give it  name and only uses 30 directly in the INaK equation
  ##################################

  #Pre-calculated Parameters
  RxT = R*Temp
  FoRT = Faradays_constant/RxT
  Reversal_potential_cons = 1/FoRT
  
  #CiPa Model initial conditions (taken from supplmentary section of Change 2017)
  V_m = -88.0145 #ORd V init = -87.84 #mV
  Na_ion_conc_i = 6.46961 #ORd init = 7.23 #mM
  Na_ion_conc_ss = 6.46967 #ORd init = 7.23 #mM
  K_ion_conc_i = 145.501 #ORd init = 143.79 #mM
  K_ion_conc_ss = 145.501 #ORd init = 143.79 #mM
  Ca_ion_conc_i = 7.45e-05 #ORd init = 8.54E-5 #mM
  Ca_ion_conc_ss = 7.3e-05 #ORd init = 8.43E-5 #mM
  Ca_ion_conc_nsr = 1.37987 #ORd init =1.61 #mM
  Ca_ion_conc_jsr = 1.37944 #ORd init =1.56 #mM
  m = 0.007335 #ORd init =0.0074621
  h_fast = 0.698542 #ORd init =0.692591
  h_slow = 0.698542 #ORd init =0.692574
  j = 0.698542 #ORd init =0.692477
  CaMK_trap = 0.003252 #ORd init =0.0124065
  m_L = 0.000188 #ORd init =0.000194015
  h_L = 0.513396 #ORd init =0.496116
  a = 0.001 #ORd init =0.00101185
  i_fast = 0.999555 #ORd init =0.999542
  i_slow = 0.871715 #ORd init =0.589579
  d = 2.33e-9 #ORd init =2.43015E-9 
  f_fast = 1.0 # same as ORd
  f_slow = 0.971897 # ORd init = 0.910671
  f_Ca_fast = 1.0 # same as ORd
  f_Ca_slow = 1 #ORd init = 0.99982
  j_Ca = 1 #ORd init = 0.999977
#   x_r_fast = 8E-06 #ORd init = 8.26608E-6 Dutta 2017
#   x_r_slow = 0.160765 #ORd init = .453268 Dutta 2017
  x_s1 = 0.130941 #ORd init = 0.270492
  x_s2 = 0.000193 #ORd init = 0.0001963
  x_K1 = 0.996756 #ORd init = 0.996801
  h_CaMK_slow = 0.455526 #0.448501 called hsp in CiPA code
  h_L_CaMK = 0.307923 #0.265885 called hLp in CiPA code
  f_CaMK_fast = 1.0 #same as ORd, called ffp in CiPA code
  f_Ca_CaMK_fast = 1.0 #same as ORd, called fcapf in CiPA code
  j_CaMK = 0.698541 #0.692413, called jp in CiPA model
  a_CaMK = 0.000515567 # same as OrD, called ap in CiPA code
  i_CaMK_fast = 0.999555 #0.999542, called iFp in CiPA model
  i_CaMK_slow = 0.905466 #0.641861, called iSp in CiPA model
  n = 0.001539 #ORd init =0.00267171, called nCa in CiPA
  J_rel_NP = 1.47e-07 #ORd init =2.53943E-5 #mM/ms 
  J_rel_CaMK = 1.83e-07 #ORd init =3.17262E-7 #mM/ms 
  
  #new parameters in CiPA not in Original ORd
  IC1 = 0.999637
  IC2 = 6.83e-05
  C1 = 1.80e-08
  C2 = 8.27e-05
  O = 0.000156
  IO = 5.68e-05
  IObound = 0
  Obound = 0
  Cbound = 0
  D = 0
  n_hERG = 1
  isepi = False
  Jup_scale = 1
  Jrel_scale = 1
  
  if (cell_type == 'EPI'):
      CMDN = 1.3*CMDN # correct
      isepi = True
      Jup_scale = 1.3
  elif (cell_type == "M"):
      Jrel_scale = 1.7
    
      #Gkb = 0.6*Gkb # correct
      #GK1 = 1.2 * GK1 # correct
      #GKr = 1.3* GKr # correct
      #P_Ca = 1.2 *P_Ca # correct
      # P_CaNa = 1.2 * P_CaNa # correct
      # P_CaK = 1.2 * P_CaK # correct
      #Gto = 4*Gto # correct
      #GNa_late = 0.6*GNa_late # correct
      #G_NaCa = 1.1*G_NaCa # correct
      #PNaK = 0.9*PNaK # note that PNak = GNaK, correct
      #GKs = 1.4*GKs # correct - but we need to only activate it when the slider is not being used
      
  # elif (cell_type == 'M'):
  #     #GK1 = 1.3 * GK1 #correct
  #     #GKr = 0.8* GKr # correct
  #     #P_Ca = 2.5 *P_Ca # correct
  #     # P_CaNa = 2.5 * P_CaNa # correct
  #     # P_CaK = 2.5 * P_CaK # correct
  #     #Gto = 4*Gto # correct
  #     #G_NaCa = 1.4 * G_NaCa # correct
  #     #PNaK = 0.7*PNaK # note that PNak = GNaK, correct
  
  P_CaNa=0.00125*P_Ca #correct
  P_CaK=3.574E-4*P_Ca #correct
  P_Ca_CaMK = 1.1*P_Ca #correct
  P_CaNa_CaMK = 0.00125*P_Ca_CaMK #correct
  P_CaK_CaMK = 3.574E-4*P_Ca_CaMK #correct

  initial_conds = [V_m, Na_ion_conc_i, Na_ion_conc_ss, K_ion_conc_i, K_ion_conc_ss, Ca_ion_conc_i, Ca_ion_conc_ss, Ca_ion_conc_nsr, Ca_ion_conc_jsr, m, h_fast, h_slow, j, CaMK_trap, m_L, h_L, a, i_fast, i_slow, d, f_fast, f_slow, f_Ca_fast, f_Ca_slow, j_Ca, x_s1, x_s2, x_K1, h_CaMK_slow, h_L_CaMK, f_CaMK_fast, f_Ca_CaMK_fast, j_CaMK, a_CaMK, i_CaMK_fast, i_CaMK_slow, n, J_rel_NP, J_rel_CaMK, IC1, IC2, C1, C2, O, IO] #, IObound, Obound, Cbound, D
  constants = [cell_type,cycle_length,GKs,J_rel_scaling_factor, amplitude, duration, Na_ion_conc_o, Ca_ion_conc_o, K_ion_conc_o, L, radius, vcell, Ageo, Acap, vmyo, vnsr, vjsr, vss, R, Temp, Faradays_constant, z_Na, z_Ca, z_K, PR_Na_K, alpha_CaMK, beta_CaMK, CaMK_0, K_mCaM, K_m_CaMK, A_h_fast, A_h_slow, Ah_CaMK_fast, Ah_CaMK_slow, GNa_fast, GNa_late, tau_h_L, tau_h_L_CaMK, Gto, GKr, GK1, K_m_n, k_plus2_n, tau_j_Ca, A_f_fast, A_f_slow, P_Ca, P_CaNa, P_CaK, P_Ca_CaMK, P_CaNa_CaMK, P_CaK_CaMK, gamma_Cai, gamma_Cao, gamma_Nai, gamma_Nao, gamma_Ki, gamma_Ko, P_Nab, P_Cab, Gkb, GpCa, Cm, q_Na, q_Ca, G_NaCa, k_Na3, k_Na1, k_Na2, k_asymm, k_Ca_on, omega_Ca, omega_NaCa, omega_Na, K_mCaAct, K_mCaAct2, K0_Nai, K0_Nao, k1_pos, k1_neg, k2_pos, k2_neg, k3_pos, k3_neg, k4_pos, k4_neg, delta, K_Ki, K_Ko, MgADP, MgATP, K_MgATP, H_conc, alpha2, beta1, beta3_alpha4_denominator, alpha4, sigmaP, K_H_P, K_Na_P, K_K_P, tau_diff_Na, tau_diff_K, tau_diff_Ca, beta_tau, alpha_rel, beta_tau_CaMK, alpha_rel_CaMK, tau_tr, deltaK_m_PLB, deltaJ_up_CaMK, CMDN, K_m_CMDN, TRPN, K_m_TRPN, BSR, K_m_BSR, BSL, K_m_BSL, CSQN, K_m_CSQN, A1, B1, q1, A2, B2, q2, A3, B3, q3, A4, B4, q4, A11, B11, q11, A21, B21, q21, A31, B31, q31, A41, B41, q41, A51, B51, q51, A52, B52, q52, A53, B53, q53, A61, B61, q61, A62, B62, q62, A63, B63, q63, temperature, shift_INa_inact, Kmax, Ku, halfmax, Kt, Vhalf, PNaK, RxT, FoRT, Reversal_potential_cons,n_hERG,isepi,Jup_scale,Jrel_scale]
  # Solve the ODE
  tspan = (0, cycles*cycle_length)
  #print(cycles)
  start_time = tm.time()
  sol = solve_ivp(fun = Dutta_Model, t_span = tspan, y0 = initial_conds, args = constants,method='BDF',rtol= 1e-6,atol = 1e-5,max_step = 0.5)
  end_time = tm.time()
  elapsed_time = end_time - start_time

  time = sol.t  # Time points
  solutions = sol.y  # Solution vectors, each row corresponds to a variable
  y_names = ['V', 'Nai', 'Na_ion_conc_ss', 'Ki', 'K_ion_conc_ss', 'Cai', 'Ca_ion_conc_ss', 'Ca_ion_conc_nsr', 'Ca_ion_conc_jsr', 'm', 'h_fast', 'h_slow', 'j', 'CaMK_trap', 'm_L', 'h_L', 'a', 'i_fast', 'i_slow', 'd', 'f_fast', 'f_slow', 'f_Ca_fast', 'f_Ca_slow', 'j_Ca', 'x_s1', 'x_s2', 'x_K1', 'h_CaMK_slow', 'h_L_CaMK', 'f_CaMK_fast', 'f_Ca_CaMK_fast', 'j_CaMK', 'a_CaMK', 'i_CaMK_fast', 'i_CaMK_slow', 'n', 'J_rel_NP', 'J_rel_CaMK', 'IC1', 'IC2', 'C1', 'C2', 'O', 'IO'] #, 'IObound', 'Obound', 'Cbound', 'D'
  
  df = pd.DataFrame(solutions.T, columns=y_names)
  df['time'] = time
  stim_duration = duration
  return df, pd.DataFrame(), stim_duration

@njit
def Dutta_Model(time, y, cell_type,cycle_length,GKs,J_rel_scaling_factor, amplitude, duration, Na_ion_conc_o, Ca_ion_conc_o, K_ion_conc_o, L, radius, vcell, Ageo, Acap, vmyo, vnsr, vjsr, vss, R, Temp, Faradays_constant, z_Na, z_Ca, z_K, PR_Na_K, alpha_CaMK, beta_CaMK, CaMK_0, K_mCaM, K_m_CaMK, A_h_fast, A_h_slow, Ah_CaMK_fast, Ah_CaMK_slow, GNa_fast, GNa_late, tau_h_L, tau_h_L_CaMK, Gto, GKr, GK1, K_m_n, k_plus2_n, tau_j_Ca, A_f_fast, A_f_slow, P_Ca, P_CaNa, P_CaK, P_Ca_CaMK, P_CaNa_CaMK, P_CaK_CaMK, gamma_Cai, gamma_Cao, gamma_Nai, gamma_Nao, gamma_Ki, gamma_Ko, P_Nab, P_Cab, Gkb, GpCa, Cm, q_Na, q_Ca, G_NaCa, k_Na3, k_Na1, k_Na2, k_asymm, k_Ca_on, omega_Ca, omega_NaCa, omega_Na, K_mCaAct, K_mCaAct2, K0_Nai, K0_Nao, k1_pos, k1_neg, k2_pos, k2_neg, k3_pos, k3_neg, k4_pos, k4_neg, delta, K_Ki, K_Ko, MgADP, MgATP, K_MgATP, H_conc, alpha2, beta1, beta3_alpha4_denominator, alpha4, sigmaP, K_H_P, K_Na_P, K_K_P, tau_diff_Na, tau_diff_K, tau_diff_Ca, beta_tau, alpha_rel, beta_tau_CaMK, alpha_rel_CaMK, tau_tr, deltaK_m_PLB, deltaJ_up_CaMK, CMDN, K_m_CMDN, TRPN, K_m_TRPN, BSR, K_m_BSR, BSL, K_m_BSL, CSQN, K_m_CSQN, A1, B1, q1, A2, B2, q2, A3, B3, q3, A4, B4, q4, A11, B11, q11, A21, B21, q21, A31, B31, q31, A41, B41, q41, A51, B51, q51, A52, B52, q52, A53, B53, q53, A61, B61, q61, A62, B62, q62, A63, B63, q63, temperature, shift_INa_inact, Kmax, Ku, halfmax, Kt, Vhalf, PNaK, RxT, FoRT, Reversal_potential_cons,n_hERG,isepi,Jup_scale,Jrel_scale): 
  
  #ydot= np.zeros_like(y) 
  
  #unpack initial conditions
  V_m, Na_ion_conc_i, Na_ion_conc_ss, K_ion_conc_i, K_ion_conc_ss, Ca_ion_conc_i, Ca_ion_conc_ss, Ca_ion_conc_nsr, Ca_ion_conc_jsr, m, h_fast, h_slow, j, CaMK_trap, m_L, h_L, a, i_fast, i_slow, d, f_fast, f_slow, f_Ca_fast, f_Ca_slow, j_Ca, x_s1, x_s2, x_K1, h_CaMK_slow, h_L_CaMK, f_CaMK_fast, f_Ca_CaMK_fast, j_CaMK, a_CaMK, i_CaMK_fast, i_CaMK_slow, n, J_rel_NP, J_rel_CaMK, IC1, IC2, C1, C2, O, IO = y # IObound, Obound, Cbound, D
  #cycle_length,amplitude, duration, Na_ion_conc_o, Ca_ion_conc_o, K_ion_conc_o, L, radius, vcell, Ageo, Acap, vmyo, vnsr, vjsr, vss,R, Temp, Faradays_constant, z_Na, z_Ca, z_K, PR_Na_K, alpha_CaMK, beta_CaMK, CaMK_0, K_mCaM, K_m_CaMK, A_h_fast, A_h_slow, Ah_CaMK_fast, Ah_CaMK_slow, GNa_fast, GNa_late, tau_h_L, tau_h_L_CaMK, Gto, GKr, GKs, GK1, K_m_n, k_plus2_n, tau_j_Ca, A_f_fast, A_f_slow, A_f_CaMK_fast, A_f_CaMK_slow, P_Ca, P_CaNa, P_CaK, P_Ca_CaMK, P_CaNa_CaMK, P_CaK_CaMK, gamma_Cai, gamma_Cao, gamma_Nai, gamma_Nao, gamma_Ki, gamma_Ko, P_Nab, P_Cab, Gkb, GpCa, Cm, q_Na, q_Ca, G_NaCa, k_Na3, k_Na1, k_Na2, k_asymm, k_Ca_on, omega_Ca, omega_NaCa, omega_Na, K_mCaAct, K_mCaAct2, K0_Nai, K0_Nao, k1_pos, k1_neg, k2_pos, k2_neg, k3_pos, k3_neg, k4_pos, k4_neg, delta, K_Ki, K_Ko, MgADP, MgATP, beta1, K_MgATP, H_conc, alpha2, beta3_alpha4_denominator, alpha4, sigmaP, K_H_P, K_Na_P, K_K_P, tau_diff_Na, tau_diff_K, tau_diff_Ca, beta_tau, alpha_rel, beta_tau_CaMK, alpha_rel_CaMK, tau_tr, deltaK_m_PLB, deltaJ_up_CaMK, CMDN, K_m_CMDN, TRPN, K_m_TRPN, BSR, K_m_BSR, BSL, K_m_BSL, CSQN, K_m_CSQN, RxT, FoRT, Reversal_potential_cons = constants
  ### The stimulus equation
  #amplitude=−80.0 μAμF, duration=0.5 ms
  #For charge conservation sake, stimulus has K+ identity as described by Hund et al.3.
  I_stim = Istim(time,cycle_length,amplitude,duration)
  
  E_Na = Reversal_potential_cons*np.log(Na_ion_conc_o/Na_ion_conc_i) # correct
  E_K = Reversal_potential_cons*np.log(K_ion_conc_o/K_ion_conc_i) #correct
  E_Ks = Reversal_potential_cons*np.log((K_ion_conc_o+PR_Na_K*Na_ion_conc_o)/(K_ion_conc_i+PR_Na_K*Na_ion_conc_i))
  #print(f"E is: {E_Na},{E_K},{E_Ks}")
  ## Precalculate constants
  V_E_Na = (V_m-E_Na)
  V_E_K = (V_m-E_K)
  V_E_Ks = (V_m-E_Ks)
  VFoRT = V_m*FoRT
  
  ###################################
  #The Calcium/Calmodulin-Depenent Protein Kinase (CaMK)
  CaMK_bound=CaMK_0*((1-CaMK_trap)/(1+(K_mCaM/Ca_ion_conc_ss))) #correct
  CaMK_active=CaMK_bound+CaMK_trap ##correct
  dCaMK_trap=alpha_CaMK*CaMK_bound*(CaMK_bound+CaMK_trap)-beta_CaMK*CaMK_trap # correct
  phi = 1/(1+(K_m_CaMK/CaMK_active))
  one_minus_phi = (1-phi)
  ###################################
  
  ###################################
  #INa Calculation - same as ORd model
  # Gating variables
  m_inf = 1 / (1 + np.exp(-(V_m + 39.57) / 9.871)) # correct
  h_inf= 1 / (1 + np.exp(((V_m + 82.9)-shift_INa_inact) / 6.086)) # correct
  m_L_inf= 1/(1+np.exp(-(V_m+42.85)/5.264)) # correct
  h_L_inf= 1/(1+np.exp((V_m+87.61)/7.488)) #correct
  h_CaMK_inf=1/(1+np.exp(((V_m+89.1)-shift_INa_inact)/6.086)) # hssp, correct
  h_L_CaMK_inf= 1/(1+np.exp((V_m+93.81)/7.488)) # correct
  j_inf = h_inf # correct jss=hss
  j_CaMK_inf = j_inf
  
  #time constants
  tau_m=1/((6.765*np.exp((V_m+11.64)/34.77))+(8.552*np.exp(-(V_m+77.42)/5.955))) # correct
  tau_m_L=tau_m# correct
  tau_h_fast= 1 / (1.432E-5 * np.exp(-((V_m + 1.196) -shift_INa_inact)/ 6.285) + 6.149 * np.exp(((V_m + 0.5096) -shift_INa_inact)/ 20.27)) # correct
  tau_h_slow=1 / (0.009764*np.exp(-((V_m+17.95)-shift_INa_inact)/28.05)+0.3343*np.exp(((V_m+5.730)-shift_INa_inact)/56.66)) # correct
  tau_j= 2.038 +( 1 / (0.02136 * np.exp(-((V_m + 100.6)-shift_INa_inact )/ 8.281) + 0.3052 * np.exp(((V_m + 0.9941)-shift_INa_inact )/ 38.45))) # correct
  tau_h_CaMK_slow= 3.0*tau_h_slow # correct 3*ths=thsp
  tau_j_CaMK= 1.46*tau_j # correct tjp = 1.46*tj
  
  #differentials
  dm = (m_inf - m)/tau_m # correct
  dh_fast = (h_inf - h_fast)/tau_h_fast # correct
  dh_slow = (h_inf - h_slow)/tau_h_slow # correct
  dj = (j_inf - j)/tau_j # correct
  dh_CaMK_slow = (h_CaMK_inf-h_CaMK_slow)/tau_h_CaMK_slow # correct
  dj_CaMK = (j_CaMK_inf-j_CaMK)/tau_j_CaMK # correct
  dmL = (m_L_inf-m_L)/tau_m_L # correct
  dhL = (h_L_inf-h_L)/tau_h_L # correct
  dhL_CaMK = (h_L_CaMK_inf-h_L_CaMK)/tau_h_L_CaMK # correct
  
  # Integrated Parameters calculations
  h = (A_h_fast*h_fast)+(A_h_slow*h_slow) # correct
  h_CaMK=Ah_CaMK_fast*h_fast+Ah_CaMK_slow*h_CaMK_slow # correct
  
  # Current Calculations
  INa_fast = GNa_fast * V_E_Na * (m**3) * (one_minus_phi*h*j+phi*h_CaMK*j_CaMK) # correct
  INa_late = GNa_late * V_E_Na * m_L * (one_minus_phi*h_L+phi*h_L_CaMK) # correct
  INa = INa_fast + INa_late # correct
  #print(f"INa is: {INa}")

  #############################
  
  #############################
  ### Transient Outward Potassium Current (Ito), exactly the same as ORd
  
  delta_epi = 1
  if (isepi): # correct
      delta_epi = 1-(0.95/(1+np.exp((V_m+70)/5)))
      
    
  #Gating variables
  a_inf=1/(1+np.exp(-(V_m-14.34)/14.82)) # correct
  i_inf=1/(1+np.exp((V_m+43.94)/5.711)) # correct
  a_CaMK_inf=1/(1+np.exp(-(V_m-24.34)/14.82)) # correct
  
  #Time constants
  tau_a=1.0515/((1/(1.2089*(1+np.exp(-(V_m-18.4099)/29.3814))))+(3.5/(1+np.exp((V_m+100)/29.3814)))) # correct
  tau_i_fast=delta_epi *(4.562+(1/(0.3933*np.exp(-(V_m+100)/100)+0.08004*np.exp((V_m+50)/16.59)))) # correct
  tau_i_slow=delta_epi *(23.62+(1/(0.001416*np.exp(-(V_m+96.52)/59.05)+1.7808E-8*np.exp((V_m+114.1)/8.079)))) # correct
  sigma_CaMK_dev=1.354+(1E-4/(np.exp((V_m-167.4)/15.89)+np.exp(-(V_m-12.23)/0.2154))) # correct
  sigma_CaMK_recov=1-(0.5/(1+np.exp((V_m+70)/20))) # correct
  sigma_product = sigma_CaMK_dev*sigma_CaMK_recov # correct
  tau_i_CaMK_fast=tau_i_fast*sigma_product # correct
  tau_i_CaMK_slow=tau_i_slow*sigma_product # correct
  
  #differentials
  da = (a_inf - a)/tau_a # correct
  di_fast = (i_inf - i_fast)/tau_i_fast # correct
  di_slow = (i_inf - i_slow)/tau_i_slow # correct
  da_CaMK = (a_CaMK_inf - a_CaMK)/tau_a # correct
  di_CaMK_fast = (i_inf - i_CaMK_fast)/tau_i_CaMK_fast # correct
  di_CaMK_slow = (i_inf - i_CaMK_slow)/tau_i_CaMK_slow # correct
   
  # Integrated Parameters calculations
  A_i_fast=1/(1+np.exp((V_m-213.6)/151.2)) # correct
  A_i_slow=1-A_i_fast # correct
  i = A_i_fast *i_fast + A_i_slow*i_slow # correct
  i_CaMK = A_i_fast *i_CaMK_fast + A_i_slow*i_CaMK_slow # correct
  
  #Current Calculations
  Ito = Gto * V_E_K * (one_minus_phi*a*i + phi*a_CaMK*i_CaMK) # correct
  #print(f"Ito is: {Ito}")
  ##############################################
  
  ###############################################
  #L-type Calcium Current (ICaL) - all correct, just one addition is that there is a threshold effect at calculating PhiCaK, PhiCaNa and PhiCaL
  #gating variables
  d_inf=1/(1+np.exp(-(V_m+3.94)/4.23)) # correct
  f_inf=1/(1+np.exp((V_m+19.58)/3.696)) # correct
  f_Ca_inf=f_inf # correct
  j_Ca_inf=f_Ca_inf # correct
  f_CaMK_inf=f_inf # correct

  #time constants
  tau_const1 = (V_m+20)/10 # correct
  tau_const2 = (V_m-4)/7
  tau_const3 = (V_m+5) # correct
  tau_d=0.6+(1/(np.exp(-0.05*(V_m+6))+np.exp(0.09*(V_m+14)))) # correct
  tau_f_fast=7+(1/(0.0045*np.exp(-tau_const1)+0.0045*np.exp(tau_const1)))# correct
  tau_f_slow=1000+(1/(0.000035*np.exp(-tau_const3/4)+0.000035*np.exp(tau_const3/6))) # correct
  tau_f_Ca_fast=7+(1/(0.04*np.exp(-tau_const2)+0.04*np.exp(tau_const2))) # correct
  tau_f_Ca_slow=100+(1/(0.00012*np.exp(-(V_m)/3)+0.00012*np.exp((V_m)/7))) # correct
  tau_f_CaMK_fast=2.5*tau_f_fast # correct
  tau_f_Ca_CaMK_fast=2.5*tau_f_Ca_fast # correct
  
  #Differentials
  dd = (d_inf - d)/tau_d # correct
  df_fast = (f_inf - f_fast)/tau_f_fast # correct
  df_slow = (f_inf - f_slow)/tau_f_slow # correct
  df_Ca_fast = (f_Ca_inf - f_Ca_fast)/tau_f_Ca_fast # correct
  df_Ca_slow = (f_Ca_inf - f_Ca_slow)/tau_f_Ca_slow # correct
  dj_Ca = (j_Ca_inf - j_Ca)/tau_j_Ca # correct
  df_CaMK_fast = (f_CaMK_inf - f_CaMK_fast)/tau_f_CaMK_fast # correct
  df_Ca_CaMK_fast = (f_inf - f_Ca_CaMK_fast)/tau_f_Ca_CaMK_fast # correct
  
  #Integrated Parameter calculations
  A_f_Ca_fast= 0.3+(0.6/(1+np.exp((V_m-10)/10))) # correct
  A_f_Ca_slow = 1-A_f_Ca_fast # correct
  f = A_f_fast*f_fast + A_f_slow*f_slow # correct
  f_Ca = A_f_Ca_fast*f_Ca_fast + A_f_Ca_slow*f_Ca_slow # correct
  f_CaMK = A_f_fast*f_CaMK_fast + A_f_slow*f_slow # correct
  f_Ca_CaMK = A_f_Ca_fast*f_Ca_CaMK_fast + A_f_Ca_slow*f_Ca_slow # correct
  k_minus2_n = j_Ca # correct
  alpha_n = 1/((k_plus2_n/k_minus2_n)+(1+(K_m_n/Ca_ion_conc_ss))**4) # correct
  dn = (alpha_n*k_plus2_n)-(n*k_minus2_n) # correct
  
  #Calculate the Psi values
  #pre-calculate some constants
  NaK_threshold = VFoRT # added in CiPA code
  Ca_threshold = 2*VFoRT # added in CiPA code
  VFort_Frdy = (VFoRT*Faradays_constant) # correct
  Ca_exp_const = np.exp(Ca_threshold) # correct
  Na_K_exp_const = np.exp(VFoRT) # correct
  Na_K_denominator = (Na_K_exp_const-1) # correct
  Ca_denominator = (Ca_exp_const-1) # correct
  Ca_numerator_const = gamma_Cao*Ca_ion_conc_o # correct
  
  #these are correct but it is worth checking them again
  # Psi_Ca = np.where((Ca_threshold>=-1e-07)&(Ca_threshold<=1e07),(2*Faradays_constant*(1-VFoRT)*Ca_ion_conc_ss*Ca_exp_const-Ca_numerator_const),4*VFort_Frdy*((gamma_Cai*Ca_ion_conc_ss*Ca_exp_const-Ca_numerator_const)/Ca_denominator)) # correct
  # Psi_CaNa = np.where((NaK_threshold>=-1e-07)&(NaK_threshold<=1e07),0.75*Faradays_constant*(Na_ion_conc_ss*Na_K_exp_const-Na_ion_conc_o),gamma_Nao*VFort_Frdy*((Na_ion_conc_ss*Na_K_exp_const-Na_ion_conc_o)/Na_K_denominator))
  # Psi_CaK = np.where((NaK_threshold>=-1e-07)&(NaK_threshold<=1e07),0.75*Faradays_constant*(K_ion_conc_ss*Na_K_exp_const-K_ion_conc_o),gamma_Ki*VFort_Frdy*((K_ion_conc_ss*Na_K_exp_const-K_ion_conc_o)/Na_K_denominator))
  Psi_Ca = 4*VFort_Frdy*((gamma_Cai*Ca_ion_conc_ss*Ca_exp_const-Ca_numerator_const)/Ca_denominator)
  Psi_CaNa = 1*VFort_Frdy*((gamma_Nai*Na_ion_conc_ss*Na_K_exp_const-gamma_Nao*Na_ion_conc_o)/Na_K_denominator)
  Psi_CaK = 1*VFort_Frdy*((gamma_Ki*K_ion_conc_ss*Na_K_exp_const-gamma_Ko*K_ion_conc_o)/Na_K_denominator)
  
  
  Ca_current_constants1 = d*one_minus_phi*(f*(1-n)+f_Ca*n*j_Ca) # correct
  Ca_current_constants2 = d*phi*(f_CaMK*(1-n)+f_Ca_CaMK*n*j_Ca) # correct
  ICaL = P_Ca*Psi_Ca *Ca_current_constants1+P_Ca_CaMK*Psi_Ca*Ca_current_constants2 # correct
  ICaNa = P_CaNa*Psi_CaNa*Ca_current_constants1+P_CaNa_CaMK*Psi_CaNa*Ca_current_constants2 # correct
  ICaK = P_CaK*Psi_CaK*Ca_current_constants1+P_CaK_CaMK * Psi_CaK*Ca_current_constants2 # correct
  #print(f"ICa is: {ICaL},{ICaNa},{ICaK}")
  ###############################################
  
  ###############################################
  #Rapid Delayed rectifier Potassium Current (Ikr)
  
  ### IKr Markov Model for channel states Li 2016
  # This is copied from the CeLLML model - need to chekc the values with Li2016
  corrected_temp = (temperature - 20.0000)/10
  # Markov_IO_exp_const = np.exp( n_hERG*np.log(D))
  # Markov_const_IO = Markov_IO_exp_const+halfmax
  #
  dIC1 = A21 * np.exp(B21 * V_m) * IC2 * (q21 ** (corrected_temp)) - A11 * np.exp(B11 * V_m) * IC1 * (q11 ** (corrected_temp)) + A51 * np.exp(B51 * V_m) * C1 * (q51 ** (corrected_temp)) - A61 * np.exp(B61 * V_m) * IC1 * (q61 ** (corrected_temp))
  dIC2 = A11 * np.exp(B11 * V_m) * IC1 * (q11 ** (corrected_temp)) - A21 * np.exp(B21 * V_m) * IC2 * (q21 ** (corrected_temp)) - A3 * np.exp(B3 * V_m) * IC2 * (q3 ** (corrected_temp)) + A4 * np.exp(B4 * V_m) * IO * (q4 ** (corrected_temp)) + A52 * np.exp(B52 * V_m) * C2 * (q52 ** (corrected_temp)) - A62 * np.exp(B62 * V_m) * IC2 * (q62 ** (corrected_temp))
  dC1 = A2 * np.exp(B2 * V_m) * C2 * (q2 ** (corrected_temp)) - A1 * np.exp(B1 * V_m) * C1 * (q1 ** (corrected_temp)) - A51 * np.exp(B51 * V_m) * C1 * (q51 ** (corrected_temp)) + A61 * np.exp(B61 * V_m) * IC1 * (q61 ** (corrected_temp))
  dC2 = A1 * np.exp(B1 * V_m) * C1 * (q1 ** (corrected_temp)) - A2 * np.exp(B2 * V_m) * C2 * (q2 ** (corrected_temp)) - A31 * np.exp(B31 * V_m) * C2 * (q31 ** (corrected_temp)) + A41 * np.exp(B41 * V_m) * O * (q41 ** (corrected_temp)) - A52 * np.exp(B52 * V_m) * C2 * (q52 ** (corrected_temp)) + A62 * np.exp(B62 * V_m) * IC2 * (q62 ** (corrected_temp))
  dO = A31 * np.exp(B31 * V_m) * C2 * (q31 ** (corrected_temp)) - A41 * np.exp(B41 * V_m) * O * (q41 ** (corrected_temp)) - A53 * np.exp(B53 * V_m) * O * (q53 ** (corrected_temp)) + A63 * np.exp(B63 * V_m) * IO * (q63 ** (corrected_temp))
  dIO = A3 * np.exp(B3 * V_m) * IC2 * (q3 ** (corrected_temp)) - A4 * np.exp(B4 * V_m) * IO * (q4 ** (corrected_temp)) + A53 * np.exp(B53 * V_m) * O * (q53 ** (corrected_temp)) - A63 * np.exp(B63 * V_m) * IO * (q63 ** (corrected_temp))

#V_m_V_half_exp_const = np.exp( - (V_m - Vhalf)/6.78900)
  # dIC1 = ( - ( A11*np.exp( B11*V_m)*IC1*np.exp(( corrected_temp*np.log(q11))) -  A21*np.exp( B21*V_m)*IC2*np.exp(( corrected_temp*np.log(q21))))+ A51*np.exp( B51*V_m)*C1*np.exp(( corrected_temp*np.log(q51)))) -  A61*np.exp( B61*V_m)*IC1*np.exp(( corrected_temp*np.log(q61)))
  # dIC2 = ((( A11*np.exp( B11*V_m)*IC1*np.exp(( corrected_temp*np.log(q11))) -  A21*np.exp( B21*V_m)*IC2*np.exp(( corrected_temp*np.log(q21)))) - ( A3*np.exp( B3*V_m)*IC2*np.exp(( corrected_temp*np.log(q3))) -  A4*np.exp( B4*V_m)*IO*np.exp(( corrected_temp*np.log(q4)))))+ A52*np.exp( B52*V_m)*C2*np.exp(( corrected_temp*np.log(q52)))) -  A62*np.exp( B62*V_m)*IC2*np.exp(( corrected_temp*np.log(q62)))
  # dC1 =  - ( A1*np.exp( B1*V_m)*C1*np.exp(( corrected_temp*np.log(q1))) -  A2*np.exp( B2*V_m)*C2*np.exp(( corrected_temp*np.log(q2)))) - ( A51*np.exp( B51*V_m)*C1*np.exp(( corrected_temp*np.log(q51))) -  A61*np.exp( B61*V_m)*IC1*np.exp(( corrected_temp*np.log(q61))))
  # dC2 = (( A1*np.exp( B1*V_m)*C1*np.exp(( corrected_temp*np.log(q1))) -  A2*np.exp( B2*V_m)*C2*np.exp(( corrected_temp*np.log(q2)))) - ( A31*np.exp( B31*V_m)*C2*np.exp(( corrected_temp*np.log(q31))) -  A41*np.exp( B41*V_m)*O*np.exp(( corrected_temp*np.log(q41))))) - ( A52*np.exp( B52*V_m)*C2*np.exp(( corrected_temp*np.log(q52))) -  A62*np.exp( B62*V_m)*IC2*np.exp(( corrected_temp*np.log(q62))))
  # dO = (( A31*np.exp( B31*V_m)*C2*np.exp(( corrected_temp*np.log(q31))) -  A41*np.exp( B41*V_m)*O*np.exp(( corrected_temp*np.log(q41)))) - ( A53*np.exp( B53*V_m)*O*np.exp(( corrected_temp*np.log(q53))) -  A63*np.exp( B63*V_m)*IO*np.exp(( corrected_temp*np.log(q63))))) - ( (( Kmax*Ku*Markov_IO_exp_const)/(Markov_const_IO))*O -  Ku*Obound)
  # dIO = ((( A3*np.exp( B3*V_m)*IC2*np.exp(( corrected_temp*np.log(q3))) -  A4*np.exp( B4*V_m)*IO*np.exp(( corrected_temp*np.log(q4))))+ A53*np.exp( B53*V_m)*O*np.exp(( corrected_temp*np.log(q53)))) -  A63*np.exp( B63*V_m)*IO*np.exp(( corrected_temp*np.log(q63)))) - ( (( Kmax*Ku*Markov_IO_exp_const)/(Markov_const_IO))*IO -  (( Ku*A53*np.exp( B53*V_m)*np.exp(( corrected_temp*np.log(q53))))/( A63*np.exp( B63*V_m)*np.exp(( corrected_temp*np.log(q63)))))*IObound)
  # dIObound = (( (( Kmax*Ku*Markov_IO_exp_const)/(Markov_const_IO))*IO -  (( Ku*A53*np.exp( B53*V_m)*np.exp(( corrected_temp*np.log(q53))))/( A63*np.exp( B63*V_m)*np.exp(( corrected_temp*np.log(q63)))))*IObound)+ (Kt/(1.00000+V_m_V_half_exp_const))*Cbound) -  Kt*IObound
  # dObound = (( (( Kmax*Ku*Markov_IO_exp_const)/(Markov_const_IO))*O -  Ku*Obound)+ (Kt/(1.00000+V_m_V_half_exp_const))*Cbound) -  Kt*Obound
  # dCbound =  - ( (Kt/(1.00000+V_m_V_half_exp_const))*Cbound -  Kt*Obound) - ( (Kt/(1.00000+V_m_V_half_exp_const))*Cbound -  Kt*IObound)
  # dD = 0 # or equals PNaK and there are differences between the script in CeLLML, FDA CiPA script and the CeLLML mathematics documentation
  #gating variables
  # In the CiPA model, the gating variable O is used instead of xr
  #xr_inf=1/(1+np.exp(-(V_m+8.337)/6.789))
  
  #time constants
#   tau_xr_fast=12.98+(1/(0.3652*np.exp((V_m-31.66)/3.869)+4.123E-5*np.exp(-(V_m-47.78)/20.38)))
#   tau_xr_slow=1.865+(1/(0.06629*np.exp((V_m-34.7)/7.355)+1.128E-5*np.exp(-(V_m-29.74)/25.94)))
  
#   #Differentials
#   dxr_fast = (xr_inf - x_r_fast)/tau_xr_fast
#   dxr_slow = (xr_inf - x_r_slow)/tau_xr_slow
  
#   #Calculated parameters
#   A_xr_fast=1/(1+np.exp((V_m+54.81)/38.21))
#   A_xr_slow= 1-A_xr_fast
#   xr = A_xr_fast*x_r_fast + A_xr_slow*x_r_slow
#   R_kr= 1/((1+np.exp((V_m+55)/75))*(1+np.exp((V_m-10)/30)))
  
  #Calculate Currents - this equation is different than the ORd model
  IKr = GKr * np.sqrt(K_ion_conc_o/5.4)*O*V_E_K
  #print(f"IKr is: {IKr}")
  ###############################################
  
  ###############################################
  # Slow delayed Rectifier Potassium Current (IKs) - no difference with ORd
  #Gating Variables
  x_s1_inf=1/(1+np.exp(-(V_m+11.6)/8.932)) # correct
  x_s2_inf=x_s1_inf # correct
  
  #time constants
  tau_x_s1=817.3+(1/(2.326E-4*np.exp((V_m+48.28)/17.8)+0.001292*np.exp(-(V_m+210)/230))) # correct
  tau_x_s2=1/(0.01*np.exp((V_m-50)/20)+0.0193*np.exp(-(V_m+66.54)/31)) # correct
  
  #Differentials
  dxs1 = (x_s1_inf - x_s1)/tau_x_s1 # correct
  dxs2 = (x_s2_inf - x_s2)/tau_x_s2 # correct
  
  #Calculate Currents
  IKs = GKs * (1+(0.6/(1+(3.8E-5/Ca_ion_conc_i)**1.4)))*x_s1*x_s2*V_E_Ks # correct
  #print(f"IKs is: {IKs}")
  ###############################################
  
  ###############################################
  #Inward Rectifier Potassium Current (I_K1) - no difference with ORd
  
  #Gating variable
  x_K1_inf=1/(1+(np.exp(-(V_m+2.5538*K_ion_conc_o+144.59)/(1.5692*K_ion_conc_o+3.8115)))) # correct
  
  #time constants
  tau_x_K1=122.2/(np.exp(-(V_m+127.2)/20.36)+np.exp((V_m+236.8)/69.33)) # correct
  
  #Differentials
  dx_K1 = (x_K1_inf-x_K1)/tau_x_K1 # correct
  
  #Calculate currents
  R_K1=1/(1+np.exp((V_m+105.8-2.6*K_ion_conc_o)/9.493)) #correct
  IK1 = GK1*np.sqrt(K_ion_conc_o)*x_K1*R_K1*V_E_K #correct
  #print(f"IK1 is: {IK1}")
  ###############################################
  
  ###############################################
  #Sodium/calcium Exchange Current (INaCa) - identical to the ORd model
  h_Na_val = np.exp(q_Na*VFoRT) # correct
  h_Ca_val = np.exp(q_Ca*VFoRT) # correct
  h7=1+(Na_ion_conc_o/k_Na3)*(1+(1/h_Na_val)) # correct
  h8=(Na_ion_conc_o)/(k_Na3*h_Na_val*h7) # correct
  h9=1/h7 # correct
  h10=k_asymm + 1+ (Na_ion_conc_o/k_Na1)*(1+(Na_ion_conc_o/k_Na2)) # correct
  h11=(Na_ion_conc_o**2)/(h10*k_Na1*k_Na2) # correct
  h12=1/h10 # correct
  k1=h12*Ca_ion_conc_o*k_Ca_on # correct
  k3_prime=h9*omega_Ca #correct
  k3_double_prime=h8*omega_NaCa # correct
  k3=k3_prime+ k3_double_prime # correct
  k8 = h8*h11*omega_Na # correct
  # the x1, x2, x3 and x4 equations are applied correctly
  x1_val_i = x1(Na_ion_conc_i,h_Na_val,h_Ca_val,Ca_ion_conc_i,k3)
  x2_val_i = x2(Na_ion_conc_i,h_Na_val,h_Ca_val,Ca_ion_conc_i,k1,k8)
  x3_val_i = x3(Na_ion_conc_i,h_Na_val,Ca_ion_conc_i,k1,k3,k8)
  x4_val_i = x4(Na_ion_conc_i,h_Na_val,h_Ca_val,k1,k3,k8)
  sum_xi = x1_val_i+x2_val_i+x3_val_i+x4_val_i
  x1_val_ss = x1(Na_ion_conc_ss,h_Na_val,h_Ca_val,Ca_ion_conc_ss,k3)
  x2_val_ss = x2(Na_ion_conc_ss,h_Na_val,h_Ca_val,Ca_ion_conc_ss,k1,k8)
  x3_val_ss = x3(Na_ion_conc_ss,h_Na_val,Ca_ion_conc_ss,k1,k3,k8)
  x4_val_ss = x4(Na_ion_conc_ss,h_Na_val,h_Ca_val,k1,k3,k8)
  sum_xss = x1_val_ss+x2_val_ss+x3_val_ss+x4_val_ss
  E1_val_i = E_function(x1_val_i,sum_xi)
  E2_val_i = E_function(x2_val_i,sum_xi)
  E3_val_i = E_function(x3_val_i,sum_xi)
  E4_val_i = E_function(x4_val_i,sum_xi)
  E1_val_ss = E_function(x1_val_ss,sum_xss)
  E2_val_ss = E_function(x2_val_ss,sum_xss)
  E3_val_ss = E_function(x3_val_ss,sum_xss)
  E4_val_ss = E_function(x4_val_ss,sum_xss)
  allo_i = allo_Y(Ca_ion_conc_i,K_mCaAct2)
  allo_ss = allo_Y(Ca_ion_conc_ss,K_mCaAct2)
  J_NaCa_Na_i = J_NaCa_Na_Y(E1_val_i,E2_val_i,E3_val_i,E4_val_i,Na_ion_conc_i,h_Na_val,k3_double_prime,k8)
  J_NaCa_Na_ss = J_NaCa_Na_Y(E1_val_ss,E2_val_ss,E3_val_ss,E4_val_ss,Na_ion_conc_ss,h_Na_val,k3_double_prime,k8)
  J_NaCa_Ca_i = J_NaCa_Ca_Y(E1_val_i,E2_val_i,k1)
  J_NaCa_Ca_ss = J_NaCa_Ca_Y(E1_val_ss,E2_val_ss,k1)
  
  # all the below equations are correct
  I_NaCa_i = G_NaCa*0.8*allo_i*(z_Na*J_NaCa_Na_i+z_Ca*J_NaCa_Ca_i)
  I_NaCa_ss = G_NaCa*0.2*allo_ss*(z_Na*J_NaCa_Na_ss+z_Ca*J_NaCa_Ca_ss)
  INaCa = I_NaCa_i + I_NaCa_ss
  #print(f"INaCa is: {I_NaCa_i},{I_NaCa_ss}")
  ###############################################
  
  ###############################################
  #Sodium/Potassium ATPase Current (INaK) - same as ORd model
  K_constants = VFoRT/3 #correct
  P = sigmaP/(1+(H_conc/K_H_P)+(Na_ion_conc_i/K_Na_P)+(K_ion_conc_i/K_K_P)) #correct
  K_Nai_val = K0_Nai*np.exp((delta*K_constants)) #correct
  K_Nao_val = K0_Nao*np.exp(((1-delta)*K_constants)) #correct
  common_denominator_1 = (((1+Na_ion_conc_o/K_Nao_val)**3)+((1+K_ion_conc_o/K_Ko)**2)-1) #correct
  common_denominator_2 = (((1+Na_ion_conc_i/K_Nai_val)**3)+((1+K_ion_conc_i/K_Ki)**2)-1) #correct
  alpha1=(k1_pos*((Na_ion_conc_i/K_Nai_val)**3))/common_denominator_2 #correct
  beta2=(k2_neg*((Na_ion_conc_o/K_Nao_val)**3))/common_denominator_1 #correct
  alpha3=(k3_pos*((K_ion_conc_o/K_Ko)**2))/common_denominator_1 #correct
  beta4=(k4_neg*((K_ion_conc_i/K_Ki)**2))/common_denominator_2 #correct
  beta3=(k3_neg*P*H_conc)/beta3_alpha4_denominator #correct
  # x vals and E vals are all correct
  x1_val = alpha4*alpha1*alpha2+beta2*beta4*beta3+alpha2*beta4*beta3+beta3*alpha1*alpha2
  x2_val = beta2*beta1*beta4+alpha1*alpha2*alpha3+alpha3*beta1*beta4+alpha2*alpha3*beta4
  x3_val = alpha2*alpha3*alpha4+beta3*beta2*beta1+beta2*beta1*alpha4+alpha3*alpha4*beta1
  x4_val = beta4*beta3*beta2+alpha3*alpha4*alpha1+beta2*alpha4*alpha1+beta3*beta2*alpha1
  sum_x_val = x1_val + x2_val + x3_val +x4_val
  E1_NAK = E_function(x1_val,sum_x_val)
  E2_NAK = E_function(x2_val,sum_x_val)
  E3_NAK = E_function(x3_val,sum_x_val)
  E4_NAK = E_function(x4_val,sum_x_val)
  J_NaK_Na=3*(E1_NAK*alpha3-E2_NAK*beta3) #correct
  J_NaK_K=2*(E4_NAK*beta1-E3_NAK*alpha1) #correct
  INaK = PNaK*(z_Na*J_NaK_Na+z_K*J_NaK_K) # correct
  #print(f"INaK is: {INaK},{J_NaK_K},{J_NaK_Na}")
  ###############################################
  
  ###############################################
  #Background Currents
  #Gating Variables
  x_kb =  1 / (1 + np.exp(-(V_m - 14.48) / 18.34)) # correct
  
  #Current calculations, INab and ICab are correct but have the 1e-7 threshold
  INab = P_Nab*1*VFort_Frdy*((Na_ion_conc_i*Na_K_exp_const-Na_ion_conc_o)/(Na_K_denominator))
  ICab = P_Cab*4*VFort_Frdy*((gamma_Cai*Ca_ion_conc_i*Ca_exp_const-Ca_numerator_const)/Ca_denominator)
  # INab = np.where((NaK_threshold>=-1e-07)&(NaK_threshold<=1e07),P_Nab*1*Faradays_constant*(1-VFoRT/2)*((Na_ion_conc_i*Na_K_exp_const-Na_ion_conc_o)),P_Nab*1*VFort_Frdy*((Na_ion_conc_i*Na_K_exp_const-Na_ion_conc_o)/(Na_K_denominator)))
  # ICab = np.where((Ca_threshold>=-1e-07)&(Ca_threshold<=1e07),(P_Cab*2*Faradays_constant*(1-VFoRT)*Ca_ion_conc_ss*Ca_exp_const-Ca_numerator_const),P_Cab*4*VFort_Frdy*((gamma_Cai*Ca_ion_conc_ss*Ca_exp_const-Ca_numerator_const)/Ca_denominator))
  Ikb = Gkb*x_kb*V_E_K # correct
  IpCa = GpCa*(Ca_ion_conc_i/(0.0005+Ca_ion_conc_i)) #correct
  #print(f"Backgroun currents is: {INab},{ICab},{Ikb},{IpCa}")
  ###############################################
  
  ###############################################
  #Voltage ##### correct (INa = INa_f+INa_L)
  dV = (-1/Cm)*(INa+Ito+ICaL+ICaNa+ICaK+IKr+IKs+IK1+INaCa+INaK+INab+ICab+Ikb+IpCa+I_stim)
  ###############################################
  
  ###############################################
  #Diffusion Fluxes
  J_diff_Na = (Na_ion_conc_ss-Na_ion_conc_i)/tau_diff_Na #correct
  J_diff_Ca = (Ca_ion_conc_ss-Ca_ion_conc_i)/tau_diff_Ca #correct
  J_diff_K = (K_ion_conc_ss-K_ion_conc_i)/tau_diff_K #correct
  ###############################################
  
  ###############################################
  #SR Calcium Release Flux via Ryanodine receptor
  J_rel_const = (1+(1.5/Ca_ion_conc_jsr)**8) #correct
  tau_rel_const = (1+(0.0123/Ca_ion_conc_jsr)) #correct
  J_rel_NP_inf = Jrel_scale *((alpha_rel*(-ICaL))/J_rel_const) #correct
  J_rel_CaMK_inf = Jrel_scale*((alpha_rel_CaMK*(-ICaL))/J_rel_const) #correct
  tau_rel_NP = np.maximum(beta_tau/tau_rel_const,0.001) #correct
  tau_rel_CaMK = np.maximum(beta_tau_CaMK/tau_rel_const,0.001) #correct
  dJ_rel_NP = (J_rel_NP_inf-J_rel_NP)/tau_rel_NP # correct
  dJ_rel_CaMK = (J_rel_CaMK_inf-J_rel_CaMK)/tau_rel_CaMK #correct
  J_rel = J_rel_scaling_factor *one_minus_phi*J_rel_NP+phi*J_rel_CaMK
  #print(f"Jrel is: {J_rel},{dJ_rel_CaMK},{dJ_rel_NP}")
  #Calcium Uptake via SERCA Pump (Jup) - same as ORd model
  J_SERCA_const = (0.004375*Ca_ion_conc_i) #correct
  J_leak = (0.0039375*Ca_ion_conc_nsr)/15 #correct
  J_up_NP = Jup_scale*(J_SERCA_const/(0.00092+Ca_ion_conc_i)) #correct
  J_up_CaMK = Jup_scale*((1+deltaJ_up_CaMK)*(J_SERCA_const/(0.00092-deltaK_m_PLB+Ca_ion_conc_i))) #correct
  J_up = one_minus_phi*J_up_NP+phi*J_up_CaMK-J_leak #correct
  
  #Calcium Translocation from NSR to JSR (Jtr)
  J_tr = (Ca_ion_conc_nsr-Ca_ion_conc_jsr)/tau_tr # correct
  #print(f"Jup,Jtr is: {J_up},{J_tr}")
  ###############################################
  
  ###############################################
  #ORd Model concentrations and buffers
  
  # some constants
  vss_div_vmyo = (vss/vmyo)
  Acap_vmyo_const = (Acap/(Faradays_constant*vmyo))
  Acap_vss_const = (Acap/(Faradays_constant*vss))
  
  #rate of change of Na ions 
  dNai = -(INa+INa_late+3*I_NaCa_i+3*INaK+INab)*Acap_vmyo_const+(J_diff_Na*vss_div_vmyo) # correct
  dNass = -(ICaNa+3*I_NaCa_ss)*Acap_vss_const-J_diff_Na # correct
  # rate of change of K ions
  # There needs to be a stimulus current in the below equation Istim)
  dKi = -(Ito+IKr+IKs+IK1+Ikb+I_stim-2*INaK)*Acap_vmyo_const+(J_diff_K*vss_div_vmyo) # correct
  dKss = -ICaK * Acap_vss_const - J_diff_K # correct
  # rate of change of Ca ions
  beta_Cai = 1/(1+(CMDN*K_m_CMDN/(K_m_CMDN+Ca_ion_conc_i)**2)+(TRPN*K_m_TRPN/(K_m_TRPN+Ca_ion_conc_i)**2))
  beta_Cass = 1/(1+(BSR*K_m_BSR/(K_m_BSR+Ca_ion_conc_ss)**2)+(BSL*K_m_BSL/(K_m_BSL+Ca_ion_conc_ss)**2))
  beta_Cajsr = 1/(1+((CSQN*K_m_CSQN)/(K_m_CSQN+Ca_ion_conc_jsr)**2))
  dCai = beta_Cai*(-(IpCa+ICab-2*I_NaCa_i)*(Acap_vmyo_const/2)-J_up*(vnsr/vmyo)+J_diff_Ca*vss_div_vmyo)
  dCass = beta_Cass *(-(ICaL-2*I_NaCa_ss)*(Acap_vss_const/2)+J_rel*(vjsr/vss)-J_diff_Ca)
  dCansr = J_up-J_tr*(vjsr/vnsr)
  dCajsr = beta_Cajsr * (J_tr - J_rel)
  ###############################################
  
  ydot = [dV, dNai, dNass, dKi, dKss, dCai, dCass, dCansr, dCajsr, dm, dh_fast, dh_slow, dj, dCaMK_trap, dmL, dhL, da, di_fast, di_slow, dd, df_fast, df_slow, df_Ca_fast, df_Ca_slow, dj_Ca, dxs1, dxs2, dx_K1, dh_CaMK_slow, dhL_CaMK, df_CaMK_fast, df_Ca_CaMK_fast, dj_CaMK, da_CaMK, di_CaMK_fast, di_CaMK_slow, dn, dJ_rel_NP, dJ_rel_CaMK, dIC1, dIC2, dC1, dC2, dO, dIO] #, dIObound, dObound, dCbound, dD
  return ydot
  

