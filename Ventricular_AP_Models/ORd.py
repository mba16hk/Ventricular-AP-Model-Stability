#ORd_Model_final

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import ode
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
    # k_Na1 * k_Na2 = 15 * 5 = 75
    return (Na_ion_con ** 2) / (h4(Na_ion_con) * 75)

@njit
def h6(Na_ion_con):
    return 1 / h4(Na_ion_con)

@njit
def k4_prime(Na_ion_con, h_Na_val, h_Ca_val):
    omega_Ca = 6E4  # Hz
    return (h3(Na_ion_con, h_Na_val) * omega_Ca) / h_Ca_val

@njit
def k4_double_prime(Na_ion_con, h_Na_val):
    omega_NaCa = 5E3  # Hz
    return h2(Na_ion_con, h_Na_val) * omega_NaCa

@njit
def k4(Na_ion_con, h_Na_val, h_Ca_val):
    return k4_prime(Na_ion_con, h_Na_val, h_Ca_val) + k4_double_prime(Na_ion_con, h_Na_val)

@njit
def k6(Na_ion_con, Ca_ion_con):
    k_Ca_on = 1.5E6  # mM/ms
    return h6(Na_ion_con) * Ca_ion_con * k_Ca_on

@njit
def k7(Na_ion_con, h_Na_val):
    omega_Na = 6E4  # Hz
    return h5(Na_ion_con) * h2(Na_ion_con, h_Na_val) * omega_Na

@njit
def x1(Na_ion_con, h_Na_val, h_Ca_val, Ca_ion_con, k3):
    k_Ca_off = 5E3  # Hz
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


@njit
def Istim(time, cycle_length,amplitude,duration):
  if (time % cycle_length <= duration): #the duration is 0.5ms
    return amplitude # the amplitude is -80
  else:
    return 0
#


def run_ORd_Model(cycles, cycle_length, cell_type, user_K_conc, amp=-80):
  model_type = "O'Hara Rudy 2010"
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
  GKr = G_Kr#0.046
  GK1 = G_K1
  Gto = G_to
  GNa_fast = G_Na_fast#was 75
  GNa_late = G_Na_late# was0.0075
  
  amplitude = amp #-80 #uA/uF
  duration = 0.5 #ms

  #External concentration
  Na_ion_conc_o = 140 #mM
  Ca_ion_conc_o = 1.8 #mM
  K_ion_conc_o = user_K_conc#5.4 #mM baseline K conc in ORD is 5.4

  # Cell Geometry
  L=0.01 #cm
  radius=0.0011 #cm 
  vcell=38E-6 #μL # π*r**2*L
  Ageo=0.767E-4 #cm2 #=2π*r**2+2π*r*L
  Acap=1.534E-4 #cm2 #2*Ageo
  vmyo=25.84E-6 #μL #0.68*vcell
  vnsr=2.098E-6 #μL #=0.0552*vcell
  vjsr=0.182E-6 #μL #=0.0048*vcell
  vss=0.76E-6 #μL #=0.02*vcell

  #Ord Model initial conditions
  V_m = -87.84 #mV
  Na_ion_conc_i = 7.23 #mM
  Na_ion_conc_ss = 7.23 #mM
  K_ion_conc_i = 143.79 #mM
  K_ion_conc_ss = 143.79 #mM
  Ca_ion_conc_i = 8.54E-5 #mM
  Ca_ion_conc_ss = 8.43E-5 #mM
  Ca_ion_conc_nsr = 1.61 #mM
  Ca_ion_conc_jsr = 1.56 #mM
  m = 0.0074621
  h_fast = 0.692591
  h_slow = 0.692574
  j = 0.692477
  h_CaMK_slow = 0.448501 #!!!!!!!!!
  j_CaMK = 0.692413
  m_L = 0.000194015
  h_L = 0.496116
  h_L_CaMK = 0.265885 #!!!!!!!!!
  a = 0.00101185
  i_fast = 0.999542
  i_slow = 0.589579
  a_CaMK = 0.000515567
  i_CaMK_fast = 0.999542 #!!!!!!!!!
  i_CaMK_slow = 0.641861 #!!!!!!!!!
  d = 2.43015E-9
  f_fast = 1.0
  f_slow = 0.910671
  f_Ca_fast = 1.0 #!!!!!!!!!
  f_Ca_slow = 0.99982 #!!!!!!!!!
  j_Ca = 0.999977
  n = 0.00267171
  f_CaMK_fast = 1.0 #!!!!!!!!!
  f_Ca_CaMK_fast = 1.0 #!!!!!!!!!
  x_r_fast = 8.26608E-6 #!!!!!!!!!
  x_r_slow = 0.453268 #!!!!!!!!!
  x_s1 = 0.270492
  x_s2 = 0.0001963
  x_K1 = 0.996801
  J_rel_NP = 2.53943E-5 #mM/ms #!!!!!!!!!
  J_rel_CaMK = 3.17262E-7 #mM/ms #!!!!!!!!!
  CaMK_trap = 0.0124065

  ####################################
  #Consants
  R = 8314  # J/(kmol*K)
  Temp = 310    # K, body temperature
  Faradays_constant = 96485  # C/mol

  # Valence values
  z_Na = 1
  z_Ca = 2
  z_K = 1

  #reversal potentials
  PR_Na_K = 0.01833

  #CaMK
  alpha_CaMK = 0.05
  beta_CaMK = 0.00068
  CaMK_0 = 0.05
  K_mCaM = 0.0015

  # Phi params
  K_m_CaMK=0.15

  #I_Na
  A_h_fast = 0.99
  A_h_slow = 0.01
  Ah_CaMK_fast=A_h_fast
  Ah_CaMK_slow=A_h_slow
  # GNa_fast = 75
  # GNa_late = 0.0075
  tau_h_L = 200 #ms
  tau_h_L_CaMK = 3*tau_h_L

  #Ito, IKr, IK1 and IKs
  #Gto = 0.02
  #GKr = 0.046
  #GKs = 0.0034
  #GK1 = 0.1908
  PNaK = GNaK#30

  #ICaL
  K_m_n = 0.002
  k_plus2_n = 1000
  tau_j_Ca = 75
  A_f_fast = 0.6
  A_f_slow = (1-A_f_fast)
  A_f_CaMK_fast = A_f_fast
  A_f_CaMK_slow = A_f_slow
  P_Ca = GCa#0.0001 #cm/s
  gamma_Cai = 1
  gamma_Cao = 0.341
  gamma_Nai=0.75
  gamma_Nao=0.75
  gamma_Ki=0.75
  gamma_Ko=0.75

  #Background Currents
  P_Nab = GNab#3.75E-10
  P_Cab = GCab#2.5E-8
  Gkb = GKb#0.003
  GpCa = GpCa_input#0.0005
  Cm = 1

  #INaCa
  q_Na = 0.5224
  q_Ca = 0.1670
  G_NaCa = GNCX#0.0008
  k_Na3 = 88.12 #mM,
  k_Na1 = 15 #mM,
  k_Na2 = 5 #mM, 
  k_asymm = 12.5
  k_Ca_on = 1.5E6 #mM/ms
  omega_Ca= 6E4 #Hz,
  omega_NaCa= 5E3 #Hz
  omega_Na = 6E4 #Hz,
  K_mCaAct = 150E-6 #mM
  K_mCaAct2 = 150E-6**2 #mM

  #I_NaK
  K0_Nai = 9.073 #mM
  K0_Nao = 27.78 #mM
  k1_pos = 949.5 #Hz
  k1_neg = 182.4 #mM-1
  k2_pos = 687.2
  k2_neg = 39.4 #Hz
  k3_pos = 1899 #Hz
  k3_neg = 79300 #Hz*mM-2
  k4_pos = 639
  k4_neg = 40 #Hz
  delta = -0.155
  K_Ki = 0.5 #mM
  K_Ki2 = K_Ki**2
  K_Ko = 0.3582 #mM
  K_Ko2 = K_Ko**2
  MgADP = 0.05
  MgATP = 9.8
  beta1 = k1_neg*MgADP
  K_MgATP = 1.698E-7 #mM
  H_conc = 1E-7 #mM
  alpha2 = k2_pos
  beta1 = k1_neg*MgADP
  beta3_alpha4_denominator = (1+(MgATP/K_MgATP))
  alpha4 = (k4_pos*(MgATP/K_MgATP))/beta3_alpha4_denominator
  H_conc = 1E-7 #mM
  sigmaP = 4.2 #mM
  K_H_P = 1.698E-7 #mM
  K_Na_P = 224 #mM
  K_K_P = 292 #mM

  #Diffusion Flxes in ms
  tau_diff_Na = 2
  tau_diff_K = 2
  tau_diff_Ca = 0.2

  #SR Calciu Release Flux
  beta_tau = 4.75
  alpha_rel = 0.5*beta_tau
  beta_tau_CaMK = 1.25*beta_tau
  alpha_rel_CaMK = 0.5*beta_tau_CaMK
  tau_tr = 100
  deltaK_m_PLB = 0.00017
  deltaJ_up_CaMK = 1.75


  #Model Concentrations
  CMDN = 0.05 #mM,
  K_m_CMDN= 0.00238 #mM #!!!!!!!!!
  TRPN = 0.07 #mM,
  K_m_TRPN= 0.0005 #mM #!!!!!!!!!
  BSR =0.047 #mM,
  K_m_BSR= 0.00087 #mM #!!!!!!!!!
  BSL = 1.124 #mM,
  K_m_BSL= 0.0087 #mM #!!!!!!!!!
  CSQN = 10.0 #mM,
  K_m_CSQN= 0.8 #mM #!!!!!!!!!
  ##################################

  #Pre-calculated Parameters
  RxT = R*Temp
  FoRT = Faradays_constant/RxT
  Reversal_potential_cons = 1/FoRT
  
  Jup_scale = 1
  isepi = False
  
  ## Some pre-calculated equations
  h10=k_asymm + 1+ (Na_ion_conc_o/k_Na1)*(1+(Na_ion_conc_o/k_Na2))
  h11=(Na_ion_conc_o**2)/(h10*k_Na1*k_Na2)
  h12=1/h10
  k1=h12*Ca_ion_conc_o*k_Ca_on
  
  xK1_const1 = 2.5538*K_ion_conc_o+144.59
  xK1_const2 = (1.5692*K_ion_conc_o+3.8115)
  RK1_const = 105.8-2.6*K_ion_conc_o
  
  if (cell_type == 'EPI'):
    CMDN = 1.3*CMDN
    Jup_scale = 1.3
    isepi = True
    #Gkb = 0.6*Gkb # correct
    #GK1 = 1.2 * GK1
    #GKr = 1.3* GKr # correct
    #P_Ca = 1.2 *P_Ca # correct
    # P_CaNa = 1.2 * P_CaNa # correct
    # P_CaK = 1.2 * P_CaK # correct
    #Gto = 4*Gto # correct
    #GNa_late = 0.6*GNa_late # correct
    #G_NaCa = 1.1*G_NaCa
    #PNaK = 0.9*PNaK # note that PNak = GNaK, correct
    #GKs = 1.4*GKs # correct - but we need to only activate it when the slider is not being used
  
  if (cell_type == 'M'): # correct
    Jrel_scale = 1.7
  else :
    Jrel_scale = 1
    
      
  # elif (cell_type == 'M'):
  #   #GK1 = 1.3 * GK1
  #   #GKr = 0.8* GKr # correct
  #   #P_Ca = 2.5 *P_Ca # correct
  #   # P_CaNa = 2.5 * P_CaNa # correct
  #   # P_CaK = 2.5 * P_CaK # correct
  #   #Gto = 4*Gto # correct
  #   #G_NaCa = 1.4 * G_NaCa
  #   #PNaK = 0.7*PNaK # note that PNak = GNaK, correct
  
  P_CaNa=0.00125*P_Ca
  P_CaK=3.574E-4*P_Ca  
  P_Ca_CaMK = 1.1*P_Ca
  P_CaNa_CaMK = 0.00125*P_Ca_CaMK
  P_CaK_CaMK = 3.574E-4*P_Ca_CaMK
  
  initial_conds = [m ,m_L ,h_fast ,h_slow ,h_CaMK_slow ,h_L ,h_L_CaMK ,j ,j_CaMK ,CaMK_trap ,V_m ,a,i_fast,i_slow ,i_CaMK_fast , i_CaMK_slow,a_CaMK,d,f_fast ,f_slow ,j_Ca,f_Ca_CaMK_fast,n, f_Ca_slow ,f_Ca_fast, f_CaMK_fast , x_r_slow ,x_r_fast ,x_s1,x_s2 ,x_K1 ,J_rel_NP,J_rel_CaMK ,Na_ion_conc_i , Na_ion_conc_ss, K_ion_conc_i , K_ion_conc_ss , Ca_ion_conc_i , Ca_ion_conc_ss , Ca_ion_conc_nsr , Ca_ion_conc_jsr]
  constants = [cell_type,cycle_length,amplitude, duration, PNaK,Na_ion_conc_o, Ca_ion_conc_o, K_ion_conc_o, L, radius, vcell, Ageo, Acap, vmyo, vnsr, vjsr, vss,R, Temp, Faradays_constant, z_Na, z_Ca, z_K, PR_Na_K, alpha_CaMK, beta_CaMK, CaMK_0, K_mCaM, K_m_CaMK, A_h_fast, A_h_slow, Ah_CaMK_fast, Ah_CaMK_slow, GNa_fast, GNa_late, tau_h_L, tau_h_L_CaMK, Gto, GKr, GKs, GK1, K_m_n, k_plus2_n, tau_j_Ca, A_f_fast, A_f_slow, A_f_CaMK_fast, A_f_CaMK_slow, P_Ca, P_CaNa, P_CaK, P_Ca_CaMK, P_CaNa_CaMK, P_CaK_CaMK, gamma_Cai, gamma_Cao, gamma_Nai, gamma_Nao, gamma_Ki, gamma_Ko, P_Nab, P_Cab, Gkb, GpCa, Cm, q_Na, q_Ca, G_NaCa, k_Na3, k_Na1, k_Na2, k_asymm, k_Ca_on, omega_Ca, omega_NaCa, omega_Na, K_mCaAct, K_mCaAct2, K0_Nai, K0_Nao, k1_pos, k1_neg, k2_pos, k2_neg, k3_pos, k3_neg, k4_pos, k4_neg, delta, K_Ki, K_Ko, MgADP, MgATP, beta1, K_MgATP, H_conc, alpha2, beta3_alpha4_denominator, alpha4, sigmaP, K_H_P, K_Na_P, K_K_P, tau_diff_Na, tau_diff_K, tau_diff_Ca, beta_tau, alpha_rel, beta_tau_CaMK, alpha_rel_CaMK, tau_tr, deltaK_m_PLB, deltaJ_up_CaMK, CMDN, K_m_CMDN, TRPN, K_m_TRPN, BSR, K_m_BSR, BSL, K_m_BSL, CSQN, K_m_CSQN, RxT, FoRT, Reversal_potential_cons,K_Ki2,K_Ko2,Jrel_scale,Jup_scale,isepi,h10,h11,h12,k1,xK1_const1,xK1_const2,RK1_const]
  # Solve the ODE
  tspan = (0, cycles*cycle_length)
  start_time = tm.time()
  sol = solve_ivp(fun = ORD_Model, t_span = tspan, y0 = initial_conds, args = constants,method='BDF',rtol= 1e-5,max_step = 0.5)
  end_time = tm.time()
  elapsed_time = end_time - start_time
  print(elapsed_time)

  time = sol.t  # Time points
  solutions = sol.y  # Solution vectors, each row corresponds to a variable
  y_names = ['dm' ,'dmL' ,'dh_fast' ,'dh_slow' ,'dh_CaMK_slow' ,'dhL' ,'dhL_CaMK' ,'dj' ,'dj_CaMK' ,'dCaMK_trap' ,'V' ,'da','di_fast','di_slow' ,'di_CaMK_fast' , 'di_CaMK_slow','da_CaMK','dd','df_fast' ,'df_slow' ,'dj_Ca','df_Ca_CaMK_fast','dn', 'df_Ca_slow' ,'df_Ca_fast', 'df_CaMK_fast' , 'dxr_slow' ,'dxr_fast' ,'dxs1','dxs2' ,'dx_K1' ,'dJ_rel_NP','dJ_rel_CaMK' ,'Nai' , 'dNass', 'Ki' , 'dKss' , 'Cai' , 'dCass' , 'dCansr' , 'dCajsr']
  
  df = pd.DataFrame(solutions.T, columns=y_names)
  
  df['time'] = time
  return df, pd.DataFrame(), duration


@njit
def ORD_Model(time, y, cell_type, cycle_length,amplitude, duration, PNaK,Na_ion_conc_o, Ca_ion_conc_o, K_ion_conc_o, L, radius, vcell, Ageo, Acap, vmyo, vnsr, vjsr, vss,R, Temp, Faradays_constant, z_Na, z_Ca, z_K, PR_Na_K, alpha_CaMK, beta_CaMK, CaMK_0, K_mCaM, K_m_CaMK, A_h_fast, A_h_slow, Ah_CaMK_fast, Ah_CaMK_slow, GNa_fast, GNa_late, tau_h_L, tau_h_L_CaMK, Gto, GKr, GKs, GK1, K_m_n, k_plus2_n, tau_j_Ca, A_f_fast, A_f_slow, A_f_CaMK_fast, A_f_CaMK_slow, P_Ca, P_CaNa, P_CaK, P_Ca_CaMK, P_CaNa_CaMK, P_CaK_CaMK, gamma_Cai, gamma_Cao, gamma_Nai, gamma_Nao, gamma_Ki, gamma_Ko, P_Nab, P_Cab, Gkb, GpCa, Cm, q_Na, q_Ca, G_NaCa, k_Na3, k_Na1, k_Na2, k_asymm, k_Ca_on, omega_Ca, omega_NaCa, omega_Na, K_mCaAct, K_mCaAct2, K0_Nai, K0_Nao, k1_pos, k1_neg, k2_pos, k2_neg, k3_pos, k3_neg, k4_pos, k4_neg, delta, K_Ki, K_Ko, MgADP, MgATP, beta1, K_MgATP, H_conc, alpha2, beta3_alpha4_denominator, alpha4, sigmaP, K_H_P, K_Na_P, K_K_P, tau_diff_Na, tau_diff_K, tau_diff_Ca, beta_tau, alpha_rel, beta_tau_CaMK, alpha_rel_CaMK, tau_tr, deltaK_m_PLB, deltaJ_up_CaMK, CMDN, K_m_CMDN, TRPN, K_m_TRPN, BSR, K_m_BSR, BSL, K_m_BSL, CSQN, K_m_CSQN, RxT, FoRT, Reversal_potential_cons,K_Ki2,K_Ko2,Jrel_scale,Jup_scale,isepi,h10,h11,h12,k1,xK1_const1,xK1_const2,RK1_const): 
  
  #ydot= np.zeros_like(y) 
  
  #unpack initial conditions
  m ,m_L ,h_fast ,h_slow ,h_CaMK_slow ,h_L ,h_L_CaMK ,j ,j_CaMK ,CaMK_trap ,V_m ,a,i_fast,i_slow ,i_CaMK_fast , i_CaMK_slow,a_CaMK,d,f_fast ,f_slow ,j_Ca,f_Ca_CaMK_fast,n, f_Ca_slow ,f_Ca_fast, f_CaMK_fast , x_r_slow ,x_r_fast ,x_s1,x_s2 ,x_K1 ,J_rel_NP,J_rel_CaMK ,Na_ion_conc_i , Na_ion_conc_ss, K_ion_conc_i , K_ion_conc_ss , Ca_ion_conc_i , Ca_ion_conc_ss , Ca_ion_conc_nsr , Ca_ion_conc_jsr = y
  #cycle_length,amplitude, duration, Na_ion_conc_o, Ca_ion_conc_o, K_ion_conc_o, L, radius, vcell, Ageo, Acap, vmyo, vnsr, vjsr, vss,R, Temp, Faradays_constant, z_Na, z_Ca, z_K, PR_Na_K, alpha_CaMK, beta_CaMK, CaMK_0, K_mCaM, K_m_CaMK, A_h_fast, A_h_slow, Ah_CaMK_fast, Ah_CaMK_slow, GNa_fast, GNa_late, tau_h_L, tau_h_L_CaMK, Gto, GKr, GKs, GK1, K_m_n, k_plus2_n, tau_j_Ca, A_f_fast, A_f_slow, A_f_CaMK_fast, A_f_CaMK_slow, P_Ca, P_CaNa, P_CaK, P_Ca_CaMK, P_CaNa_CaMK, P_CaK_CaMK, gamma_Cai, gamma_Cao, gamma_Nai, gamma_Nao, gamma_Ki, gamma_Ko, P_Nab, P_Cab, Gkb, GpCa, Cm, q_Na, q_Ca, G_NaCa, k_Na3, k_Na1, k_Na2, k_asymm, k_Ca_on, omega_Ca, omega_NaCa, omega_Na, K_mCaAct, K_mCaAct2, K0_Nai, K0_Nao, k1_pos, k1_neg, k2_pos, k2_neg, k3_pos, k3_neg, k4_pos, k4_neg, delta, K_Ki, K_Ko, MgADP, MgATP, beta1, K_MgATP, H_conc, alpha2, beta3_alpha4_denominator, alpha4, sigmaP, K_H_P, K_Na_P, K_K_P, tau_diff_Na, tau_diff_K, tau_diff_Ca, beta_tau, alpha_rel, beta_tau_CaMK, alpha_rel_CaMK, tau_tr, deltaK_m_PLB, deltaJ_up_CaMK, CMDN, K_m_CMDN, TRPN, K_m_TRPN, BSR, K_m_BSR, BSL, K_m_BSL, CSQN, K_m_CSQN, RxT, FoRT, Reversal_potential_cons = constants
  ### The stimulus equation
  #amplitude=−80.0 μAμF, duration=0.5 ms
  #For charge conservation sake, stimulus has K+ identity as described by Hund et al.3.
  I_stim = Istim(time,cycle_length,amplitude,duration)
  
  E_Na = Reversal_potential_cons*np.log(Na_ion_conc_o/Na_ion_conc_i)
  E_K = Reversal_potential_cons*np.log(K_ion_conc_o/K_ion_conc_i)
  E_Ks = Reversal_potential_cons*np.log((K_ion_conc_o+PR_Na_K*Na_ion_conc_o)/(K_ion_conc_i+PR_Na_K*Na_ion_conc_i))
  
  ## Precalculate constants
  V_E_Na = (V_m-E_Na)
  V_E_K = (V_m-E_K)
  V_E_Ks = (V_m-E_Ks)
  VFoRT = V_m*FoRT
  
  ###################################
  #The Calcium/Calmodulin-Depenent Protein Kinase (CaMK)
  CaMK_bound=CaMK_0*((1-CaMK_trap)/(1+(K_mCaM/Ca_ion_conc_ss)))
  CaMK_active=CaMK_bound+CaMK_trap
  dCaMK_trap=alpha_CaMK*CaMK_bound*(CaMK_bound+CaMK_trap)-beta_CaMK*CaMK_trap
  phi = 1/(1+(K_m_CaMK/CaMK_active))
  one_minus_phi = 1-phi
  ###################################
  
  ###################################
  #INa Calculation
  # Gating variables
  m_inf = 1 / (1 + np.exp(-(V_m + 39.57) / 9.871))
  h_inf= 1 / (1 + np.exp((V_m + 82.9) / 6.086))
  m_L_inf= 1/(1+np.exp(-(V_m+42.85)/5.264))
  h_L_inf= 1/(1+np.exp((V_m+87.61)/7.488)) #fixed
  h_CaMK_inf=1/(1+np.exp((V_m+89.1)/6.086))
  h_L_CaMK_inf= 1/(1+np.exp((V_m+93.81)/7.488)) #fixed
  j_inf = h_inf
  j_CaMK_inf = j_inf
  
  #time constants
  tau_m=1/((6.765*np.exp((V_m+11.64)/34.77))+(8.552*np.exp(-(V_m+77.42)/5.955)))
  tau_m_L=tau_m
  tau_h_fast= 1 / (1.432E-5 * np.exp(-(V_m + 1.196) / 6.285) + 6.149 * np.exp((V_m + 0.5096) / 20.27))
  tau_h_slow=1 / (0.009764*np.exp(-(V_m+17.95)/28.05)+0.3343*np.exp((V_m+5.730)/56.66))
  tau_j= 2.038 +( 1 / (0.02136 * np.exp(-(V_m + 100.6) / 8.281) + 0.3052 * np.exp((V_m + 0.9941) / 38.45)))
  tau_h_CaMK_slow= 3.0*tau_h_slow
  tau_j_CaMK= 1.46*tau_j
  
  #differentials
  dm = (m_inf - m)/tau_m
  dh_fast = (h_inf - h_fast)/tau_h_fast
  dh_slow = (h_inf - h_slow)/tau_h_slow
  dj = (j_inf - j)/tau_j
  dh_CaMK_slow = (h_CaMK_inf-h_CaMK_slow)/tau_h_CaMK_slow
  dj_CaMK = (j_CaMK_inf-j_CaMK)/tau_j_CaMK
  dmL = (m_L_inf-m_L)/tau_m_L
  dhL = (h_L_inf-h_L)/tau_h_L
  dhL_CaMK = (h_L_CaMK_inf-h_L_CaMK)/tau_h_L_CaMK
  
  # Integrated Parameters calculations
  h = (A_h_fast*h_fast)+(A_h_slow*h_slow)
  h_CaMK_fast = h_fast
  h_CaMK=Ah_CaMK_fast*h_CaMK_fast+Ah_CaMK_slow*h_CaMK_slow
  
  # Current Calculations
  INa_fast = GNa_fast * V_E_Na * (m**3) * (one_minus_phi*h*j+phi*h_CaMK*j_CaMK)
  INa_late = GNa_late * V_E_Na * m_L * (one_minus_phi*h_L+phi*h_L_CaMK)
  INa = INa_fast + INa_late
  #############################
  
  #############################
  ### Transient Outward Potassium Current (Ito)
  
  delta_epi = 1
  if isepi:
    delta_epi = 1-(0.95/(1+np.exp((V_m+70)/5)))
    
  #Gating variables
  a_inf=1/(1+np.exp(-(V_m-14.34)/14.82))
  i_inf=1/(1+np.exp((V_m+43.94)/5.711))
  i_CaMK_inf=i_inf
  a_CaMK_inf=1/(1+np.exp(-(V_m-24.34)/14.82))
  
  #Time constants
  tau_a=1.0515/((1/(1.2089*(1+np.exp(-(V_m-18.4099)/29.3814))))+(3.5/(1+np.exp((V_m+100)/29.3814))))
  tau_a_CaMK=tau_a
  tau_i_fast=delta_epi*(4.562+(1/(0.3933*np.exp(-(V_m+100)/100)+0.08004*np.exp((V_m+50)/16.59))))
  tau_i_slow=delta_epi*(23.62+(1/(0.001416*np.exp(-(V_m+96.52)/59.05)+1.7808E-8*np.exp((V_m+114.1)/8.079))))
  sigma_CaMK_dev=1.354+(1E-4/(np.exp((V_m-167.4)/15.89)+np.exp(-(V_m-12.23)/0.2154)))
  sigma_CaMK_recov=1-(0.5/(1+np.exp((V_m+70)/20)))
  sigma_product = sigma_CaMK_dev*sigma_CaMK_recov
  tau_i_CaMK_fast=tau_i_fast*sigma_product
  tau_i_CaMK_slow=tau_i_slow*sigma_product
  
  #differentials
  da = (a_inf - a)/tau_a
  di_fast = (i_inf - i_fast)/tau_i_fast
  di_slow = (i_inf - i_slow)/tau_i_slow
  da_CaMK = (a_CaMK_inf - a_CaMK)/tau_a_CaMK
  di_CaMK_fast = (i_CaMK_inf - i_CaMK_fast)/tau_i_CaMK_fast
  di_CaMK_slow = (i_CaMK_inf - i_CaMK_slow)/tau_i_CaMK_slow
  
  # Integrated Parameters calculations
  A_i_fast=1/(1+np.exp((V_m-213.6)/151.2))
  A_i_CaMK_fast=A_i_fast
  A_i_slow=1-A_i_fast
  A_i_CaMK_slow=A_i_slow
  i = A_i_fast *i_fast + A_i_slow*i_slow
  i_CaMK = A_i_CaMK_fast *i_CaMK_fast + A_i_CaMK_slow*i_CaMK_slow
  
  #Current Calculations
  Ito = Gto * V_E_K * (one_minus_phi*a*i + phi*a_CaMK*i_CaMK)
  ##############################################
  
  ###############################################
  #L-type Calcium Current (ICaL)
  #gating variables
  d_inf=1/(1+np.exp(-(V_m+3.94)/4.23))
  f_inf=1/(1+np.exp((V_m+19.58)/3.696))
  f_Ca_inf=f_inf
  j_Ca_inf=f_Ca_inf
  f_CaMK_inf=f_inf
  f_Ca_CaMK_inf=f_inf

  #time constants
  tau_const1 = (V_m+20)/10
  tau_const2 = (V_m-4)/7
  tau_const3 = (V_m+5)
  tau_d=0.6+(1/(np.exp(-0.05*(V_m+6))+np.exp(0.09*(V_m+14))))
  tau_f_fast=7+(1/(0.0045*np.exp(-tau_const1)+0.0045*np.exp(tau_const1)))
  tau_f_slow=1000+(1/(0.000035*(np.exp(-tau_const3/4)+np.exp(tau_const3/6))))
  tau_f_Ca_fast=7+(1/(0.04*(np.exp(-tau_const2)+np.exp(tau_const2))))
  tau_f_Ca_slow=100+(1/(0.00012*(np.exp(-(V_m)/3)+np.exp((V_m)/7))))
  tau_f_CaMK_fast=2.5*tau_f_fast
  tau_f_Ca_CaMK_fast=2.5*tau_f_Ca_fast
  
  #Differentials
  dd = (d_inf - d)/tau_d
  df_fast = (f_inf - f_fast)/tau_f_fast
  df_slow = (f_inf - f_slow)/tau_f_slow
  df_Ca_fast = (f_Ca_inf - f_Ca_fast)/tau_f_Ca_fast
  df_Ca_slow = (f_Ca_inf - f_Ca_slow)/tau_f_Ca_slow
  dj_Ca = (j_Ca_inf - j_Ca)/tau_j_Ca
  df_CaMK_fast = (f_CaMK_inf - f_CaMK_fast)/tau_f_CaMK_fast
  df_Ca_CaMK_fast = (f_Ca_CaMK_inf - f_Ca_CaMK_fast)/tau_f_Ca_CaMK_fast
  
  #Integrated Parameter calculations
  A_f_Ca_fast= 0.3+(0.6/(1+np.exp((V_m-10)/10)))
  A_f_Ca_CaMK_fast=A_f_Ca_fast
  A_f_Ca_slow = 1-A_f_Ca_fast
  A_f_Ca_CaMK_slow = A_f_Ca_slow
  f = A_f_fast*f_fast + A_f_slow*f_slow
  f_Ca = A_f_Ca_fast*f_Ca_fast + A_f_Ca_slow*f_Ca_slow
  f_CaMK_slow = f_slow
  f_CaMK = A_f_CaMK_fast*f_CaMK_fast + A_f_CaMK_slow*f_CaMK_slow
  f_Ca_CaMK_slow = f_Ca_slow
  f_Ca_CaMK = A_f_Ca_CaMK_fast*f_Ca_CaMK_fast + A_f_Ca_CaMK_slow*f_Ca_CaMK_slow
  k_minus2_n = j_Ca
  alpha_n = 1/((k_plus2_n/k_minus2_n)+(1+(K_m_n/Ca_ion_conc_ss))**4)
  dn = (alpha_n*k_plus2_n)-(n*k_minus2_n)
  
  #Calculate the Psi values
  #pre-calculate some constants
  VFort_Frdy = (VFoRT*Faradays_constant)
  Ca_exp_const = np.exp(2*VFoRT)
  Na_K_exp_const = np.exp(VFoRT)
  Na_K_denominator = (Na_K_exp_const-1)
  Ca_denominator = (Ca_exp_const-1)
  Ca_numerator_const = gamma_Cao*Ca_ion_conc_o
  Psi_Ca = 4*VFort_Frdy*((gamma_Cai*Ca_ion_conc_ss*Ca_exp_const-Ca_numerator_const)/Ca_denominator)
  Psi_CaNa = 1*VFort_Frdy*((gamma_Nai*Na_ion_conc_ss*Na_K_exp_const-gamma_Nao*Na_ion_conc_o)/Na_K_denominator)
  Psi_CaK = 1*VFort_Frdy*((gamma_Ki*K_ion_conc_ss*Na_K_exp_const-gamma_Ko*K_ion_conc_o)/Na_K_denominator)
  
  #Calculate currents
  I_bar_CaL = P_Ca*Psi_Ca
  I_bar_CaNa = P_CaNa*Psi_CaNa
  I_bar_CaK = P_CaK*Psi_CaK
  I_bar_CaL_CaMK = P_Ca_CaMK*Psi_Ca
  I_bar_CaNa_CaMK = P_CaNa_CaMK*Psi_CaNa
  I_bar_CaK_CaMK = P_CaK_CaMK * Psi_CaK
  
  n_x_jCa = n*j_Ca
  Ca_current_constants1 = d*one_minus_phi*(f*(1-n)+f_Ca*n_x_jCa)
  Ca_current_constants2 = d*phi*(f_CaMK*(1-n)+f_Ca_CaMK*n_x_jCa)
  ICaL = I_bar_CaL *Ca_current_constants1+I_bar_CaL_CaMK*Ca_current_constants2
  ICaNa = I_bar_CaNa*Ca_current_constants1+I_bar_CaNa_CaMK*Ca_current_constants2
  ICaK = I_bar_CaK*Ca_current_constants1+I_bar_CaK_CaMK*Ca_current_constants2
  ###############################################
  
  ###############################################
  #Rapid Delayed rectifier Potassium Current (Ikr)
  
  #gating variables
  xr_inf=1/(1+np.exp(-(V_m+8.337)/6.789))
  
  #time constants
  tau_xr_fast=12.98+(1/(0.3652*np.exp((V_m-31.66)/3.869)+4.123E-5*np.exp(-(V_m-47.78)/20.38)))
  tau_xr_slow=1.865+(1/(0.06629*np.exp((V_m-34.7)/7.355)+1.128E-5*np.exp(-(V_m-29.74)/25.94)))
  
  #Differentials
  dxr_fast = (xr_inf - x_r_fast)/tau_xr_fast
  dxr_slow = (xr_inf - x_r_slow)/tau_xr_slow
  
  #Calculated parameters
  A_xr_fast=1/(1+np.exp((V_m+54.81)/38.21))
  A_xr_slow= 1-A_xr_fast
  xr = A_xr_fast*x_r_fast + A_xr_slow*x_r_slow
  R_kr= 1/((1+np.exp((V_m+55)/75))*(1+np.exp((V_m-10)/30)))
  
  #Calculate Currents
  IKr = GKr * np.sqrt(K_ion_conc_o/5.4)*xr*R_kr*V_E_K
  ###############################################
  
  ###############################################
  # Slow delayed Rectifier Potassium Current (IKs)
  #Gating Variables
  x_s1_inf=1/(1+np.exp(-(V_m+11.6)/8.932))
  x_s2_inf=x_s1_inf
  
  #time constants
  tau_x_s1=817.3+(1/(2.326E-4*np.exp((V_m+48.28)/17.8)+0.001292*np.exp(-(V_m+210)/230)))
  tau_x_s2=1/(0.01*np.exp((V_m-50)/20)+0.0193*np.exp(-(V_m+66.54)/31))
  
  #Differentials
  dxs1 = (x_s1_inf - x_s1)/tau_x_s1
  dxs2 = (x_s2_inf - x_s2)/tau_x_s2
  
  #Calculate Currents
  IKs = GKs * (1+(0.6/(1+(3.8E-5/Ca_ion_conc_i)**1.4)))*x_s1*x_s2*V_E_Ks
  ###############################################
  
  
  ###############################################
  #Inward Rectifier Potassium Current (I_K1)
  
  #Gating variable
  x_K1_inf=1/(1+(np.exp(-(V_m+xK1_const1)/(xK1_const2))))
  
  #time constants
  tau_x_K1=122.2/(np.exp(-(V_m+127.2)/20.36)+np.exp((V_m+236.8)/69.33))
  
  #Differentials
  dx_K1 = (x_K1_inf-x_K1)/tau_x_K1
  
  #Calculate currents
  
  R_K1=1/(1+np.exp((V_m+RK1_const)/9.493))
  IK1 = GK1*np.sqrt(K_ion_conc_o)*x_K1*R_K1*V_E_K
  ###############################################
  
  
  ###############################################
  #Sodium/calcium Exchange Current (INaCa)
  h_Na_val = np.exp(q_Na*VFoRT)
  h_Ca_val = np.exp(q_Ca*VFoRT)
  h7=1+(Na_ion_conc_o/k_Na3)*(1+(1/h_Na_val))
  h8=(Na_ion_conc_o)/(k_Na3*h_Na_val*h7)
  h9=1/h7
  k3_prime=h9*omega_Ca
  k3_double_prime=h8*omega_NaCa
  k3=k3_prime+ k3_double_prime
  k8 = h8*h11*omega_Na
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
  I_NaCa_i = G_NaCa*0.8*allo_i*(z_Na*J_NaCa_Na_i+z_Ca*J_NaCa_Ca_i)
  I_NaCa_ss = G_NaCa*0.2*allo_ss*(z_Na*J_NaCa_Na_ss+z_Ca*J_NaCa_Ca_ss)
  INaCa = I_NaCa_i + I_NaCa_ss
  
  ###############################################
  
  ###############################################
  #Sodium/Potassium ATPase Current (INaK)
  K_constants = VFoRT/3
  P = sigmaP/(1+(H_conc/K_H_P)+(Na_ion_conc_i/K_Na_P)+(K_ion_conc_i/K_K_P))
  K_Nai_val = K0_Nai*np.exp((delta*K_constants))
  K_Nao_val = K0_Nao*np.exp(((1-delta)*K_constants))
  K_Nao_val3 = K_Nao_val**3
  K_Nai_val3 = K_Nai_val**3
  common_denominator_1 = (((1+Na_ion_conc_o)**3/K_Nao_val3)+((1+K_ion_conc_o)**2/K_Ko2)-1)
  common_denominator_2 = (((1+Na_ion_conc_i)**3/K_Nai_val3)+((1+K_ion_conc_i)**2/K_Ki2)-1)
  alpha1=(k1_pos*((Na_ion_conc_i)**3/K_Nai_val3))/common_denominator_2
  beta2=(k2_neg*((Na_ion_conc_o)**3/K_Nao_val3))/common_denominator_1
  alpha3=(k3_pos*((K_ion_conc_o)**2/K_Ko2))/common_denominator_1
  beta4=(k4_neg*((K_ion_conc_i)**2/K_Ki2))/common_denominator_2
  beta3=(k3_neg*P*H_conc)/beta3_alpha4_denominator
  alpha4_x_alpha1 = alpha4*alpha1
  alpha2_x_alpha3 = alpha2*alpha3
  beta2_x_beta1 = beta2*beta1
  x1_val = alpha4_x_alpha1*alpha2+beta2*beta4*beta3+alpha2*beta4*beta3+beta3*alpha1*alpha2
  x2_val = beta2_x_beta1*beta4+alpha1*alpha2_x_alpha3+alpha3*beta1*beta4+alpha2_x_alpha3*beta4
  x3_val = alpha2_x_alpha3*alpha4+beta3*beta2_x_beta1+beta2_x_beta1*alpha4+alpha3*alpha4*beta1
  x4_val = beta4*beta3*beta2+alpha3*alpha4_x_alpha1+beta2*alpha4_x_alpha1+beta3*beta2*alpha1
  ## need to check if these are correct
  sum_x_val = x1_val + x2_val + x3_val +x4_val
  E1_NAK = E_function(x1_val,sum_x_val)
  E2_NAK = E_function(x2_val,sum_x_val)
  E3_NAK = E_function(x3_val,sum_x_val)
  E4_NAK = E_function(x4_val,sum_x_val)
  J_NaK_Na=3*(E1_NAK*alpha3-E2_NAK*beta3)
  J_NaK_K=2*(E4_NAK*beta1-E3_NAK*alpha1)
  INaK = PNaK*(z_Na*J_NaK_Na+z_K*J_NaK_K)
  ###############################################
  
  ###############################################
  #Background Currents
  #Gating Variables
  x_kb =  1 / (1 + np.exp(-(V_m - 14.48) / 18.34))
  
  #Current calculations
  INab = P_Nab*1*VFort_Frdy*((Na_ion_conc_i*Na_K_exp_const-Na_ion_conc_o)/(Na_K_denominator))
  ICab = P_Cab*4*VFort_Frdy*((gamma_Cai*Ca_ion_conc_i*Ca_exp_const-Ca_numerator_const)/Ca_denominator)
  Ikb = Gkb*x_kb*V_E_K
  IpCa = GpCa*(Ca_ion_conc_i/(0.0005+Ca_ion_conc_i))
  ###############################################
  
  ###############################################
  #Voltage
  dV = (-1/Cm)*(INa+Ito+ICaL+ICaNa+ICaK+IKr+IKs+IK1+INaCa+INaK+INab+ICab+Ikb+IpCa+I_stim)
  ###############################################
  
  ###############################################
  #Diffusion Fluxes
  J_diff_Na = (Na_ion_conc_ss-Na_ion_conc_i)/tau_diff_Na
  J_diff_Ca = (Ca_ion_conc_ss-Ca_ion_conc_i)/tau_diff_Ca
  J_diff_K = (K_ion_conc_ss-K_ion_conc_i)/tau_diff_K
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
  J_rel = one_minus_phi*J_rel_NP+phi*J_rel_CaMK
  
  #Calcium Uptake via SERCA Pump (Jup)
      
  J_SERCA_const = (0.004375*Ca_ion_conc_i)
  J_leak = (0.0039375*Ca_ion_conc_nsr)/15
  J_up_NP = Jup_scale*(J_SERCA_const/(0.00092+Ca_ion_conc_i))
  J_up_CaMK = Jup_scale*((1+deltaJ_up_CaMK)*(J_SERCA_const/(0.00092-deltaK_m_PLB+Ca_ion_conc_i)))
  J_up = one_minus_phi*J_up_NP+phi*J_up_CaMK-J_leak
  
  #Calcium Translocation from NSR to JSR (Jtr)
  J_tr = (Ca_ion_conc_nsr-Ca_ion_conc_jsr)/tau_tr
  ###############################################
  
  ###############################################
  #ORd Model concentrations and buffers
  
  # some constants
  vss_div_vmyo = (vss/vmyo)
  Acap_vmyo_const = (Acap/(Faradays_constant*vmyo))
  Acap_vss_const = (Acap/(Faradays_constant*vss))
  
  #rate of change of Na ions 
  dNai = -(INa+INa_late+3*I_NaCa_i+3*INaK+INab)*Acap_vmyo_const+(J_diff_Na*vss_div_vmyo)
  dNass = -(ICaNa+3*I_NaCa_ss)*Acap_vss_const-J_diff_Na
  # rate of change of K ions
  # There needs to be a stimulus current in the below equation Istim)
  dKi = -(Ito+IKr+IKs+IK1+Ikb+I_stim-2*INaK)*Acap_vmyo_const+(J_diff_K*vss_div_vmyo)
  dKss = -ICaK * Acap_vss_const - J_diff_K
  # rate of change of Ca ions
  beta_Cai = 1/(1+(CMDN*K_m_CMDN/(K_m_CMDN+Ca_ion_conc_i)**2)+(TRPN*K_m_TRPN/(K_m_TRPN+Ca_ion_conc_i)**2))
  beta_Cass = 1/(1+(BSR*K_m_BSR/(K_m_BSR+Ca_ion_conc_ss)**2)+(BSL*K_m_BSL/(K_m_BSL+Ca_ion_conc_ss)**2))
  beta_Cajsr = 1/(1+((CSQN*K_m_CSQN)/(K_m_CSQN+Ca_ion_conc_jsr)**2))
  dCai = beta_Cai*(-(IpCa+ICab-2*I_NaCa_i)*(Acap_vmyo_const/2)-J_up*(vnsr/vmyo)+J_diff_Ca*vss_div_vmyo)
  dCass = beta_Cass *(-(ICaL-2*I_NaCa_ss)*(Acap_vss_const/2)+J_rel*(vjsr/vss)-J_diff_Ca)
  dCansr = J_up-J_tr*(vjsr/vnsr)
  dCajsr = beta_Cajsr * (J_tr - J_rel)
  ###############################################
  
  ydot = [dm ,dmL ,dh_fast ,dh_slow ,dh_CaMK_slow ,dhL ,dhL_CaMK ,dj ,dj_CaMK ,dCaMK_trap ,dV ,da,di_fast,di_slow ,di_CaMK_fast , di_CaMK_slow,da_CaMK,dd,df_fast ,df_slow ,dj_Ca,df_Ca_CaMK_fast,dn, df_Ca_slow ,df_Ca_fast, df_CaMK_fast , dxr_slow ,dxr_fast ,dxs1,dxs2 ,dx_K1 ,dJ_rel_NP,dJ_rel_CaMK ,dNai , dNass, dKi , dKss , dCai , dCass , dCansr , dCajsr]
  return ydot
  
# cycles = 10
# cycle_length = 500
# df, currents_df, duration = run_ORd_Model(cycles,cycle_length,1,"EPI", 0.00476,0.0598,0.22896,0.08,0.0045,75,3.75E-10,2.5E-8,1.2*0.0001,1.1*0.0008,0.0005,0.6*0.003,0.9*30,5.4)
