import numpy as np
import math
from scipy.integrate import solve_ivp
from scipy.integrate import ode
import time as tm
import pandas as pd
import os
import matplotlib.pyplot as plt
from numba import njit
from conductances import *

# A novel computational model of the human ventricular action potential and Ca transient.
# Grandi E, Pasqualini FS, Bers DM.
# J Mol Cell Cardiol. 2010 Jan48(1):112-21. Epub 2009 Oct 14.
# PMID:19835882

# 02/12/2019 corrected equation for J_CaB_cytosol
#7.6 minutes for 1010 cycles

# @njit
# def Iapp(time,cycle_length, amplitude, duration):
#   if time % (cycle_length) <= duration:
#     I_app = 9.5
#   else:
#     I_app = 0.0
#   return (I_app)
#

def run_GB_model(cycles, cycle_length, cell_type, amp=9.5):

  model_type = "Grandi Bers 2010"
  G_ks                    = GKs_conductance(model_type, cell_type)
  G_Kr                    = GKr_conductance(model_type, cell_type)
  G_K1                    = GK1_conductance(model_type, cell_type)
  G_to_Slow, G_to_Fast    = Gto_conductance(model_type, cell_type)
  G_Na                    = GNa_conductance(model_type, cell_type)
  GCaL, GCaNa, GCaK       = GCa_conductance(model_type, cell_type)
  GNCX                    = GNCX_conductance(model_type, cell_type)
  GNaK                    = GNaK_conductance(model_type, cell_type)
  GKb                     = GKb_conductance(model_type, cell_type)
  GNab                    = GNab_conductance(model_type, cell_type)
  GCab                    = GCab_conductance(model_type, cell_type)
  GpCa                    = GpCa_conductance(model_type, cell_type)
  GClCa_input             = GClCa_conductance(model_type, cell_type)
  GClb                    = GClb_conductance(model_type, cell_type)

  #default GKs params:
  # gks_junc=1*0.0035
  # gks_sl=1*0.0035 #FRA
  amplitude = amp
  gks_junc = G_ks
  gks_sl = G_ks
  G_Kr = G_Kr ### original GKr value is 0.0350
  GKi = G_K1 #original value was 0.35
  GtoSlow = G_to_Slow
  GtoFast = G_to_Fast
  # if celltype=='EPI':
  #     GtoSlow = 1.0*0.13*0.12
  #     GtoFast = 1.0*0.13*0.88
  # else:
  #     GtoSlow = 0.13*0.3*0.964
  #     GtoFast = 0.13*0.3*0.036
  GNa= G_Na # was23        # [mS/uF]
  duration = 5
  ## Initial conditions
  mo=1.405627e-3
  ho= 9.867005e-1
  jo=9.915620e-1
  do=7.175662e-6
  fo=1.000681
  fcaBjo=2.421991e-2
  fcaBslo=1.452605e-2
  xtoso=4.051574e-3
  ytoso=9.945511e-1
  xtofo=4.051574e-3
  ytofo= 9.945511e-1
  xkro=8.641386e-3
  xkso= 5.412034e-3
  RyRro=8.884332e-1
  RyRoo=8.156628e-7
  RyRio=1.024274e-7
  NaBjo=3.539892
  NaBslo=7.720854e-1
  TnCLo=8.773191e-3
  TnCHco=1.078283e-1
  TnCHmo=1.524002e-2
  CaMo=2.911916e-4
  Myoco=1.298754e-3
  Myomo=1.381982e-1
  SRBo=2.143165e-3
  SLLjo=9.566355e-3
  SLLslo=1.110363e-1
  SLHjo=7.347888e-3
  SLHslo=7.297378e-2
  Csqnbo= 1.242988
  Ca_sro=0.1e-1 #5.545201e-1
  Najo=9.06#8.80329
  Naslo=9.06#8.80733
  Naio=9.06#8.80853
  Kio=120
  Cajo=1.737475e-4
  Caslo= 1.031812e-4
  Caio=8.597401e-5
  Vmo=-8.09763e+1
  rtoso=0.9946
  ICajuncinto=1
  ICaslinto=0
  C1o=0.0015       # []
  C2o=0.0244       # []
  C3o=0.1494       # []
  C4o=0.4071       # []
  C5o=0.4161       # []
  C7o=0.0001       # []
  C8o=0.0006       # []
  C9o=0.0008       # []
  C10o=0           # []
  C11o=0           # []
  C12o=0           # []
  C13o=0           # []
  C14o=0           # []
  C15o=0           # []
  O1o=0            # []
  O2o=0            # []
  C6o=1-(C1o+C2o+C3o+C4o+C5o+C7o+C8o+C9o+C10o+C11o+C12o+C13o+C14o+C15o+O1o+O2o)       # []

  # Gating variables
  #   1       2       3       4       5       6       7       8       9       10      11      12      13
  ##   m       h       j       d       f       fcaBj   fcaBsl   xtos    ytos    xtof    ytof    xkr     xks
  #y10=[1.2e-30.99   0.99   0.0    1.0    0.0141 0.0141     0      1      0.0    1.0    0.0    6e-3]
  y10=[1,mo, ho, jo, do, fo, fcaBjo, fcaBslo, xtoso, ytoso, xtofo, ytofo, xkro, xkso]
  # RyR and Buffering variables
  #   14      15      16      17      18      19      20      21      22      23      24
  ##   RyRr    RyRo    RyRi    NaBj    NaBsl   TnCL    TnCHc   TnCHm   CaM     Myoc    Myom
  y20=[RyRro, RyRoo, RyRio, NaBjo, NaBslo, TnCLo, TnCHco, TnCHmo, CaMo, Myoco, Myomo]
  #y20=[1     0      0      1.8   0.8    0.012   0.112  0.01   0.4e-3 1.9e-3 0.135]
  # More buffering variables
  #   25      26      27      28      29      30
  ##   SRB     SLLj   SLLsl    SLHj    SLHsl  Csqnb
  y30=[SRBo, SLLjo, SLLslo, SLHjo, SLHslo, Csqnbo]
  #y30=[3.3e-3 0.012 0.012 0.13  0.13  1.5]
  #   Intracellular concentrations/ Membrane voltage
  #    31      32      33      34      35      36      37     38     39    40   41
  ##    Ca_sr   Naj     Nasl    Nai     Ki      Caj    Casl    Cai   Vm  rtos ?
  y40=[Ca_sro, Najo, Naslo, Naio, Kio, Cajo, Caslo, Caio, Vmo, rtoso, 1]
  # y50=[C1o, C2o, C3o, C4o ,C5o ,C6o ,C7o, C8o ,C9o, C10o ,C11o ,C12o ,C13o ,C14o, C15o, O1o]
  #y40=[0.9    8.8    8.8    8.8    135    0.1e-3 0.1e-3 0.1e-3 -88  0.89 0          0]
  # y50=[UIC3o UIC2o UIFo UIM1o UC3o UC2o UC1o UOo UIM2o LC3o LC2o LC1o LOo ]

  # Put everything together
  y0  = np.concatenate([y10,y20,y30,y40])

  ### Parameters
  ## Model Parameters
  # EPI or ENDO?
  #celltype = 'EPI'

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
  Vjunc = 0.0539*.01*Vcell
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
  Ko = 5.4   # Extracellular K   [mM]
  Nao = 140  # Extracellular Na  [mM]
  Cao = 1.8  # Extracellular Ca  [mM]1.8
  Mgi = 1    # Intracellular Mg  [mM]

  # Na currents/transport parameters
  GNaB = GNab#0.597e-3    # [mS/uF] 0.897e-3
  IbarNaK = GNaK#1.0*1.8#1.90719     # [uA/uF]
  KmNaip = 11         # [mM]11
  KmKo =1.5         # [mM]1.5
  Q10NaK = 1.63
  Q10KmNai = 1.39

  # K current parameters
  pNaK = 0.01833
  gkp = GKb#2*0.001

  # Cl current parameters
  GClCa =GClCa_input#0.5* 0.109625   # [mS/uF]
  GClB = GClb#1*9e-3        # [mS/uF]
  KdClCa = 100e-3    # [mM]

  # I_Ca parameters
  pNa = GCaNa#0.50*1.5e-8       # [cm/sec]
  pCa = GCaL#0.50*5.4e-4       # [cm/sec]
  pK = GCaK#0.50*2.7e-7        # [cm/sec]
  Q10CaL = 1.8

  # Ca transport parameters
  IbarNCX = GNCX#1.0*4.5      # [uA/uF]5.5 before - 9 in rabbit
  KmCai = 3.59e-3    # [mM]
  KmCao = 1.3        # [mM]
  KmNai = 12.29      # [mM]
  KmNao = 87.5       # [mM]
  ksat = 0.32        # [none]
  nu = 0.27          # [none]
  Kdact = 0.150e-3   # [mM]
  Q10NCX = 1.57      # [none]
  IbarSLCaP = GpCa #0.0673 # IbarSLCaP FEI changed [uA/uF](2.2 umol/L cytosol/sec) jeff 0.093 [uA/uF]
  KmPCa = 0.5e-3     # [mM]
  GCaB = GCab#5.513e-4    # [uA/uF] 3
  Q10SLCaP = 2.35    # [none]

  # SR flux parameters
  Q10SRCaP = 2.6          # [none]
  Vmax_SRCaP = 1.0*5.3114e-3  # [mM/msec] (286 umol/L cytosol/sec)
  Kmf = 0.246e-3          # [mM] default
  #Kmf = 0.175e-3          # [mM]
  Kmr = 1.7               # [mM]L cytosol
  hillSRCaP = 1.787       # [mM]
  ks = 25                 # [1/ms]
  koCa = 10               # [mM**-2 1/ms]   #default 10   modified 20
  kom = 0.06              # [1/ms]
  kiCa = 0.5              # [1/mM/ms]
  kim = 0.005             # [1/ms]
  ec50SR = 0.45           # [mM]

  # Buffering parameters
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
  Bmax_SLlowsl = 37.4e-3*Vmyo/Vsl        # [mM]    # SL buffering
  Bmax_SLlowj = 4.6e-3*Vmyo/Vjunc*0.1    # [mM]    #Fei *0.1!!! junction reduction factor
  koff_sll = 1300e-3     # [1/ms]
  kon_sll = 100          # [1/mM/ms]
  Bmax_SLhighsl = 13.4e-3*Vmyo/Vsl       # [mM]
  Bmax_SLhighj = 1.65e-3*Vmyo/Vjunc*0.1  # [mM] #Fei *0.1!!! junction reduction factor
  koff_slh = 30e-3       # [1/ms]
  kon_slh = 100          # [1/mM/ms]
  Bmax_Csqn = 140e-3*Vmyo/Vsr            # [mM] # Bmax_Csqn = 2.6      # Csqn buffering
  koff_csqn = 65         # [1/ms]
  kon_csqn = 100         # [1/mM/ms]

  # precalculated parameters removed from ODE to speed up run times
  FoRT_reciprocal = (1/FoRT)
  ecl = FoRT_reciprocal*np.log(Cli/Clo)            # [mV]
  sigma = (np.exp(Nao/67.3)-1)/7
  KmNaip4 = KmNaip**4
  KoKmKo = (Ko+KmKo)
  IbarNaKKo = IbarNaK*Ko
  gkr =1.0*G_Kr*np.sqrt(Ko/5.4)
  # gks_junc=1*0.0035
  # gks_sl=1*0.0035 #FRA
  Kdact2 = Kdact**2
  Nao3 = Nao**3
  KmPCa16 = KmPCa**1.6
  KmNai3 = KmNai**3
  MaxSR = 15
  MinSR = 1
  KmNao3 = KmNao**3
  constants = [cycle_length,cell_type, R, Frdy, Temp, FoRT, Cmem, Qpow, cellLength, cellRadius, junctionLength, junctionRadius, distSLcyto, distJuncSL, DcaJuncSL, DcaSLcyto,
               DnaJuncSL, DnaSLcyto, Vcell, Vmyo, Vsr, Vsl, Vjunc, SAjunc, SAsl, J_ca_juncsl, J_ca_slmyo, J_na_juncsl, J_na_slmyo, Fjunc, Fsl, Fjunc_CaL, Fsl_CaL, Cli, Clo, Ko, Nao,
               Cao, Mgi, GNa, GNaB, IbarNaK, KmNaip, KmKo, Q10NaK, Q10KmNai, pNaK, gkp, GClCa, GClB, KdClCa, pNa, pCa, pK, Q10CaL, IbarNCX, KmCai, KmCao, KmNai, KmNao, ksat, nu, Kdact,
               Q10NCX, IbarSLCaP, KmPCa, GCaB, Q10SLCaP, Q10SRCaP, Vmax_SRCaP, Kmf, Kmr, hillSRCaP, ks, koCa, kom, kiCa, kim, ec50SR, Bmax_Naj, Bmax_Nasl, koff_na, kon_na, Bmax_TnClow,
               koff_tncl, kon_tncl, Bmax_TnChigh, koff_tnchca, kon_tnchca, koff_tnchmg, kon_tnchmg, Bmax_CaM, koff_cam, kon_cam, Bmax_myosin, koff_myoca, kon_myoca, koff_myomg, kon_myomg,
               Bmax_SR, koff_sr, kon_sr, Bmax_SLlowsl, Bmax_SLlowj, koff_sll, kon_sll, Bmax_SLhighsl, Bmax_SLhighj, koff_slh, kon_slh, Bmax_Csqn, koff_csqn, kon_csqn, GtoSlow, GtoFast,
               FoRT_reciprocal, ecl, sigma, KmNaip4, KoKmKo, IbarNaKKo, gkr, gks_junc, gks_sl, Kdact2, Nao3, KmPCa16, KmNai3, MaxSR, MinSR, KmNao3,GKi,duration,amplitude]
  tspan = (0, cycles*cycle_length)
  start_time = tm.time()
  sol = solve_ivp(GB_ode, tspan, y0, args = constants, method='BDF',rtol= 1e-5,max_step = 1)
  end_time = tm.time()
  elapsed_time = end_time - start_time

  time = sol.t  # Time points
  solutions = sol.y  # Solution vectors, each row corresponds to a variable

  # Plot each solution vector against time
  num_variables = solutions.shape[0]  # Number of variables in the system

  y10_n = ['nothing','mo', 'ho', 'jo', 'do', 'fo', 'fcaBjo', 'fcaBslo', 'xtoso', 'ytoso', 'xtofo', 'ytofo', 'xkro', 'xkso'] #12
  y20_n = ['RyRro', 'RyRoo', 'RyRio', 'NaBjo', 'NaBslo', 'TnCLo', 'TnCHco', 'TnCHmo', 'CaMo', 'Myoco', 'Myomo'] #23
  y30_n = ['SRBo', 'SLLjo', 'SLLslo', 'SLHjo', 'SLHslo', 'Csqnbo'] #29
  y40_n = ['Ca_sro', 'Najo', 'Naslo', 'Nai', 'Ki', 'Cajo', 'Caslo', 'Cai', 'V','rtos','nothing2'] #38

  y_names = np.concatenate([y10_n, y20_n, y30_n, y40_n])
  df = pd.DataFrame(solutions.T, columns=y_names)

  df['time'] = time
  stim_duration = 2
  return df, pd.DataFrame(), stim_duration

@njit
def GB_ode(t, y, cycle_length,celltype, R, Frdy, Temp, FoRT, Cmem, Qpow, cellLength, cellRadius, junctionLength, junctionRadius, distSLcyto, distJuncSL, DcaJuncSL, DcaSLcyto, DnaJuncSL, DnaSLcyto, Vcell, Vmyo, Vsr, Vsl, Vjunc, SAjunc, SAsl, J_ca_juncsl, J_ca_slmyo, J_na_juncsl, J_na_slmyo, Fjunc, Fsl, Fjunc_CaL, Fsl_CaL, Cli, Clo, Ko, Nao, Cao, Mgi, GNa, GNaB, IbarNaK, KmNaip, KmKo, Q10NaK, Q10KmNai, pNaK, gkp, GClCa, GClB, KdClCa, pNa, pCa, pK, Q10CaL, IbarNCX, KmCai, KmCao, KmNai, KmNao, ksat, nu, Kdact, Q10NCX, IbarSLCaP, KmPCa, GCaB, Q10SLCaP, Q10SRCaP, Vmax_SRCaP, Kmf, Kmr, hillSRCaP, ks, koCa, kom, kiCa, kim, ec50SR, Bmax_Naj, Bmax_Nasl, koff_na, kon_na, Bmax_TnClow, koff_tncl, kon_tncl, Bmax_TnChigh, koff_tnchca, kon_tnchca, koff_tnchmg, kon_tnchmg, Bmax_CaM, koff_cam, kon_cam, Bmax_myosin, koff_myoca, kon_myoca, koff_myomg, kon_myomg, Bmax_SR, koff_sr, kon_sr, Bmax_SLlowsl, Bmax_SLlowj, koff_sll, kon_sll, Bmax_SLhighsl, Bmax_SLhighj, koff_slh, kon_slh, Bmax_Csqn, koff_csqn, kon_csqn, GtoSlow, GtoFast, FoRT_reciprocal, ecl, sigma, KmNaip4, KoKmKo, IbarNaKKo, gkr, gks_junc, gks_sl, Kdact2, Nao3, KmPCa16, KmNai3, MaxSR, MinSR, KmNao3,GKi,duration, amplitude):

    ydot = np.zeros_like(y)
    # cycle_length,epi, R, Frdy, Temp, FoRT, Cmem, Qpow, cellLength, cellRadius, junctionLength, junctionRadius, distSLcyto, distJuncSL, DcaJuncSL, DcaSLcyto, DnaJuncSL, DnaSLcyto, Vcell, Vmyo, Vsr, Vsl, Vjunc, SAjunc, SAsl, J_ca_juncsl, J_ca_slmyo, J_na_juncsl, J_na_slmyo, Fjunc, Fsl, Fjunc_CaL, Fsl_CaL, Cli, Clo, Ko, Nao, Cao, Mgi, GNa, GNaB, IbarNaK, KmNaip, KmKo, Q10NaK, Q10KmNai, pNaK, gkp, GClCa, GClB, KdClCa, pNa, pCa, pK, Q10CaL, IbarNCX, KmCai, KmCao, KmNai, KmNao, ksat, nu, Kdact, Q10NCX, IbarSLCaP, KmPCa, GCaB, Q10SLCaP, Q10SRCaP, Vmax_SRCaP, Kmf, Kmr, hillSRCaP, ks, koCa, kom, kiCa, kim, ec50SR, Bmax_Naj, Bmax_Nasl, koff_na, kon_na, Bmax_TnClow, koff_tncl, kon_tncl, Bmax_TnChigh, koff_tnchca, kon_tnchca, koff_tnchmg, kon_tnchmg, Bmax_CaM, koff_cam, kon_cam, Bmax_myosin, koff_myoca, kon_myoca, koff_myomg, kon_myomg, Bmax_SR, koff_sr, kon_sr, Bmax_SLlowsl, Bmax_SLlowj, koff_sll, kon_sll, Bmax_SLhighsl, Bmax_SLhighj, koff_slh, kon_slh, Bmax_Csqn, koff_csqn, kon_csqn, GtoSlow, GtoFast, FoRT_reciprocal, ecl, sigma, KmNaip4, KoKmKo, IbarNaKKo, gkr, gks_junc, gks_sl, Kdact2, Nao3, KmPCa16, KmNai3, MaxSR, MinSR, KmNao3 = constants

    # Nernst Potentials
    ena_junc = FoRT_reciprocal*np.log(Nao/y[32])     # [mV]
    ena_sl = FoRT_reciprocal*np.log(Nao/y[33])       # [mV]
    ek = FoRT_reciprocal*np.log(Ko/y[35])	        # [mV]
    eca_junc = (FoRT_reciprocal/2)*np.log(Cao/y[36])   # [mV]
    eca_sl = (FoRT_reciprocal/2)*np.log(Cao/y[37])     # [mV]
    eks = FoRT_reciprocal*np.log((Ko+pNaK*Nao)/(y[35]+pNaK*y[34]))

    ## Pre-calculate some constants
    V_FoRT = y[39]*FoRT
    V_FoRT_Frdy = (V_FoRT*Frdy)
    V_EKs = (y[39]-eks)
    V_ENa_junc = (y[39]-ena_junc)
    V_ENa_sl = (y[39]-ena_sl)
    V_Ek = (y[39]-ek)
    V_ECl = (y[39]-ecl)

    #INa
    mss = 1 / ((1 + np.exp( -(56.86 + y[39]) / 9.03 ))**2)
    taum = 0.1292 * np.exp(-((y[39]+45.79)/15.54)**2) + 0.06487 * np.exp(-((y[39]-4.823)/51.12)**2)

    ah = np.where((y[39] >= -40) ,0,(0.057 * np.exp( -(y[39] + 80) / 6.8 )) )
    bh =  np.where((y[39] >= -40) , (0.77 / (0.13*(1 + np.exp( -(y[39] + 10.66) / 11.1 )))), ((2.7 * np.exp( 0.079 * y[39]) + 3.1*10**5 * np.exp(0.3485 * y[39]))) )
    tauh = 1 / (ah + bh)
    hss = 1 / ((1 + np.exp( (y[39] + 71.55)/7.43 ))**2)

    aj = np.where((y[39] >= -40) , 0, (((-2.5428 * 10**4*np.exp(0.2444*y[39]) - 6.948*10**-6 * np.exp(-0.04391*y[39])) * (y[39] + 37.78)) / (1 + np.exp( 0.311 * (y[39] + 79.23) ))))
    bj = np.where((y[39] >= -40) , ((0.6 * np.exp( 0.057 * y[39])) / (1 + np.exp( -0.1 * (y[39] + 32) ))) , ((0.02424 * np.exp( -0.01052 * y[39] )) / (1 + np.exp( -0.1378 * (y[39] + 40.14) ))) )
    tauj = 1 / (aj + bj)
    jss = hss #1 / ((1 + np.exp( (y[39] + 71.55)/7.43 ))**2)

    ydot[1] = (mss - y[1]) / taum
    ydot[2] = (hss - y[2]) / tauh
    ydot[3] = (jss - y[3]) / tauj

    INa_constants =  GNa*(y[1]**3)*y[2]*y[3]
    I_Na_junc = Fjunc*INa_constants*V_ENa_junc
    I_Na_sl = Fsl*INa_constants*V_ENa_sl
    I_Na = I_Na_junc+I_Na_sl

    # I_nabk: Na Background Current
    I_nabk_junc = Fjunc*GNaB*V_ENa_junc
    I_nabk_sl = Fsl*GNaB*V_ENa_sl
    I_nabk = I_nabk_junc+I_nabk_sl

    # I_nak: Na/K Pump Current
    fnak = 1/(1+0.1245*np.exp(-0.1*V_FoRT)+0.0365*sigma*np.exp(-V_FoRT))
    I_nak_constants = IbarNaKKo*fnak
    I_nak_junc = 1*Fjunc*I_nak_constants /(1+(KmNaip4/(y[32]**4)) )/KoKmKo
    I_nak_sl = 1*Fsl*I_nak_constants /(1+(KmNaip4/(y[33]**4))) /KoKmKo
    I_nak = I_nak_junc+I_nak_sl

    ## I_kr: Rapidly Activating K Current
    xrss = 1/(1+np.exp(-(y[39]+10)/5))
    tauxr = 550/(1+np.exp((-22-y[39])/9))*6/(1+np.exp((y[39]-(-11))/9))+230/(1+np.exp((y[39]-(-40))/20))
    ydot[12] = (xrss-y[12])/tauxr
    rkr = 1/(1+np.exp((y[39]+74)/24))
    I_kr = gkr*y[12]*rkr*V_Ek

    ## I_ks: Slowly Activating K Current
    xsss = 1 / (1+np.exp(-(y[39] + 3.8)/14.25)) # fitting Fra
    tauxs=990.1/(1+np.exp(-(y[39]+2.436)/14.12))
    ydot[13] = (xsss-y[13])/tauxs
    IKs_constant = y[13]**2*V_EKs
    I_ks_junc = Fjunc*gks_junc*IKs_constant
    I_ks_sl = Fsl*gks_sl*IKs_constant
    I_ks = I_ks_junc+I_ks_sl
    #markov_iks=0
    # pcaks_junc = -np.log10(y[36])+3.0
    # pcaks_sl = -np.log10(y[37])+3.0
    # gks_junc = 0.07*(0.057 +0.19/(1+ np.exp((-7.2+pcaks_junc)/0.6)))
    # gks_sl = 0.07*(0.057 +0.19/(1+ np.exp((-7.2+pcaks_sl)/0.6)))

    # if markov_iks==0:
    #     gks_junc=1*0.0035
    #     gks_sl=1*0.0035 #FRA
    #     xsss = 1 / (1+np.exp(-(y[39] + 3.8)/14.25)) # fitting Fra
    #     tauxs=990.1/(1+np.exp(-(y[39]+2.436)/14.12))
    #     ydot[13] = (xsss-y[13])/tauxs
    #     I_ks_junc = Fjunc*gks_junc*y[13]**2*V_EKs
    #     I_ks_sl = Fsl*gks_sl*y[13]**2*V_EKs
    #     I_ks = I_ks_junc+I_ks_sl

    # print('Calculated Iks')
    # print(I_ks)
    # else
    #     gks_junc=0.0065
    #     gks_sl=0.0065 #FRA
    #     alpha=3.98e-4*np.exp(3.61e-1*V_FoRT)
    #     beta=5.74e-5*np.exp(-9.23e-2*V_FoRT)
    #     gamma=3.41e-3*np.exp(8.68e-1*V_FoRT)
    #     delta=1.2e-3*np.exp(-3.3e-1*V_FoRT)
    #     teta=6.47e-3
    #     eta=1.25e-2*np.exp(-4.81e-1*V_FoRT)
    #     psi=6.33e-3*np.exp(1.27*V_FoRT)
    #     omega=4.91e-3*np.exp(-6.79e-1*V_FoRT)

    #     ydot[42)=-4*alpha*y[42)+beta*y[43)
    #     ydot[43)=4*alpha*y[42)-(beta+gamma+3*alpha)*y[43)+2*beta*y[44)
    #     ydot[44)=3*alpha*y[43)-(2*beta+2*gamma+2*alpha)*y[44)+3*beta*y[45)
    #     ydot[45)=2*alpha*y[44)-(3*beta+3*gamma+alpha)*y[45)+4*beta*y[46)
    #     ydot[46)=1*alpha*y[44)-(4*beta+4*gamma)*y[46)+delta*y[50)
    #     ydot[47)=gamma*y[43)-(delta+3*alpha)*y[47)+beta*y[48)
    #     ydot[48)=2*gamma*y[44)+3*alpha*y[47)-(delta+beta+2*alpha+gamma)*y[48)+2*beta*y[49)+2*delta*y[51)
    #     ydot[49)=3*gamma*y[45)+2*alpha*y[48)-(delta+2*beta+1*alpha+2*gamma)*y[49)+3*beta*y[50)+2*delta*y[52)
    #     ydot[50)=4*gamma*y[46)+1*alpha*y[49)-(delta+3*beta+0*alpha+3*gamma)*y[50)+2*delta*y[53)
    #     ydot[51)=1*gamma*y[48)-(2*delta+2*alpha)*y[51)+beta*y[52)
    #     ydot[52)=2*gamma*y[49)+2*alpha*y[51)-(2*delta+beta+1*alpha+gamma)*y[52)+2*beta*y[53)+3*delta*y[54)
    #     ydot[53)=3*gamma*y[50)+1*alpha*y[52)-(2*delta+2*beta+2*gamma)*y[53)+3*delta*y[55)
    #     ydot[54)=1*gamma*y[52)-(3*delta+1*alpha)*y[54)+beta*y[55)
    #     ydot[55)=2*gamma*y[53)+1*alpha*y[54)-(3*delta+1*beta+1*gamma)*y[55)+4*delta*y[56)
    #     ydot[56)=1*gamma*y[55)-(4*delta+teta)*y[56)+eta*y[57)
    #     O2=1-(y[42)+y[43)+y[44)+y[45)+y[46)+y[47)+y[49)+y[48)+y[50)+y[51)+y[52)+y[53)+y[54)+y[55)+y[56)+y[57))
    #     ydot[57)=1*teta*y[56)-(eta+psi)*y[57)+omega*O2
    #     I_ks_junc = Fjunc*gks_junc*(y[57)+O2)*V_EKs
    #     I_ks_sl = Fsl*gks_sl*(y[57)+O2)*V_EKs
    #     I_ks = I_ks_junc+I_ks_sl
    # end
    #I_kp: Plateau K current
    kp_kp = 1/(1+np.exp(7.488-y[39]/5.98))
    I_Kp_constants = gkp*kp_kp*V_Ek
    I_kp_junc = Fjunc*I_Kp_constants
    I_kp_sl = Fsl*I_Kp_constants
    I_kp = I_kp_junc+I_kp_sl

    ## I_to: Transient Outward K Current (slow and fast components)
    # modified for human myocytes

    xtoss = 1/(1+np.exp(-(y[39]-19.0)/13))
    ytoss = 1/(1+np.exp((y[39]+19.5)/5))
    # rtoss = 1/(1+np.exp((y[39]+33.5)/10))
    tauxtos = 9/(1+np.exp((y[39]+3.0)/15))+0.5
    tauytos = 800/(1+np.exp((y[39]+60.0)/10))+30
    # taurtos = 2.8e3/(1+np.exp((y[39]+60.0)/10))+220 #Fei changed here!! time-dependent gating variable
    ydot[8] = (xtoss-y[8])/tauxtos
    ydot[9] = (ytoss-y[9])/tauytos
    # ydot[40)=0
    I_tos = GtoSlow*y[8]*y[9]*V_Ek    # [uA/uF]
    tauxtof = 8.5*np.exp(-((y[39]+45)/50)**2)+0.5
    #tauxtof = 3.5*np.exp(-((y[39]+3)/30)**2)+1.5
    tauytof = 85*np.exp((-(y[39]+40)**2/220))+7
    #tauytof = 20.0/(1+np.exp((y[39]+33.5)/10))+20.0
    ydot[10] = (xtoss-y[10])/tauxtof
    ydot[11] = (ytoss-y[11])/tauytof
    I_tof = GtoFast*y[10]*y[11]*V_Ek
    I_to = I_tos + I_tof

    ## I_ki: Time-Independent K Current
    aki = 1.02/(1+np.exp(0.2385*(V_Ek-59.215)))
    bki =(0.49124*np.exp(0.08032*(V_Ek+5.476)) + np.exp(0.06175*(V_Ek-594.31))) /(1 + np.exp(-0.5143*(V_Ek+4.753)))
    kiss = aki/(aki+bki)
    I_ki =GKi*np.sqrt(Ko/5.4)*kiss*V_Ek

    # I_ClCa: Ca-activated Cl Current, I_Clbk: background Cl Current
    I_ClCa_junc = Fjunc*GClCa/(1+KdClCa/y[36])*V_ECl
    I_ClCa_sl = Fsl*GClCa/(1+KdClCa/y[37])*V_ECl
    I_ClCa = I_ClCa_junc+I_ClCa_sl
    I_Clbk = GClB*V_ECl

    # ## I_Ca: L-type Calcium Current
    fss = 1/(1+np.exp((y[39]+35)/9))+0.6/(1+np.exp((50-y[39])/20))
    dss = 1/(1+np.exp(-(y[39]+5)/6.0))
    taud = dss*(1-np.exp(-(y[39]+5)/6.0))/(0.035*(y[39]+5))
    # fss = 1/(1+np.exp((y[39]+35.06)/3.6))+0.6/(1+np.exp((50-y[39])/20))
    tauf = 1/(0.0197*np.exp( -(0.0337*(y[39]+14.5))**2 )+0.02)
    ydot[4] = (dss-y[4])/taud
    ydot[5] = (fss-y[5])/tauf
    fcabsl_1 = (1-y[7])
    fcaj_1 = (1-y[6])
    ydot[6] = 1.7*y[36]*fcaj_1-11.9e-3*y[6] # fCa_junc   koff!!!!!!!!
    ydot[7] = 1.7*y[37]*fcabsl_1-11.9e-3*y[7] # fCa_sl
    # fcaCaMSL= 0.1/(1+(0.01/y[37]))
    # fcaCaj= 0.1/(1+(0.01/y[36]))
    fcaCaMSL=0
    fcaCaj= 0

    #some constants
    exp_VFoRT = np.exp(V_FoRT)
    exp_2VFoRT = np.exp(2*V_FoRT)
    Na_K_denominator = (exp_VFoRT-1)
    Na_K_numerator = 0.75*exp_VFoRT
    Ca_denominator = (exp_2VFoRT-1)
    Ca_numerator = 0.341*exp_2VFoRT
    Q10_const = Q10CaL**Qpow
    dxf = y[4]*y[5]
    pNa_V_FoRT_Frdy = pNa*V_FoRT_Frdy
    pCa_4_V_FoRT_Frdy = pCa*4*V_FoRT_Frdy
    fCaBsl_const = (fcabsl_1+fcaCaMSL)
    fCabj_const = (fcaj_1+fcaCaj)
    sl_const = Fsl_CaL*dxf*fCaBsl_const*Q10_const
    junc_const = Fjunc_CaL*dxf*fCabj_const*Q10_const
    cao_const = 0.341*Cao
    Nao_const = 0.75*Nao

    ibarca_j = pCa_4_V_FoRT_Frdy * (Ca_numerator*y[36]-cao_const) /Ca_denominator
    ibarca_sl = pCa_4_V_FoRT_Frdy * (Ca_numerator*y[37]-cao_const) /Ca_denominator
    ibark = pK*V_FoRT_Frdy*(Na_K_numerator*y[35]-0.75*Ko) /Na_K_denominator
    ibarna_j = pNa_V_FoRT_Frdy *(Na_K_numerator*y[32]-Nao_const)  /Na_K_denominator
    ibarna_sl = pNa_V_FoRT_Frdy *(Na_K_numerator*y[33]-Nao_const)  /Na_K_denominator
    I_Ca_junc = (ibarca_j*junc_const)*0.45#*1
    I_Ca_sl = (ibarca_sl*sl_const)*0.45#*1
    I_Ca = I_Ca_junc+I_Ca_sl
    I_CaK = (ibark*(junc_const+sl_const))*0.45#*1
    I_CaNa_junc = (ibarna_j*junc_const)*0.45#*1
    I_CaNa_sl = (ibarna_sl*sl_const)*0.45#*1
    I_CaNa = I_CaNa_junc+I_CaNa_sl
    I_Catot = I_Ca+I_CaK+I_CaNa

    # I_ncx: Na/Ca Exchanger flux

    #pre-calculate fixed constants
    Naj3 = y[32]**3
    Nasl3 = y[33]**3
    exp_nuVFoRT =  np.exp(nu*V_FoRT)
    exp_nu_1_VFoRT = np.exp((nu-1)*V_FoRT)
    INCX_denom = (1+ksat*exp_nu_1_VFoRT)
    s1_constants = exp_nuVFoRT*Cao
    s2_constants = exp_nu_1_VFoRT*Nao3
    s3_constants = KmCai*Nao3
    INCX_constants = IbarNCX*Q10NCX**Qpow
    Caj2 = y[36]**2

    Ka_junc = 1/(1+Kdact2/(Caj2))
    Ka_sl = 1/(1+Kdact2/(y[37]**2))
    s1_junc = s1_constants*Naj3
    s1_sl = s1_constants*Nasl3
    s2_junc = s2_constants*y[36]
    s3_junc = s3_constants*(1+Naj3/KmNai3) + KmNao3*y[36]*(1+y[36]/KmCai)+KmCao*Naj3+Naj3*Cao+Nao3*y[36]
    s2_sl =s2_constants*y[37]
    s3_sl = s3_constants*(1+Nasl3/KmNai3) + KmNao3*y[37]*(1+y[37]/KmCai)+KmCao*Nasl3+Nasl3*Cao+Nao3*y[37]

    I_ncx_junc = Fjunc*INCX_constants*Ka_junc*(s1_junc-s2_junc)/s3_junc/INCX_denom
    I_ncx_sl = Fsl*INCX_constants*Ka_sl*(s1_sl-s2_sl)/s3_sl/INCX_denom
    I_ncx = I_ncx_junc+I_ncx_sl

    # I_pca: Sarcolemmal Ca Pump Current
    Ipca_constants = Q10SLCaP**Qpow*IbarSLCaP
    Caj16 = y[36]**1.6
    Casl16 = y[37]**1.6
    I_pca_junc = Fjunc*Ipca_constants*Caj16/(KmPCa16+Caj16)
    I_pca_sl = Fsl*Ipca_constants*Casl16/(KmPCa16+Casl16)
    I_pca = I_pca_junc+I_pca_sl

    # I_cabk: Ca Background Current
    I_cabk_junc = Fjunc*GCaB*(y[39]-eca_junc)
    I_cabk_sl = Fsl*GCaB*(y[39]-eca_sl)
    I_cabk = I_cabk_junc+I_cabk_sl

    ## SR fluxes: Calcium Release, SR Ca pump, SR Ca leak
    casr_caj = (y[31]-y[36])
    kCaSR = MaxSR - (MaxSR-MinSR)/(1+(ec50SR/y[31])**2.5)
    koSRCa = koCa/kCaSR
    kiSRCa = kiCa*kCaSR
    RI = 1-y[14]-y[15]-y[16]
    const1 = (koSRCa*Caj2*y[14]-kom*y[15])
    const2 = (kiSRCa*y[36]*y[15]-kim*y[16])
    ydot[14] = (kim*RI-kiSRCa*y[36]*y[14])-const1   # R
    ydot[15] = const1-const2# O
    ydot[16] = const2-(kom*y[16]-koSRCa*Caj2*RI)   # I
    J_SRCarel = ks*y[15]*casr_caj          # [mM/ms]
    Kmf_const = (y[38]/Kmf)**hillSRCaP
    Kmr_const = (y[31]/Kmr)**hillSRCaP
    J_serca = 1*Q10SRCaP**Qpow*Vmax_SRCaP*(Kmf_const-Kmr_const)/(1+Kmf_const+Kmr_const)
    J_SRleak = 5.348e-6*casr_caj           #   [mM/ms]

    ## Sodium and Calcium Buffering
    ydot[17] = kon_na*y[32]*(Bmax_Naj-y[17])-koff_na*y[17]        # NaBj      [mM/ms]
    ydot[18] = kon_na*y[33]*(Bmax_Nasl-y[18])-koff_na*y[18]       # NaBsl     [mM/ms]

    # Cytosolic Ca Buffers
    Bmax_myosin_const = Bmax_myosin-y[23]-y[24]
    Bmax_TnChigh_const = Bmax_TnChigh-y[20]-y[21]
    ydot[19] = kon_tncl*y[38]*(Bmax_TnClow-y[19])-koff_tncl*y[19]            # TnCL      [mM/ms]
    ydot[20] = kon_tnchca*y[38]*(Bmax_TnChigh_const)-koff_tnchca*y[20] # TnCHc     [mM/ms]
    ydot[21] = kon_tnchmg*Mgi*(Bmax_TnChigh_const)-koff_tnchmg*y[21]   # TnCHm     [mM/ms]
    ydot[22] = kon_cam*y[38]*(Bmax_CaM-y[22])-koff_cam*y[22]                 # CaM       [mM/ms]
    ydot[23] = kon_myoca*y[38]*(Bmax_myosin_const)-koff_myoca*y[23]    # Myosin_ca [mM/ms]
    ydot[24] = kon_myomg*Mgi*(Bmax_myosin_const)-koff_myomg*y[24]      # Myosin_mg [mM/ms]
    ydot[25] = kon_sr*y[38]*(Bmax_SR-y[25])-koff_sr*y[25]                    # SRB       [mM/ms]
    #J_CaB_cytosol = sum(ydot[19:25)) # wrong formulation
    J_CaB_cytosol = ydot[19]+ydot[20]+ydot[22]+ydot[23]+ydot[25]

    # Junctional and SL Ca Buffers
    ydot[26] = kon_sll*y[36]*(Bmax_SLlowj-y[26])-koff_sll*y[26]       # SLLj      [mM/ms]
    ydot[27] = kon_sll*y[37]*(Bmax_SLlowsl-y[27])-koff_sll*y[27]      # SLLsl     [mM/ms]
    ydot[28] = kon_slh*y[36]*(Bmax_SLhighj-y[28])-koff_slh*y[28]      # SLHj      [mM/ms]
    ydot[29] = kon_slh*y[37]*(Bmax_SLhighsl-y[29])-koff_slh*y[29]     # SLHsl     [mM/ms]
    J_CaB_junction = ydot[26]+ydot[28]
    J_CaB_sl = ydot[27]+ydot[29]

    ## Ion concentrations
    # SR Ca Concentrations
    ydot[30] = kon_csqn*y[31]*(Bmax_Csqn-y[30])-koff_csqn*y[30]       # Csqn      [mM/ms]
    ydot[31] = J_serca-(J_SRleak*Vmyo/Vsr+J_SRCarel)-ydot[30]         # Ca_sr     [mM/ms] #Ratio 3 leak current
    # ydot[31)=0

    # Sodium Concentrations
    I_Na_tot_junc = I_Na_junc+I_nabk_junc+3*I_ncx_junc+3*I_nak_junc+I_CaNa_junc   # [uA/uF]
    I_Na_tot_sl = I_Na_sl+I_nabk_sl+3*I_ncx_sl+3*I_nak_sl+I_CaNa_sl   # [uA/uF]
    # I_Na_tot_sl2 = 3*I_ncx_sl+3*I_nak_sl+I_CaNa_sl   # [uA/uF]
    # I_Na_tot_junc2 = 3*I_ncx_junc+3*I_nak_junc+I_CaNa_junc   # [uA/uF]

    Cmem_junc_const = Cmem/(Vjunc*Frdy)
    Cmem_sl_const = Cmem/(Vsl*Frdy)

    ydot[32] = -I_Na_tot_junc*Cmem_junc_const+J_na_juncsl/Vjunc*(y[33]-y[32])-ydot[17]
    ydot[33] = -I_Na_tot_sl*Cmem_sl_const+J_na_juncsl/Vsl*(y[32]-y[33])+J_na_slmyo/Vsl*(y[34]-y[33])-ydot[18]
    ydot[34] = J_na_slmyo/Vmyo*(y[33]-y[34])             # [mM/msec]

    # Potassium Concentration
    I_K_tot = I_to+I_kr+I_ks+I_ki-2*I_nak+I_CaK+I_kp     # [uA/uF]
    # ydot[35) = 0 #-I_K_tot*Cmem/(Vmyo*Frdy)           # [mM/msec]
    ydot[35] =0 # -I_K_tot*Cmem/(Vmyo*Frdy)

    # Calcium Concentrations
    I_Ca_tot_junc = I_Ca_junc+I_cabk_junc+I_pca_junc-2*I_ncx_junc                   # [uA/uF]
    I_Ca_tot_sl = I_Ca_sl+I_cabk_sl+I_pca_sl-2*I_ncx_sl            # [uA/uF]
    ydot[36] = -I_Ca_tot_junc*Cmem_junc_const/2+J_ca_juncsl/Vjunc*(y[37]-y[36]) -J_CaB_junction+(J_SRCarel)*Vsr/Vjunc+J_SRleak*Vmyo/Vjunc  # Ca_j
    ydot[37] = -I_Ca_tot_sl*Cmem_sl_const/2+J_ca_juncsl/Vsl*(y[36]-y[37])+ J_ca_slmyo/Vsl*(y[38]-y[37])-J_CaB_sl   # Ca_sl
    ydot[38] = -J_serca*Vsr/Vmyo-J_CaB_cytosol +J_ca_slmyo/Vmyo*(y[37]-y[38])


    if t % (cycle_length) <= duration:
      I_app = amplitude
    else:
      I_app = 0.0

    ## Membrane Potential
    I_Na_tot = I_Na_tot_junc+I_Na_tot_sl          # [uA/uF]
    I_Cl_tot = I_ClCa+I_Clbk                        # [uA/uF]
    I_Ca_tot = I_Ca_tot_junc+I_Ca_tot_sl
    I_tot = I_Na_tot+I_Cl_tot+I_Ca_tot+I_K_tot
    #ydot[39] = -(I_Ca_tot+I_K_tot+I_Na_tot-I_app)
    ydot[39] = -(I_tot-I_app)
    return ydot
