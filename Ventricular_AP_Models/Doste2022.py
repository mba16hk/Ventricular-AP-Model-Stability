import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import ode
from .Doste2022_signaling import *
from .ORd import Istim
import math
import time as tm
import pandas as pd
from numba import njit
from conductances import *

# code translated from MATLAB from Github page: https://github.com/rdoste/ToR-ORd-BARS
# NOTE that even though the model provdes parameters for M cels, it does not provide initial states. Running it without BARS for M cells is fine, but running it with BARS causes EADs

def GetStartingState_DynCL(cell_type):
   if cell_type == 'ENDO':
        X0 = [-90.2610631103136,12.6104737146693,12.6108144383834,150.153498961714,150.153455537464,7.44545085449596e-05,6.48598870105768e-05,1.55855629215003,1.55637168593098,0.000583364759888665,0.717289909066744,0.856387110491620,0.856248291782132,0.856036277427299,0.000122575491209903,0.573096605589374,0.325541162736861,0.000859674719043351,0.999699784457874,0.602611094668993,0.000437997062095915,0.999699790735512,0.667617331114495,5.38757122921842e-33,0.999999995047883,0.941524647592581,0.999999995047914,0.999904380012983,0.999978306045816,0.000486748305163032,0.000829368708547468,0.999999995045147,0.999999995046113,0.241643250013222,0.000149760845750349,-5.92994670509191e-25,0.0114045602562791,0.998345893868994,0.000765189598839647,0.000628331733884328,0.000252348419834982,8.23608332499044e-06,2.63697969117824e-24,29.2069793887423,29.2069557375901,0.394984055405210,-7.05809047188543e-32,0.999999956865775,0.933408929581152,0.999999956866041,0.999854614620516,0.999999956849038,0.999999956850700,0.745762621045509,0.856248291782131,0.545264167134426,0.856036277427299,1.52638884959147e-25,-1.03375480257661e-24]
   if cell_type == 'EPI':
        X0 = [-90.6382160024659,13.3768727751355,13.3771886950160,151.081840923245,151.081791542168,6.58999529464348e-05,5.72505786272457e-05,1.82963344246430,1.82794885037311,0.000537676082983256,0.728291371745131,0.862750017533519,0.862695532472151,0.862591321987368,0.000114101448449066,0.588453200471855,0.345184178201681,0.000838088197472356,0.999719003354595,0.999718255395732,0.000426994385265556,0.999719003466272,0.999718826022699,-1.56497194391289e-32,0.999999995528930,0.951865728647970,0.999999995528940,0.999939871692426,0.999984012183261,0.000299855457412818,0.000517523100408780,0.999999995517324,0.999999995526524,0.222124872250935,0.000143538597298202,4.85001305397468e-24,0.0129067865640192,0.998460677017900,0.000744974318734279,0.000607510327766145,0.000181054224815059,5.78323125882101e-06,-1.15589115528518e-21,34.3172099045416,34.3171879616709,0.370015473901015,2.94618862079189e-33,0.999999961055796,0.946239077084497,0.999999961055876,0.999911737067179,0.999999961039081,0.999999961047713,0.755944224162249,0.862695532472151,0.559672701074154,0.862591321987367,-2.23044696760782e-26,7.55851039422359e-24]

    # I have used the same initial conditions as ENDO, Ask BARS authors what the initial conditions are, but M does not work under BARS stimulation 
   if cell_type == 'M':
        X0 = [-90.2610631103136,12.6104737146693,12.6108144383834,150.153498961714,150.153455537464,7.44545085449596e-05,6.48598870105768e-05,1.55855629215003,1.55637168593098,0.000583364759888665,0.717289909066744,0.856387110491620,0.856248291782132,0.856036277427299,0.000122575491209903,0.573096605589374,0.325541162736861,0.000859674719043351,0.999699784457874,0.602611094668993,0.000437997062095915,0.999699790735512,0.667617331114495,5.38757122921842e-33,0.999999995047883,0.941524647592581,0.999999995047914,0.999904380012983,0.999978306045816,0.000486748305163032,0.000829368708547468,0.999999995045147,0.999999995046113,0.241643250013222,0.000149760845750349,-5.92994670509191e-25,0.0114045602562791,0.998345893868994,0.000765189598839647,0.000628331733884328,0.000252348419834982,8.23608332499044e-06,2.63697969117824e-24,29.2069793887423,29.2069557375901,0.394984055405210,-7.05809047188543e-32,0.999999956865775,0.933408929581152,0.999999956866041,0.999854614620516,0.999999956849038,0.999999956850700,0.745762621045509,0.856248291782131,0.545264167134426,0.856036277427299,1.52638884959147e-25,-1.03375480257661e-24]
   
   X0_Signaling= [0.00685533455220118,0.0184630401160325,0.000731426797266862,0.00746268345094940,0.0191020788696145,0.00115141961261304,0.000607348898749425,0.000639038753581265,0.000419992815346110,0.344257659177271,9.62262190945345,0.474028267773051,0.0148474493496437,0.203015985725400,0.00944459882156118,1.76916170568022e-10,8.36801924219387e-10,5.01719476559362e-11,0.0898875954193269,0.00272422687231002,0.225041998219388,0.0322381220102320,0.192803876209063,0.205457169881723,0.174050238959555,0.817148066343192,0.567236181979005,0.249911884364108,0.0646981828366729,0.0664977613514265,0.489057632872032,0.362107101574369,0.126950531297531,0.0233954992478800,0.0128401592747216,0.00629647926927854,4.29166115483698e-05,0.00917030613498568,0.0123536190564101,0.000664158274421826,0.000765842738691197,0.666165471397222,0.673477756497978,0.236980176272067,0.124710628782511,0.00404925913372347,0.0589106047787742,0.0274407333314977,6.32124571143896e-10,0.00159025206300466,0.00209267447556971,0.000502422412564797,0.0110248493074472,8.04005829146876e-11,0.000364313646402573,0.000705654325530757,0.000341340679127998]
   X0 = X0 + X0_Signaling
   #X0 = X0[:57]
   return X0

@njit
def Doste_Model_T(time,X0,cycle_length, clo, nao, cao, ko, R, T, F, L, rad,
    vcell, Ageo, Acap, vmyo, vnsr, vjsr, vss,
    KmCaMK, aCaMK, bCaMK, CaMKo, KmCaM, frt, 
    cmdnmax, amp, duration, kmcmdn, trpnmax, kmtrpn,
    BSRmax, KmBSR, BSLmax, KmBSL, csqnmax, kmcsqn, zna, k1p, k1m, k2p, k2m, k3p, k3m, k4p, k4m,
    Knai0, Knao0, delta, Pnak, GKb, PNab, PCab, GpCa,
    bt, a_rel, btp, a_relp, Kki, Kko, MgADP, MgATP,
    Kmgatp, H, eP, Khp, Knap, Kxkur,reversal_potential_const, 
    ICaL_fractionSS, INaCa_fractionSS, PKNa, GNaL, Gto, Kmn, k2n, GKs, zca, kna1, kna2, kna3, kasymm, wna, wca, wnaca, kcaon, kcaoff, qna, qca,
    Aff, Afs, constA, GK1, Gncx, KmCaAct, zk, jsrMidpoint, GClCa, GClB, KdClCa, PCa, GNa,GKr,Jrel_inf_scale,Jup_scale, celltype, PCaNa, PCaK, PCaNap, PCaKp, PCap, PCa_P, PCaNa_P, PCaK_P,GNa_P,
    runSignalingPathway, cycles, ISO, radiusmultiplier,alpha1,beta1,a_relP,a_relBP,isepi,txs1_P_const,
    R_wvtovcv,R_wvtovec,R_evtovcyt,PP1_EC,PP1_CV,PP1_CYT,PP1_aff
    ):
    
    #initial_conds = X0[0:59]
    #X0_Signaling = X0[59:]
    v,nai,nass,ki,kss,cai,cass,cansr,cajsr,m,hp,h,j,jp,mL,hL,hLp,a,iF,iS,ap,iFp,iSp,d,ff,fs,fcaf,fcas,jca,nca,nca_i,ffp,fcafp,xs1,xs2,Jrel_np,CaMKt,ikr_c0,ikr_c1,ikr_c2,ikr_o,ikr_i,Jrel_p,cli,clss,xs1_P, d_P,ff_P,fs_P,fcaf_P,fcas_P,fBPf,fcaBPf, h_P, j_P, hp_P, jp_P, Jrel_np_P, Jrel_p_P, cond1,cond2,cond3,cond4,cond5,cond6,cond7,cond8,cond9,cond10,cond11,cond12,cond13,cond14,cond15,cond16,cond17,cond18,cond19,cond20,cond21,cond22,cond23,cond24,cond25,cond26,cond27,cond28,cond29,cond30,cond31,cond32,cond33,cond34,cond35,cond36,cond37,cond38,cond39,cond40,cond41,cond42,cond43,cond44,cond45,cond46,cond47,cond48,cond49,cond50,cond51,cond52,cond53,cond54,cond55,cond56,cond57 = X0

    ##%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%


    index = int(time // cycle_length)  # integer division to get current 500ms window index
    if index >= cycles:
        index = cycles - 1 
    signalling_pathway_perbeat = runSignalingPathway[index]
    current_iso = ISO[index]
    c = Constants_SignalingMyokit2(current_iso, radiusmultiplier)
    
    if signalling_pathway_perbeat == 1:
        dXSignaling = Model_SignalingMyokit2(cond1,cond2,cond3,cond4,cond5,cond6,cond7,cond8,cond9,cond10,cond11,cond12,cond13,cond14,cond15,
                           cond16,cond17,cond18,cond19,cond20,cond21,cond22,cond23,cond24,cond25,cond26,cond27,cond28,cond29,
                           cond30,cond31,cond32,cond33,cond34,cond35,cond36,cond37,cond38,cond39,cond40,cond41,cond42,cond43,
                           cond44,cond45,cond46,cond47,cond48,cond49,cond50,cond51,cond52,cond53,cond54,cond55,cond56,cond57, c)
        dcond1,dcond2,dcond3,dcond4,dcond5,dcond6,dcond7,dcond8,dcond9,dcond10,dcond11,dcond12,dcond13,dcond14,dcond15,dcond16,dcond17,dcond18,dcond19,dcond20,dcond21,dcond22,dcond23,dcond24,dcond25,dcond26,dcond27,dcond28,dcond29,dcond30,dcond31,dcond32,dcond33,dcond34,dcond35,dcond36,dcond37,dcond38,dcond39,dcond40,dcond41,dcond42,dcond43,dcond44,dcond45,dcond46,dcond47,dcond48,dcond49,dcond50,dcond51,dcond52,dcond53,dcond54,dcond55,dcond56,dcond57 = dXSignaling
        fICaLP,fIKsP,fPLBP,fTnIP,fINaP,fINaKP,fRyRP,fIKurP =  EffectiveFraction_Torord2(cond40,cond41,cond42,cond43,cond44,cond45,cond46,cond47, c) 
        # # Concentration of uninhibited PP1 in the cytosolic compartment
        pp1_PP1f_cyt_sum = PP1_aff - PP1_CYT + cond39
        # # Concentration of uninhibited PP1 in the cytosolic compartment
        PP1f_cyt = 0.5 * (np.sqrt(pp1_PP1f_cyt_sum ** 2.0 + 4.0 * PP1_aff * PP1_CYT) - pp1_PP1f_cyt_sum)
        Whole_cell_PP1  = PP1_CV / R_wvtovcv + PP1_EC / R_wvtovec + PP1f_cyt / R_evtovcyt
        # if abs(time % 1000) < 0.1:
        #     # t is approximately a multiple of 1000
        #     print(dXSignaling)
    else:
        dcond1=dcond2=dcond3=dcond4=dcond5=dcond6=dcond7=dcond8=dcond9=dcond10=dcond11=dcond12=dcond13=dcond14=dcond15=dcond16=dcond17=dcond18=dcond19=dcond20=dcond21=dcond22=dcond23=dcond24=dcond25=dcond26=dcond27=dcond28=dcond29=dcond30=dcond31=dcond32=dcond33=dcond34=dcond35=dcond36=dcond37=dcond38=dcond39=dcond40=dcond41=dcond42=dcond43=dcond44=dcond45=dcond46=dcond47=dcond48=dcond49=dcond50=dcond51=dcond52=dcond53=dcond54=dcond55=dcond56=dcond57 =0
        fICaLP,fIKsP,fPLBP,fTnIP,fINaP,fINaKP,fRyRP,fIKurP =  EffectiveFraction_Torord2(cond40,cond41,cond42,cond43,cond44,cond45,cond46,cond47, c) 
        Whole_cell_PP1 = 0.1371
        #dXSignaling = 0
    
    
    
    vfrt=v*frt
    vffrt=vfrt*F
    
    ##%update CaMK
    CaMKb=CaMKo*(1.0-CaMKt)/(1.0+KmCaM/cass)
    CaMKa=CaMKb+CaMKt
    betaCaMKII =  bCaMK * (0.1 + (0.9  * Whole_cell_PP1 / 0.1371))
    dCaMKt=aCaMK*CaMKb*(CaMKb+CaMKt)-betaCaMKII*CaMKt

    ##%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%

    ##%reversal potentials
    ENa=reversal_potential_const*np.log(nao/nai)
    EK=reversal_potential_const*np.log(ko/ki)
    EKs=reversal_potential_const*np.log((ko+PKNa*nao)/(ki+PKNa*nai))
    ecl = reversal_potential_const*np.log(cli/clo)            #% [mV]
    eclss = reversal_potential_const*np.log(clss/clo);           # % [mV]
    #%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%
    V_ENa = (v-ENa)
    fINap=(1.0/(1.0+KmCaMK/CaMKa))
    fINaLp=fINap
    fItop=fINap
    fICaLp=fINap
    fJrelp= fINap

    mss = 1 / ((1 + np.exp( -(56.86 + v) / 9.03 ))**2)
    taum = 0.1292 * np.exp(-((v+45.79)/15.54)**2) + 0.06487 * np.exp(-((v-4.823)/51.12)**2)
    dm = (mss - m) / taum

    #%#% h gate
    ah = np.where(v >= -40,0,0.057 * np.exp( -(v + 80) / 6.8 ))
    bh =  np.where(v >= -40 , (0.77 / (0.13*(1 + np.exp( -(v + 10.66) / 11.1 )))) ,((2.7 * np.exp( 0.079 * v) + 3.1*10**5 * np.exp(0.3485 * v))))
    tauh = 1 / (ah + bh)
    v_plus_71 = v + 71.55
    hss = 1 / ((1 + np.exp( (v_plus_71)/7.43 ))**2)
    dh = (hss - h) / tauh
    #%#% j gate
    aj = np.where(v >= -40, (0) ,(((-2.5428 * 10**4*np.exp(0.2444*v) - 6.948*10**-6 * np.exp(-0.04391*v)) * (v + 37.78)) / (1 + np.exp( 0.311 * (v + 79.23) ))))
    bj = np.where(v >= -40 , ((0.6 * np.exp( 0.057 * v)) / (1 + np.exp( -0.1 * (v + 32) ))) ,((0.02424 * np.exp( -0.01052 * v )) / (1 + np.exp( -0.1378 * (v + 40.14) ))))
    tauj = 1 / (aj + bj)
    jss = hss#1 / ((1 + np.exp( (v + 71.55)/7.43 ))**2)
    dj = (jss - j) / tauj

    #%#% h phosphorylated
    hssp = 1 / ((1 + np.exp( (v_plus_71 + 6)/7.43 ))**2)
    dhp = (hssp - hp) / tauh
    #%#% j phosphorylated
    taujp = 1.46 * tauj
    djp = (jss - jp) / taujp
    
    
    #### BARS ##############
     ## gating PKA
    hss_P = 1 / ((1 + np.exp( (v_plus_71+5.0)/7.43 ))**2)  #BetaAdrenergic
    dh_P=(hss_P-h_P)/tauh   #BetaAdrenergic
    jss_P=hss_P
    dj_P = (jss_P - j_P) / tauj

    ## Both Phosphorilated
    hssp_P = 1 / ((1 + np.exp( (v_plus_71 + 6 + 5.0)/7.43 ))**2) #BetaAdrenergic
    dhp_P = (hssp_P - hp_P) / tauh
    jssp_P=hssp_P
    djp_P = (jssp_P - jp_P) / taujp

    fINa_P = fINaP # PKA-P fraction as assigned as input, take the value 0 or 1
    fINa_BP = fINap*fINa_P 
    fINa_CaMKonly = fINap-fINa_BP 
    fINa_PKAonly = fINa_P-fINa_BP 

    m_cubed = m**3 
    V_ENa_mcubed = (V_ENa)*m_cubed
    INa_NP = GNa*V_ENa_mcubed*h*j  # Non-Phosphorylated 
    INa_CaMK = GNa*V_ENa_mcubed*hp*jp 
    INa_PKA = GNa_P*V_ENa_mcubed*h_P*j_P 
    INa_BP = GNa_P*V_ENa_mcubed*hp_P*jp_P 
    
    # 4 population 
    INa = ((1-fINa_CaMKonly-fINa_PKAonly-fINa_BP)*INa_NP + fINa_CaMKonly*INa_CaMK + fINa_PKAonly*INa_PKA + fINa_BP*INa_BP) 
    
    #INa=GNa*(v-ENa)*m**3.0*((1.0-fINap)*h*j+fINap*hp*jp)


    mLss=1.0/(1.0+np.exp((-(v+42.85))/5.264))
    tm = 0.1292 * np.exp(-((v+45.79)/15.54)**2) + 0.06487 * np.exp(-((v-4.823)/51.12)**2) 
    tmL=tm
    dmL=(mLss-mL)/tmL
    hLss=1.0/(1.0+np.exp((v+87.61)/7.488))
    #thL=200.0
    dhL=(hLss-hL)/200.0
    hLssp=1.0/(1.0+np.exp((v+93.81)/7.488))
    #thLp=3.0*thL
    dhLp=(hLssp-hLp)/600.0

    INaL=GNaL*(v-ENa)*mL*((1.0-fINaLp)*hL+fINaLp*hLp)


    #%#% ITo
    #%calculate Ito
    ass=1.0/(1.0+np.exp((-(v-14.34))/14.82))
    ta=1.0515/(1.0/(1.2089*(1.0+np.exp(-(v-18.4099)/29.3814)))+3.5/(1.0+np.exp((v+100.0)/29.3814)))
    da=(ass-a)/ta
    iss=1.0/(1.0+np.exp((v+43.94)/5.711))
    delta_epi=1.0
    if isepi:
        delta_epi=1.0-(0.95/(1.0+np.exp((v+70.0)/5.0)))
        
    tiF=4.562+1/(0.3933*np.exp((-(v+100.0))/100.0)+0.08004*np.exp((v+50.0)/16.59))
    tiS=23.62+1/(0.001416*np.exp((-(v+96.52))/59.05)+1.780e-8*np.exp((v+114.1)/8.079))
    tiF=tiF*delta_epi
    tiS=tiS*delta_epi
    AiF=1.0/(1.0+np.exp((v-213.6)/151.2))
    AiS=1.0-AiF
    diF=(iss-iF)/tiF
    diS=(iss-iS)/tiS
    i=AiF*iF+AiS*iS
    assp=1.0/(1.0+np.exp((-(v-24.34))/14.82))
    dap=(assp-ap)/ta
    dti_develop=1.354+1.0e-4/(np.exp((v-167.4)/15.89)+np.exp(-(v-12.23)/0.2154))
    dti_recover=1.0-0.5/(1.0+np.exp((v+70.0)/20.0))
    develop_recover_product = dti_develop*dti_recover
    tiFp=develop_recover_product*tiF
    tiSp=develop_recover_product*tiS
    diFp=(iss-iFp)/tiFp
    diSp=(iss-iSp)/tiSp
    ip=AiF*iFp+AiS*iSp

    Ito=Gto*(v-EK)*((1.0-fItop)*a*i+fItop*ap*ip)
 
    dss = np.where(v >31.4978,1.00,1.0763*np.exp(-1.0070*np.exp(-0.0829*(v)))) #% magyar
    td= 0.6+1.0/(np.exp(-0.05*(v+6.0))+np.exp(0.09*(v+14.0)))

    dd=(dss-d)/td
    fss=1.0/(1.0+np.exp((v+19.58)/3.696))
    v_20 = (v+20.0)
    v_5 = (v+5.0)
    tff=7.0+1.0/(0.0045*np.exp(-v_20/10.0)+0.0045*np.exp(v_20/10.0))
    tfs=1000.0+1.0/(0.000035*np.exp(-v_5/4.0)+0.000035*np.exp(v_5/6.0))
    
    dff=(fss-ff)/tff
    dfs=(fss-fs)/tfs
    f=Aff*ff+Afs*fs
    fcass=fss
    v_4 = (v-4.0)
    tfcaf=7.0+1.0/(0.04*np.exp(-v_4/7.0)+0.04*np.exp(v_4/7.0))
    tfcas=100.0+1.0/(0.00012*np.exp(-v/3.0)+0.00012*np.exp(v/7.0))

    Afcaf=0.3+0.6/(1.0+np.exp((v-10.0)/10.0))

    Afcas=1.0-Afcaf
    dfcaf=(fcass-fcaf)/tfcaf
    dfcas=(fcass-fcas)/tfcas
    fca=Afcaf*fcaf+Afcas*fcas

    jcass = 1.0/(1.0+np.exp((v+18.08)/(2.7916)))   
    djca=(jcass-jca)/72.5# reduced from 75
    tffp=2.5*tff
    dffp=(fss-ffp)/tffp
    fp=Aff*ffp+Afs*fs
    tfcafp=2.5*tfcaf
    dfcafp=(fcass-fcafp)/tfcafp
    fcap=Afcaf*fcafp+Afcas*fcas

    #%#% SS nca

    km2n=jca*1
    anca=1.0/(k2n/km2n+(1.0+Kmn/cass)**4.0)
    dnca=anca*k2n-nca*km2n

    #%#% myoplasmic nca
    anca_i = 1.0/(k2n/km2n+(1.0+Kmn/cai)**4.0)
    dnca_i = anca_i*k2n-nca_i*km2n

    #%#% SS driving force
    Io = 0.5*(nao + ko + clo + 4*cao)/1000  #% ionic strength outside. /1000 is for things being in micromolar
    Ii = 0.5*(nass + kss + clss + 4*cass)/1000  #% ionic strength outside. /1000 is for things being in micromolar
    #% The ionic strength is too high for basic DebHuc. We'll use Davies
    
    Ii_const = (np.sqrt(Ii)/(1+np.sqrt(Ii))-0.3*Ii)
    Io_const = (np.sqrt(Io)/(1+np.sqrt(Io))-0.3*Io)
    gamma_cai = np.exp(-constA * 4 * Ii_const)
    gamma_cao = np.exp(-constA * 4 * Io_const)
    gamma_nai = np.exp(-constA * 1 * Ii_const)
    gamma_nao = np.exp(-constA * 1 * Io_const)
    gamma_ki = gamma_nai#np.exp(-constA * 1 * (np.sqrt(Ii)/(1+np.sqrt(Ii))-0.3*Ii))
    gamma_kao = gamma_nao#np.exp(-constA * 1 * (np.sqrt(Io)/(1+np.sqrt(Io))-0.3*Io))

    Na_K_exp_const = np.exp(1.0*vfrt)
    Ca_exp_const = np.exp(2.0*vfrt)
    Ca_denom = (Ca_exp_const-1.0)
    Na_K_denom = (Na_K_exp_const-1.0)
    PhiCaL_ss =  4.0*vffrt*(gamma_cai*cass*Ca_exp_const-gamma_cao*cao)/Ca_denom
    PhiCaNa_ss =  vffrt*(gamma_nai*nass*Na_K_exp_const-gamma_nao*nao)/Na_K_denom
    PhiCaK_ss =  vffrt*(gamma_ki*kss*Na_K_exp_const-gamma_kao*ko)/Na_K_denom

    #%#% Myo driving force
    #Io = 0.5*(nao + ko + clo + 4*cao)/1000  #% ionic strength outside. /1000 is for things being in micromolar
    Ii = 0.5*(nai + ki + cli + 4*cai)/1000  #% ionic strength outside. /1000 is for things being in micromolar
    #% The ionic strength is too high for basic DebHuc. We'll use Davies
    Ii_const = (np.sqrt(Ii)/(1+np.sqrt(Ii))-0.3*Ii)
    gamma_cai = np.exp(-constA * 4 * Ii_const)
    #gamma_cao = np.exp(-constA * 4 * (np.sqrt(Io)/(1+np.sqrt(Io))-0.3*Io))
    gamma_nai = np.exp(-constA * 1 * Ii_const)
    #gamma_nao = np.exp(-constA * 1 * (np.sqrt(Io)/(1+np.sqrt(Io))-0.3*Io))
    gamma_ki = gamma_nai#np.exp(-constA * 1 * (np.sqrt(Ii)/(1+np.sqrt(Ii))-0.3*Ii))
    #gamma_kao = gamma_nao#np.exp(-constA * 1 * (np.sqrt(Io)/(1+np.sqrt(Io))-0.3*Io))

    gammaCaoMyo = gamma_cao
    gammaCaiMyo = gamma_cai

    PhiCaL_i =  4.0*vffrt*(gamma_cai*cai*Ca_exp_const-gamma_cao*cao)/Ca_denom
    PhiCaNa_i =  vffrt*(gamma_nai*nai*Na_K_exp_const-gamma_nao*nao)/Na_K_denom
    PhiCaK_i =  vffrt*(gamma_ki*ki*Na_K_exp_const-gamma_kao*ko)/Na_K_denom


    ## #PHOSPHORILATED##############################################################################################
    #      dPss=1.0763*np.exp(-1.0070*np.exp(-0.0829*(v+9.5)))  # #BetaAdrenergic
    dPss=1.0323*np.exp(-1.0553*np.exp(-0.0810*(v+9.5)))     #BetaAdrenergic ,fitted curve, why is the 9.5 added?
    dPss=np.minimum(dPss,1)
    dd_P=(dPss-d_P)/td
    #   fss_P=1.0/(1.0+np.exp((v+19.58+8.0)/3.696)) #BetaAdrenergic 
    fss_P=1.0/(1.0+np.exp((v+19.58+4.0)/3.696)) #BetaAdrenergic 
    dff_P=(fss_P-ff_P)/tff
    dfs_P=(fss_P-fs_P)/tfs
    fcass_P=fss_P
    dfcaf_P=(fcass_P-fcaf_P)/tfcaf
    dfcas_P=(fcass_P-fcas_P)/tfcas
    f_P=Aff*ff_P+Afs*fs_P
    fcap_P=Afcaf*fcaf_P+Afcas*fcas_P

    ## PHOSPHORILATION ##########################################
    # Both-P population takes on dP, for f gate, takes ss from PKA and tau from CaMK (only fast component was modified)
    fBPss = fss_P
    dfBPf = (fBPss-fBPf)/tffp
    fBP = Aff*fBPf+Afs*fs_P # only fast component modified, slow component same as fPs
    fcaBPss = fcass_P
    dfcaBPf = (fcaBPss-fcaBPf)/tfcafp  
    fcaBP = Afcaf*fcaBPf+Afcas*fcas_P
    #############
    #fICaLp=fICaLp # CaMK-P fraction
    fICaL_P = fICaLP # PKA-P fraction
    fICaL_BP = fICaLp*fICaL_P
    fICaL_CaMKonly = fICaLp-fICaL_BP
    fICaL_PKAonly = fICaL_P-fICaL_BP
    
    ############################################
    const1 = (1.0-nca)
    const2 = (1.0-nca_i)
    BP_const = d_P*(fBP*const2+jca*fcaBP*nca_i)
    BP_ss_const = d_P*(fBP*const1+jca*fcaBP*nca)
    NP_ss_const = d*(f*const1+jca*fca*nca)
    NP_i_const = d*(f*const2+jca*fca*nca_i)
    CaMK_ss_const = d*(fp*const1+jca*fcap*nca)
    CaMK_i_const = d*(fp*const2+jca*fcap*nca_i)
    PKA_ss_const = d_P*(f_P*const1+jca*fcap_P*nca)
    PKA_i_const = d_P*(f_P*const2+jca*fcap_P*nca_i)
    
    ICaL_ss_NP=PCa*PhiCaL_ss*NP_ss_const
    ICaL_ss_CaMK=PCap*PhiCaL_ss*CaMK_ss_const
    ICaL_ss_PKA=PCa_P*PhiCaL_ss*PKA_ss_const
    ICaL_ss_BP=PCa_P*PhiCaL_ss*BP_ss_const
    ICaL_i_NP=PCa*PhiCaL_i*NP_i_const
    ICaL_i_CaMK=PCap*PhiCaL_i*CaMK_i_const
    ICaL_i_PKA=PCa_P*PhiCaL_i*PKA_i_const
    ICaL_i_BP=PCa_P*PhiCaL_i*BP_const

    
    ICaNa_ss_NP=PCaNa*PhiCaNa_ss*NP_ss_const
    ICaNa_ss_CaMK=PCaNap*PhiCaNa_ss*CaMK_ss_const
    ICaNa_ss_PKA=PCaNa_P*PhiCaNa_ss*PKA_ss_const
    ICaNa_ss_BP=PCaNa_P*PhiCaNa_ss*BP_ss_const
    ICaNa_i_NP=PCaNa*PhiCaNa_i*NP_i_const
    ICaNa_i_CaMK=PCaNap*PhiCaNa_i*CaMK_i_const
    ICaNa_i_PKA=PCaNa_P*PhiCaNa_i*PKA_i_const
    ICaNa_i_BP=PCaNa_P*PhiCaNa_i*BP_const


    ICaK_ss_NP=PCaK*PhiCaK_ss*NP_ss_const
    ICaK_ss_CaMK=PCaKp*PhiCaK_ss*CaMK_ss_const
    ICaK_ss_PKA=PCaK_P*PhiCaK_ss*PKA_ss_const
    ICaK_ss_BP=PCaK_P*PhiCaK_ss*BP_ss_const
    ICaK_i_NP=PCaK*PhiCaK_i*NP_i_const
    ICaK_i_CaMK=PCaKp*PhiCaK_i*CaMK_i_const
    ICaK_i_PKA=PCaK_P*PhiCaK_i*d_P*PKA_i_const
    ICaK_i_BP=PCaK_P*PhiCaK_i*BP_const
    
    # 4 population combination######################################################
    Ca_i_ss_consts = (1-fICaL_CaMKonly-fICaL_PKAonly-fICaL_BP)
    ICaL_ss = Ca_i_ss_consts*ICaL_ss_NP + fICaL_CaMKonly*ICaL_ss_CaMK + fICaL_PKAonly*ICaL_ss_PKA + fICaL_BP*ICaL_ss_BP
    ICaNa_ss = Ca_i_ss_consts*ICaNa_ss_NP + fICaL_CaMKonly*ICaNa_ss_CaMK + fICaL_PKAonly*ICaNa_ss_PKA + fICaL_BP*ICaNa_ss_BP
    ICaK_ss = Ca_i_ss_consts*ICaK_ss_NP + fICaL_CaMKonly*ICaK_ss_CaMK + fICaL_PKAonly*ICaK_ss_PKA + fICaL_BP*ICaK_ss_BP
    # 
    ICaL_i = Ca_i_ss_consts*ICaL_i_NP + fICaL_CaMKonly*ICaL_i_CaMK + fICaL_PKAonly*ICaL_i_PKA + fICaL_BP*ICaL_i_BP
    ICaNa_i = Ca_i_ss_consts*ICaNa_i_NP + fICaL_CaMKonly*ICaNa_i_CaMK + fICaL_PKAonly*ICaNa_i_PKA + fICaL_BP*ICaNa_i_BP
    ICaK_i = Ca_i_ss_consts*ICaK_i_NP + fICaL_CaMKonly*ICaK_i_CaMK + fICaL_PKAonly*ICaK_i_PKA + fICaL_BP*ICaK_i_BP

    #% And we weight ICaL (in ss) and ICaL_i
    ICa_i_fraction = (1-ICaL_fractionSS)
    ICaL_i = ICaL_i * ICa_i_fraction
    ICaNa_i = ICaNa_i * ICa_i_fraction
    ICaK_i = ICaK_i * ICa_i_fraction
    ICaL_ss = ICaL_ss * ICaL_fractionSS
    ICaNa_ss = ICaNa_ss * ICaL_fractionSS
    ICaK_ss = ICaK_ss * ICaL_fractionSS

    ICaL = ICaL_ss + ICaL_i
    ICaNa = ICaNa_ss + ICaNa_i
    ICaK = ICaK_ss + ICaK_i
    #ICaL_tot = ICaL + ICaNa + ICaK

    ##%#% IKr
    #% transition rates
    #% from c0 to c1 in l-v model,
    alpha = 0.1161 * np.exp(0.2990 * vfrt)
    #% from c1 to c0 in l-v/
    beta =  0.2442 * np.exp(-1.604 * vfrt)

    #% from c1 to c2 in l-v/
    # alpha1 = 1.25 * 0.1235 
    # #% from c2 to c1 in l-v/
    # beta1 =  0.1911

    #% from c2 to o/           c1 to o
    alpha2 =0.0578 * np.exp(0.9710 * vfrt) #%
    #% from o to c2/
    beta2 = 0.349e-3* np.exp(-1.062 * vfrt) #%

    #% from o to i
    alphai = 0.2533 * np.exp(0.5953 * vfrt) #%
    #% from i to o
    betai = 1.25* 0.0522 * np.exp(-0.8209 * vfrt) #%

    #% from c2 to i (from c1 in orig)
    alphac2ToI = 0.52e-4 * np.exp(1.525 * vfrt) #%
    #% from i to c2
    #% betaItoC2 = 0.85e-8 * np.exp(-1.842 * vfrt) #%
    betaItoC2 = (beta2 * betai * alphac2ToI)/(alpha2 * alphai) #%
    #% transitions themselves
    #% for reason of backward compatibility of naming of an older version of a
    #% MM IKr, c3 in code is c0 in article diagram, c2 is c1, c1 is c2.

    dc0 = ikr_c1 * beta - ikr_c0 * alpha #% delta for c0
    dc1 = ikr_c0 * alpha + ikr_c2*beta1 - ikr_c1*(beta+alpha1) #% c1
    dc2 = ikr_c1 * alpha1 + ikr_o*beta2 + ikr_i*betaItoC2 - ikr_c2 * (beta1 + alpha2 + alphac2ToI) #% subtraction is into c2, to o, to i. #% c2
    do = ikr_c2 * alpha2 + ikr_i*betai - ikr_o*(beta2+alphai)
    di = ikr_c2*alphac2ToI + ikr_o*alphai - ikr_i*(betaItoC2 + betai)
    

    IKr = GKr * np.sqrt(ko/5)* ikr_o  * (v-EK)

    ##%#% IKs
    V_EKs = (v-EKs)
    IKs_const = (2.326e-4*np.exp((v+48.28)/17.80)+0.001292*np.exp((-(v+210.0))/230.0))
    xs1ss=1.0/(1.0+np.exp((-(v+11.60))/8.932))
    txs1=817.3+1.0/IKs_const
    dxs1=(xs1ss-xs1)/txs1
    xs2ss=xs1ss
    txs2=1.0/(0.01*np.exp((v-50.0)/20.0)+0.0193*np.exp((-(v+66.54))/31.0))
    dxs2=(xs2ss-xs2)/txs2
    KsCa=1.0+0.6/(1.0+(3.8e-5/cai)**1.4)
    
    IKs_NP=GKs*KsCa*xs1*xs2*V_EKs
    #IKs=GKs*KsCa*xs1*xs2*(v-EKs)
    
    ##### BARS for IKs
    txs1_P=txs1_P_const+ 2.75/IKs_const#BetaAdrenergic 
    # txs1_P=0.6*txs1
    dxs1_P=(xs1ss-xs1_P)/txs1_P
    GKs_P=GKs*10
    IKs_P= GKs_P*KsCa*xs1_P*xs2*V_EKs
    IKs =(1 - fIKsP) * IKs_NP + fIKsP * IKs_P

    ##%#% IK1
    aK1 = 4.094/(1+np.exp(0.1217*(v-EK-49.934)))
    bK1 = (15.72*np.exp(0.0674*(v-EK-3.257))+np.exp(0.0618*(v-EK-594.31)))/(1+np.exp(-0.1629*(v-EK+14.207)))
    K1ss = aK1/(aK1+bK1)

    
    IK1=GK1*np.sqrt(ko/5)*K1ss*(v-EK)

    ##%#% INaCa
    hca=np.exp(qca*vfrt)
    hna=np.exp(qna*vfrt)
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
    E1=x1/x_sum#(x1+x2+x3+x4)
    E2=x2/x_sum#(x1+x2+x3+x4)
    E3=x3/x_sum#(x1+x2+x3+x4)
    E4=x4/x_sum#(x1+x2+x3+x4)
    KmCaAct=150.0e-6
    allo=1.0/(1.0+(KmCaAct/cai)**2.0)
    
    JncxNa=3.0*(E4*k7-E1*k8)+E3*k4pp-E2*k3pp
    JncxCa=E2*k2-E1*k1


    INaCa_i=(1-INaCa_fractionSS)*Gncx*allo*(zna*JncxNa+zca*JncxCa)

    #%calculate INaCa_ss
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
    x_sum = x1+x2+x3+x4
    E1=x1/x_sum
    E2=x2/x_sum
    E3=x3/x_sum
    E4=x4/x_sum

    allo=1.0/(1.0+(KmCaAct/cass)**2.0)
    JncxNa=3.0*(E4*k7-E1*k8)+E3*k4pp-E2*k3pp
    JncxCa=E2*k2-E1*k1
    INaCa_ss=INaCa_fractionSS*Gncx*allo*(zna*JncxNa+zca*JncxCa)

    ##%#% INaK
    fINaK_PKA = fINaKP
    if fINaK_PKA != 0:
        Knai0_P = Knai0*0.7#BetaAdrenergic
        Knai0=(1-fINaK_PKA)*Knai0 + fINaK_PKA*Knai0_P#BetaAdrenergic
    Knai=Knai0*np.exp((delta*vfrt)/(3.0))
    Knao=Knao0*np.exp(((1.0-delta)*vfrt)/(3.0))
    ATP_const1 = MgATP/Kmgatp
    ATP_const2 = 1+ATP_const1
    nai_div_Knai = nai/Knai
    ki_div_kki = ki/Kki
    nao_div_knao = nao/Knao
    
    P=eP/(1.0+H/Khp+nai/Knap+ki/Kxkur)
    a1_b4_conc = ((1.0+nai_div_Knai)**3.0+(1.0+ki_div_kki)**2.0-1.0)
    a1=(k1p*(nai_div_Knai)**3.0)/a1_b4_conc
    b1=k1m*MgADP
    a2=k2p
    b2_a3_const = ((1.0+nao_div_knao)**3.0+(1.0+ko/Kko)**2.0-1.0)
    b2=(k2m*(nao_div_knao)**3.0)/b2_a3_const
    a3=(k3p*(ko/Kko)**2.0)/b2_a3_const
    b3=(k3m*P*H)/ATP_const2
    a4=(k4p*ATP_const1)/ATP_const2
    b4=(k4m*(ki_div_kki)**2.0)/a1_b4_conc
    x1=a4*a1*a2+b2*b4*b3+a2*b4*b3+b3*a1*a2
    x2=b2*b1*b4+a1*a2*a3+a3*b1*b4+a2*a3*b4
    x3=a2*a3*a4+b3*b2*b1+b2*b1*a4+a3*a4*b1
    x4=b4*b3*b2+a3*a4*a1+b2*a4*a1+b3*b2*a1
    x_sum = x1+x2+x3+x4
    E1=x1/x_sum
    E2=x2/x_sum
    E3=x3/x_sum
    E4=x4/x_sum
    JnakNa=3.0*(E1*a3-E2*b3)
    JnakK=2.0*(E4*b1-E3*a1)

    INaK=Pnak*(zna*JnakNa+zk*JnakK)

    ##%#% Minor/background currents
    ##%calculate IKb with BARS effect
    xkb=1.0/(1.0+np.exp(-(v-10.8968)/(23.9871)))
    xkb_V_EK = xkb*(v-EK)
    GKbP=GKb*1.2
    IKb_P=GKbP*xkb_V_EK#xkb*(V_EK)
    IKb_NP=GKb*xkb_V_EK#xkb*(V_EK)
    IKb =(1 - fIKurP) * IKb_NP + fIKurP* IKb_P
    
    # Calculate other ackground currents
    INab=PNab*vffrt*(nai*Na_K_exp_const-nao)/(Na_K_exp_const-1.0)
    ICab=PCab*4.0*vffrt*(gammaCaiMyo*cai*Ca_exp_const-gammaCaoMyo*cao)/(Ca_exp_const-1.0)
    IpCa=GpCa*cai/(0.0005+cai)

    #%#% Chloride

    Fjunc = 1 
    Fsl = 1-Fjunc #% fraction in SS and in myoplasm - as per literature, I(Ca)Cl is in junctional subspace
    KdClCa = 0.1    #% [mM]

    I_ClCa_junc = Fjunc*GClCa/(1+KdClCa/cass)*(v-eclss)
    I_ClCa_sl = Fsl*GClCa/(1+KdClCa/cai)*(v-ecl)

    I_ClCa = I_ClCa_junc+I_ClCa_sl
    I_Clbk = GClB*(v-ecl)

    #%#% Calcium handling
    #%calculate ryanodione receptor calcium induced calcium release from the jsr

    #%#% Jrel
    
    J_rel_const = (-ICaL_ss)/(1.0+(jsrMidpoint/cajsr)**8.0)
    tau_rel_const = (1.0+0.0123/cajsr)
    
    
    Jrel_inf=Jrel_inf_scale * a_rel*J_rel_const
    tau_rel=np.maximum(bt/tau_rel_const,0.001)#(1.0+0.0123/cajsr)
    dJrelnp=(Jrel_inf-Jrel_np)/tau_rel
    
    Jrel_infp=Jrel_inf_scale *a_relp*J_rel_const #(Jrel_inf_scale) did not seem to be present in TOR model, need to check
    tau_relp=np.maximum(btp/tau_rel_const,0.001)#(1.0+0.0123/cajsr)
    dJrelp=(Jrel_infp-Jrel_p)/tau_relp
    
    #####################################
    # PKA-P channel behavior
    #####################################
    #a_relP=a_rel*1.4#BetaAdrenergic  (#Gong--> 2.5)
    Jrel_inf_P=Jrel_inf_scale*a_relP*J_rel_const
    tau_rel_P=np.maximum((0.75)*bt/tau_rel_const,0.001)#BetaAdrenergic
    dJrelnp_P=(Jrel_inf_P-Jrel_np_P)/tau_rel_P

    #a_relBP=a_relp*1.4#BetaAdrenergic  (#Gong--> 2.5)
    Jrel_infBP=Jrel_inf_scale*a_relBP*J_rel_const
    tau_relBP=np.maximum((0.75)*btp/tau_rel_const,0.001)#BetaAdrenergic
    dJrelp_P=(Jrel_infBP-Jrel_p_P)/tau_relBP

    fJrel_PKA = fRyRP
    fJrel_BP = fJrelp*fJrel_PKA
    fJrel_CaMKonly = fJrelp - fJrel_BP
    fJrel_PKAonly = fJrel_PKA - fJrel_BP
    Jrel= 1.5378* ((1.0-fJrel_CaMKonly-fJrel_PKAonly-fJrel_BP)*Jrel_np + fJrel_CaMKonly*Jrel_p + fJrel_PKAonly*Jrel_np_P + fJrel_BP*Jrel_p_P )
    
    #Jrel=1.5378 * ((1.0-fJrelp)*Jrel_np+fJrelp*Jrel_p)
    cai_const1 = 0.005425*cai
    cai_const2 = cai+0.00092
    fJupp=fJrelp#(1.0/(1.0+KmCaMK/CaMKa))
    Jupnp=Jup_scale*cai_const1/(cai_const2)
    Jupp= Jup_scale*2.75*cai_const1/(cai_const2-0.00017)
    
    ######################################
    # PKA-P SERCA
    #calculate serca pump, ca uptake flux
    Jup_P= Jup_scale*cai_const1/(cai_const2*0.7)#BetaAdrenergic
    Jup_BP= Jup_scale*2.75*cai_const1/(cai+(0.00092-0.00017)*0.7)#BetaAdrenergic
 
    # 4 population
    fJup_P = fPLBP # PKA-P channel fraction
    fJup_BP = fJupp*fJup_P
    fJup_CaMKonly = fJupp - fJup_BP
    fJup_PKAonly = fJup_P - fJup_BP

    Jleak=0.0048825*cansr/15.0
    Jup= ((1.0-fJup_CaMKonly-fJup_PKAonly-fJup_BP)*Jupnp + fJup_CaMKonly*Jupp + fJup_PKAonly*Jup_P + fJup_BP*Jup_BP) - Jleak
    #Jup=(1.0-fJupp)*Jupnp+fJupp*Jupp-Jleak

    #%calculate tranlocation flux
    Jtr=(cansr-cajsr)/60

    #%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%

    #%#% calculate the stimulus current, Istim
    I_stim = Istim(time, cycle_length,amp,duration)


    #%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%

    #%update the membrane voltage
    dv=-(INa+INaL+Ito+ICaL+ICaNa+ICaK+IKr+IKs+IK1+INaCa_i+INaCa_ss+INaK+INab+IKb+IpCa+ICab+I_ClCa+I_Clbk+I_stim)
    #%calculate diffusion fluxes
    JdiffNa=(nass-nai)/2.0
    JdiffK=(kss-ki)/2.0
    Jdiff=(cass-cai)/0.2
    JdiffCl=(clss-cli)/2.0

    #%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%

    #%calcium buffer constants 
    #calcium buffer constants when FTnIP is 0, the value of kmtrpn is same as TOR DynCL 
    kmtrpn = (1 -  fTnIP) * kmtrpn +  fTnIP * 1.6 * kmtrpn #BetaAdrenergic
    
    #%update intracellular concentrations, using buffers for cai, cass, cajsr
    F_vmyo = (F*vmyo)
    Acap_div_Fvmyo = Acap/F_vmyo
    Acap_div_2Fvmyo = Acap_div_Fvmyo/2.0
    vss_div_vmyo = vss/vmyo
    dnai=-(ICaNa_i+INa+INaL+3.0*INaCa_i+3.0*INaK+INab)*Acap_div_Fvmyo+JdiffNa*vss_div_vmyo
    dnass=-(ICaNa_ss+3.0*INaCa_ss)*Acap/(F*vss)-JdiffNa

    dki=-(ICaK_i+Ito+IKr+IKs+IK1+IKb+I_stim-2.0*INaK)*Acap_div_Fvmyo+JdiffK*vss_div_vmyo
    dkss=-(ICaK_ss)*Acap/(F*vss)-JdiffK

    Bcai=1.0/(1.0+cmdnmax*kmcmdn/(kmcmdn+cai)**2.0+trpnmax*kmtrpn/(kmtrpn+cai)**2.0)
    dcai=Bcai*(-(ICaL_i + IpCa+ICab-2.0*INaCa_i)*Acap_div_2Fvmyo-Jup*vnsr/vmyo+Jdiff*vss_div_vmyo)

    Bcass=1.0/(1.0+BSRmax*KmBSR/(KmBSR+cass)**2.0+BSLmax*KmBSL/(KmBSL+cass)**2.0)
    dcass=Bcass*(-(ICaL_ss-2.0*INaCa_ss)*Acap/(2.0*F*vss)+Jrel*vjsr/vss-Jdiff)

    #z_cl is -1
    dcli = - (I_Clbk + I_ClCa_sl)*Acap/(-F_vmyo)+JdiffCl*vss_div_vmyo
    dclss = - I_ClCa_junc*Acap/(-1*F*vss)-JdiffCl

    dcansr=Jup-Jtr*vjsr/vnsr

    Bcajsr=1.0/(1.0+csqnmax*kmcsqn/(kmcsqn+cajsr)**2.0)
    dcajsr=Bcajsr*(Jtr-Jrel)

    #%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%

    #%output the state vector when ode_flag==1, and the calculated currents and fluxes otherwise

    output=[dv, dnai, dnass, dki, dkss ,dcai ,dcass, dcansr ,dcajsr, dm ,dhp, dh ,dj, djp, dmL ,dhL, dhLp, da, diF, diS, dap, diFp, diSp,dd, dff, dfs, dfcaf, dfcas, djca, dnca ,dnca_i, dffp, dfcafp ,dxs1 ,dxs2 ,dJrelnp, dCaMKt,dc0, dc1, dc2, do, di, dJrelp, dcli,dclss,dxs1_P, dd_P, dff_P, dfs_P, dfcaf_P, dfcas_P, dfBPf, dfcaBPf, dh_P, dj_P, dhp_P, djp_P, dJrelnp_P, dJrelp_P,
            dcond1,dcond2,dcond3,dcond4,dcond5,dcond6,dcond7,dcond8,dcond9,dcond10,dcond11,dcond12,dcond13,dcond14,dcond15,dcond16,dcond17,dcond18,dcond19,dcond20,dcond21,dcond22,dcond23,dcond24,dcond25,dcond26,dcond27,dcond28,dcond29,dcond30,dcond31,dcond32,dcond33,dcond34,dcond35,dcond36,dcond37,dcond38,dcond39,dcond40,dcond41,dcond42,dcond43,dcond44,dcond45,dcond46,dcond47,dcond48,dcond49,dcond50,dcond51,dcond52,dcond53,dcond54,dcond55,dcond56,dcond57]
    return output



def run_Doste_Model(cycles, cycle_length, cell_type, BARS, amp=-53):
    # ISO_conc is the steady-state isoproterenol concentration (uM) reached at the end of the
    # ramp when BARS signalling is active. Hardcoded to 0.1 uM (typical experimental level used
    # to fully activate beta-AR signalling); change here if a different ISO concentration is needed.
    ISO_conc = 0.1
    model_type   = "BARS 2022"
    G_Ks         = GKs_conductance(model_type, cell_type)
    G_Kr         = GKr_conductance(model_type, cell_type)
    G_K1         = GK1_conductance(model_type, cell_type)
    G_to         = Gto_conductance(model_type, cell_type)
    G_Na_late, G_Na_fast = GNa_conductance(model_type, cell_type)
    GCa          = GCa_conductance(model_type, cell_type)
    GNCX         = GNCX_conductance(model_type, cell_type)
    GNaK         = GNaK_conductance(model_type, cell_type)
    GKb_input    = GKb_conductance(model_type, cell_type)
    GNab         = GNab_conductance(model_type, cell_type)
    GCab         = GCab_conductance(model_type, cell_type)
    GpCa_input   = GpCa_conductance(model_type, cell_type)
    GClCa_input  = GClCa_conductance(model_type, cell_type)
    GClb         = GClb_conductance(model_type, cell_type)

    #cli = 24   ##% Intracellular Cl  [mM]
    clo = 150  ##% Extracellular Cl  [mM]
    nao = 140#; if (isfield(parameters,'nao')) nao = parameters.nao; end
    cao = 1.8#;if (isfield(parameters,'cao')) cao = parameters.cao; end
    ko = 5#;if (isfield(parameters,'ko')) ko = parameters.ko; end
    R=8314.0
    T=310.0
    F=96485.0
    L=0.01
    rad=0.0011
    vcell=1000.0*math.pi*(rad**2)*L
    Ageo=2*math.pi*rad*rad+2*math.pi*rad*L
    Acap=2*Ageo
    vmyo=0.68*vcell
    vnsr=0.0552*vcell
    vjsr=0.0048*vcell
    vss=0.02*vcell
    ##%CaMK constants
    KmCaMK=0.15
    aCaMK=0.05
    bCaMK=0.00068
    CaMKo=0.05
    KmCaM=0.0015
    
    #%convenient shorthand calculations
    frt = F/(R*T)
    
    cmdnmax= 0.05 
    amp=amp #-53
    duration=1
    kmcmdn=0.00238 
    trpnmax=0.07
    kmtrpn=0.0005
    BSRmax=0.047
    KmBSR = 0.00087
    BSLmax=1.124
    KmBSL = 0.0087
    csqnmax=10.0
    kmcsqn=0.8

    zna=1.0
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
    delta=-0.1550
    Pnak= GNaK
    GKb=GKb_input
    PNab=GNab
    PCab=GCab
    GpCa=GpCa_input
    bt=4.75
    a_rel=0.5*bt
    btp=1.25*bt
    a_relp=0.5*btp
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
    
    reversal_potential_const = (R*T/F)
    
    ICaL_fractionSS = 0.8#; if (isfield(parameters, 'ICaL_fractionSS')) ICaL_fractionSS = parameters.ICaL_fractionSS; end
    INaCa_fractionSS = 0.35#; if (isfield(parameters, 'INaCa_fractionSS')) INaCa_fractionSS = parameters.INaCa_fractionSS; end
    PKNa=0.01833
    GNaL=G_Na_late
    Gto=G_to
    Kmn=0.002
    k2n=500.0
    GKs= G_Ks
    zca = 2.0
    kna1=15.0
    kna2=5.0
    kna3=88.12
    kasymm=12.5
    wna=6.0e4
    wca=6.0e4
    wnaca=5.0e3
    kcaon=1.5e6
    kcaoff=5.0e3
    qna=0.5224
    qca=0.1670
    txs1_P_const = 817.3-1.75/(2.326e-4*np.exp((10+48.28)/17.80)+0.001292*np.exp((-(10+210.0))/230.0))
    
    Aff=0.6
    Afs=1.0-Aff
    dielConstant = 74 #% water at 37°.
    constA = 1.82*10**6*(dielConstant*T)**(-1.5)
    GK1= G_K1
    Gncx= GNCX
    KmCaAct=150.0e-6 
    zk=1.0
    jsrMidpoint = 1.7
    GClCa = GClCa_input
    GClB = GClb
    KdClCa = 0.1    #% [mM]
    #%#% The rest
    PCa=GCa
    PCa_P=GCa * 1.1
    GNa = G_Na_fast
    GNa_P= GNa*1.7 #BetaAdrenergic
    GKr = G_Kr
    Jrel_inf_scale = 1
    Jup_scale = 1
    isepi = False
    #% from c1 to c2 in l-v/
    alpha1 = 1.25 * 0.1235 
    #% from c2 to c1 in l-v/
    beta1 =  0.1911
    a_relP=a_rel*1.4#BetaAdrenergic  (#Gong--> 2.5)
    a_relBP=a_relp*1.4#BetaAdrenergic  (#Gong--> 2.5)
    
    radiusmultiplier = 1
    ## Constantssignalling function
    radius = 0.0011*radiusmultiplier# Cell radius
    volume = 1000.0 * math.pi * (radius**2) * L# Cell volume

    Vcv = 0.02 * volume# Volume of the caveolar subspace, 2
    Vec = 0.04 * volume# Volume of the extracaveolar subspace, 3
    Vcyt = volume * 0.678# Volume of the Cytoplasm / Myoplasm, 4

    R_wvtovcv = volume / Vcv# Ratio of whole volume to caveolar subspace volume, 5
    R_wvtovec = volume / Vec# Ratio of whole volume to extracaveolar subspace volume, 6
    R_evtovcyt = volume / Vcyt# Ratio of whole volume to cytoplasm volume, 7
    
    PP1_EC = 0.1# PP1 concentration in the extracaveolar compartment, 35
    PP1_CV = 0.25# PP1 concentration in the caveolar compartment, 36
    PP1_CYT = 0.2# PP1 concentration in the cytosolic compartment, 37
    PP1_aff = 0.001# Affinity foor PP1 Inhibitor 1 binding, 38
    
    if cell_type == 'EPI':
        #GNaL = GNaL * 0.6
        #Gto = Gto * 2.0
        #PCa = PCa * 1.2
        #GKr = GKr * 1.3
        #GKs = GKs * 1.4
        #GK1 = GK1 * 1.2
        #Gncx = Gncx * 1.1
        cmdnmax = cmdnmax * 1.3
        Jup_scale = Jup_scale * 1.3
        PCa_P=PCa_P*1.2
        isepi = True
        #Pnak = Pnak * 0.9
        #GKb = GKb * 0.6

    elif cell_type == 'M':
        #Gto = Gto * 2.0
        #PCa = PCa * 2
        #GKr = GKr * 0.8
        #GK1 = GK1 * 1.3
        #Gncx = Gncx * 1.4
        Jrel_inf_scale = Jrel_inf_scale * 1.7
        PCa_P=PCa_P*2
        #Pnak = Pnak * 0.7

    PCap=1.1*PCa
    PCaNa=0.00125*PCa
    PCaK=3.574e-4*PCa
    PCaNap=0.00125*PCap
    PCaKp=3.574e-4*PCap
    PCaNa_P=0.00125*PCa_P
    PCaK_P=3.574e-4*PCa_P
    
    #### Signalling from BARS model
    ISO=np.zeros(cycles)
    #%Gradual administration of ISO (lineal variation over time):
    if cycles<=33:# just setting 33 as an arbitrary number, unlikely for anyone to run less than 33 heart beats
        ISO = ISO
    else:
        initISO=20 #%Beat of start of ISO application
        finalISO=np.min([200,round(0.65*cycles)])#cycles #%Beat when max levels of ISO are reach
        ISO[initISO:]=0.1 #%Iso values for each beat  (uM/l)
        ISO[initISO:finalISO]=np.linspace(0,ISO_conc,finalISO-initISO)#%Iso values for each beat  (uM/l) 
    
    #print(ISO)
    cumprod = np.cumprod(ISO == 0)
    NOISO = np.sum(cumprod)
    # Initialize runSignalingPathway with 1s, runsignalling pathway where there is ISO
    runSignalingPathway = np.ones(cycles, dtype=int)
    runSignalingPathway[:NOISO] = 0
    #print(runSignalingPathway)
    
    #print(len(ISO))
    
    constants_T = [
    cycle_length, clo, nao, cao, ko, R, T, F, L, rad,
    vcell, Ageo, Acap, vmyo, vnsr, vjsr, vss,
    KmCaMK, aCaMK, bCaMK, CaMKo, KmCaM, frt, 
    cmdnmax, amp, duration, kmcmdn, trpnmax, kmtrpn,
    BSRmax, KmBSR, BSLmax, KmBSL, csqnmax, kmcsqn, zna, k1p, k1m, k2p, k2m, k3p, k3m, k4p, k4m,
    Knai0, Knao0, delta, Pnak, GKb, PNab, PCab, GpCa,
    bt, a_rel, btp, a_relp, Kki, Kko, MgADP, MgATP,
    Kmgatp, H, eP, Khp, Knap, Kxkur,reversal_potential_const, 
    ICaL_fractionSS, INaCa_fractionSS, PKNa, GNaL, Gto, Kmn, k2n, GKs, zca, kna1, kna2, kna3, kasymm, wna, wca, wnaca, kcaon, kcaoff, qna, qca,
    Aff, Afs, constA, GK1, Gncx, KmCaAct, zk, jsrMidpoint, GClCa, GClB, KdClCa, PCa, GNa,GKr,Jrel_inf_scale,Jup_scale, cell_type,
    PCaNa, PCaK, PCaNap, PCaKp, PCap, PCa_P, PCaNa_P, PCaK_P, GNa_P, runSignalingPathway, cycles,
    ISO, radiusmultiplier,alpha1,beta1,a_relP,a_relBP,isepi,txs1_P_const,
    R_wvtovcv,R_wvtovec,R_evtovcyt,PP1_EC,PP1_CV,PP1_CYT,PP1_aff
    ]
    
    

    initial_conds = GetStartingState_DynCL(cell_type)#[m ,m_L ,h_fast ,h_slow ,h_CaMK_slow ,h_L ,h_L_CaMK ,j ,j_CaMK ,CaMK_trap ,V_m ,a,i_fast,i_slow ,i_CaMK_fast , i_CaMK_slow,a_CaMK,d,f_fast ,f_slow ,j_Ca,f_Ca_CaMK_fast,n, f_Ca_slow ,f_Ca_fast, f_CaMK_fast , x_r_slow ,x_r_fast ,x_s1,x_s2 ,x_K1 ,J_rel_NP,J_rel_CaMK ,Na_ion_conc_i , Na_ion_conc_ss, K_ion_conc_i , K_ion_conc_ss , Ca_ion_conc_i , Ca_ion_conc_ss , Ca_ion_conc_nsr , Ca_ion_conc_jsr]
    tspan = (0, cycles*cycle_length)
    
    # Truthy check: accepts boolean True/False, integer 1/0, or legacy 'True'/'False'.
    if BARS:
        sol = solve_ivp(fun = Doste_Model_T, t_span = tspan, y0 = initial_conds, args = constants_T,method='LSODA',rtol= 1e-5,max_step = 1)
    else:
        c = Constants_SignalingMyokit2(0, radiusmultiplier)
        v,nai,nass,ki,kss,cai,cass,cansr,cajsr,m,hp,h,j,jp,mL,hL,hLp,a,iF,iS,ap,iFp,iSp,d,ff,fs,fcaf,fcas,jca,nca,nca_i,ffp,fcafp,xs1,xs2,Jrel_np,CaMKt,ikr_c0,ikr_c1,ikr_c2,ikr_o,ikr_i,Jrel_p,cli,clss,xs1_P, d_P,ff_P,fs_P,fcaf_P,fcas_P,fBPf,fcaBPf, h_P, j_P, hp_P, jp_P, Jrel_np_P, Jrel_p_P, cond1,cond2,cond3,cond4,cond5,cond6,cond7,cond8,cond9,cond10,cond11,cond12,cond13,cond14,cond15,cond16,cond17,cond18,cond19,cond20,cond21,cond22,cond23,cond24,cond25,cond26,cond27,cond28,cond29,cond30,cond31,cond32,cond33,cond34,cond35,cond36,cond37,cond38,cond39,cond40,cond41,cond42,cond43,cond44,cond45,cond46,cond47,cond48,cond49,cond50,cond51,cond52,cond53,cond54,cond55,cond56,cond57 = initial_conds
        fICaLP,fIKsP,fPLBP,fTnIP,fINaP,fINaKP,fRyRP,fIKurP =  EffectiveFraction_Torord2(cond40,cond41,cond42,cond43,cond44,cond45,cond46,cond47, c) 
        Whole_cell_PP1 = 0.1371
        constants_F = [
        cycle_length, clo, nao, cao, ko, R, T, F, L, rad,
        vcell, Ageo, Acap, vmyo, vnsr, vjsr, vss,
        KmCaMK, aCaMK, bCaMK, CaMKo, KmCaM, frt, 
        cmdnmax, amp, duration, kmcmdn, trpnmax, kmtrpn,
        BSRmax, KmBSR, BSLmax, KmBSL, csqnmax, kmcsqn, zna, k1p, k1m, k2p, k2m, k3p, k3m, k4p, k4m,
        Knai0, Knao0, delta, Pnak, GKb, PNab, PCab, GpCa,
        bt, a_rel, btp, a_relp, Kki, Kko, MgADP, MgATP,
        Kmgatp, H, eP, Khp, Knap, Kxkur,reversal_potential_const, 
        ICaL_fractionSS, INaCa_fractionSS, PKNa, GNaL, Gto, Kmn, k2n, GKs, zca, kna1, kna2, kna3, kasymm, wna, wca, wnaca, kcaon, kcaoff, qna, qca,
        Aff, Afs, constA, GK1, Gncx, KmCaAct, zk, jsrMidpoint, GClCa, GClB, KdClCa, PCa, GNa,GKr,Jrel_inf_scale,Jup_scale, cell_type,
        PCaNa, PCaK, PCaNap, PCaKp, PCap, PCa_P, PCaNa_P, PCaK_P, GNa_P, cycles,c,fICaLP,fIKsP,fPLBP,fTnIP,fINaP,fINaKP,fRyRP,fIKurP,Whole_cell_PP1,alpha1,beta1,a_relP,a_relBP,isepi,txs1_P_const
        ]
        sol = solve_ivp(fun = Doste_Model_F, t_span = tspan, y0 = initial_conds, args = constants_F,method='LSODA',rtol= 1e-5,max_step = 1)
        
    time = sol.t  # Time points
    solutions = sol.y  # Solution vectors, each row corresponds to a variable
    y_names = ["V","Nai","Na_ss","Ki","K_ss","Cai","Ca_ss","Ca_nsr","Ca_jsr","m","hp","h","j","jp","m_L","h_L","h_L_CaMK","a","i_fast","i_slow","a_CaMK","i_CaMK_fast","i_CaMK_slow","d","f_fast","f_slow","f_Ca_fast","f_Ca_slow","j_Ca","n","nca_i","f_CaMK_fast","f_Ca_CaMK_fast","x_s1","x_s2","J_rel_NP","CaMK_trap","C0","C1","C2","O","I","J_rel_CaMK","Cl_i","Cl_ss",'Xs_P',"dd_P", "dff_P", "dfs_P", "dfcaf_P", "dfcas_P", "dfBPf", "dfcaBPf","dh_P", "dj_P", "dhp_P", "djp_P","dJrelnp_P", "dJrelp_P",
               "dcond1","dcond2","dcond3","dcond4","dcond5","dcond6","dcond7","dcond8","dcond9","dcond10","dcond11","dcond12","dcond13","dcond14","dcond15","dcond16","dcond17","dcond18","dcond19","dcond20","dcond21","dcond22","dcond23","dcond24","dcond25","dcond26","dcond27","dcond28","dcond29","dcond30","dcond31","dcond32","dcond33","dcond34","dcond35","dcond36","dcond37","dcond38","dcond39","dcond40","dcond41","dcond42","dcond43","dcond44","dcond45","dcond46","dcond47","dcond48","dcond49","dcond50","dcond51","dcond52","dcond53","dcond54","dcond55","dcond56","dcond57"]
    
    df = pd.DataFrame(solutions.T, columns=y_names)
    
    df['time'] = time
    stim_duration = duration

    return df, pd.DataFrame(), stim_duration


@njit
def Doste_Model_F(time,X0,cycle_length, clo, nao, cao, ko, R, T, F, L, rad,
    vcell, Ageo, Acap, vmyo, vnsr, vjsr, vss,
    KmCaMK, aCaMK, bCaMK, CaMKo, KmCaM, frt, 
    cmdnmax, amp, duration, kmcmdn, trpnmax, kmtrpn,
    BSRmax, KmBSR, BSLmax, KmBSL, csqnmax, kmcsqn, zna, k1p, k1m, k2p, k2m, k3p, k3m, k4p, k4m,
    Knai0, Knao0, delta, Pnak, GKb, PNab, PCab, GpCa,
    bt, a_rel, btp, a_relp, Kki, Kko, MgADP, MgATP,
    Kmgatp, H, eP, Khp, Knap, Kxkur,reversal_potential_const, 
    ICaL_fractionSS, INaCa_fractionSS, PKNa, GNaL, Gto, Kmn, k2n, GKs, zca, kna1, kna2, kna3, kasymm, wna, wca, wnaca, kcaon, kcaoff, qna, qca,
    Aff, Afs, constA, GK1, Gncx, KmCaAct, zk, jsrMidpoint, GClCa, GClB, KdClCa, PCa, GNa,GKr,Jrel_inf_scale,Jup_scale, celltype,
    PCaNa, PCaK, PCaNap, PCaKp, PCap, PCa_P, PCaNa_P, PCaK_P, GNa_P, cycles,c,fICaLP,fIKsP,fPLBP,fTnIP,fINaP,fINaKP,fRyRP,fIKurP,Whole_cell_PP1,alpha1,beta1,a_relP,a_relBP,isepi,
    txs1_P_const):
    
    v,nai,nass,ki,kss,cai,cass,cansr,cajsr,m,hp,h,j,jp,mL,hL,hLp,a,iF,iS,ap,iFp,iSp,d,ff,fs,fcaf,fcas,jca,nca,nca_i,ffp,fcafp,xs1,xs2,Jrel_np,CaMKt,ikr_c0,ikr_c1,ikr_c2,ikr_o,ikr_i,Jrel_p,cli,clss,xs1_P, d_P,ff_P,fs_P,fcaf_P,fcas_P,fBPf,fcaBPf, h_P, j_P, hp_P, jp_P, Jrel_np_P, Jrel_p_P, cond1,cond2,cond3,cond4,cond5,cond6,cond7,cond8,cond9,cond10,cond11,cond12,cond13,cond14,cond15,cond16,cond17,cond18,cond19,cond20,cond21,cond22,cond23,cond24,cond25,cond26,cond27,cond28,cond29,cond30,cond31,cond32,cond33,cond34,cond35,cond36,cond37,cond38,cond39,cond40,cond41,cond42,cond43,cond44,cond45,cond46,cond47,cond48,cond49,cond50,cond51,cond52,cond53,cond54,cond55,cond56,cond57 = X0

    ##%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%
    
    dcond1=dcond2=dcond3=dcond4=dcond5=dcond6=dcond7=dcond8=dcond9=dcond10=dcond11=dcond12=dcond13=dcond14=dcond15=dcond16=dcond17=dcond18=dcond19=dcond20=dcond21=dcond22=dcond23=dcond24=dcond25=dcond26=dcond27=dcond28=dcond29=dcond30=dcond31=dcond32=dcond33=dcond34=dcond35=dcond36=dcond37=dcond38=dcond39=dcond40=dcond41=dcond42=dcond43=dcond44=dcond45=dcond46=dcond47=dcond48=dcond49=dcond50=dcond51=dcond52=dcond53=dcond54=dcond55=dcond56=dcond57 =0

    vfrt=v*frt
    vffrt=vfrt*F
    
    ##%update CaMK
    CaMKb=CaMKo*(1.0-CaMKt)/(1.0+KmCaM/cass)
    CaMKa=CaMKb+CaMKt
    betaCaMKII =  bCaMK * (0.1 + (0.9  * Whole_cell_PP1 / 0.1371))
    dCaMKt=aCaMK*CaMKb*(CaMKb+CaMKt)-betaCaMKII*CaMKt

    ##%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%

    ##%reversal potentials
    ENa=reversal_potential_const*np.log(nao/nai)
    EK=reversal_potential_const*np.log(ko/ki)
    EKs=reversal_potential_const*np.log((ko+PKNa*nao)/(ki+PKNa*nai))
    ecl = reversal_potential_const*np.log(cli/clo)            #% [mV]
    eclss = reversal_potential_const*np.log(clss/clo);           # % [mV]
    #%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%
    V_ENa = (v-ENa)
    V_EK = (v-EK)
    fINap=(1.0/(1.0+KmCaMK/CaMKa))
    fINaLp=fINap
    fItop=fINap
    fICaLp=fINap
    fJrelp= fINap

    mss = 1 / ((1 + np.exp( -(56.86 + v) / 9.03 ))**2)
    taum = 0.1292 * np.exp(-((v+45.79)/15.54)**2) + 0.06487 * np.exp(-((v-4.823)/51.12)**2)
    dm = (mss - m) / taum

    #%#% h gate
    ah = np.where(v >= -40,0,0.057 * np.exp( -(v + 80) / 6.8 ))
    bh =  np.where(v >= -40 , (0.77 / (0.13*(1 + np.exp( -(v + 10.66) / 11.1 )))) ,((2.7 * np.exp( 0.079 * v) + 3.1*10**5 * np.exp(0.3485 * v))))
    tauh = 1 / (ah + bh)
    v_plus_71 = v + 71.55
    hss = 1 / ((1 + np.exp( (v_plus_71)/7.43 ))**2)
    dh = (hss - h) / tauh
    #%#% j gate
    aj = np.where(v >= -40, (0) ,(((-2.5428 * 10**4*np.exp(0.2444*v) - 6.948*10**-6 * np.exp(-0.04391*v)) * (v + 37.78)) / (1 + np.exp( 0.311 * (v + 79.23) ))))
    bj = np.where(v >= -40 , ((0.6 * np.exp( 0.057 * v)) / (1 + np.exp( -0.1 * (v + 32) ))) ,((0.02424 * np.exp( -0.01052 * v )) / (1 + np.exp( -0.1378 * (v + 40.14) ))))
    tauj = 1 / (aj + bj)
    jss = hss#1 / ((1 + np.exp( (v + 71.55)/7.43 ))**2)
    dj = (jss - j) / tauj

    #%#% h phosphorylated
    hssp = 1 / ((1 + np.exp( (v_plus_71 + 6)/7.43 ))**2)
    dhp = (hssp - hp) / tauh
    #%#% j phosphorylated
    taujp = 1.46 * tauj
    djp = (jss - jp) / taujp
    
    
    #### BARS ##############
     ## gating PKA
    hss_P = 1 / ((1 + np.exp( (v_plus_71+5.0)/7.43 ))**2)  #BetaAdrenergic
    dh_P=(hss_P-h_P)/tauh   #BetaAdrenergic
    jss_P=hss_P
    dj_P = (jss_P - j_P) / tauj

    ## Both Phosphorilated
    hssp_P = 1 / ((1 + np.exp( (v_plus_71 + 6 + 5.0)/7.43 ))**2) #BetaAdrenergic
    dhp_P = (hssp_P - hp_P) / tauh
    jssp_P=hssp_P
    djp_P = (jssp_P - jp_P) / taujp

    fINa_P = fINaP # PKA-P fraction as assigned as input, take the value 0 or 1
    fINa_BP = fINap*fINa_P 
    fINa_CaMKonly = fINap-fINa_BP 
    fINa_PKAonly = fINa_P-fINa_BP 

    m_cubed = m**3 
    INa_NP = GNa*(V_ENa)*m_cubed*h*j  # Non-Phosphorylated 
    INa_CaMK = GNa*(V_ENa)*m_cubed*hp*jp 
    INa_PKA = GNa_P*(V_ENa)*m_cubed*h_P*j_P 
    INa_BP = GNa_P*(V_ENa)*m_cubed*hp_P*jp_P 
    
    # 4 population 
    INa = ((1-fINa_CaMKonly-fINa_PKAonly-fINa_BP)*INa_NP + fINa_CaMKonly*INa_CaMK + fINa_PKAonly*INa_PKA + fINa_BP*INa_BP) 
    
    #INa=GNa*(v-ENa)*m**3.0*((1.0-fINap)*h*j+fINap*hp*jp)


    mLss=1.0/(1.0+np.exp((-(v+42.85))/5.264))
    tm = 0.1292 * np.exp(-((v+45.79)/15.54)**2) + 0.06487 * np.exp(-((v-4.823)/51.12)**2) 
    tmL=tm
    dmL=(mLss-mL)/tmL
    hLss=1.0/(1.0+np.exp((v+87.61)/7.488))
    #thL=200.0
    dhL=(hLss-hL)/200.0
    hLssp=1.0/(1.0+np.exp((v+93.81)/7.488))
    #thLp=3.0*thL
    dhLp=(hLssp-hLp)/600.0

    INaL=GNaL*(v-ENa)*mL*((1.0-fINaLp)*hL+fINaLp*hLp)


    #%#% ITo
    #%calculate Ito
    ass=1.0/(1.0+np.exp((-(v-14.34))/14.82))
    ta=1.0515/(1.0/(1.2089*(1.0+np.exp(-(v-18.4099)/29.3814)))+3.5/(1.0+np.exp((v+100.0)/29.3814)))
    da=(ass-a)/ta
    iss=1.0/(1.0+np.exp((v+43.94)/5.711))
    delta_epi=1.0
    if isepi:
        delta_epi=1.0-(0.95/(1.0+np.exp((v+70.0)/5.0)))
        
    tiF=4.562+1/(0.3933*np.exp((-(v+100.0))/100.0)+0.08004*np.exp((v+50.0)/16.59))
    tiS=23.62+1/(0.001416*np.exp((-(v+96.52))/59.05)+1.780e-8*np.exp((v+114.1)/8.079))
    tiF=tiF*delta_epi
    tiS=tiS*delta_epi
    AiF=1.0/(1.0+np.exp((v-213.6)/151.2))
    AiS=1.0-AiF
    diF=(iss-iF)/tiF
    diS=(iss-iS)/tiS
    i=AiF*iF+AiS*iS
    assp=1.0/(1.0+np.exp((-(v-24.34))/14.82))
    dap=(assp-ap)/ta
    dti_develop=1.354+1.0e-4/(np.exp((v-167.4)/15.89)+np.exp(-(v-12.23)/0.2154))
    dti_recover=1.0-0.5/(1.0+np.exp((v+70.0)/20.0))
    develop_recover_product = dti_develop*dti_recover
    tiFp=develop_recover_product*tiF
    tiSp=develop_recover_product*tiS
    diFp=(iss-iFp)/tiFp
    diSp=(iss-iSp)/tiSp
    ip=AiF*iFp+AiS*iSp

    Ito=Gto*V_EK*((1.0-fItop)*a*i+fItop*ap*ip)

    dss=np.where(v >31.4978,1.00,1.0763*np.exp(-1.0070*np.exp(-0.0829*(v))))  #% magyar
    td= 0.6+1.0/(np.exp(-0.05*(v+6.0))+np.exp(0.09*(v+14.0)))

    dd=(dss-d)/td
    fss=1.0/(1.0+np.exp((v+19.58)/3.696))
    tff=7.0+1.0/(0.0045*np.exp(-(v+20.0)/10.0)+0.0045*np.exp((v+20.0)/10.0))
    tfs=1000.0+1.0/(0.000035*np.exp(-(v+5.0)/4.0)+0.000035*np.exp((v+5.0)/6.0))
    
    dff=(fss-ff)/tff
    dfs=(fss-fs)/tfs
    f=Aff*ff+Afs*fs
    fcass=fss
    tf_const = (v-4.0)/7.0
    tfcaf=7.0+1.0/(0.04*np.exp(-tf_const)+0.04*np.exp(tf_const))
    tfcas=100.0+1.0/(0.00012*np.exp(-v/3.0)+0.00012*np.exp(v/7.0))

    Afcaf=0.3+0.6/(1.0+np.exp((v-10.0)/10.0))

    Afcas=1.0-Afcaf
    dfcaf=(fcass-fcaf)/tfcaf
    dfcas=(fcass-fcas)/tfcas
    fca=Afcaf*fcaf+Afcas*fcas

    jcass = 1.0/(1.0+np.exp((v+18.08)/(2.7916)))   
    djca=(jcass-jca)/72.5# reduced from 75
    tffp=2.5*tff
    dffp=(fss-ffp)/tffp
    fp=Aff*ffp+Afs*fs
    tfcafp=2.5*tfcaf
    dfcafp=(fcass-fcafp)/tfcafp
    fcap=Afcaf*fcafp+Afcas*fcas

    #%#% SS nca

    km2n=jca#*1
    anca=1.0/(k2n/km2n+(1.0+Kmn/cass)**4.0)
    dnca=anca*k2n-nca*km2n

    #%#% myoplasmic nca
    anca_i = 1.0/(k2n/km2n+(1.0+Kmn/cai)**4.0)
    dnca_i = anca_i*k2n-nca_i*km2n

    #%#% SS driving force
    Io = 0.5*(nao + ko + clo + 4*cao)/1000  #% ionic strength outside. /1000 is for things being in micromolar
    Ii = 0.5*(nass + kss + clss + 4*cass)/1000  #% ionic strength outside. /1000 is for things being in micromolar
    #% The ionic strength is too high for basic DebHuc. We'll use Davies
    
    Ii_const = (np.sqrt(Ii)/(1+np.sqrt(Ii))-0.3*Ii)
    Io_const = (np.sqrt(Io)/(1+np.sqrt(Io))-0.3*Io)
    gamma_cai = np.exp(-constA * 4 * Ii_const)
    gamma_cao = np.exp(-constA * 4 * Io_const)
    gamma_nai = np.exp(-constA * 1 * Ii_const)
    gamma_nao = np.exp(-constA * 1 * Io_const)
    gamma_ki = gamma_nai#np.exp(-constA * 1 * (np.sqrt(Ii)/(1+np.sqrt(Ii))-0.3*Ii))
    gamma_kao = gamma_nao#np.exp(-constA * 1 * (np.sqrt(Io)/(1+np.sqrt(Io))-0.3*Io))

    Na_K_exp_const = np.exp(1.0*vfrt)
    Ca_exp_const = np.exp(2.0*vfrt)
    PhiCaL_ss =  4.0*vffrt*(gamma_cai*cass*Ca_exp_const-gamma_cao*cao)/(Ca_exp_const-1.0)
    PhiCaNa_ss =  1.0*vffrt*(gamma_nai*nass*Na_K_exp_const-gamma_nao*nao)/(Na_K_exp_const-1.0)
    PhiCaK_ss =  1.0*vffrt*(gamma_ki*kss*Na_K_exp_const-gamma_kao*ko)/(Na_K_exp_const-1.0)

    #%#% Myo driving force
    #Io = 0.5*(nao + ko + clo + 4*cao)/1000  #% ionic strength outside. /1000 is for things being in micromolar
    Ii = 0.5*(nai + ki + cli + 4*cai)/1000  #% ionic strength outside. /1000 is for things being in micromolar
    #% The ionic strength is too high for basic DebHuc. We'll use Davies
    Ii_const = (np.sqrt(Ii)/(1+np.sqrt(Ii))-0.3*Ii)
    gamma_cai = np.exp(-constA * 4 * Ii_const)
    #gamma_cao = np.exp(-constA * 4 * (np.sqrt(Io)/(1+np.sqrt(Io))-0.3*Io))
    gamma_nai = np.exp(-constA * 1 * Ii_const)
    #gamma_nao = np.exp(-constA * 1 * (np.sqrt(Io)/(1+np.sqrt(Io))-0.3*Io))
    gamma_ki = gamma_nai#np.exp(-constA * 1 * (np.sqrt(Ii)/(1+np.sqrt(Ii))-0.3*Ii))
    #gamma_kao = gamma_nao#np.exp(-constA * 1 * (np.sqrt(Io)/(1+np.sqrt(Io))-0.3*Io))

    gammaCaoMyo = gamma_cao
    gammaCaiMyo = gamma_cai

    PhiCaL_i =  4.0*vffrt*(gamma_cai*cai*Ca_exp_const-gamma_cao*cao)/(Ca_exp_const-1.0)
    PhiCaNa_i =  1.0*vffrt*(gamma_nai*nai*Na_K_exp_const-gamma_nao*nao)/(Na_K_exp_const-1.0)
    PhiCaK_i =  1.0*vffrt*(gamma_ki*ki*Na_K_exp_const-gamma_kao*ko)/(Na_K_exp_const-1.0)


    ## #PHOSPHORILATED##############################################################################################
    #      dPss=1.0763*np.exp(-1.0070*np.exp(-0.0829*(v+9.5)))  # #BetaAdrenergic
    dPss=1.0323*np.exp(-1.0553*np.exp(-0.0810*(v+9.5)))     #BetaAdrenergic ,fitted curve, why is the 9.5 added?
    dPss=np.minimum(dPss,1.0)
    dd_P=(dPss-d_P)/td
    #   fss_P=1.0/(1.0+np.exp((v+19.58+8.0)/3.696)) #BetaAdrenergic 
    fss_P=1.0/(1.0+np.exp((v+19.58+4.0)/3.696)) #BetaAdrenergic 
    dff_P=(fss_P-ff_P)/tff
    dfs_P=(fss_P-fs_P)/tfs
    fcass_P=fss_P
    dfcaf_P=(fcass_P-fcaf_P)/tfcaf
    dfcas_P=(fcass_P-fcas_P)/tfcas
    f_P=Aff*ff_P+Afs*fs_P
    fcap_P=Afcaf*fcaf_P+Afcas*fcas_P

    ## PHOSPHORILATION ##########################################
    # Both-P population takes on dP, for f gate, takes ss from PKA and tau from CaMK (only fast component was modified)
    fBPss = fss_P
    dfBPf = (fBPss-fBPf)/tffp
    fBP = Aff*fBPf+Afs*fs_P # only fast component modified, slow component same as fPs
    fcaBPss = fcass_P
    dfcaBPf = (fcaBPss-fcaBPf)/tfcafp  
    fcaBP = Afcaf*fcaBPf+Afcas*fcas_P
    #############
    #fICaLp=fICaLp # CaMK-P fraction
    fICaL_P = fICaLP # PKA-P fraction
    fICaL_BP = fICaLp*fICaL_P
    fICaL_CaMKonly = fICaLp-fICaL_BP
    fICaL_PKAonly = fICaL_P-fICaL_BP
    
    ############################################
    const1 = (1.0-nca)
    const2 = (1.0-nca_i)
    BP_const = d_P*(fBP*const2+jca*fcaBP*nca_i)
    BP_ss_const = d_P*(fBP*const1+jca*fcaBP*nca)
    NP_ss_const = d*(f*const1+jca*fca*nca)
    NP_i_const = d*(f*const2+jca*fca*nca_i)
    CaMK_ss_const = d*(fp*const1+jca*fcap*nca)
    CaMK_i_const = d*(fp*const2+jca*fcap*nca_i)
    PKA_ss_const = d_P*(f_P*const1+jca*fcap_P*nca)
    PKA_i_const = d_P*(f_P*const2+jca*fcap_P*nca_i)
    
    ICaL_ss_NP=PCa*PhiCaL_ss*NP_ss_const
    ICaL_ss_CaMK=PCap*PhiCaL_ss*CaMK_ss_const
    ICaL_ss_PKA=PCa_P*PhiCaL_ss*PKA_ss_const
    ICaL_ss_BP=PCa_P*PhiCaL_ss*BP_ss_const
    ICaL_i_NP=PCa*PhiCaL_i*NP_i_const
    ICaL_i_CaMK=PCap*PhiCaL_i*CaMK_i_const
    ICaL_i_PKA=PCa_P*PhiCaL_i*PKA_i_const
    ICaL_i_BP=PCa_P*PhiCaL_i*BP_const

    
    ICaNa_ss_NP=PCaNa*PhiCaNa_ss*NP_ss_const
    ICaNa_ss_CaMK=PCaNap*PhiCaNa_ss*CaMK_ss_const
    ICaNa_ss_PKA=PCaNa_P*PhiCaNa_ss*PKA_ss_const
    ICaNa_ss_BP=PCaNa_P*PhiCaNa_ss*BP_ss_const
    ICaNa_i_NP=PCaNa*PhiCaNa_i*NP_i_const
    ICaNa_i_CaMK=PCaNap*PhiCaNa_i*CaMK_i_const
    ICaNa_i_PKA=PCaNa_P*PhiCaNa_i*PKA_i_const
    ICaNa_i_BP=PCaNa_P*PhiCaNa_i*BP_const


    ICaK_ss_NP=PCaK*PhiCaK_ss*NP_ss_const
    ICaK_ss_CaMK=PCaKp*PhiCaK_ss*CaMK_ss_const
    ICaK_ss_PKA=PCaK_P*PhiCaK_ss*PKA_ss_const
    ICaK_ss_BP=PCaK_P*PhiCaK_ss*BP_ss_const
    ICaK_i_NP=PCaK*PhiCaK_i*NP_i_const
    ICaK_i_CaMK=PCaKp*PhiCaK_i*CaMK_i_const
    ICaK_i_PKA=PCaK_P*PhiCaK_i*d_P*PKA_i_const
    ICaK_i_BP=PCaK_P*PhiCaK_i*BP_const
    
    # 4 population combination######################################################
    Ca_i_ss_consts = (1-fICaL_CaMKonly-fICaL_PKAonly-fICaL_BP)
    ICaL_ss = Ca_i_ss_consts*ICaL_ss_NP + fICaL_CaMKonly*ICaL_ss_CaMK + fICaL_PKAonly*ICaL_ss_PKA + fICaL_BP*ICaL_ss_BP
    ICaNa_ss = Ca_i_ss_consts*ICaNa_ss_NP + fICaL_CaMKonly*ICaNa_ss_CaMK + fICaL_PKAonly*ICaNa_ss_PKA + fICaL_BP*ICaNa_ss_BP
    ICaK_ss = Ca_i_ss_consts*ICaK_ss_NP + fICaL_CaMKonly*ICaK_ss_CaMK + fICaL_PKAonly*ICaK_ss_PKA + fICaL_BP*ICaK_ss_BP
    # 
    ICaL_i = Ca_i_ss_consts*ICaL_i_NP + fICaL_CaMKonly*ICaL_i_CaMK + fICaL_PKAonly*ICaL_i_PKA + fICaL_BP*ICaL_i_BP
    ICaNa_i = Ca_i_ss_consts*ICaNa_i_NP + fICaL_CaMKonly*ICaNa_i_CaMK + fICaL_PKAonly*ICaNa_i_PKA + fICaL_BP*ICaNa_i_BP
    ICaK_i = Ca_i_ss_consts*ICaK_i_NP + fICaL_CaMKonly*ICaK_i_CaMK + fICaL_PKAonly*ICaK_i_PKA + fICaL_BP*ICaK_i_BP

    #% And we weight ICaL (in ss) and ICaL_i
    ICa_i_fraction = (1-ICaL_fractionSS)
    ICaL_i = ICaL_i * ICa_i_fraction
    ICaNa_i = ICaNa_i * ICa_i_fraction
    ICaK_i = ICaK_i * ICa_i_fraction
    ICaL_ss = ICaL_ss * ICaL_fractionSS
    ICaNa_ss = ICaNa_ss * ICaL_fractionSS
    ICaK_ss = ICaK_ss * ICaL_fractionSS

    ICaL = ICaL_ss + ICaL_i
    ICaNa = ICaNa_ss + ICaNa_i
    ICaK = ICaK_ss + ICaK_i
    #ICaL_tot = ICaL + ICaNa + ICaK

    ##%#% IKr
    #% transition rates
    #% from c0 to c1 in l-v model,
    alpha = 0.1161 * np.exp(0.2990 * vfrt)
    #% from c1 to c0 in l-v/
    beta =  0.2442 * np.exp(-1.604 * vfrt)

    #% from c2 to o/           c1 to o
    alpha2 =0.0578 * np.exp(0.9710 * vfrt) #%
    #% from o to c2/
    beta2 = 0.349e-3* np.exp(-1.062 * vfrt) #%

    #% from o to i
    alphai = 0.2533 * np.exp(0.5953 * vfrt) #%
    #% from i to o
    betai = 1.25* 0.0522 * np.exp(-0.8209 * vfrt) #%

    #% from c2 to i (from c1 in orig)
    alphac2ToI = 0.52e-4 * np.exp(1.525 * vfrt) #%
    #% from i to c2
    #% betaItoC2 = 0.85e-8 * np.exp(-1.842 * vfrt) #%
    betaItoC2 = (beta2 * betai * alphac2ToI)/(alpha2 * alphai) #%
    #% transitions themselves
    #% for reason of backward compatibility of naming of an older version of a
    #% MM IKr, c3 in code is c0 in article diagram, c2 is c1, c1 is c2.

    dc0 = ikr_c1 * beta - ikr_c0 * alpha #% delta for c0
    dc1 = ikr_c0 * alpha + ikr_c2*beta1 - ikr_c1*(beta+alpha1) #% c1
    dc2 = ikr_c1 * alpha1 + ikr_o*beta2 + ikr_i*betaItoC2 - ikr_c2 * (beta1 + alpha2 + alphac2ToI) #% subtraction is into c2, to o, to i. #% c2
    do = ikr_c2 * alpha2 + ikr_i*betai - ikr_o*(beta2+alphai)
    di = ikr_c2*alphac2ToI + ikr_o*alphai - ikr_i*(betaItoC2 + betai)
    

    IKr = GKr * np.sqrt(ko/5)* ikr_o  * V_EK

    ##%#% IKs
    V_EKs = (v-EKs)
    IKs_const = (2.326e-4*np.exp((v+48.28)/17.80)+0.001292*np.exp((-(v+210.0))/230.0))
    xs1ss=1.0/(1.0+np.exp((-(v+11.60))/8.932))
    txs1=817.3+1.0/IKs_const
    dxs1=(xs1ss-xs1)/txs1
    xs2ss=xs1ss
    txs2=1.0/(0.01*np.exp((v-50.0)/20.0)+0.0193*np.exp((-(v+66.54))/31.0))
    dxs2=(xs2ss-xs2)/txs2
    KsCa=1.0+0.6/(1.0+(3.8e-5/cai)**1.4)
    
    IKs_NP=GKs*KsCa*xs1*xs2*V_EKs
    #IKs=GKs*KsCa*xs1*xs2*(v-EKs)
    
    ##### BARS for IKs
    txs1_P=txs1_P_const+ 2.75/IKs_const#BetaAdrenergic 
    # txs1_P=0.6*txs1
    dxs1_P=(xs1ss-xs1_P)/txs1_P
    GKs_P=GKs*10
    IKs_P= GKs_P*KsCa*xs1_P*xs2*V_EKs
    IKs =(1 - fIKsP) * IKs_NP + fIKsP * IKs_P

    ##%#% IK1
    aK1 = 4.094/(1+np.exp(0.1217*(V_EK-49.934)))
    bK1 = (15.72*np.exp(0.0674*(V_EK-3.257))+np.exp(0.0618*(V_EK-594.31)))/(1+np.exp(-0.1629*(V_EK+14.207)))
    K1ss = aK1/(aK1+bK1)
    IK1=GK1*np.sqrt(ko/5)*K1ss*V_EK

    ##%#% INaCa
    hca=np.exp(qca*vfrt)
    hna=np.exp(qna*vfrt)
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
    E1=x1/x_sum#(x1+x2+x3+x4)
    E2=x2/x_sum#(x1+x2+x3+x4)
    E3=x3/x_sum#(x1+x2+x3+x4)
    E4=x4/x_sum#(x1+x2+x3+x4)
    
    allo=1.0/(1.0+(KmCaAct/cai)**2.0)
    
    JncxNa=3.0*(E4*k7-E1*k8)+E3*k4pp-E2*k3pp
    JncxCa=E2*k2-E1*k1
    INaCa_i=(1-INaCa_fractionSS)*Gncx*allo*(zna*JncxNa+zca*JncxCa)

    #%calculate INaCa_ss
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
    x_sum = x1+x2+x3+x4
    E1=x1/x_sum
    E2=x2/x_sum
    E3=x3/x_sum
    E4=x4/x_sum

    allo=1.0/(1.0+(KmCaAct/cass)**2.0)
    JncxNa=3.0*(E4*k7-E1*k8)+E3*k4pp-E2*k3pp
    JncxCa=E2*k2-E1*k1
    INaCa_ss=INaCa_fractionSS*Gncx*allo*(zna*JncxNa+zca*JncxCa)

    ##%#% INaK
    #fINaKP = 0 when there is no BARS
    # fINaK_PKA = fINaKP
    # if fINaK_PKA != 0:
    #     Knai0_P = Knai0*0.7#BetaAdrenergic
    #     Knai0=(1-fINaK_PKA)*Knai0 + fINaK_PKA*Knai0_P#BetaAdrenergic
    Knai=Knai0*np.exp((delta*vfrt)/(3.0))
    Knao=Knao0*np.exp(((1.0-delta)*vfrt)/(3.0))
    ATP_const1 = MgATP/Kmgatp
    ATP_const2 = 1+ATP_const1
    nai_div_Knai = nai/Knai
    ki_div_kki = ki/Kki
    nao_div_knao = nao/Knao
    
    P=eP/(1.0+H/Khp+nai/Knap+ki/Kxkur)
    a1_b4_conc = ((1.0+nai_div_Knai)**3.0+(1.0+ki_div_kki)**2.0-1.0)
    a1=(k1p*(nai_div_Knai)**3.0)/a1_b4_conc
    b1=k1m*MgADP
    a2=k2p
    b2_a3_const = ((1.0+nao_div_knao)**3.0+(1.0+ko/Kko)**2.0-1.0)
    b2=(k2m*(nao_div_knao)**3.0)/b2_a3_const
    a3=(k3p*(ko/Kko)**2.0)/b2_a3_const
    b3=(k3m*P*H)/ATP_const2
    a4=(k4p*ATP_const1)/ATP_const2
    b4=(k4m*(ki_div_kki)**2.0)/a1_b4_conc
    x1=a4*a1*a2+b2*b4*b3+a2*b4*b3+b3*a1*a2
    x2=b2*b1*b4+a1*a2*a3+a3*b1*b4+a2*a3*b4
    x3=a2*a3*a4+b3*b2*b1+b2*b1*a4+a3*a4*b1
    x4=b4*b3*b2+a3*a4*a1+b2*a4*a1+b3*b2*a1
    x_sum = x1+x2+x3+x4
    E1=x1/x_sum
    E2=x2/x_sum
    E3=x3/x_sum
    E4=x4/x_sum
    JnakNa=3.0*(E1*a3-E2*b3)
    JnakK=2.0*(E4*b1-E3*a1)

    INaK=Pnak*(zna*JnakNa+zk*JnakK)

    ##%#% Minor/background currents
    ##%calculate IKb with BARS effect
    xkb=1.0/(1.0+np.exp(-(v-10.8968)/(23.9871)))
    xkb_V_EK = xkb*V_EK
    GKbP=GKb*1.2
    IKb_P=GKbP*xkb_V_EK#xkb*(V_EK)
    IKb_NP=GKb*xkb_V_EK#xkb*(V_EK)
    IKb =(1 - fIKurP) * IKb_NP + fIKurP* IKb_P
    
    # Calculate other ackground currents
    INab=PNab*vffrt*(nai*Na_K_exp_const-nao)/(Na_K_exp_const-1.0)
    ICab=PCab*4.0*vffrt*(gammaCaiMyo*cai*Ca_exp_const-gammaCaoMyo*cao)/(Ca_exp_const-1.0)
    IpCa=GpCa*cai/(0.0005+cai)

    #%#% Chloride

    Fjunc = 1 
    Fsl = 1-Fjunc #% fraction in SS and in myoplasm - as per literature, I(Ca)Cl is in junctional subspace
    KdClCa = 0.1    #% [mM]

    I_ClCa_junc = Fjunc*GClCa/(1+KdClCa/cass)*(v-eclss)
    I_ClCa_sl = Fsl*GClCa/(1+KdClCa/cai)*(v-ecl)

    I_ClCa = I_ClCa_junc+I_ClCa_sl
    I_Clbk = GClB*(v-ecl)

    #%#% Calcium handling
    #%calculate ryanodione receptor calcium induced calcium release from the jsr

    #%#% Jrel
    
    J_rel_const = (-ICaL_ss)/(1.0+(jsrMidpoint/cajsr)**8.0)
    tau_rel_const = (1.0+0.0123/cajsr)
    
    Jrel_inf=Jrel_inf_scale * a_rel*J_rel_const
    tau_rel=np.maximum(bt/tau_rel_const,0.001)#(1.0+0.0123/cajsr)
    dJrelnp=(Jrel_inf-Jrel_np)/tau_rel
    
    Jrel_infp=Jrel_inf_scale *a_relp*J_rel_const #(Jrel_inf_scale) did not seem to be present in TOR model, need to check
    tau_relp=np.maximum(btp/tau_rel_const,0.001)#(1.0+0.0123/cajsr)
    dJrelp=(Jrel_infp-Jrel_p)/tau_relp
    
    #####################################
    # PKA-P channel behavior
    #####################################
    
    Jrel_inf_P=Jrel_inf_scale*a_relP*J_rel_const
    tau_rel_P=np.maximum((0.75)*bt/tau_rel_const,0.001)#BetaAdrenergic
    dJrelnp_P=(Jrel_inf_P-Jrel_np_P)/tau_rel_P

    
    Jrel_infBP=Jrel_inf_scale*a_relBP*J_rel_const
    tau_relBP=np.maximum((0.75)*btp/tau_rel_const,0.001)#BetaAdrenergic
    dJrelp_P=(Jrel_infBP-Jrel_p_P)/tau_relBP

    fJrel_PKA = fRyRP
    fJrel_BP = fJrelp*fJrel_PKA
    fJrel_CaMKonly = fJrelp - fJrel_BP
    fJrel_PKAonly = fJrel_PKA - fJrel_BP
    Jrel= 1.5378* ((1.0-fJrel_CaMKonly-fJrel_PKAonly-fJrel_BP)*Jrel_np + fJrel_CaMKonly*Jrel_p + fJrel_PKAonly*Jrel_np_P + fJrel_BP*Jrel_p_P )
    
    #Jrel=1.5378 * ((1.0-fJrelp)*Jrel_np+fJrelp*Jrel_p)
    cai_const1 = 0.005425*cai
    fJupp=fJrelp#(1.0/(1.0+KmCaMK/CaMKa))
    Jup_const = cai+0.00092
    Jupnp=Jup_scale*cai_const1/Jup_const
    Jupp= Jup_scale*2.75*cai_const1/(Jup_const-0.00017)
    
    ######################################
    # PKA-P SERCA
    #calculate serca pump, ca uptake flux
    Jup_P= Jup_scale*cai_const1/(Jup_const*0.7)#BetaAdrenergic
    Jup_BP= Jup_scale*2.75*cai_const1/(cai+(0.00092-0.00017)*0.7)#BetaAdrenergic
 
    # 4 population
    fJup_P = fPLBP # PKA-P channel fraction
    fJup_BP = fJupp*fJup_P
    fJup_CaMKonly = fJupp - fJup_BP
    fJup_PKAonly = fJup_P - fJup_BP

    Jleak=0.0048825*cansr/15.0
    Jup= ((1.0-fJup_CaMKonly-fJup_PKAonly-fJup_BP)*Jupnp + fJup_CaMKonly*Jupp + fJup_PKAonly*Jup_P + fJup_BP*Jup_BP) - Jleak
    #Jup=(1.0-fJupp)*Jupnp+fJupp*Jupp-Jleak

    #%calculate tranlocation flux
    Jtr=(cansr-cajsr)/60

    #%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%

    #%#% calculate the stimulus current, Istim
    I_stim = Istim(time, cycle_length,amp,duration)


    #%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%

    #%update the membrane voltage
    dv=-(INa+INaL+Ito+ICaL+ICaNa+ICaK+IKr+IKs+IK1+INaCa_i+INaCa_ss+INaK+INab+IKb+IpCa+ICab+I_ClCa+I_Clbk+I_stim)
    #%calculate diffusion fluxes
    JdiffNa=(nass-nai)/2.0
    JdiffK=(kss-ki)/2.0
    Jdiff=(cass-cai)/0.2
    JdiffCl=(clss-cli)/2.0

    #%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%

    #%calcium buffer constants 
    #calcium buffer constants when FTnIP is 0, the value of kmtrpn is same as TOR DynCL 
    kmtrpn = (1 -  fTnIP) * kmtrpn +  fTnIP * 1.6 * kmtrpn #BetaAdrenergic
    
    #%update intracellular concentrations, using buffers for cai, cass, cajsr
    dnai=-(ICaNa_i+INa+INaL+3.0*INaCa_i+3.0*INaK+INab)*Acap/(F*vmyo)+JdiffNa*vss/vmyo
    dnass=-(ICaNa_ss+3.0*INaCa_ss)*Acap/(F*vss)-JdiffNa

    dki=-(ICaK_i+Ito+IKr+IKs+IK1+IKb+I_stim-2.0*INaK)*Acap/(F*vmyo)+JdiffK*vss/vmyo
    dkss=-(ICaK_ss)*Acap/(F*vss)-JdiffK

    Bcai=1.0/(1.0+cmdnmax*kmcmdn/(kmcmdn+cai)**2.0+trpnmax*kmtrpn/(kmtrpn+cai)**2.0)
    dcai=Bcai*(-(ICaL_i + IpCa+ICab-2.0*INaCa_i)*Acap/(2.0*F*vmyo)-Jup*vnsr/vmyo+Jdiff*vss/vmyo)

    Bcass=1.0/(1.0+BSRmax*KmBSR/(KmBSR+cass)**2.0+BSLmax*KmBSL/(KmBSL+cass)**2.0)
    dcass=Bcass*(-(ICaL_ss-2.0*INaCa_ss)*Acap/(2.0*F*vss)+Jrel*vjsr/vss-Jdiff)

    #z_cl is -1
    dcli = - (I_Clbk + I_ClCa_sl)*Acap/(-1*F*vmyo)+JdiffCl*vss/vmyo
    dclss = - I_ClCa_junc*Acap/(-1*F*vss)-JdiffCl

    dcansr=Jup-Jtr*vjsr/vnsr

    Bcajsr=1.0/(1.0+csqnmax*kmcsqn/(kmcsqn+cajsr)**2.0)
    dcajsr=Bcajsr*(Jtr-Jrel)

    #%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%

    #%output the state vector when ode_flag==1, and the calculated currents and fluxes otherwise

    output=[dv, dnai, dnass, dki, dkss ,dcai ,dcass, dcansr ,dcajsr, dm ,dhp, dh ,dj, djp, dmL ,dhL, dhLp, da, diF, diS, dap, diFp, diSp,dd, dff, dfs, dfcaf, dfcas, djca, dnca ,dnca_i, dffp, dfcafp ,dxs1 ,dxs2 ,dJrelnp, dCaMKt,dc0, dc1, dc2, do, di, dJrelp, dcli,dclss,dxs1_P, dd_P, dff_P, dfs_P, dfcaf_P, dfcas_P, dfBPf, dfcaBPf, dh_P, dj_P, dhp_P, djp_P, dJrelnp_P, dJrelp_P,
            dcond1,dcond2,dcond3,dcond4,dcond5,dcond6,dcond7,dcond8,dcond9,dcond10,dcond11,dcond12,dcond13,dcond14,dcond15,dcond16,dcond17,dcond18,dcond19,dcond20,dcond21,dcond22,dcond23,dcond24,dcond25,dcond26,dcond27,dcond28,dcond29,dcond30,dcond31,dcond32,dcond33,dcond34,dcond35,dcond36,dcond37,dcond38,dcond39,dcond40,dcond41,dcond42,dcond43,dcond44,dcond45,dcond46,dcond47,dcond48,dcond49,dcond50,dcond51,dcond52,dcond53,dcond54,dcond55,dcond56,dcond57]
    return output

# start_time = tm.time()
# df, currents_df = run_TOR_Model(100,500,1,G_Ks=1.4 * 0.0011 ,G_Kr = 1.3 * 0.0321,
#                                 G_K1=1.2*0.6992,G_to=2.0*0.16,G_Na_late=0.6*0.0279,G_Na_fast=11.7802,cell_type="EPI")
# end_time = tm.time()
# elapsed_time = end_time - start_time
# print(elapsed_time)
# df_tail = df.tail(1200)
# plt.plot(df_tail['time'], df_tail['V'])
# plt.xlabel('Time')
# plt.ylabel('V')
# plt.title('V vs Time')
# plt.grid(True)
# plt.show()

