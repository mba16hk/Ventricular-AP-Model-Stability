import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import ode
from .ORd import Istim
import time as tm
import pandas as pd
from numba import njit
from conductances import *

def GetStartingState_DCl(cell_type):
    if cell_type == 'ENDO':
        X0 = [-89.7480828168753, 12.3973564703237, 12.3976953330256, 147.711474407329, 147.711433360862, 7.45348093580908e-05, 6.49734059262396e-05, 1.52800059077496, 1.52569253436820, 0.000651715361520436, 0.701845427662345, 0.847326693399099, 0.847165668535608, 0.846901438940383, 0.000135120261194560, 0.556601650459093, 0.311549070692613, 0.000889925901945137, 0.999671576537361, 0.598890849399207, 0.000453416519958675, 0.999671583620741, 0.662069197871101, 1.58884121071311e-31, 0.999999994310640, 0.940179072110115, 0.999999994310551, 0.999901376748386, 0.999984619576572, 0.000489937844976504, 0.000832600925714172, 0.999999994343118, 0.999999994312043, 0.243959016060676, 0.000158616694622612, 1.80824774078763e-22, 0.0109502564192484, 0.998251134783098, 0.000793602046257213, 0.000653214338491609, 0.000292244895058368, 9.80408293117642e-06, 4.35860784390795e-21, 29.2069793887423, 29.2069557375901]

    if cell_type == 'EPI':
        X0 = [-90.7456332496277, 13.4006197127101, 13.4009366375605, 152.363889991411, 152.363844281265, 6.62181569329109e-05, 5.74992101186664e-05, 1.80679354108253, 1.80504703846161, 0.000525323142244124, 0.731365613931879, 0.864514804478946, 0.864457115015362, 0.864352715733040, 0.000111796946200753, 0.591653600903076, 0.347681236790268, 0.000832040787904695, 0.999724237643709, 0.999723490323258, 0.000423912054921203, 0.999724237767365, 0.999724054768924, -2.48652723154831e-36, 0.999999995656968, 0.951060213798798, 0.999999995656976, 0.999937692440505, 0.999988620215189, 0.000304952292598877, 0.000527266836293694, 0.999999995656685, 0.999999995656399, 0.223358448594285, 0.000141824743798310, 6.77882673136480e-25, 0.0127354136594547, 0.998473340797544, 0.000739304520671507, 0.000602907876760312, 0.000178778329437106, 5.67825481839631e-06, -1.58194108230975e-23, 34.3172099045416, 34.3171879616709]

    if cell_type == 'M':
        X0 = [-91.3391803967798, 15.9486733631902, 15.9492201219199, 156.713067183465, 156.713012917889, 8.29757614119249e-05, 6.64218738239930e-05, 2.01222465016069, 2.01641461550468, 0.000461956534829239, 0.747897241513971, 0.873907657623256, 0.873784149746068, 0.873537507470071, 9.98770874860868e-05, 0.598611759466370, 0.333989914508018, 0.000799404188333773, 0.999751418648699, 0.570253781914665, 0.000407277681198651, 0.999751424044492, 0.635192659820007, -8.33460419955820e-30, 0.999999996299523, 0.918358695116332, 0.999999996300839, 0.999754043234341, 0.999974289100453, 0.000533651965198686, 0.00125786136830066, 0.999999996207845, 0.999999996298728, 0.264229298875232, 0.000132734770983313, -1.30048587380675e-21, 0.0201882046405105, 0.998345129712001, 0.000708645966770289, 0.000591004733696651, 0.000344573276330018, 1.06423014130810e-05, -7.61071425848620e-20, 48.9127681450109, 48.9127355777872]

    return X0

@njit
def TOR_DCl_Model(time,X0,cycle_length, clo, nao, cao, ko, R, T, F, L, rad,
    vcell, Ageo, Acap, vmyo, vnsr, vjsr, vss,
    KmCaMK, aCaMK, bCaMK, CaMKo, KmCaM, frt, 
    cmdnmax, amp, duration, kmcmdn, trpnmax, kmtrpn,
    BSRmax, KmBSR, BSLmax, KmBSL, csqnmax, kmcsqn, zna, k1p, k1m, k2p, k2m, k3p, k3m, k4p, k4m,
    Knai0, Knao0, delta, Pnak, GKb, PNab, PCab, GpCa,
    bt, a_rel, btp, a_relp, Kki, Kko, MgADP, MgATP,
    Kmgatp, H, eP, Khp, Knap, Kxkur,reversal_potential_const, 
    ICaL_fractionSS, INaCa_fractionSS, PKNa, GNaL, Gto, Kmn, k2n, GKs, zca, kna1, kna2, kna3, kasymm, wna, wca, wnaca, kcaon, kcaoff, qna, qca,
    Aff, Afs, constA, GK1, Gncx, KmCaAct, zk, jsrMidpoint, GClCa, GClB, KdClCa, PCa, GNa,GKr,Jrel_inf_scale,Jup_scale, celltype):
    
    
    v,nai,nass,ki,kss,cai,cass,cansr,cajsr,m,hp,h,j,jp,mL,hL,hLp,a,iF,iS,ap,iFp,iSp,d,ff,fs,fcaf,fcas,jca,nca,nca_i,ffp,fcafp,xs1,xs2,Jrel_np,CaMKt,ikr_c0,ikr_c1,ikr_c2,ikr_o,ikr_i,Jrel_p,cli,clss = X0
    ##%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%

    vfrt=v*frt
    vffrt=vfrt*F
    
    ##%update CaMK
    CaMKb=CaMKo*(1.0-CaMKt)/(1.0+KmCaM/cass)
    CaMKa=CaMKb+CaMKt
    dCaMKt=aCaMK*CaMKb*(CaMKb+CaMKt)-bCaMK*CaMKt

    ##%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%

    ##%reversal potentials
    ENa=reversal_potential_const*np.log(nao/nai)
    EK=reversal_potential_const*np.log(ko/ki)
    EKs=reversal_potential_const*np.log((ko+PKNa*nao)/(ki+PKNa*nai))
    ecl = reversal_potential_const*np.log(cli/clo)            #% [mV]
    eclss = reversal_potential_const*np.log(clss/clo);           # % [mV]
    #%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%
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
    hss = 1 / ((1 + np.exp( (v + 71.55)/7.43 ))**2)
    dh = (hss - h) / tauh
    #%#% j gate
    aj = np.where(v >= -40, (0) ,(((-2.5428 * 10**4*np.exp(0.2444*v) - 6.948*10**-6 * np.exp(-0.04391*v)) * (v + 37.78)) / (1 + np.exp( 0.311 * (v + 79.23) ))))
    bj = np.where(v >= -40 , ((0.6 * np.exp( 0.057 * v)) / (1 + np.exp( -0.1 * (v + 32) ))) ,((0.02424 * np.exp( -0.01052 * v )) / (1 + np.exp( -0.1378 * (v + 40.14) ))))
    tauj = 1 / (aj + bj)
    jss = hss#1 / ((1 + np.exp( (v + 71.55)/7.43 ))**2)
    dj = (jss - j) / tauj

    #%#% h phosphorylated
    hssp = 1 / ((1 + np.exp( (v + 71.55 + 6)/7.43 ))**2)
    dhp = (hssp - hp) / tauh
    #%#% j phosphorylated
    taujp = 1.46 * tauj
    djp = (jss - jp) / taujp
    INa=GNa*(v-ENa)*m**3.0*((1.0-fINap)*h*j+fINap*hp*jp)


    mLss=1.0/(1.0+np.exp((-(v+42.85))/5.264))
    tm = 0.1292 * np.exp(-((v+45.79)/15.54)**2) + 0.06487 * np.exp(-((v-4.823)/51.12)**2) 
    tmL=tm
    dmL=(mLss-mL)/tmL
    hLss=1.0/(1.0+np.exp((v+87.61)/7.488))
    thL=200.0
    dhL=(hLss-hL)/thL
    hLssp=1.0/(1.0+np.exp((v+93.81)/7.488))
    thLp=3.0*thL
    dhLp=(hLssp-hLp)/thLp

    INaL=GNaL*(v-ENa)*mL*((1.0-fINaLp)*hL+fINaLp*hLp)


    #%#% ITo
    #%calculate Ito
    ass=1.0/(1.0+np.exp((-(v-14.34))/14.82))
    ta=1.0515/(1.0/(1.2089*(1.0+np.exp(-(v-18.4099)/29.3814)))+3.5/(1.0+np.exp((v+100.0)/29.3814)))
    da=(ass-a)/ta
    iss=1.0/(1.0+np.exp((v+43.94)/5.711))
    if celltype=='EPI':
        delta_epi=1.0-(0.95/(1.0+np.exp((v+70.0)/5.0)))
    else:
        delta_epi=1.0

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

    dss=1.0763*np.exp(-1.0070*np.exp(-0.0829*(v)))  #% magyar
    if (v >31.4978): #% activation cannot be greater than 1
        dss = 1

    td= 0.6+1.0/(np.exp(-0.05*(v+6.0))+np.exp(0.09*(v+14.0)))

    dd=(dss-d)/td
    fss=1.0/(1.0+np.exp((v+19.58)/3.696))
    tff=7.0+1.0/(0.0045*np.exp(-(v+20.0)/10.0)+0.0045*np.exp((v+20.0)/10.0))
    tfs=1000.0+1.0/(0.000035*np.exp(-(v+5.0)/4.0)+0.000035*np.exp((v+5.0)/6.0))
    
    dff=(fss-ff)/tff
    dfs=(fss-fs)/tfs
    f=Aff*ff+Afs*fs
    fcass=fss
    tfcaf=7.0+1.0/(0.04*np.exp(-(v-4.0)/7.0)+0.04*np.exp((v-4.0)/7.0))
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

    PCap=1.1*PCa
    PCaNa=0.00125*PCa
    PCaK=3.574e-4*PCa
    PCaNap=0.00125*PCap
    PCaKp=3.574e-4*PCap

    ICa_ss_const = (1.0-fICaLp)*d*(f*(1.0-nca)+jca*fca*nca)
    ICa_ss_const2 = fICaLp*d*(fp*(1.0-nca)+jca*fcap*nca)
    ICaL_ss=PCa*PhiCaL_ss*ICa_ss_const+PCap*PhiCaL_ss*ICa_ss_const2
    ICaNa_ss=PCaNa*PhiCaNa_ss*ICa_ss_const+PCaNap*PhiCaNa_ss*ICa_ss_const2
    ICaK_ss=PCaK*PhiCaK_ss*ICa_ss_const+PCaKp*PhiCaK_ss*ICa_ss_const2

    ICa_i_const = (1.0-fICaLp)*d*(f*(1.0-nca_i)+jca*fca*nca_i)
    ICa_i_const2 = fICaLp*d*(fp*(1.0-nca_i)+jca*fcap*nca_i)
    ICaL_i=PCa*PhiCaL_i*ICa_i_const+PCap*PhiCaL_i*ICa_i_const2
    ICaNa_i=PCaNa*PhiCaNa_i*ICa_i_const+PCaNap*PhiCaNa_i*ICa_i_const2
    ICaK_i=PCaK*PhiCaK_i*ICa_i_const+PCaKp*PhiCaK_i*ICa_i_const2


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
    alpha1 = 1.25 * 0.1235 
    #% from c2 to c1 in l-v/
    beta1 =  0.1911

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
    xs1ss=1.0/(1.0+np.exp((-(v+11.60))/8.932))
    txs1=817.3+1.0/(2.326e-4*np.exp((v+48.28)/17.80)+0.001292*np.exp((-(v+210.0))/230.0))
    dxs1=(xs1ss-xs1)/txs1
    xs2ss=xs1ss
    txs2=1.0/(0.01*np.exp((v-50.0)/20.0)+0.0193*np.exp((-(v+66.54))/31.0))
    dxs2=(xs2ss-xs2)/txs2
    KsCa=1.0+0.6/(1.0+(3.8e-5/cai)**1.4)
    
    IKs=GKs*KsCa*xs1*xs2*(v-EKs)

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
    ##%calculate IKb
    xkb=1.0/(1.0+np.exp(-(v-10.8968)/(23.9871)))
    IKb=GKb*xkb*(v-EK)
    INab=PNab*vffrt*(nai*Na_K_exp_const-nao)/(Na_K_exp_const-1.0)
    ICab=PCab*4.0*vffrt*(gammaCaiMyo*cai*Ca_exp_const-gammaCaoMyo*cao)/(Ca_exp_const-1.0)
    IpCa=GpCa*cai/(0.0005+cai)

    #%#% Chloride

    Fjunc = 1 
    Fsl = 1-Fjunc #% fraction in SS and in myoplasm - as per literature, I(Ca)Cl is in junctional subspace

    GClCa = 0.2843  #% [mS/uF]
    GClB =  1.98e-3       # % [mS/uF] %
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
    cai_const1 = 0.005425*cai
    Jrel_inf=Jrel_inf_scale * a_rel*J_rel_const
    tau_rel=np.maximum(bt/tau_rel_const,0.001)#(1.0+0.0123/cajsr)
    dJrelnp=(Jrel_inf-Jrel_np)/tau_rel
    Jrel_infp=a_relp*J_rel_const
    tau_relp=np.maximum(btp/tau_rel_const,0.001)#(1.0+0.0123/cajsr)
    dJrelp=(Jrel_infp-Jrel_p)/tau_relp
    Jrel=1.5378 * ((1.0-fJrelp)*Jrel_np+fJrelp*Jrel_p)

    fJupp=fJrelp#(1.0/(1.0+KmCaMK/CaMKa))
    Jupnp=Jup_scale*cai_const1/(cai+0.00092)
    Jupp= Jup_scale*2.75*cai_const1/(cai+0.00092-0.00017)
    

    Jleak=0.0048825*cansr/15.0
    Jup=(1.0-fJupp)*Jupnp+fJupp*Jupp-Jleak

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

    output=[dv, dnai, dnass, dki, dkss ,dcai ,dcass, dcansr ,dcajsr, dm ,dhp, dh ,dj, djp, dmL ,dhL, dhLp, da, diF, diS, dap, diFp, diSp,dd, dff, dfs, dfcaf, dfcas, djca, dnca ,dnca_i, dffp, dfcafp ,dxs1 ,dxs2 ,dJrelnp, dCaMKt,dc0, dc1, dc2, do, di, dJrelp, dcli,dclss]
    return output

def run_TOR_DCl_Model(cycles, cycle_length, cell_type, amp=-53):

    model_type   = "TOR-DynCL 2020"
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
    vcell=1000*3.14*rad*rad*L
    Ageo=2*3.14*rad*rad+2*3.14*rad*L
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
    amp=amp#-53
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
    Pnak= GNaK#15.4509 
    GKb=GKb_input#0.0189
    PNab=GNab#1.9239e-09
    PCab=GCab#5.9194e-08
    GpCa=GpCa_input#5e-04
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
    GNaL=G_Na_late#0.0279 
    Gto=G_to#0.16 
    Kmn=0.002
    k2n=500.0
    GKs= G_Ks#0.0011 
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
    
    Aff=0.6
    Afs=1.0-Aff
    dielConstant = 74 #% water at 37°.
    constA = 1.82*10**6*(dielConstant*T)**(-1.5)
    GK1= G_K1#0.6992 #%0.7266 #%* np.sqrt(5/5.4))
    Gncx= GNCX#0.0034
    KmCaAct=150.0e-6 
    zk=1.0
    jsrMidpoint = 1.7
    GClCa = GClCa_input #0.2843   #% [mS/uF]
    GClB = GClb#1.98e-3        #% [mS/uF] #%
    KdClCa = 0.1    #% [mM]
    #%#% The rest
    PCa=GCa#8.3757e-05 
    GNa = G_Na_fast#11.7802
    GKr = G_Kr#0.0321  #% 1st element compensates for change to ko (np.sqrt(5/5.4)* 0.0362)
    Jrel_inf_scale = 1
    Jup_scale = 1
    
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
        #Pnak = Pnak * 0.9
        #GKb = GKb * 0.6

    elif cell_type == 'M':
        #Gto = Gto * 2.0
        #PCa = PCa * 2
        #GKr = GKr * 0.8
        #GK1 = GK1 * 1.3
        #Gncx = Gncx * 1.4
        Jrel_inf_scale = Jrel_inf_scale * 1.7
        #Pnak = Pnak * 0.7

    constants = [
    cycle_length, clo, nao, cao, ko, R, T, F, L, rad,
    vcell, Ageo, Acap, vmyo, vnsr, vjsr, vss,
    KmCaMK, aCaMK, bCaMK, CaMKo, KmCaM, frt, 
    cmdnmax, amp, duration, kmcmdn, trpnmax, kmtrpn,
    BSRmax, KmBSR, BSLmax, KmBSL, csqnmax, kmcsqn, zna, k1p, k1m, k2p, k2m, k3p, k3m, k4p, k4m,
    Knai0, Knao0, delta, Pnak, GKb, PNab, PCab, GpCa,
    bt, a_rel, btp, a_relp, Kki, Kko, MgADP, MgATP,
    Kmgatp, H, eP, Khp, Knap, Kxkur,reversal_potential_const, 
    ICaL_fractionSS, INaCa_fractionSS, PKNa, GNaL, Gto, Kmn, k2n, GKs, zca, kna1, kna2, kna3, kasymm, wna, wca, wnaca, kcaon, kcaoff, qna, qca,
    Aff, Afs, constA, GK1, Gncx, KmCaAct, zk, jsrMidpoint, GClCa, GClB, KdClCa, PCa, GNa,GKr,Jrel_inf_scale,Jup_scale, cell_type
    ]

    initial_conds = GetStartingState_DCl(cell_type)#[m ,m_L ,h_fast ,h_slow ,h_CaMK_slow ,h_L ,h_L_CaMK ,j ,j_CaMK ,CaMK_trap ,V_m ,a,i_fast,i_slow ,i_CaMK_fast , i_CaMK_slow,a_CaMK,d,f_fast ,f_slow ,j_Ca,f_Ca_CaMK_fast,n, f_Ca_slow ,f_Ca_fast, f_CaMK_fast , x_r_slow ,x_r_fast ,x_s1,x_s2 ,x_K1 ,J_rel_NP,J_rel_CaMK ,Na_ion_conc_i , Na_ion_conc_ss, K_ion_conc_i , K_ion_conc_ss , Ca_ion_conc_i , Ca_ion_conc_ss , Ca_ion_conc_nsr , Ca_ion_conc_jsr]
    tspan = (0, cycles*cycle_length)
    sol = solve_ivp(fun = TOR_DCl_Model, t_span = tspan, y0 = initial_conds, args = constants,method='BDF',rtol= 1e-5,max_step = 1)

    time = sol.t  # Time points
    solutions = sol.y  # Solution vectors, each row corresponds to a variable
    y_names = ["V","Nai","Na_ss","Ki","K_ss","Cai","Ca_ss","Ca_nsr","Ca_jsr","m","hp","h","j","jp","m_L","h_L","h_L_CaMK","a","i_fast","i_slow","a_CaMK","i_CaMK_fast","i_CaMK_slow","d","f_fast","f_slow","f_Ca_fast","f_Ca_slow","j_Ca","n","nca_i","f_CaMK_fast","f_Ca_CaMK_fast","x_s1","x_s2","J_rel_NP","CaMK_trap","C0","C1","C2","O","I","J_rel_CaMK","Cl_i","Cl_ss"]
    
    df = pd.DataFrame(solutions.T, columns=y_names)
    
    df['time'] = time

    return df, pd.DataFrame(), duration


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

