# adapted from:  https://github.com/JQXGong/Ohara-beta-adrenergic
#
# Quantitative analysis of variability in an integrated model of 
# human ventricular electrophysiology and B-adrenergic signaling
# by Jingqi Q.X. Gong, Monica E. Susilo, Anna Sher, Cynthia J. Musante, Eric A. Sobie
# DOI:https://doi.org/10.1016/j.yjmcc.2020.04.009
#
import numpy as np
from numba import njit

@njit
def Model_SignalingMyokit2(cond1,cond2,cond3,cond4,cond5,cond6,cond7,cond8,cond9,cond10,cond11,cond12,cond13,cond14,cond15,
                           cond16,cond17,cond18,cond19,cond20,cond21,cond22,cond23,cond24,cond25,cond26,cond27,cond28,cond29,
                           cond30,cond31,cond32,cond33,cond34,cond35,cond36,cond37,cond38,cond39,cond40,cond41,cond42,cond43,
                           cond44,cond45,cond46,cond47,cond48,cond49,cond50,cond51,cond52,cond53,cond54,cond55,cond56,cond57, c):

    # y = np.array(y)
    # # Create derivatives vector
    # ydot = np.zeros(y.shape[0], y.shape[1])
    #y = np.array(y)
    #c = np.array(c)
    #print('hello')
    ydot = np.zeros(57+1)
    #print('ydot_initiated')
    ydot[0] = 0
    ##
    #print(cond1)
    #####################################################################################################################################################################################
    ## Start of the Signaling state variables
    beta_cav_Gs_aGTP = cond1#y[0]   #  1: Gs_aGTP_CAV
    beta_eca_Gs_aGTP = cond2#y[1]   #  2: Gs_aGTP_ECAV
    beta_cyt_Gs_aGTP = cond3#y[2]   #  3: Gs_a_GTP_CYT
    beta_cav_Gs_bg = cond4#y[3]     #  4: Gs_bg_CAV
    beta_eca_Gs_bg = cond5#y[4]     #  5: Gs_bg_ECAV
    beta_cyt_Gs_bg = cond6#y[5]     #  6: Gs_bg_CYT
    beta_cav_Gs_aGDP = cond7#y[6]   #  7: Gs_aGDP_CAV
    beta_eca_Gs_aGDP = cond8#y[7]   #  8: Gs_aGDP_ECAV
    beta_cyt_Gs_aGDP = cond9#y[8]   #  9: Gs_aGDP_CYT

    cAMP_cav = cond10#y[9]   # 10: cAMP_CAVVV
    cAMP_eca = cond11#y[10]   # 11: cAMP_ECAV
    cAMP_cyt = cond12#y[11]  # 12: cAMP_CYT

    beta_cav_Rb1_pka_tot = cond13#y[12]  # 13: R_pkap_tot_CAV
    beta_eca_Rb1_pka_tot = cond14#y[13]  # 14: R_pkap_tot_ECAV
    beta_cyt_Rb1_pka_tot = cond15#y[14]  # 15: R_pkap_tot_CYT
    beta_cav_Rb1_grk_tot = cond16#y[15]  # 16: R_grkp_tot_CAV
    beta_eca_Rb1_grk_tot = cond17#y[16]  # 17: R_grkp_tot_ECAV
    beta_cyt_Rb1_grk_tot = cond18#y[17]  # 18: R_grkp_tot_CYT

    pka_cav_ARC = cond19#y[18]   # 19: RLC_CAV
    pka_cav_A2RC = cond20#y[19]  # 20: L2RC_CAV
    pka_cav_A2R = cond21#y[20]   # 21: L2R_CAV
    pka_cav_C = cond22#y[21]     # 22: C_CAV
    pka_cav_PKIC = cond23#y[22]  # 23: PKI_CAV
    pka_eca_ARC = cond24#y[23]   # 24: RLC_ECAV
    pka_eca_A2RC = cond25#y[24]  # 25: L2RC_ECAV
    pka_eca_A2R = cond26#y[25]   # 26: L2R_ECAV
    pka_eca_C = cond27#y[26]     # 27: C_ECAV
    pka_eca_PKIC = cond28#y[27]  # 28: PKI_ECAV
    pka_cyt_ARC = cond29#y[28]   # 29: RLC_CYT
    pka_cyt_A2RC = cond30#y[29]  # 30: L2RC_CYT
    pka_cyt_A2R = cond31#y[30]   # 31: L2R_CYT
    pka_cyt_C = cond32#y[31]     # 32: C_CYT
    pka_cyt_PKIC = cond33#y[32]  # 33: PKI_CYT

    PDE3_P_cav = cond34#y[33]    #34   34: PDE3_P_CAV
    PDE3_P_cyt = cond35#y[34]    #35   35: PDE3_P_CYT
    PDE4_P_cav = cond36#y[35]    #36   36: PDE4_P_CAV
    PDE4_P_eca = cond37#y[36]    #37   37: PDE4_P_ECAV
    PDE4_P_cyt = cond38#y[37]    #38   38: PDE4_P_CYT

    inhib1_p = cond39#y[38]  #39       39: Inhib1_P_CYT

    ICaLp = cond40#y[39]     # 40: fLCC_P
    IKsp = cond41#y[40]      # 41: fIKS_P
    iup_f_plb = cond42#y[41] # 42: fPLB_P
    f_tni = cond43#y[42]     # 43: fTnI_P
    ina_f_ina = cond44#y[43] # 44: fINa_P
    f_inak = cond45#y[44]    # 45: fINaK_P
    RyRp = cond46#y[45]      # 46: fRyR_P
    f_ikur = cond47#y[46]    # 47: fIKur_P
    #################################################
    #### update here to make the original PKA fractions bound between [0,1]
    #### use 0.0001 and 0.9999
    #### for AKAP related 3 channels implement the constraint in EffectiveFraction function
    ### ICaLp IKsp RyRp

    ### iup_f_plb
    if iup_f_plb < 0.0:
        iup_f_plb=0.0001
    elif iup_f_plb > 1.0:
        iup_f_plb= 0.9999

    ### f_tni
    if f_tni  < 0.0:
        f_tni =0.0001
    elif f_tni  > 1.0:
        f_tni = 0.9999
    

    ### ina_f_ina
    if ina_f_ina  < 0.0:
        ina_f_ina =0.0001
    elif ina_f_ina > 1.0:
        ina_f_ina = 0.9999
    

    ### f_inak
    if f_inak  < 0.0:
        f_inak =0.0001
    elif f_inak > 1.0:
        f_inak = 0.9999
    

    ### f_ikur
    if f_ikur  < 0.0:
        f_ikur =0.0001
    elif f_ikur > 1.0:
        f_ikur= 0.9999
    
    #################################################

    beta_cav_Rb2_pka_tot = cond48#y[47]  # 48: Rb2_pkap_tot_CAV
    beta_cav_Rb2_grk_tot = cond49#y[48]  # 49: Rb2_grkp_tot_CAV
    beta_cav_Gi_aGTP = cond50#y[49]      # 50: Gi_aGTP_CAV
    beta_cav_Gi_bg = cond51#y[50]        # 51: Gi_bg_CAV
    beta_cav_Gi_aGDP = cond52#y[51]      # 52: Gi_aGDP_CAV
    beta_eca_Rb2_pka_tot = cond53 #y[52]  # 53: Rb2_pkap_tot_ECAV
    beta_eca_Rb2_grk_tot = cond54 #y[53]  # 54: Rb2_grkp_tot_ECAV
    beta_eca_Gi_aGTP = cond55 #y[54]      # 55: Gi_aGTP_ECAV
    beta_eca_Gi_bg =cond56 # y[55]        # 56: Gi_bg_ECAV
    beta_eca_Gi_aGDP = cond57 #y[56]      # 57: Gi_aGDP_ECAV

    ## 

    #print('All values assigned')
    ## CAVEOLAR ##############################

    # Total concentration of non-phosphorylated B1AR in the caveolar subspace
    beta_cav_Rb1_np_tot = c[87] - beta_cav_Rb1_pka_tot - beta_cav_Rb1_grk_tot  # working
    # Total concentration of non-phosphorylated B2AR in the caveolar subspace
    beta_cav_Rb2_np_tot = c[89] - beta_cav_Rb2_pka_tot - beta_cav_Rb2_grk_tot # working
    # Concentration of Gi holoenzyme in the caveolar subspace
    beta_cav_Gi_abg = c[63] * c[59] * c[5] - beta_cav_Gi_aGTP - beta_cav_Gi_aGDP # working
    # Concentration of Gs holoenzyme in the caveolar subspace
    beta_cav_Gs_abg = c[61] * c[58] * c[5] - beta_cav_Gs_aGTP - beta_cav_Gs_aGDP

    beta_cav_Gs_f_d = beta_cav_Gs_abg * c[93] / c[92]
    beta_cav_Gs_f_b = (c[94] + c[91]) / c[92] + beta_cav_Rb1_np_tot + beta_cav_Rb2_np_tot - beta_cav_Gs_abg
    beta_cav_Gs_f_c = (c[91] * (beta_cav_Rb1_np_tot - beta_cav_Gs_abg) + c[94] * (beta_cav_Rb2_np_tot - beta_cav_Gs_abg) + c[93]) / c[92]
    beta_cav_Gs_f_rr = -beta_cav_Gs_f_d / 27.0 * beta_cav_Gs_f_b ** 3.0 - beta_cav_Gs_f_b * beta_cav_Gs_f_b * beta_cav_Gs_f_c * beta_cav_Gs_f_c / 108.0 + beta_cav_Gs_f_b * beta_cav_Gs_f_c * beta_cav_Gs_f_d / 6.0 + beta_cav_Gs_f_c ** 3.0 / 27.0 + beta_cav_Gs_f_d * beta_cav_Gs_f_d / 4.0
    if beta_cav_Gs_f_rr > 0.0:
        tmp = np.sqrt(beta_cav_Gs_f_rr)
    else:
        tmp = 0.0
    
    beta_cav_Gs_f_yr = tmp + beta_cav_Gs_f_d / 2.0 + beta_cav_Gs_f_b * beta_cav_Gs_f_c / 6.0 - beta_cav_Gs_f_b ** 3.0 / 27.0
    if beta_cav_Gs_f_rr < 0.0:
        beta_cav_Gs_f_yi = np.sqrt(-beta_cav_Gs_f_rr)
    else:
        beta_cav_Gs_f_yi = 0.0
    beta_cav_Gs_f_mag = (beta_cav_Gs_f_yr * beta_cav_Gs_f_yr + beta_cav_Gs_f_yi * beta_cav_Gs_f_yi) ** (1.0 / 6.0)
    beta_cav_Gs_f_arg = np.arctan(beta_cav_Gs_f_yi / beta_cav_Gs_f_yr) / 3.0
    beta_cav_Gs_f_x = (beta_cav_Gs_f_c / 3.0 - beta_cav_Gs_f_b * beta_cav_Gs_f_b / 9.0) / (beta_cav_Gs_f_mag * beta_cav_Gs_f_mag)
    beta_cav_Gs_f_r = beta_cav_Gs_f_mag * np.cos(beta_cav_Gs_f_arg) * (1.0 - beta_cav_Gs_f_x) - beta_cav_Gs_f_b / 3.0
    beta_cav_Gs_f_i = beta_cav_Gs_f_mag * np.sin(beta_cav_Gs_f_arg) * (1.0 + beta_cav_Gs_f_x)

    # Concentration of free Gs in the caveolar subspace
    beta_cav_Gs_f = np.sqrt(beta_cav_Gs_f_r * beta_cav_Gs_f_r + beta_cav_Gs_f_i * beta_cav_Gs_f_i)
    # Concentration of free non-phosphorylated beta1AR in caveolar subspace
    beta_cav_Rb1_f = beta_cav_Rb1_np_tot / (1.0 + c[1] / c[65] + beta_cav_Gs_f * (c[67] + c[1]) / (c[66] * c[67]))
    # Concentration of non-phosphorylated Ligand / Receptor1 complexes in caveolar subspace
    beta_cav_LRb1 = c[1] * beta_cav_Rb1_f / c[65]
    # Concentration of non-phosphorylated Ligand / Receptor1 / G-protein complexes in caveolar subspace
    beta_cav_LRb1Gs = c[1] * beta_cav_Rb1_f * beta_cav_Gs_f / (c[66] * c[67])
    # Concentration of free non-phosphorylated beta2AR in caveolar subspace
    beta_cav_Rb2_f = beta_cav_Rb2_np_tot / (1.0 + c[1] / c[72] + beta_cav_Gs_f * (c[69] + c[1]) / (c[71] * c[69]))
    # Concentration of non-phosphorylated Ligand / Receptor2 complexes in caveolar subspace
    beta_cav_LRb2 = c[1] * beta_cav_Rb2_f / c[72]

    # Concentration of non-phosphorylated Receptor2 / G-protein complexes in caveolar subspace
    beta_cav_Rb2Gs = beta_cav_Rb2_f * beta_cav_Gs_f / c[71]
    # Concentration of non-phosphorylated Receptor1 / G-protein complexes in caveolar subspace
    beta_cav_Rb1Gs = beta_cav_Rb1_f * beta_cav_Gs_f / c[66]
    # Concentration of non-phosphorylated Ligand / Receptor2 / G-protein complexes in caveolar subspace
    beta_cav_LRb2Gs = c[1] * beta_cav_Rb2_f * beta_cav_Gs_f / (c[71] * c[69])

    #print('some values calculated')
    # Concentration of total PKA-phosphorylated beta1 receptors
    ydot[13] = 0.001 * (c[84] * pka_cav_C * beta_cav_Rb1_np_tot - c[85] * beta_cav_Rb1_pka_tot)
    # Concentration of total GRK-phosphorylated beta1 receptors
    ydot[16] = 0.001 * (c[83] * c[86] * (beta_cav_LRb1 + beta_cav_LRb1Gs) - c[82] * beta_cav_Rb1_grk_tot)
    # Concentration of total PKA-phosphorylated beta2 receptors
    ydot[48] = 0.001 * (c[84] * pka_cav_C * beta_cav_Rb2_np_tot - c[85] * beta_cav_Rb2_pka_tot)
    # Concentration of total GRK-phosphorylated beta2 receptors
    ydot[49] = 0.001 * (c[83] * c[86] * (beta_cav_LRb2 + beta_cav_LRb2Gs) - c[82] * beta_cav_Rb2_grk_tot)

    ## EXTRACAVEOLAR ##################
    #
    # beta_eca
    #
    # Concentration of Gs holoenzyme in the extracaveolar space
    beta_eca_Gs_abg = c[60] * c[58] * c[6] - beta_eca_Gs_aGTP - beta_eca_Gs_aGDP
    # Total concentration of non-phosphorylated B2AR in the extracaveolar space
    beta_eca_Rb2_np_tot = c[97] - beta_eca_Rb2_pka_tot - beta_eca_Rb2_grk_tot
    # Total concentration of non-phosphorylated B1AR in the extracaveolar space
    beta_eca_Rb1_np_tot = c[98] - beta_eca_Rb1_pka_tot - beta_eca_Rb1_grk_tot
    beta_eca_Gs_f_d = beta_eca_Gs_abg * c[101] / c[102]
    beta_eca_Gs_f_c = (c[103] * (beta_eca_Rb1_np_tot - beta_eca_Gs_abg) + c[100] * (beta_eca_Rb2_np_tot - beta_eca_Gs_abg) + c[101]) / c[102]
    beta_eca_Gs_f_b = (c[100] + c[103]) / c[102] + beta_eca_Rb1_np_tot + beta_eca_Rb2_np_tot - beta_eca_Gs_abg
    beta_eca_Gs_f_rr = -beta_eca_Gs_f_d / 27.0 * beta_eca_Gs_f_b ** 3.0 - beta_eca_Gs_f_b * beta_eca_Gs_f_b * beta_eca_Gs_f_c * beta_eca_Gs_f_c / 108.0 + beta_eca_Gs_f_b * beta_eca_Gs_f_c * beta_eca_Gs_f_d / 6.0 + beta_eca_Gs_f_c ** 3.0 / 27.0 + beta_eca_Gs_f_d * beta_eca_Gs_f_d / 4.0
    if beta_eca_Gs_f_rr < 0.0:
        beta_eca_Gs_f_yi = np.sqrt(-beta_eca_Gs_f_rr)
    else:
        beta_eca_Gs_f_yi = 0.0
        
    if beta_eca_Gs_f_rr > 0.0:
        tmp = np.sqrt(beta_eca_Gs_f_rr)
    else:
        tmp = 0.0

    beta_eca_Gs_f_yr = tmp + beta_eca_Gs_f_d / 2.0 + beta_eca_Gs_f_b * beta_eca_Gs_f_c / 6.0 - beta_eca_Gs_f_b ** 3.0 / 27.0
    beta_eca_Gs_f_mag = (beta_eca_Gs_f_yr * beta_eca_Gs_f_yr + beta_eca_Gs_f_yi * beta_eca_Gs_f_yi) ** (1.0 / 6.0)
    beta_eca_Gs_f_arg = np.arctan(beta_eca_Gs_f_yi / beta_eca_Gs_f_yr) / 3.0
    beta_eca_Gs_f_x = (beta_eca_Gs_f_c / 3.0 - beta_eca_Gs_f_b * beta_eca_Gs_f_b / 9.0) / (beta_eca_Gs_f_mag * beta_eca_Gs_f_mag)
    beta_eca_Gs_f_i = beta_eca_Gs_f_mag * np.sin(beta_eca_Gs_f_arg) * (1.0 + beta_eca_Gs_f_x)
    beta_eca_Gs_f_r = beta_eca_Gs_f_mag * np.cos(beta_eca_Gs_f_arg) * (1.0 - beta_eca_Gs_f_x) - beta_eca_Gs_f_b / 3.0
    # Concentration of free Gs in the caveolar subspace
    beta_eca_Gs_f = np.sqrt(beta_eca_Gs_f_r * beta_eca_Gs_f_r + beta_eca_Gs_f_i * beta_eca_Gs_f_i)
    # Concentration of free non-phosphorylated beta1AR in the extracaveolar space
    beta_eca_Rb1_f = beta_eca_Rb1_np_tot / (1.0 + c[1] / c[65] + beta_eca_Gs_f * (c[67] + c[1]) / (c[66] * c[67]))
    # Concentration of free non-phosphorylated beta2AR in the extracaveolar space
    beta_eca_Rb2_f = beta_eca_Rb2_np_tot / (1.0 + c[1] / c[72] + beta_eca_Gs_f * (c[69] + c[1]) / (c[71] * c[69]))
    # Concentration of non-phosphorylated Ligand / Receptor1 complexes in the extracaveolar space
    beta_eca_LRb1 = c[1] * beta_eca_Rb1_f / c[65]
    # Concentration of non-phosphorylated Ligand / Receptor2 complexes in the extracaveolar space
    beta_eca_LRb2 = c[1] * beta_eca_Rb2_f / c[72]
    # Concentration of non-phosphorylated Ligand / Receptor2 / G-protein complexes in the extracaveolar space
    beta_eca_LRb2Gs = c[1] * beta_eca_Rb2_f * beta_eca_Gs_f / (c[71] * c[69])
    # Concentration of non-phosphorylated Ligand / Receptor1 / G-protein complexes in the extracaveolar space
    beta_eca_LRb1Gs = c[1] * beta_eca_Rb1_f * beta_eca_Gs_f / (c[66] * c[67])
    # Concentration of non-phosphorylated Receptor2 / G-protein complexes in the extracaveolar space
    beta_eca_Rb2Gs = beta_eca_Rb2_f * beta_eca_Gs_f / c[71]
    # Concentration of non-phosphorylated Receptor1 / G-protein complexes in the extracaveolar space
    beta_eca_Rb1Gs = beta_eca_Rb1_f * beta_eca_Gs_f / c[66]
    beta_eca_RGs_tot = beta_eca_Rb1Gs + c[96] * beta_eca_Rb2Gs
    beta_eca_LRGs_tot = beta_eca_LRb1Gs + c[96] * beta_eca_LRb2Gs
    # Concentration of Gi holoenzyme in the extracaveolar space
    beta_eca_Gi_abg = c[64] * c[59] * c[6] - beta_eca_Gi_aGTP - beta_eca_Gi_aGDP


    beta_eca_Rb2_pka_f_c = -beta_eca_Rb2_pka_tot * c[70] * c[73]
    beta_eca_Rb2_pka_f_b = beta_eca_Gi_abg * (c[1] + c[73]) - beta_eca_Rb2_pka_tot * (c[73] + c[1]) + c[70] * c[73] * (1.0 + c[1] / c[68])
    # Concentration of free PKA-phosphorylated beta2AR in the extracaveolar space
    beta_eca_Rb2_pka_f = (-beta_eca_Rb2_pka_f_b + np.sqrt(beta_eca_Rb2_pka_f_b * beta_eca_Rb2_pka_f_b - 4.0 * c[99] * beta_eca_Rb2_pka_f_c)) / (2.0 * c[99])

    # Concentration of total PKA-phosphorylated beta1 receptors in the extracaveolar space
    ydot[14] = 0.001 * (c[84] * pka_eca_C * beta_eca_Rb1_np_tot - c[85] * beta_eca_Rb1_pka_tot)
    # Concentration of total GRK-phosphorylated beta1 receptors in the extracaveolar space
    ydot[17] = 0.001 * (c[83] * c[95] * (beta_eca_LRb1 + beta_eca_LRb1Gs) - c[82] * beta_eca_Rb1_grk_tot)
    # Concentration of total PKA-phosphorylated beta2 receptors in the extracaveolar space
    ydot[53] = 0.001 * (c[84] * pka_eca_C * beta_eca_Rb2_np_tot - c[85] * beta_eca_Rb2_pka_tot)
    # Concentration of total GRK-phosphorylated beta2 receptors in the extracaveolar space
    ydot[54] = 0.001 * (c[83] * c[95] * (beta_eca_LRb2 + beta_eca_LRb2Gs) - c[82] * beta_eca_Rb2_grk_tot)


    ## CYTOPLASM ###########
    # Concentration of Gs holoenzyme in the cytoplasm
    beta_cyt_Gs_abg = c[62] * c[58] * c[7] - beta_cyt_Gs_aGTP - beta_cyt_Gs_aGDP
    # Total concentration of non-phosphorylated beta-1 AR in the cytoplasm
    beta_cyt_Rb1_np_tot = c[104] - beta_cyt_Rb1_pka_tot - beta_cyt_Rb1_grk_tot

    beta_cyt_Rb1_np_f_b = beta_cyt_Gs_abg * (c[67] + c[1]) - beta_cyt_Rb1_np_tot * (c[67] + c[1]) + c[66] * c[67] * (1.0 + c[1] / c[65])
    beta_cyt_Rb1_np_f_c = -beta_cyt_Rb1_np_tot * c[67] * c[66]
    # Concentration of free non-phosphorylated beta-1 AR in the cytoplasm
    Rb1_np_f = (-beta_cyt_Rb1_np_f_b + np.sqrt(beta_cyt_Rb1_np_f_b * beta_cyt_Rb1_np_f_b - 4.0 * c[106] * beta_cyt_Rb1_np_f_c)) / (2.0 * c[106])
    # Concentration of free (non-complexed) Gi in the cytoplasm
    beta_cyt_Gs_f = beta_cyt_Gs_abg / (1.0 + Rb1_np_f / c[66] * (1.0 + c[1] / c[67]))
    # Concentration of non-phosphorylated ligand / beta-1 AR complexes in the cytoplasm
    LRb1_np = c[1] * Rb1_np_f / c[65]
    # Concentration of non-phosphorylated ligand / beta-1 AR / Gs complexes in the cytoplasm
    LRb1Gs_np = c[1] * Rb1_np_f * beta_cyt_Gs_f / (c[66] * c[67])
    # Concentration of non-phosphorylated beta-1 AR / Gs complexes in the cytoplasm
    Rb1Gs_np = beta_cyt_Gs_f * Rb1_np_f / c[66]

    # Concentration of total PKA-phosphorylated receptors in the cytoplasm
    ydot[15] = 0.001 * (c[84] * pka_cyt_C * beta_cyt_Rb1_np_tot - c[85] * beta_cyt_Rb1_pka_tot)
    # Concentration of total GRK-phosphorylated receptors in the cytoplasm
    ydot[18] = 0.001 * (c[83] * c[105] * (LRb1_np + LRb1Gs_np) - c[82] * beta_cyt_Rb1_grk_tot)


    ## function Mod_GprotAct() in the other version ###

    beta_cav_RGs_tot = beta_cav_Rb1Gs + c[88] * beta_cav_Rb2Gs
    beta_cav_LRGs_tot = beta_cav_LRb1Gs + c[88] * beta_cav_LRb2Gs
    beta_cav_Rb2_pka_f_c = -beta_cav_Rb2_pka_tot * c[70] * c[73]
    beta_cav_Rb2_pka_f_b = beta_cav_Gi_abg * (c[1] + c[73]) - beta_cav_Rb2_pka_tot * (c[73] + c[1]) + c[70] * c[73] * (1.0 + c[1] / c[68])
    # Concentration of free PKA-phosphorylated beta2AR in the caveolar subspace
    beta_cav_Rb2_pka_f = (-beta_cav_Rb2_pka_f_b + np.sqrt(beta_cav_Rb2_pka_f_b * beta_cav_Rb2_pka_f_b - 4.0 * c[90] * beta_cav_Rb2_pka_f_c)) / (2.0 * c[90])
    # Concentration of free (non-complexed) Gi in the caveolar subspace
    beta_cav_Gi_f = beta_cav_Gi_abg / (1.0 + beta_cav_Rb2_pka_f / c[70] * (1.0 + c[1] / c[73]))
    # Concentration of PKA phosphorylated b2AR/Gi complexes in caveolar subspace
    beta_cav_Rb2Gi = beta_cav_Rb2_pka_f * beta_cav_Gi_f / c[70]
    # Concentration of PKA phosphorylated Ligand/b2AR/Gi complexes in caveolar subspace
    beta_cav_LRb2Gi = beta_cav_Rb2Gi * c[1] / c[73]
    # Concentration of free (non-complexed) Gi in the extracaveolar space
    beta_eca_Gi_f = beta_eca_Gi_abg / (1.0 + beta_eca_Rb2_pka_f / c[70] * (1.0 + c[1] / c[73]))
    # Concentration of PKA phosphorylated b2AR/Gi complexes in extracaveolar space
    beta_eca_Rb2Gi = beta_eca_Rb2_pka_f * beta_eca_Gi_f / c[70]
    # Concentration of PKA phosphorylated Ligand/b2AR/Gi complexes in extracaveolar space
    beta_eca_LRb2Gi = c[1] / c[73] * beta_eca_Rb2Gi

    # Concentration of active Gs alpha subunit in caveolar subspace (tri-phosphate)
    ydot[1] = 0.001 * (c[75] * beta_cav_RGs_tot + c[74] * beta_cav_LRGs_tot - c[78] * beta_cav_Gs_aGTP)
    # Concentration of active Gi alpha subunit in caveolar subspace
    ydot[50] = 0.001 * (c[77] * beta_cav_Rb2Gi + c[76] * beta_cav_LRb2Gi - c[79] * beta_cav_Gi_aGTP)
    # Concentration of active Gs alpha subunit in the extracaveolar space (tri-phosphate)
    ydot[2] = 0.001 * (c[75] * beta_eca_RGs_tot + c[74] * beta_eca_LRGs_tot - c[78] * beta_eca_Gs_aGTP)
    # Concentration of active Gi alpha subunit in the extracaveolar space
    ydot[55] = 0.001 * (c[77] * beta_eca_Rb2Gi + c[76] * beta_eca_LRb2Gi - c[79] * beta_eca_Gi_aGTP)
    # Concentration of active Gs alpha subunit in cytoplasm
    ydot[3] = 0.001 * (c[75] * Rb1Gs_np + c[74] * LRb1Gs_np - c[78] * beta_cyt_Gs_aGTP)

    # Concentration of active Gs beta-gamma subunit in caveolar subspace
    ydot[4] = 0.001 * (c[75] * beta_cav_RGs_tot + c[74] * beta_cav_LRGs_tot - c[80] * beta_cav_Gs_bg * beta_cav_Gs_aGDP)
    # Concentration of active Gi beta-gamma subunit in caveolar subspace
    ydot[51] = 0.001 * (c[77] * beta_cav_Rb2Gi + c[76] * beta_cav_LRb2Gi - c[81] * beta_cav_Gi_bg * beta_cav_Gi_aGDP)
    # Concentration of active Gs beta-gamma subunit in the extracaveolar space
    ydot[5] = 0.001 * (c[75] * beta_eca_RGs_tot + c[74] * beta_eca_LRGs_tot - c[80] * beta_eca_Gs_bg * beta_eca_Gs_aGDP)
    # Concentration of active Gi beta-gamma subunit in the extracaveolar space
    ydot[56] = 0.001 * (c[77] * beta_eca_Rb2Gi + c[76] * beta_eca_LRb2Gi - c[81] * beta_eca_Gi_bg * beta_eca_Gi_aGDP)
    # Concentration of active Gs beta-gamma subunit in cytoplasm
    ydot[6] = 0.001 * (c[75] * Rb1Gs_np + c[74] * LRb1Gs_np - c[80] * beta_cyt_Gs_bg * beta_cyt_Gs_aGDP)

    # Concentration of inactive Gs alpha subunit in caveolar subspace (di-phosphate)
    ydot[7] = 0.001 * (c[78] * beta_cav_Gs_aGTP - c[80] * beta_cav_Gs_bg * beta_cav_Gs_aGDP)
    # Concentration of inactive Gi alpha subunit in caveolar subspace
    ydot[52] = 0.001 * (c[79] * beta_cav_Gi_aGTP - c[81] * beta_cav_Gi_bg * beta_cav_Gi_aGDP)
    # Concentration of inactive Gs alpha subunit in the extracaveolar space (di-phosphate)
    ydot[8] = 0.001 * (c[78] * beta_eca_Gs_aGTP - c[80] * beta_eca_Gs_bg * beta_eca_Gs_aGDP)
    # Concentration of inactive Gi alpha subunit in the extracaveolar space
    ydot[57] = 0.001 * (c[79] * beta_eca_Gi_aGTP - c[81] * beta_eca_Gi_bg * beta_eca_Gi_aGDP)
    # Concentration of inactive Gs alpha subunit in cytoplasm
    ydot[9] = 0.001 * (c[78] * beta_cyt_Gs_aGTP - c[80] * beta_cyt_Gs_bg * beta_cyt_Gs_aGDP)

    ## function Mod_AC() 

    # only calculating constants, but does not compute derivative of state
    # variables

    ## function Mod_PKA() in the other version ##########
    # Concentration of free PKA RC subunits in the caveolar compartment
    pka_cav_RCf = c[12] - pka_cav_ARC - pka_cav_A2RC - pka_cav_A2R

    # Caveolar concentration of PKA RC dimer with 1 cAMP molecule bound
    ydot[19] = 0.001 * (c[19] * pka_cav_RCf * cAMP_cav - c[22] * pka_cav_ARC - c[20] * pka_cav_ARC * cAMP_cav + c[23] * pka_cav_A2RC)
    # Caveolar concentration of PKA RC dimer with 2 cAMP molecules bound
    ydot[20] = 0.001 * (c[20] * pka_cav_ARC * cAMP_cav - (c[23] + c[21]) * pka_cav_A2RC + c[24] * pka_cav_A2R * pka_cav_C)
    # Caveolar concentration of PKA R subunit with 2 cAMP molecules bound
    ydot[21] = 0.001 * (c[21] * pka_cav_A2RC - c[24] * pka_cav_A2R * pka_cav_C)
    # Caveolar concentration of free PKA catalytic subunit
    ydot[22] = 0.001 * (c[21] * pka_cav_A2RC - c[24] * pka_cav_A2R * pka_cav_C + c[18] * pka_cav_PKIC - c[17] * (c[14] - pka_cav_PKIC) * pka_cav_C)
    # Caveolar concentration of free PKI inactivated PKA C subunit
    ydot[23] = 0.001 * (c[17] * (c[14] - pka_cav_PKIC) * pka_cav_C - c[18] * pka_cav_PKIC)

    # Concentration of free PKA RC subunits in the Extracaveolar compartment
    pka_eca_RCf = c[11] - pka_eca_ARC - pka_eca_A2RC - pka_eca_A2R
    # Extracaveolar rate of change in free cAMP through binding by PKA
    pka_eca_dcAMP = -c[19] * pka_eca_RCf * cAMP_eca + c[25] * pka_eca_ARC - c[20] * pka_eca_ARC * cAMP_eca + c[26] * pka_eca_A2RC

    # Extracaveolar concentration of PKA RC dimer with 1 cAMP molecule bound
    ydot[24] = 0.001 * (c[19] * pka_eca_RCf * cAMP_eca - c[25] * pka_eca_ARC - c[20] * pka_eca_ARC * cAMP_eca + c[26] * pka_eca_A2RC)
    # Extracaveolar concentration of PKA RC dimer with 2 cAMP molecules bound
    ydot[25] = 0.001 * (c[20] * pka_eca_ARC * cAMP_eca - (c[26] + c[21]) * pka_eca_A2RC + c[27] * pka_eca_A2R * pka_eca_C)
    # Extracaveolar concentration of PKA R subunit with 2 cAMP molecules bound
    ydot[26] = 0.001 * (c[21] * pka_eca_A2RC - c[27] * pka_eca_A2R * pka_eca_C)
    # Extracaveolar concentration of free PKA catalytic subunit
    ydot[27] = 0.001 * (c[21] * pka_eca_A2RC - c[27] * pka_eca_A2R * pka_eca_C + c[18] * pka_eca_PKIC - c[17] * (c[15] - pka_eca_PKIC) * pka_eca_C)
    # Extracaveolar concentration of free PKI inactivated PKA C subunit
    ydot[28] = 0.001 * (c[17] * (c[15] - pka_eca_PKIC) * pka_eca_C - c[18] * pka_eca_PKIC)

    # Concentration of free PKA RC subunits in the Cytosolic compartment
    pka_cyt_RCf = c[13] - pka_cyt_ARC - pka_cyt_A2RC - pka_cyt_A2R

    # Cytosolic concentration of PKA RC dimer with 1 cAMP molecule bound
    ydot[29] = 0.001 * (c[19] * pka_cyt_RCf * cAMP_cyt - c[28] * pka_cyt_ARC - c[20] * pka_cyt_ARC * cAMP_cyt + c[29] * pka_cyt_A2RC)
    # Cytosolic concentration of PKA RC dimer with 2 cAMP molecules bound
    ydot[30] = 0.001 * (c[20] * pka_cyt_ARC * cAMP_cyt - (c[29] + c[21]) * pka_cyt_A2RC + c[30] * pka_cyt_A2R * pka_cyt_C)
    # Cytosolic concentration of PKA R subunit with 2 cAMP molecules bound
    ydot[31] = 0.001 * (c[21] * pka_cyt_A2RC - c[30] * pka_cyt_A2R * pka_cyt_C)
    # Cytosolic concentration of free PKA catalytic subunit
    ydot[32] = 0.001 * (c[21] * pka_cyt_A2RC - c[30] * pka_cyt_A2R * pka_cyt_C + c[18] * pka_cyt_PKIC - c[17] * (c[16] - pka_cyt_PKIC) * pka_cyt_C)
    # Cytosolic concentration of free PKI inactivated PKA C subunit
    ydot[33] = 0.001 * (c[17] * (c[16] - pka_cyt_PKIC) * pka_cyt_C - c[18] * pka_cyt_PKIC)

    ## function Mod_cAMP() in the other version


    # Caveolar rate of change in free cAMP through binding by PKA
    pka_cav_dcAMP = -c[19] * pka_cav_RCf * cAMP_cav + c[22] * pka_cav_ARC - c[20] * pka_cav_ARC * cAMP_cav + c[23] * pka_cav_A2RC

    # Cytosolic rate of change in free cAMP through binding by PKA
    pka_cyt_dcAMP = -c[19] * pka_cyt_RCf * cAMP_cyt + c[28] * pka_cyt_ARC - c[20] * pka_cyt_ARC * cAMP_cyt + c[29] * pka_cyt_A2RC

    #PDE
    # Rate of cAMP degradation by PDE2 in cytosolic subspace
    dcAMP_PDE2_cyt = c[52] * c[41] / (1.0 + c[44] / cAMP_cyt)
    # Rate of cAMP degradation by PDE2 in extracaveolar subspace
    dcAMP_PDE2_eca = c[51] * c[41] / (1.0 + c[44] / cAMP_eca)
    # Rate of cAMP degradation by PDE2 in caveolar subspace
    dcAMP_PDE2_cav = c[50] * c[41] / (1.0 + c[44] / cAMP_cav)
    # Rate of cAMP degradation by PDE3 in caveolar subspace
    dcAMP_PDE3_cav = (c[53] + (c[47] - 1.0) * PDE3_P_cav) * c[42] / (1.0 + c[45] / cAMP_cav)
    # Rate of cAMP degradation by PDE4 in cytosolic subspace
    dcAMP_PDE4_cyt = (c[57] + (c[47] - 1.0) * PDE4_P_cyt) * c[43] / (1.0 + c[46] / cAMP_cyt)
    # Rate of cAMP degradation by PDE4 in extracaveolar subspace
    dcAMP_PDE4_eca = (c[56] + (c[47] - 1.0) * PDE4_P_eca) * c[43] / (1.0 + c[46] / cAMP_eca)
    # Rate of cAMP degradation by PDE4 in caveolar subspace
    dcAMP_PDE4_cav = (c[55] + (c[47] - 1.0) * PDE4_P_cav) * c[43] / (1.0 + c[46] / cAMP_cav)
    # Rate of cAMP degradation by PDE3 in cytosolic subspace
    dcAMP_PDE3_cyt = (c[54] + (c[47] - 1.0) * PDE3_P_cyt) * c[42] / (1.0 + c[45] / cAMP_cyt)

    camp_cAMP_cyt_pde = dcAMP_PDE2_cyt + dcAMP_PDE3_cyt + dcAMP_PDE4_cyt
    camp_cAMP_cyt_j1 = c[9] * (cAMP_cav - cAMP_cyt) / c[4]
    camp_cAMP_cyt_j2 = c[10] * (cAMP_eca - cAMP_cyt) / c[4]

    camp_cAMP_eca_pde = dcAMP_PDE2_eca + dcAMP_PDE4_eca
    camp_cAMP_eca_j2 = c[10] * (cAMP_eca - cAMP_cyt) / c[3]
    camp_cAMP_eca_j1 = c[8] * (cAMP_cav - cAMP_eca) / c[3]

    camp_cAMP_cav_pde = dcAMP_PDE2_cav + dcAMP_PDE3_cav + dcAMP_PDE4_cav
    camp_cAMP_cav_j2 = c[9] * (cAMP_cav - cAMP_cyt) / c[2]
    camp_cAMP_cav_j1 = c[8] * (cAMP_cav - cAMP_eca) / c[2]
    # ac
    #
    ac_kAC47_cyt_gsa = beta_cyt_Gs_aGTP ** c[107]
    kAC47_cyt = c[116] * (c[114] + ac_kAC47_cyt_gsa / (c[110] + ac_kAC47_cyt_gsa))
    ac_kAC56_cav_gsa = beta_cav_Gs_aGTP ** c[108]
    gsi = beta_cav_Gs_aGTP ** c[109]
    kAC56_cav = c[117] * (c[115] + ac_kAC56_cav_gsa / (c[111] + ac_kAC56_cav_gsa)) * (1.0 - (1.0 - c[118] * gsi / (c[113] + gsi)) * beta_cav_Gi_bg / (c[112] + beta_cav_Gi_bg))
    ac_kAC47_eca_gsa = beta_eca_Gs_aGTP ** c[107]
    kAC47_eca = c[116] * (c[114] + ac_kAC47_eca_gsa / (c[110] + ac_kAC47_eca_gsa))
    ac_kAC56_cyt_gsa = beta_cyt_Gs_aGTP ** c[108]
    kAC56_cyt = c[117] * (c[115] + ac_kAC56_cyt_gsa / (c[111] + ac_kAC56_cyt_gsa))

    # Rate of cAMP production by AC type 4/7 in cytoplasm
    dcAMP_AC47_cyt = kAC47_cyt * c[119] * c[121]
    # Rate of cAMP production by AC type 5/6 in cytoplasm
    dcAMP_AC56_cyt = kAC56_cyt * c[123] * c[121]
    # Rate of cAMP production by AC type 5/6 in caveolar subspace
    dcAMP_AC56_cav = kAC56_cav * c[120] * c[121]
    # Rate of cAMP production by AC type 4/7 in extracaveolar subspace
    dcAMP_AC47_eca = kAC47_eca * c[122] * c[121]


    # Caveolar concentration of cAMP
    ydot[10] = 0.001 * (pka_cav_dcAMP + dcAMP_AC56_cav - camp_cAMP_cav_pde - camp_cAMP_cav_j1 - camp_cAMP_cav_j2)
    # Extracaveolar concentration of cAMP
    ydot[11] = 0.001 * (pka_eca_dcAMP + dcAMP_AC47_eca - camp_cAMP_eca_pde + camp_cAMP_eca_j1 - camp_cAMP_eca_j2)
    # Cytosolic concentration of cAMP
    ydot[12] = 0.001 * (pka_cyt_dcAMP + dcAMP_AC47_cyt + dcAMP_AC56_cyt - camp_cAMP_cyt_pde + camp_cAMP_cyt_j1 + camp_cAMP_cyt_j2)


    ## Mod_PDE_Phosphorylation() function

    # Concentration of phosphorylated PDE3 in the caveolar subspace
    ydot[34] = 0.001 * (c[48] * pka_cav_C * (c[53] - PDE3_P_cav) - c[49] * PDE3_P_cav)
    # Concentration of phosphorylated PDE3 in the cytosolic subspace
    ydot[35] = 0.001 * (c[48] * pka_cyt_C * (c[54] - PDE3_P_cyt) - c[49] * PDE3_P_cyt)
    # Concentration of phosphorylated PDE4 in the caveolar subspace
    ydot[36] = 0.001 * (c[48] * pka_cav_C * (c[55] - PDE4_P_cav) - c[49] * PDE4_P_cav)
    # Concentration of phosphorylated PDE4 in the extracaveolar subspace
    ydot[37] = 0.001 * (c[48] * pka_eca_C * (c[56] - PDE4_P_eca) - c[49] * PDE4_P_eca)
    # Concentration of phosphorylated PDE4 in the cytosolic subspace
    ydot[38] = 0.001 * (c[48] * pka_cyt_C * (c[57] - PDE4_P_cyt) - c[49] * PDE4_P_cyt)

    ## Mod_PP1_Inhibition()

    pp1_PP1f_cyt_sum = c[38] - c[37] + inhib1_p
    # Concentration of uninhibited PP1 in the cytosolic compartment
    PP1f_cyt = 0.5 * (np.sqrt(pp1_PP1f_cyt_sum ** 2.0 + 4.0 * c[38] * c[37]) - pp1_PP1f_cyt_sum)
    di = c[39] - inhib1_p
    # Concentration of phosphorylated PP1 inhibitor 1 (cytoplasmic)
    ydot[39] = 0.001 * (c[31] * pka_cyt_C * di / (c[33] + di) - c[32] * c[40] * inhib1_p / (c[34] + inhib1_p))

    ## Mod_Channel_Phosphorylation()      

    # Substrates without AKAP
    # Fraction of phosphorylated PLB
    ydot[42] = 0.001 * (c[132] * pka_cyt_C * (1.0 - iup_f_plb) / (c[134] + 1.0 - iup_f_plb) - c[133] * PP1f_cyt * iup_f_plb / (c[135] + iup_f_plb))
    # Fraction of phosphorylated Troponin
    ydot[43] = 0.001 * (c[147] * pka_cyt_C * (1.0 - f_tni) / (c[149] + 1.0 - f_tni) - c[148] * c[40] * f_tni / (c[150] + f_tni))
    # Fraction of phosphorylated INa channels
    ydot[44] = 0.001 * (c[130] * pka_cav_C * (1.0 - ina_f_ina) / (c[128] + 1.0 - ina_f_ina) - c[131] * c[36] * ina_f_ina / (c[129] + ina_f_ina))
    # Fraction of phosphorylated INaK
    ydot[45] = 0.001 * (c[124] * pka_cav_C * (1.0 - f_inak) / (c[126] + 1.0 - f_inak) - c[125] * c[36] * f_inak / (c[127] + f_inak))
    # Fraction of phosphorylated IKur channels
    ydot[47] = 0.001 * (c[136] * pka_eca_C * (1.0 - f_ikur) / (c[138] + 1.0 - f_ikur) - c[137] * c[35] * f_ikur / (c[139] + f_ikur))


    #Substrates with AKAP
    iks_sig_IKsp_dif = c[146] - IKsp
    # Concentration of phosphorylated IKs channels
    ydot[41] = 0.001 * (c[140] * pka_eca_C * iks_sig_IKsp_dif / (c[142] + iks_sig_IKsp_dif) - c[141] * c[35] * IKsp / (c[143] + IKsp))
    akap_sig_RyRp_dif = c[162] - RyRp
    ydot[46] = 0.001 * (c[152] * pka_cav_C * akap_sig_RyRp_dif / (c[154] + akap_sig_RyRp_dif) - c[153] * c[36] * RyRp / (c[155] + RyRp))
    akap_sig_ICaLp_dif = c[164] - ICaLp
    # Concentration of phosphorylated L-type Calcium channels
    ydot[40] = 0.001 * (c[157] * pka_cav_C * akap_sig_ICaLp_dif / (c[159] + akap_sig_ICaLp_dif) - c[158] * c[36] * ICaLp / (c[160] + ICaLp))
    ydot = ydot[1:]

    for i in range(57):
        if np.isnan(ydot[i]) or np.isinf(ydot[i]):
            # inside njit you can't print, so you might raise an error or handle differently
            # print(i)
            # print(ydot[i])
            print("beta_cav_Gs_abg =", beta_cav_Gs_abg)
            # print("c[61] =", c[61])
            # print("c[58] =", c[58])
            # print("c[5] =", c[5])
            print("beta_cav_Gs_aGTP =", beta_cav_Gs_aGTP)
            # print("beta_cav_Gs_aGDP =", beta_cav_Gs_aGDP)

            # print("beta_cav_Rb1Gs =", beta_cav_Rb1Gs)
            # print("beta_cav_Rb2Gs =", beta_cav_Rb2Gs)
            # print("beta_cav_Rb1_f =", beta_cav_Rb1_f)
            # print("beta_cav_Gs_f =", beta_cav_Gs_f)
            # print("beta_cav_Rb2_f =", beta_cav_Rb2_f)
            # print("beta_cav_Gs_f =", beta_cav_Gs_f)
            # print("beta_cav_Gs_f_r =", beta_cav_Gs_f_r)
            # print("beta_cav_Gs_f_i =", beta_cav_Gs_f_i)
            # print("beta_cav_Gs_f_mag =", beta_cav_Gs_f_mag)
            # print("beta_cav_Gs_f_arg =", beta_cav_Gs_f_arg)
            # print("np.cos(beta_cav_Gs_f_arg) =", np.cos(beta_cav_Gs_f_arg))
            # print("np.sin(beta_cav_Gs_f_arg) =", np.sin(beta_cav_Gs_f_arg))
            # print("beta_cav_Gs_f_x =", beta_cav_Gs_f_x)
            # print("beta_cav_Gs_f_b =", beta_cav_Gs_f_b)
            # print("np.arctan(beta_cav_Gs_f_yi / beta_cav_Gs_f_y) =", np.arctan(beta_cav_Gs_f_yi / beta_cav_Gs_f_yr))
            # print("beta_cav_Gs_f_yi =", beta_cav_Gs_f_yi)
            # print("beta_cav_Gs_f_yr =", beta_cav_Gs_f_yr)
            # print("beta_cav_Gs_f_rr =", beta_cav_Gs_f_rr)
            # print("beta_cav_Gs_abg =", beta_cav_Gs_abg)
            # print("beta_cav_Gi_abg =", beta_cav_Gi_abg)
            # print("beta_cav_Rb2_np_tot =", beta_cav_Rb2_np_tot)
            # print("beta_cav_Rb1_np_tot =", beta_cav_Rb1_np_tot)
            # print("beta_cav_Rb1_pka_tot =", beta_cav_Rb1_pka_tot)
            # print("beta_eca_Rb1_pka_tot =", beta_eca_Rb1_pka_tot)
            # print("beta_cyt_Rb1_pka_tot =", beta_cyt_Rb1_pka_tot)
            # print("beta_cav_Rb1_grk_tot =", beta_cav_Rb1_grk_tot)
            # print("beta_eca_Rb1_grk_tot =", beta_eca_Rb1_grk_tot)
            # print("beta_cyt_Rb1_grk_tot =", beta_cyt_Rb1_grk_tot)


            #(c[75] * beta_cav_RGs_tot + c[74] * beta_cav_LRGs_tot - c[78] * beta_cav_Gs_aGTP)
            raise ValueError("NaN or Inf detected in dydt")
    
    return(ydot)


# Constants for Heijman2011
#
#
# adapted from:  https://github.com/JQXGong/Ohara-beta-adrenergic
#
# Quantitative analysis of variability in an integrated model of 
# human ventricular electrophysiology and B-adrenergic signaling
# by Jingqi Q.X. Gong, Monica E. Susilo, Anna Sher, Cynthia J. Musante, Eric A. Sobie
# DOI:https://doi.org/10.1016/j.yjmcc.2020.04.009
@njit
def Constants_SignalingMyokit2(iso_conc,radiusmultiplier):

    c = np.zeros(168)
    # iso
    c[0] = 0
    c[1]=iso_conc
    # IBMX
    ibmx = 0.0# Concentration of IBMX (in the range 0 to 100) not used in OharaBA

    ## Cell geometry
    length = 0.01# Cell length
    cell_pi =  3.14159265358979312#pi
    radius = 0.0011*radiusmultiplier# Cell radius
    volume = 1000.0 * cell_pi * radius * radius * length# Cell volume

    c[2] = 0.02 * volume# Volume of the caveolar subspace
    c[3] = 0.04 * volume# Volume of the extracaveolar subspace
    c[4] = volume * 0.678# Volume of the Cytoplasm / Myoplasm

    c[5] = volume / c[2]# Ratio of whole volume to caveolar subspace volume
    c[6] = volume / c[3]# Ratio of whole volume to extracaveolar subspace volume
    c[7] = volume / c[4]# Ratio of whole volume to cytoplasm volume


    ## cAMP
    c[8] = 5e-15 * 1000000.0# Rate of cAMP diffusion between caveolar and cytosolic compartments
    c[9] = 7.5e-14 * 1000000.0# Rate of cAMP diffusion between caveolar and extracaveolar compartments
    c[10] = 9e-15 * 1000000.0# Rate of cAMP diffusion between extracaveolar and cytosolic compartments


    ## PKA (See pg 29 - 32 of Heijman supplementary document)
    PKA_tot = 0.5                    # Total cellular concentration of PKA holoenzyme (umol/L)
    f_cav = 0.0388                   # Fraction of PKA located in caveolar compartment
    f_eca = 0.1                      # Fraction of PKA located in extracaveolar compartment
    f_cyt = 1.0 - f_cav - f_eca  # Fraction of PKA located in cytosolic compartment

    c[11] = f_eca * PKA_tot * c[6]# Concentration of PKA in the extracaveolar compartment
    c[12] = f_cav * PKA_tot * c[5]# Concentration of PKA in the caveolar compartment
    c[13] = f_cyt * PKA_tot * c[7]# Concentration of PKA in the cytosolic compartment

    PKI_tot = 0.2 * PKA_tot# Total cellular concentration of PKA inhibitor
    f_pki_cav = f_cav              # Fraction of PKI located in caveolar compartment
    f_pki_eca = f_eca              # Fraction of PKI located in extracaveolar compartment
    f_pki_cyt = 1.0 - f_pki_cav - f_pki_eca# Fraction of PKI located in cytosolic compartment

    c[14] = f_pki_cav * PKI_tot * c[5]# Concentration of protein kinase inhibitor in caveolar compartment
    c[15] = f_pki_eca * PKI_tot * c[6]# Concentration of protein kinase inhibitor in extracaveolar compartment
    c[16] = f_pki_cyt * PKI_tot * c[7]# Concentration of protein kinase inhibitor in cytosolic compartment

    c[17] = 50.0                     # Forward rate for inhibition of C subunit by PKI
    K_pki = 0.01 / 50.0              # Equilibrium value for inhibition of C subunit by PKI (umol/L)
    c[18] = c[17] * K_pki        # Backward rate for inhibition of C subunit by PKI

    # pka_cav
    c[19] = 100.0# Caveolar forward rate for binding of the first cAMP to PKA
    c[20] = 100.0# Caveolar forward rate for binding of the second cAMP to PKA
    c[21] = 100.0# Caveolar forward rate for dissociation of C subunit

    pka_cav_K1 = 2.4984# Caveolar equilibrium value for the binding of the first cAMP to PKA
    pka_cav_K2 = 11.359# Caveolar equilibrium value for the binding of the second cAMP to PKA
    pka_cav_K3 = 0.3755# Caveolar equilibrium value for dissociation of C subunit

    c[22] = c[19] * pka_cav_K1# Caveolar backward rate for binding of the first cAMP to PKA
    c[23] = c[20] * pka_cav_K2# Caveolar backward rate for binding of the second cAMP to PKA
    c[24] = c[21] * pka_cav_K3# Caveolar backward rate for dissociation of C subunit


    pka_eca_K1 = pka_cav_K1# Extracaveolar equilibrium value for the binding of the first cAMP to PKA
    pka_eca_K2 = pka_cav_K2# Extracaveolar equilibrium value for the binding of the second cAMP to PKA
    pka_eca_K3 = pka_cav_K3# Extracaveolar equilibrium value for dissociation of C subunit

    c[25] = c[19] * pka_eca_K1# Extracaveolar backward rate for binding of the first cAMP to PKA
    c[26] = c[20] * pka_eca_K2# Extracaveolar backward rate for binding of the second cAMP to PKA
    c[27] = c[21] * pka_eca_K3# Extracaveolar backward rate for dissociation of C subunit

    pka_cyt_K1 = 0.1088# Cytosolic equilibrium value for the binding of the first cAMP to PKA
    pka_cyt_K2 = 0.4612# Cytosolic equilibrium value for binding of the second cAMP to PKA
    pka_cyt_K3 = 0.3755# Cytosolic equilibrium value for dissociation of C subunit

    c[28] = c[19] * pka_cyt_K1# Cytosolic backward rate for binding of the first cAMP to PKA
    c[29] = c[20] * pka_cyt_K2# Cytosolic backward rate for binding of the second cAMP to PKA
    c[30] = c[21] * pka_cyt_K3# Cytosolic backward rate for dissociation of C subunit

    ## PP1 (see pg 32-33 of Heijman Supplementary document)
    f = 0.3# Fractional increase in PP1 after Inh1 knockout
    c[31] = 0.010145# Rate of phosphorylation of inhibitor 1 by PKA
    c[32] = 0.0035731# Rate of dephosphorylation if inhibitor 1
    c[33] = 0.001469# Affinity of inhibitor 1 for PKA catalytic subunit
    c[34] =  1.95259999999999991e-05# Affinity of inhibitor 1 for PP2A

    c[35] = 0.1# PP1 concentration in the extracaveolar compartment
    c[36] = 0.25# PP1 concentration in the caveolar compartment
    c[37] = 0.2# PP1 concentration in the cytosolic compartment

    c[38] = 0.001# Affinity foor PP1 Inhibitor 1 binding

    c[39] = f / (1.0 - f) * c[38] + f * c[37]# Concentration of phosphatase inhibitor 1 in the cytosolic compartment
    c[40] = 1.0 # PP2A concentration? 


    ## PDE (see pg 26 - 29 of Heijman supplementary document)
    PDE2_tot = 0.029268# Total cellular concentration of PDE2
    f_pde2_cav = 0.16957# Fraction of PDE2 located in caveolar compartment
    f_pde2_eca =  2.12570000000000006e-04# Fraction of PDE2 located in extracaveolar compartment
    f_pde2_cyt = 1.0 - f_pde2_cav - f_pde2_eca# Fraction of PDE2 located in cytosolic compartment
    f_pde2_part = f_pde2_cav + f_pde2_eca# Fraction of PDE2 located in the "particulate fraction" (cav + eca)

    f_pde_part = 0.2# Fraction of total PDE located in the "particulate fraction" (cav + eca)
    f_pde4_part = 0.125# Fraction of PDE4 located in the "particulate fraction" (cav + eca)

    f_pde4_cav = 0.12481# Fraction of PDE4 in caveolar compartment
    f_pde4_eca = f_pde4_part - f_pde4_cav# Fraction of PDE4 located in the extracaveolar compartment
    f_pde4_cyt = 1.0 - f_pde4_part# Fraction of PDE4 in cytosolic compartment

    c[41] = 20.0# Rate of cAMP hydrolysis by PDE2
    c[42] = 2.5# Rate of cAMP hydrolysis by PDE3
    c[43] = 4.0# Rate of cAMP hydrolysis by PDE4

    c[44] = 50.0# Affinity of PDE2 for cAMP
    c[45] = 0.8# Affinity of PDE3 for cAMP
    c[46] = 1.4# Affinity of PDE4 for cAMP

    KmIbmxPde2 = 21.58# Affinity of IBMX for PDE2
    h_ibmx_pde2 = 1.167# Hill coefficient for inhibition of PDE2 by IBMX
    h_ibmx_pde3 = 0.7629# Hill coefficient for inhibition of PDE3 by IBMX
    h_ibmx_pde4 = 0.9024# Hill coefficient for inhibition of PDE4 by IBMX

    KmIbmxPde3 = 2.642# Affinity of IBMX for PDE3
    KmIbmxPde4 = 11.89# Affinity of IBMX for PDE4

    KPDEp = 0.52218
    c[47] = 3.0# Increase in PDE3 / PDE4 activity after phosphorylation
    ff_pde3_cyt = 0.35# Fraction of PDE in cytosol that is of type 3
    c[48] = 0.0196# Rate of phosphorylation by PKA of PDE3 and PDE4
    r_pde34_frac = 3.71# Ratio of PDE3 to PDE4 in particulate fraction (78:21)
    r_pde3_cyt = ff_pde3_cyt / (1.0 - ff_pde3_cyt)# Relative contribution of PDE3 to cytosolic PDE3
    c[49] = KPDEp * c[48]# Rate of dephosphorylation of PDE3 and PDE4

    pde_PDE3_tot_alpha = r_pde3_cyt * (f_pde4_part * (1.0 + r_pde34_frac - r_pde34_frac * f_pde2_part - f_pde_part) + f_pde2_part * (f_pde_part - 1.0)) + r_pde34_frac * f_pde4_part * (f_pde_part - f_pde2_part)
    pde_PDE3_tot_beta = f_pde4_part * (1.0 + r_pde34_frac + f_pde_part * (r_pde3_cyt - r_pde34_frac)) - f_pde_part * (1.0 + r_pde3_cyt)

    PDE3_tot = pde_PDE3_tot_alpha / pde_PDE3_tot_beta * PDE2_tot# Total cellular concentration of PDE3
    PDE4_tot = ((f_pde_part - f_pde2_part) * PDE2_tot + f_pde_part * PDE3_tot) / ((1.0 + r_pde34_frac) * f_pde4_part - f_pde_part)# Total cellular concentration of PDE4

    ibmx_h2 = ibmx ** h_ibmx_pde2
    ibmx_h3 = ibmx ** h_ibmx_pde3
    ibmx_h4 = ibmx ** h_ibmx_pde4

    ibmx2 = (1.0 - ibmx_h2 / (KmIbmxPde2 + ibmx_h2)) * PDE2_tot
    ibmx3 = (1.0 - ibmx_h3 / (KmIbmxPde3 + ibmx_h3)) * PDE3_tot
    ibmx4 = (1.0 - ibmx_h4 / (KmIbmxPde4 + ibmx_h4)) * PDE4_tot

    f_pde3_cav = r_pde34_frac * f_pde4_part * PDE4_tot / PDE3_tot
    f_pde3_cyt = 1.0 - f_pde3_cav

    c[50] = ibmx2 * f_pde2_cav * c[5]# Concentration of PDE2 in Caveolar subspace
    c[51] = ibmx2 * f_pde2_eca * c[6]# Concentration of PDE2 in Extracaveolar subspace
    c[52] = ibmx2 * f_pde2_cyt * c[7]# Concentration of PDE2 in Cytosolic subspace

    c[53] = ibmx3 * f_pde3_cav * c[5]# Concentration of PDE3 in Caveolar subspace
    c[54] = ibmx3 * f_pde3_cyt * c[7]# Concentration of PDE3 in Cytosolic subspace

    c[55] = ibmx4 * f_pde4_cav * c[5]# Concentration of PDE4 in Caveolar subspace
    c[56] = ibmx4 * f_pde4_eca * c[6]# Concentration of PDE4 in Extracaveolar subspace
    c[57] = ibmx4 * f_pde4_cyt * c[7]# Concentration of PDE4 in Cytosolic subspace


    ## Adrenergic Receptor and G Protein Activation (pg 18 - 24 of Heijman supplementary document) 
    beta_R_b1_tot = 0.85 * 0.025         # Total cellular beta-1 adrenergic receptor concentration
    beta_R_b2_tot = 0.15 * 0.025# Total cellular beta-2 adrenergic receptor concentration
    c[58] = 224.0 * beta_R_b1_tot     # Total Gs protein concentration
    c[59] = 0.5                         # Total Gi protein concentration

    c[60] = 0.5664# Fraction of Gs proteins located in extracaveolar space
    c[61] = 0.0011071# Fraction of Gs proteins located in caveolar subspace
    c[62] = 1.0 - c[61] - c[60]# Fraction of Gs proteins located in cytoplasm

    c[63] = 0.85# Fraction of Gi proteins located in caveolar subspace
    c[64] = 1.0 - c[63]# Fraction of Gi proteins located in extracaveolar space

    f_Rb1_cav = 0.081161# Fraction of beta-1 adrenergic receptors located in caveolar subspace
    f_Rb1_eca = 0.48744# Fraction of beta-1 adrenergic receptors located in extra caveolar space
    f_Rb1_cyt = 1.0 - f_Rb1_cav - f_Rb1_eca# Fraction of beta-1 adrenergic receptors located in cytoplasm

    f_Rb2_cav = 0.85# Fraction of beta-2 adrenergic receptors located in caveolar subspace
    f_Rb2_eca = 1.0 - f_Rb2_cav# Fraction of beta-2 adrenergic receptors located in extracaveolar space

    c[65] = 0.567# beta-1 receptor / ligand low affinity constant
    c[66] = 2.449# beta-1 receptor / G-protein affinity constant
    c[67] = 0.062# beta-1 receptor / ligand high affinity constant

    c[68] = 1.053# Phosph. beta-2 receptor / ligand low affinity constant
    c[69] = 0.012# beta-2 receptor / ligand high affinity constant
    c[70] = 1.6655# Phosph. beta-2 receptor / Gi-protein affinity constant
    c[71] = 1.8463# beta-2 receptor / G-protein affinity constant
    c[72] = 1.053# beta-2 receptor / ligand low affinity constant
    c[73] = 0.1# Phosph. beta-2 receptor / ligand high affinity constant

    c[74] = 4.9054# Activation rate for Gs by high affinity complex
    c[75] = 0.25945# Activation rate for Gs by low affinity complex

    c[76] = 4.0# Activation rate for Gi by high affinity complex
    c[77] = 0.05# Activation rate for Gi by low affinity complex

    c[78] = 0.8# Gs GTP to GDP hydrolysis constant
    c[79] = c[78]# Gi GTP to GDP hydrolysis constant

    c[80] = 1210.0# Reassociation rate for Gs subunits
    c[81] = c[80]# Reassociation rate for Gi subunits

    rate_bds = 0.35

    c[82] = rate_bds * 0.0009833# Rate for GRK dephosphorylation
    c[83] = rate_bds * 0.00133# Rate for GRK dependent receptor desensitization

    c[84] = rate_bds * 0.0065# Rate for (PKA dependent receptor) desensitization
    c[85] = 0.15629 * c[84]# Rate for PKA dephosphorylation

    # beta_cav
    c[86] = 1.0
    c[87] = f_Rb1_cav * beta_R_b1_tot * c[5]# Total concentration of beta1AR in the caveolar subspace
    c[88] = 1.0# Ratio of beta2AR incorporated in RGs_Tot and LRGs_Tot
    c[89] = f_Rb2_cav * beta_R_b2_tot * c[5]# Total concentration of beta2AR in the caveolar subspace
    c[90] = (c[73] + c[1]) * (c[68] + c[1]) / c[68]
    c[91] = c[71] * c[69] * c[65] * (c[67] + c[1]) * (c[72] + c[1])
    c[92] = c[65] * c[72] * (c[67] + c[1]) * (c[69] + c[1])
    c[93] = c[66] * c[71] * c[67] * c[69] * (c[65] + c[1]) * (c[72] + c[1])
    c[94] = c[66] * c[67] * c[72] * (c[69] + c[1]) * (c[65] + c[1])

    # beta_eca
    c[95] = 1.0
    c[96] = 1.0# Ratio of beta2AR incorporated in RGs_Tot and LRGs_Tot
    c[97] = f_Rb2_eca * beta_R_b2_tot * c[6]# Total concentration of beta2AR in the extracaveolar space
    c[98] = f_Rb1_eca * beta_R_b1_tot * c[6]# Total concentration of beta1AR in the extracaveolar space
    c[99] = (c[73] + c[1]) * (c[68] + c[1]) / c[68]
    c[100] = c[66] * c[67] * c[72] * (c[69] + c[1]) * (c[65] + c[1])
    c[101] = c[66] * c[71] * c[67] * c[69] * (c[65] + c[1]) * (c[72] + c[1])
    c[102] = c[65] * c[72] * (c[67] + c[1]) * (c[69] + c[1])
    c[103] = c[71] * c[69] * c[65] * (c[67] + c[1]) * (c[72] + c[1])

    # beta_cyt
    c[104] = f_Rb1_cyt * beta_R_b1_tot * c[7]# Total concentration of beta-1 AR in the cytoplasm
    c[105] = 1.0
    c[106] = (c[67] + c[1]) * (c[65] + c[1]) / c[65]


    ## AC (see pg 24-26 of Heijman supplementary document)
    ATP = 5000.0# Concentration of ATP
    KmATP = 315.0# AC affinity for ATP
    AC_tot = 3.0 * beta_R_b1_tot# Total cellular AC concentration

    f_AC47_eca = 0.16479# Fraction of AC47 in extracaveolar space
    f_AC56_cav = 0.087459# Fraction of AC56 located in caveolae
    f_AC56_AC47 = 1.0 / (1.0 + 0.35)# Fraction of AC that is of type 5/6

    c[107] = 1.0043# Hill coefficient for AC47 activation
    c[108] = 1.3574# Hill coefficient for AC56 activation
    c[109] = 0.6623# Hill coefficient for Gs/Gi interaction of AC56

    c[110] = 0.031544# AC47 affinity for Gs
    c[111] = 0.0852# AC56 affinity for Gs
    c[112] = 0.0465# AC56 affinity for inhibition by Gi
    c[113] = 0.4824# Gs-dependence of inactivation by Gi for AC56

    c[114] = 0.03135# Basal AC47 activity
    c[115] = 0.037696# Basal AC56 activity

    c[116] = 3.3757# Amplification factor for AC47
    c[117] = 41.32# Amplification factor for AC56

    c[118] = 0.8569# Maximum reduction in Gi inhibition, by Gs

    c[119] = (1.0 - f_AC47_eca) * (1.0 - f_AC56_AC47) * AC_tot * c[7]# Concentration of AC56 in the cytoplasm
    c[120] = f_AC56_cav * f_AC56_AC47 * AC_tot * c[5]# Concentration of AC56 in the caveolar subspace

    c[121] = ATP / (KmATP + ATP)
    c[122] = f_AC47_eca * (1.0 - f_AC56_AC47) * AC_tot * c[6]# Concentration of AC56 in the extracaveolar space
    c[123] = (1.0 - f_AC56_cav) * f_AC56_AC47 * AC_tot * c[7]# Concentration of AC56 in the cytoplasm


    ## Substrate Phosphorylation
    ##  iNaK 
    c[124] = 0.015265               # Rate of INaK phosphorylation by PKA
    c[125] = 0.092455               # Rate of INaK dephosphorylation by phosphatases
    c[126] = 0.0011001              # Affinity of INaK for phosphorylation by PKA
    c[127] = 5.7392                 # Affinity of INaK for dephosphorylation by phosphatases

    ## iNa
    c[128] = 0.10988                 # Affinity of INa for phosphorylation by PKA
    c[129] = 7.8605                  # Affinity of INa for dephosphorylation by phosphatases
    c[130] = 0.01368                 # Rate of INa phosphorylation by PKA
    c[131] = 0.052811                # Rate of INa dephosphorylation by phosphatases


    ## iup
    c[132] = 0.11348                 # Rate of PLB phosphorylation by PKA
    c[133] = 0.48302                 # Rate of PLB dephosphorylation by phosphatases
    c[134] =  9.88539999999999992e-04# Affinity of PLB for phosphorylation by PKA
    c[135] = 0.80737                 # Affinity of PLB for dephosphorylation by phosphatases


    ## iKur
    c[136] = 0.069537               # Rate of IKur phosphorylation by PKA
    c[137] = 0.317                  # Rate of IKur dephosphorylation by phosphatases
    c[138] = 0.27623                # Affinity of IKur for phosphorylation by PKA
    c[139] = 0.002331               # Affinity of IKur for dephosphorylation by phosphatases


    ## iKs
    c[140] = 0.16305                 # Rate of IKs channel phosphorylation by PKA
    c[141] = 1.0542                  # Rate of IKs channel dephosphorylation by phosphatases
    c[142] =  9.97940000000000003e-05# Affinity of IKs channels for phosphorylation by PKA
    c[143] =  1.11470000000000002e-04# Affinity of IKs channels for dephosphorylation by phosphatases

    M = 0.01                 # Binding affinity between PKA and Yotiao
    iks_sig_L = 0.0001       # Binding affinity between PKA and Yotiao
    iks_sig_K = 0.01         # Binding affinity between PP1 and Yotiao
    Yotiao = 0.025           # Total concentration of Yotiao
    c[144] = 0.025          # Total concentration of IKs channels
    iks_sig_PKAf_sum = 1.0 + (Yotiao - c[11]) / M
    iks_sig_PKAf = M / 2.0 * (np.sqrt(iks_sig_PKAf_sum ** 2.0 + 4.0 * c[11] / M) - iks_sig_PKAf_sum)# Concentration of PKA not bound to AKAPs in the extracaveolar compartment
    iks_sig_PP1f_eca_sum = 1.0 + (Yotiao - c[35]) / iks_sig_K
    PP1f_eca = iks_sig_K / 2.0 * (np.sqrt(iks_sig_PP1f_eca_sum ** 2.0 + 4.0 * c[35] / iks_sig_K) - iks_sig_PP1f_eca_sum)# Concentration of PP1 not bound to AKAPs in the extracaveolar compartment
    iks_sig_IKsf_sum = 1.0 + (Yotiao - c[144]) / iks_sig_L
    IKsf = iks_sig_L / 2.0 * (np.sqrt(iks_sig_IKsf_sum ** 2.0 + 4.0 * c[144] / iks_sig_L) - iks_sig_IKsf_sum)# Concentration of IKs not bound to AKAPs in the extracaveolar compartment
    Yotiaof = (Yotiao - c[144] + IKsf) / ((1.0 + PP1f_eca / iks_sig_K) * (1.0 + iks_sig_PKAf / M))
    c[145] = IKsf * Yotiaof * iks_sig_PKAf / (iks_sig_L * M)# Concentration of IKs channels that have an AKAP and local PKA but no PP1 available
    c[146] = c[145] * PP1f_eca / iks_sig_K# Concentration of IKs channels that have an AKAP, local PKA and PP1 available


    ## Troponin
    c[147] = 0.10408                 # Rate of Troponin phosphorylation by PKA
    c[148] = 0.052633                # Rate of Troponin dephosphorylation by phosphatases
    c[149] =  2.71430000000000008e-05# Affinity of Troponin for phosphorylation by PKA
    c[150] = 0.26714                 # Affinity of Troponin for dephosphorylation by phosphatases


    ## RyR and iCaL  -  Substrates with A-Kinase Anchoring Protein (AKAP) 
    c[151] = 0.125          # Total concentration of RyRs
    RyR_akap = 0.125         # Total concentration of RyR AKAP
    c[152] = 0.0025548       # Rate of RyR phosphorylation by PKA
    c[153] = 0.0038257       # Rate of RyR dephosphorylation by phosphatases
    c[154] =  6.62979999999999944e-05# Affinity of RyR for phosphorylation by PKA
    c[155] = 0.043003        # Affinity of RyR for dephosphorylation by phosphatases
    Mr = 0.01                # Binding affinity between PKA and ICaL AKAP
    Lr = 0.0001              # Binding affinity between RyR and AKAP
    Kr = 0.01                # Binding affinity between PP1 and RyR AKAP

    akap_sig_RyRf_sum = 1.0 + (RyR_akap - c[151]) / Lr
    RyRf = Lr / 2.0 * (np.sqrt(akap_sig_RyRf_sum ** 2.0 + 4.0 * c[151] / Lr) - akap_sig_RyRf_sum)# Caveolar concentration of free RyR

    c[156] = 0.025         # Total concentration of ICaL channels
    ICaL_akap = 0.025        # Total concentration of ICaL AKAP
    c[157] =  5.10090000000000044e-04# Rate of ICaL channel phosphorylation by PKA
    c[158] = 0.0006903      # Rate of ICaL channel dephosphorylation by phosphatases
    c[159] =  1.27019999999999993e-06# Affinity of ICaL channels for phosphorylation by PKA
    c[160] = 0.0063064      # Affinity of ICaL channels for dephosphorylation by phosphatases
    Mi = 0.01                # Binding affinity between PKA and ICaL AKAP
    Li = 0.0001              # Binding affinity between ICaL channel and AKAP
    Ki = 0.01                # Binding affinity between PP1 and ICaL AKAP
    akap_sig_ICaLf_sum = 1.0 + (ICaL_akap - c[156]) / Li
    ICaLf = Li / 2.0 * (np.sqrt(akap_sig_ICaLf_sum ** 2.0 + 4.0 * c[156] / Li) - akap_sig_ICaLf_sum)# Caveolar concentration of free ICaL

    akap_sig_PP1f_cav_b = ICaL_akap + RyR_akap + Ki + Kr - c[36]
    akap_sig_PP1f_cav_c = ICaL_akap * Kr + RyR_akap * Ki + Ki * Kr - c[36] * (Ki + Kr)
    akap_sig_PP1f_cav_d = c[36] * Ki * Kr
    akap_sig_PP1f_cav_rr = -akap_sig_PP1f_cav_d / 27.0 * akap_sig_PP1f_cav_b ** 3.0 - akap_sig_PP1f_cav_b * akap_sig_PP1f_cav_b * akap_sig_PP1f_cav_c * akap_sig_PP1f_cav_c / 108.0 + akap_sig_PP1f_cav_b * akap_sig_PP1f_cav_c * akap_sig_PP1f_cav_d / 6.0 + akap_sig_PP1f_cav_c ** 3.0 / 27.0 + akap_sig_PP1f_cav_d * akap_sig_PP1f_cav_d / 4.0
    if akap_sig_PP1f_cav_rr < 0.0:
        akap_sig_PP1f_cav_yi = np.sqrt(-akap_sig_PP1f_cav_rr)
    else:
        akap_sig_PP1f_cav_yi = 0.0
    
    if akap_sig_PP1f_cav_rr > 0.0:
        tmp = np.sqrt(akap_sig_PP1f_cav_rr)
    else:
        tmp = 0.0
    akap_sig_PP1f_cav_yr = tmp + akap_sig_PP1f_cav_d / 2.0 + akap_sig_PP1f_cav_b * akap_sig_PP1f_cav_c / 6.0 - akap_sig_PP1f_cav_b ** 3.0 / 27.0
    akap_sig_PP1f_cav_mag = (akap_sig_PP1f_cav_yr * akap_sig_PP1f_cav_yr + akap_sig_PP1f_cav_yi * akap_sig_PP1f_cav_yi) ** (1.0 / 6.0)
    akap_sig_PP1f_cav_arg = np.arctan(akap_sig_PP1f_cav_yi / akap_sig_PP1f_cav_yr) / 3.0
    akap_sig_PP1f_cav_x = (akap_sig_PP1f_cav_c / 3.0 - akap_sig_PP1f_cav_b * akap_sig_PP1f_cav_b / 9.0) / (akap_sig_PP1f_cav_mag * akap_sig_PP1f_cav_mag)

    PP1f_cav = akap_sig_PP1f_cav_mag * np.cos(akap_sig_PP1f_cav_arg) * (1.0 - akap_sig_PP1f_cav_x) - akap_sig_PP1f_cav_b / 3.0# Caveolar concentration of free PP1
    akap_sig_PKAf_d = c[12] * Mi * Mr
    akap_sig_PKAf_b = ICaL_akap + RyR_akap + Mi + Mr - c[12]
    akap_sig_PKAf_c = ICaL_akap * Mr + RyR_akap * Mi + Mi * Mr - c[12] * (Mi + Mr)
    akap_sig_PKAf_rr = -akap_sig_PKAf_d / 27.0 * akap_sig_PKAf_b ** 3.0 - akap_sig_PKAf_b * akap_sig_PKAf_b * akap_sig_PKAf_c * akap_sig_PKAf_c / 108.0 + akap_sig_PKAf_b * akap_sig_PKAf_c * akap_sig_PKAf_d / 6.0 + akap_sig_PKAf_c ** 3.0 / 27.0 + akap_sig_PKAf_d * akap_sig_PKAf_d / 4.0
    if akap_sig_PKAf_rr > 0.0:
        tmp = np.sqrt(akap_sig_PKAf_rr)
    else:
        tmp = 0.0
    akap_sig_PKAf_yr = tmp + akap_sig_PKAf_d / 2.0 + akap_sig_PKAf_b * akap_sig_PKAf_c / 6.0 - akap_sig_PKAf_b ** 3.0 / 27.0
    
    if akap_sig_PKAf_rr < 0.0:
        akap_sig_PKAf_yi = np.sqrt(-akap_sig_PKAf_rr)
    else:
        akap_sig_PKAf_yi = 0.0
    akap_sig_PKAf_mag = (akap_sig_PKAf_yr * akap_sig_PKAf_yr + akap_sig_PKAf_yi * akap_sig_PKAf_yi) ** (1.0 / 6.0)
    akap_sig_PKAf_arg = np.arctan(akap_sig_PKAf_yi / akap_sig_PKAf_yr) / 3.0
    akap_sig_PKAf_x = (akap_sig_PKAf_c / 3.0 - akap_sig_PKAf_b * akap_sig_PKAf_b / 9.0) / (akap_sig_PKAf_mag * akap_sig_PKAf_mag)

    akap_sig_PKAf = akap_sig_PKAf_mag * np.cos(akap_sig_PKAf_arg) * (1.0 - akap_sig_PKAf_x) - akap_sig_PKAf_b / 3.0# Caveolar concentration of free PKA

    RyR_akapf = (RyR_akap - c[151] + RyRf) / ((PP1f_cav / Kr + 1.0) * (akap_sig_PKAf / Mr + 1.0))# Caveolar concentration of free RyR AKAP
    c[161] = RyRf * RyR_akapf * akap_sig_PKAf / (Lr * Mr)# Concentration of RyR that have an AKAP, local PKA but no PP1 available
    c[162] = c[161] * PP1f_cav / Kr# Concentration of RyR that have an AKAP, local PKA and PP1

    ICaL_akapf = (ICaL_akap - c[156] + ICaLf) / ((PP1f_cav / Ki + 1.0) * (akap_sig_PKAf / Mi + 1.0))# Caveolar concentration of free ICaL AKAP
    c[163] = ICaLf * ICaL_akapf * akap_sig_PKAf / (Li * Mi)# Concentration of ICaL channels that have an AKAP, local PKA but no PP1 available
    c[164] = c[163] * PP1f_cav / Ki# Concentration of ICaL channels that have an AKAP, local PKA and PP1

    ## These are for electrophysiology.. check these to make sure that the one for O'Hara is consisten
    # irel

    c[165] = 0.0329 + c[161] / c[151]

    # ical
    c[166] = 0.0269 + c[163] / c[156]

    #iks
    c[167] = 0.0306 + c[145] / c[144]

    return(c)


# adapted from:  https://github.com/JQXGong/Ohara-beta-adrenergic
#
# Quantitative analysis of variability in an integrated model of 
# human ventricular electrophysiology and B-adrenergic signaling
# by Jingqi Q.X. Gong, Monica E. Susilo, Anna Sher, Cynthia J. Musante, Eric A. Sobie
# DOI:https://doi.org/10.1016/j.yjmcc.2020.04.009
#

@njit
def EffectiveFraction_Torord2(ICaLp,IKsp,iup_f_plb,f_tni,ina_f_ina,f_inak,RyRp,f_ikur, c):
    # Calculating effective fraction of phosphorylated substrates 
    #################################################################
    #################################################################

    ## ICaL
    #ICaLp = y[39] #y[40]
    fp_ICaL = (ICaLp + c[163]) / c[156] # Fraction of phosphorylated ICaL channels
    ################################
    if fp_ICaL<0.0:
        fp_ICaL = 0.0001
    elif fp_ICaL > 1.0:
        fp_ICaL = 0.9999

    ical_f_hat = (fp_ICaL - c[166]) / (0.9273 - c[166]) # Effective fraction of phosphorylated ICaL channels
    if ical_f_hat < 0.0:
        ical_f_hat = 0
    elif ical_f_hat > 1.0:
        ical_f_hat = 1.0
    
    ## IKs
    #IKsp = y[40]#y[41]
    fp_iks = (IKsp + c[145]) / c[144]# Fraction of phosphorylated IKs
    if fp_iks < 0.0:
        fp_iks = 0.0001
    elif fp_iks > 1.0:
        fp_iks = 0.9999
        
    iks_f_hat = (fp_iks - c[167]) / (0.785 - c[167]) # Effective fraction of phosphorylated IKs channels
    if iks_f_hat < 0.0:
        iks_f_hat = 0
    elif iks_f_hat > 1.0:
        iks_f_hat = 1.0
    ## Iup (PLB)
    #iup_f_plb = y[41]#y[42]
    iup_f_pka = (iup_f_plb - 0.6591) / (0.9945 - 0.6591)
    if iup_f_pka < 0.0:
        iup_f_pka = 0
    elif iup_f_pka > 1.0:
        iup_f_pka = 1.0
    ## Tni
    #f_tni = y[42]#y[43]
    calcium_fhat = (f_tni - 0.6735188) / (0.9991797 - 0.6735188)# Effective fraction of phosphorylated Troponin
    if calcium_fhat < 0.0:
        calcium_fhat = 0
    elif calcium_fhat > 1.0:
        calcium_fhat = 1.0
    ## INa
    #ina_f_ina = y[43]#y[44]
    ina_f_pka = (ina_f_ina - 0.2394795) / (0.9501431 - 0.2394795) # Effective fraction of phosphorylated INa channels
    if ina_f_pka<0.0:
        ina_f_pka = 0.0
    elif ina_f_pka>1.0:
        ina_f_pka = 1.0
    ## INaK
    #f_inak = y[44]#y[45]
    inak_fhat = (f_inak - 0.1263453) / (0.9980137 - 0.1263453) # Effective fraction of phosphorylated INaK pumps
    if inak_fhat<0.0:
        inak_fhat = 0.0
    elif inak_fhat>1.0:
        inak_fhat = 1.0
    ## RyR
    #RyRp = y[45]#y[46]
    fp_RyR = (RyRp + c[161]) / c[151]# Fraction of phosphorylated RyR channels
    if fp_RyR<0.0:
        fp_RyR = 0.0001
    elif fp_RyR>1.0:
        fp_RyR = 0.9999

    irel_fhat = (fp_RyR - c[165]) / (0.9586 - c[165])# Effective fraction of phosphorylated ryr channels
    if irel_fhat<0.0:
        irel_fhat = 0.0
    elif irel_fhat>1.0:
        irel_fhat = 1.0
    ## IKur
    #f_ikur = y[46]#y[47]
    ikur_fhat = (f_ikur -  5.89379800000000009e-02) / (0.393747 -  5.89379800000000009e-02) # Effective fraction of phosphorylated IKur channels
    if ikur_fhat<0.0:
        ikur_fhat = 0.0
    elif ikur_fhat>1.0:
        ikur_fhat = 1.0

    # Applying the effective fraction of phosphorylated substrates into
    # settings struct...  Note: the commented equations below are from the
    # code downloaded from Rudy website... The ones above are from Myokit

    fICaLP = ical_f_hat#min(max(((y(98) + sigdata.ICaL_AKAP_PKA) / sigdata.ICaL_tot - (0.0269 + sigdata.ICaL_AKAP_PKA / sigdata.ICaL_tot)) / (0.9273 - (0.0269 + sigdata.ICaL_AKAP_PKA / sigdata.ICaL_tot)), 0), 1)
    fIKsP = iks_f_hat#min(max(((y(99) + sigdata.IKs_AKAP_PKA) / sigdata.IKs_tot - (0.0306 + sigdata.IKs_AKAP_PKA / sigdata.IKs_tot)) / (0.7850 - (0.0306 + sigdata.IKs_AKAP_PKA / sigdata.IKs_tot)), 0), 1)
    fPLBP = iup_f_pka#min(max((y(100) - 6.591000e-001) / (9.945000e-001 - 6.591000e-001), 0), 1)
    fTnIP = calcium_fhat#min(max((y(101) - 6.735188e-001) / (9.991797e-001 - 6.735188e-001), 0), 1)
    fINaP = ina_f_pka#min(max((y(102) - 2.394795e-001) / (9.501431e-001 - 2.394795e-001), 0), 1)
    fINaKP = inak_fhat#min(max((y(103) - 1.263453e-001) / (9.980137e-001 - 1.263453e-001), 0), 1)
    fRyRP = irel_fhat#min(max(((y(104) + sigdata.RyR_AKAP_PKA) / sigdata.RyR_tot - (0.0329 + sigdata.RyR_AKAP_PKA / sigdata.RyR_tot)) / (0.9586 - (0.0329 + sigdata.RyR_AKAP_PKA / sigdata.RyR_tot)), 0), 1)
    fIKurP = ikur_fhat#min(max((y(105) - 5.893798e-002) / (3.937470e-001 - 5.893798e-002), 0), 1)
    
    return fICaLP,fIKsP,fPLBP,fTnIP,fINaP,fINaKP,fRyRP,fIKurP
