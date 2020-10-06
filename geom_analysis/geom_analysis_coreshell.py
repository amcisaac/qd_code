import numpy as np
import sys
import matplotlib.pyplot as plt
from qd_helper import *
from geom_helper import *
import copy

'''
Script to do geometry analysis of CdSe QD's and determine if any surface atoms are
undercoordinated.

Usage: python3 geom_analysis_opt_clean.py [xyz file of crystal structure] [xyz file of optimized structure]
'''
def plot_underc(Eex,sum_frac,n_underc1,n_atomtot,n_atom1,atomname,w=0.01,savefig=False):
    # nex = Eex[-1]
    hole_sum_frac = sum_frac[:,1]
    e_sum_frac = sum_frac[:,0]

    plt.figure(figsize=(10,5))
    plt.bar(Eex,hole_sum_frac,width=w,color='turquoise')
    plt.plot([Eex[0],Eex[-1]],[n_underc1/(n_atomtot),n_underc1/(n_atomtot)],'k--',label='hole evenly distributed on all atoms')
    plt.plot([Eex[0],Eex[-1]],[n_underc1/(n_atom1),n_underc1/(n_atom1)],'r--',label='hole evenly distributed on all {}'.format(atomname))
    plt.legend()
    plt.xlabel('Excitation number')
    plt.ylabel('Fraction of charge on {} undercoordinated {}'.format(n_underc1,atomname))
    if savefig: plt.savefig(savefig[0])

    plt.figure(figsize=(10,5))
    plt.bar(Eex,e_sum_frac,width=w,color='turquoise')
    plt.plot([Eex[0],Eex[-1]],[n_underc1/(n_atomtot),n_underc1/(n_atomtot)],'k--',label='e evenly distributed on all atoms')
    plt.plot([Eex[0],Eex[-1]],[n_underc1/(n_atom1),n_underc1/(n_atom1)],'r--',label='e evenly distributed on all {}'.format(atomname))
    plt.legend()
    plt.xlabel('Excitation number')
    plt.ylabel('Fraction of charge on {} undercoordinated {}'.format(n_underc1,atomname))
    if savefig: plt.savefig(savefig[1])
    return

def plot_underc_compare(Eex,sum_frac,sum_frac2,n_underc1,n_underc2,n_atomtot,n_atom1,atomname,savefig=False,w=0.01,a=1):
    hole_sum_frac = sum_frac[:,1]
    e_sum_frac = sum_frac[:,0]

    hole_sum_frac2 = sum_frac2[:,1]
    e_sum_frac2 = sum_frac2[:,0]

    n_underc = n_underc1+n_underc2

    plt.figure(figsize=(10,5))
    plt.bar(Eex,hole_sum_frac2+hole_sum_frac,width=w,label='smaller cutoff',color='darkviolet') # all atoms underc w/ cutoff = 2.8
    plt.bar(Eex,hole_sum_frac,width=w,label='larger cutoff', color='turquoise',alpha=a) # subset of atoms that are still underc w/ cutoff = 3
    plt.plot([Eex[0],Eex[-1]],[n_underc/(n_atomtot),n_underc/(n_atomtot)],'m--')#,label='evenly distributed, small cutoff')
    plt.plot([Eex[0],Eex[-1]],[n_underc1/(n_atomtot),n_underc1/(n_atomtot)],'b--')#,label='evenly distributed, larger cutoff')
    plt.legend()
    plt.xlabel('Excitation number')
    plt.ylabel('Fraction of charge on {} undercoordinated {}'.format(n_underc,atomname))
    plt.title('hole, {}'.format(atomname))
    if savefig: plt.savefig(savefig[0])

    plt.figure(figsize=(10,5))
    plt.bar(Eex,e_sum_frac+e_sum_frac2,width=w,label='smaller cutoff',color='darkviolet')
    plt.bar(Eex,e_sum_frac,width=w,label='larger cutoff', color='turquoise',alpha=a)
    plt.plot([Eex[0],Eex[-1]],[n_underc/(n_atomtot),n_underc/(n_atomtot)],'m--')#,label='evenly distributed, small cutoff')
    plt.plot([Eex[0],Eex[-1]],[n_underc2/(n_atomtot),n_underc2/(n_atomtot)],'b--')#,label='evenly distributed, larger cutoff')
    plt.legend()
    plt.xlabel('Excitation number')
    plt.ylabel('Fraction of charge on {} undercoordinated {}'.format(n_underc,atomname))
    plt.title('electron, {}'.format(atomname))
    if savefig: plt.savefig(savefig[1])
    return

###
### USER SPECIFIED INFO
###

cutoff = 3.0 # nearest neighbor cutoff distance (lowest)
print('cutoff: ',cutoff)
cutoff2 = 2.8 # nearest neighbor cutoff distance (highest)
nncutoff = 3  # number of nearest neighbors to be considered "unpassivated" (incl. ligands)
lig_atom = "O" # atom that attaches to the Cd in the ligand

QD_file_start=sys.argv[1] # QD crystal xyz file
core_xyz=sys.argv[2]
QD_file_end=sys.argv[3]   # QD optimized xyz file
charges_input = sys.argv[4]
spectrum = sys.argv[5]
# savename=sys.argv[4]

QD_xyz_start,atom_name_start = read_input_xyz(QD_file_start)
core_xyz_start,core_atom_name= read_input_xyz(core_xyz)
QD_xyz_end,atom_name_end = read_input_xyz(QD_file_end)

Eex = np.loadtxt(spectrum,delimiter=',',usecols=2,skiprows=1,dtype=float)

# UGH SO JANKY
ind_core=np.full(atom_name_start.shape,False)
for i,coord in enumerate(QD_xyz_start):
    # print(i)
    for coord2 in core_xyz_start:
        if np.all(coord2==coord):
            # print(coord)
            ind_core[i]=True

ind_shell = np.logical_not(ind_core)
ind_Cd = (atom_name_start == 'Cd')
ind_Se = (atom_name_start == 'Se')
ind_S  = (atom_name_start == 'S')
ind_chal = np.logical_or(ind_S,ind_Se)
ind_shell_Cd = np.logical_and(ind_shell,ind_Cd)
ind_core_Cd = np.logical_and(ind_core,ind_Cd)
ind_lig = False
ind_attach = False


n_cdse_core = float(np.count_nonzero(ind_core))
n_secore = float(np.count_nonzero(ind_Se))
n_cdcore = float(np.count_nonzero(ind_core_Cd))
n_cds_shell = float(np.count_nonzero(ind_shell))
n_sshell = float(np.count_nonzero(ind_S))
n_cdshell = float(np.count_nonzero(ind_shell_Cd))




# write_xyz('core_opt.xyz',atom_name_start[ind_core],QD_xyz_end[ind_core])

# ####
# #
# # PLOT A HISTOGRAM OF NEAREST NEIGHBOR DISTANCES--to determine cutoff
# #
# ####
#
# # core only
# nn_histogram(QD_xyz_start,ind_core_Cd,ind_Se,label1='crystal (core)',xyz2=QD_xyz_end,label2='optimized (core)')
#
# #shell only
# nn_histogram(QD_xyz_start,ind_shell_Cd,ind_S,label1='crystal (shell)',xyz2=QD_xyz_end,label2='optimized (shell)')
#
# # core shell
# nn_histogram(QD_xyz_start,ind_Cd,ind_chal,label1='crystal (core/shell)',xyz2=QD_xyz_end,label2='optimized (core/shell)')
#
# # cd (shell) to se (core)
# nn_histogram(QD_xyz_start,ind_shell_Cd,ind_Se,label1='crystal (se core-cd shell)',xyz2=QD_xyz_end,label2='optimized (core/shell)')
#
# #cd (core) to s (shell)
# nn_histogram(QD_xyz_start,ind_core_Cd,ind_S,label1='crystal (cd core-s shell)',xyz2=QD_xyz_end,label2='optimized (core/shell)')
#
# plt.show()

# np.savetxt('hist.csv',cdse_dists.flatten())

####
#
# ANALYZING GEOMETRY
#   Note: to switch between structures (crystal/opt), change dist_list
#
####

# # core/shell
dist_list = get_dists_cs(QD_xyz_end,ind_core_Cd,ind_Se,ind_shell_Cd,ind_S)
cdcore_underc_ind,secore_underc_ind,cdcore_wshell_underc_ind,secore_wshell_underc_ind,cdshell_underc_ind,sshell_underc_ind=get_underc_index_cs(ind_core_Cd,ind_Se,ind_shell_Cd,ind_S,cutoff,nncutoff,dist_list)


np.save('cdshell_underc_ind_3',get_underc_ind_large(ind_shell_Cd,cdshell_underc_ind) )
np.save('sshell_underc_ind_3',get_underc_ind_large(ind_S,sshell_underc_ind))

n_underc_cdcore_co = float(np.count_nonzero(cdcore_underc_ind))
n_underc_secore_co = float(np.count_nonzero(secore_underc_ind))
n_underc_cdshell_sc = float(np.count_nonzero(cdshell_underc_ind))
n_underc_sshell_sc = float(np.count_nonzero(sshell_underc_ind))

print('Optimized geometry, bare core:')
print('Undercoordinated Cd:',np.count_nonzero(cdcore_underc_ind))
print('Undercoordinated Se:',np.count_nonzero(secore_underc_ind))

print('')
print('Optimized geometry, core with shell:')
print('Undercoordinated Cd:',np.count_nonzero(cdcore_wshell_underc_ind))
print('Undercoordinated Se:',np.count_nonzero(secore_wshell_underc_ind))

print('')
print('Optimized geometry, shell with core:')
print('Undercoordinated Cd:',np.count_nonzero(cdshell_underc_ind))
print('Undercoordinated S :',np.count_nonzero(sshell_underc_ind))
# # '''
# beg_e = '.'.join(QD_file_end.split('.')[0:-1])
# comment_e='Undercoordinated atoms from '+QD_file_end + ' cutoff '+str(cutoff)
# write_underc_xyz(QD_xyz_end,atom_name_end,ind_shell_Cd,ind_S,cdshell_underc_ind,sshell_underc_ind,beg_e,comment_e)
#
# beg_es = '.'.join(QD_file_end.split('.')[0:-1])+"_start"
# comment_es='Undercoordinated atoms from '+QD_file_start + ' in positions from end. cutoff '+str(cutoff)
# write_underc_xyz(QD_xyz_end,atom_name_end,ind_Cd,ind_Se,cd_underc_ind_s,se_underc_ind_s,beg_es,comment_es)
#
# '''


####
#
# COMPARING TWO STRUCTURES/CUTOFFS
# looks at atoms that change their number of nearest neighbors
# based on cutoff distance or optimization
#
####

# to change to pre/post opt, use a new dist_list with same cutoff
cdcore_underc_ind2,secore_underc_ind2,cdcore_wshell_underc_ind2,secore_wshell_underc_ind2,cdshell_underc_ind2,sshell_underc_ind2=get_underc_index_cs(ind_core_Cd,ind_Se,ind_shell_Cd,ind_S,cutoff2,nncutoff,dist_list)

cdcore_underc_ind_amb=np.logical_xor(cdcore_underc_ind,cdcore_underc_ind2)
secore_underc_ind_amb=np.logical_xor(secore_underc_ind,secore_underc_ind2)
cdcore_wshell_underc_ind_amb=np.logical_xor(cdcore_wshell_underc_ind,cdcore_wshell_underc_ind2)
secore_wshell_underc_ind_amb=np.logical_xor(secore_wshell_underc_ind,secore_wshell_underc_ind2)
cdshell_underc_ind_amb=np.logical_xor(cdshell_underc_ind,cdshell_underc_ind2)
sshell_underc_ind_amb=np.logical_xor(sshell_underc_ind,sshell_underc_ind2)

n_underc_cdcore_co_amb = float(np.count_nonzero(cdcore_underc_ind_amb))
n_underc_secore_co_amb = float(np.count_nonzero(secore_underc_ind_amb))
n_underc_cdshell_sc_amb = float(np.count_nonzero(cdshell_underc_ind_amb))
n_underc_sshell_sc_amb = float(np.count_nonzero(sshell_underc_ind_amb))

# np.save('cdshell_underc_ind_2p8',get_underc_ind_large(ind_shell_Cd,cdshell_underc_ind_amb))
# np.save('sshell_underc_ind_2p8',get_underc_ind_large(ind_S,sshell_underc_ind_amb))

print('')
print('Cutoff between ',cutoff2, 'and ',cutoff)
print('Optimized geometry, bare core:')
print('Undercoordinated Cd:',np.count_nonzero(cdcore_underc_ind_amb))
print('Undercoordinated Se:',np.count_nonzero(secore_underc_ind_amb))

print('')
print('Optimized geometry, core with shell:')
print('Undercoordinated Cd:',np.count_nonzero(cdcore_wshell_underc_ind_amb))
print('Undercoordinated Se:',np.count_nonzero(secore_wshell_underc_ind_amb))

print('')
print('Optimized geometry, shell with core:')
print('Undercoordinated Cd:',np.count_nonzero(cdshell_underc_ind_amb))
print('Undercoordinated S :',np.count_nonzero(sshell_underc_ind_amb))


####
#
# CHARGE ANALYSIS
#
####


# # reading in charges (same as surf vs bulk)
Charges_full=np.loadtxt(charges_input,delimiter=',',skiprows=1,dtype=str)
Charges = Charges_full[:-1,1:].astype(float)

# # sum over charges
sum_charge = np.sum(Charges,axis=0)
sum_charge[np.nonzero(np.abs(sum_charge)<=1e-15)] = 1e-8 # sometimes delta is too small, replace with 1e-8
                                                         # we never use delta so shouldn't matter
# # calculate charge fractions
chargefrac_tot = Charges/sum_charge

# charge fraction sum for hole on undercoordinated atoms
# PLAIN CUTOFF
sum_underc_sshell_sc = sum_chargefrac(chargefrac_tot,ind_S,sshell_underc_ind)
sum_underc_cdshell_sc = sum_chargefrac(chargefrac_tot,ind_shell_Cd,cdshell_underc_ind)

sum_underc_secore_co = sum_chargefrac(chargefrac_tot,ind_Se,secore_underc_ind)
sum_underc_cdcore_co = sum_chargefrac(chargefrac_tot,ind_core_Cd,cdcore_underc_ind)

sum_underc_secore_cs = sum_chargefrac(chargefrac_tot,ind_Se,secore_wshell_underc_ind)
sum_underc_cdcore_cs = sum_chargefrac(chargefrac_tot,ind_core_Cd,cdcore_wshell_underc_ind)


#AMBIGUOUS--BETWEEN 2 CUTOFFS
sum_underc_sshell_sc_amb = sum_chargefrac(chargefrac_tot,ind_S,sshell_underc_ind_amb)
sum_underc_cdshell_sc_amb = sum_chargefrac(chargefrac_tot,ind_shell_Cd,cdshell_underc_ind_amb)

sum_underc_secore_co_amb = sum_chargefrac(chargefrac_tot,ind_Se,secore_underc_ind_amb)
sum_underc_cdcore_co_amb = sum_chargefrac(chargefrac_tot,ind_core_Cd,cdcore_underc_ind_amb)

sum_underc_secore_cs_amb = sum_chargefrac(chargefrac_tot,ind_Se,secore_wshell_underc_ind_amb)
sum_underc_cdcore_cs_amb = sum_chargefrac(chargefrac_tot,ind_core_Cd,cdcore_wshell_underc_ind_amb)


# ####
# #
# # PLOTTING CHARGE FRACTIONS FOR ALL EXCITATIONS
# # if not using undercoordination as your metric, change axis titles
# #
# ####
# '''
nex = range(0,len(Eex))
# BARE CORE--if there were no shell

# plot_underc(nex,sum_underc_secore_co,n_underc_secore_co,n_cdse_core,n_secore,'Se',w=1,savefig=['secore_underc_co_h_3cut.pdf','secore_underc_co_e_3cut.pdf'])
# plot_underc(nex,sum_underc_cdcore_co,n_underc_cdcore_co,n_cdse_core,n_cdcore,'Cd',w=1,savefig=['cdcore_underc_co_h_3cut.pdf','cdcore_underc_co_e_3cut.pdf'])
#

# SHELL--including core as NN
plot_underc(Eex,sum_underc_sshell_sc,n_underc_sshell_sc,n_cds_shell,n_sshell,'S')
# plot_underc(Eex,sum_underc_cdshell_sc,n_underc_cdshell_sc,n_cds_shell,n_cdshell,'Cd')

# plot_underc(Eex,sum_underc_sshell_sc_amb,n_underc_sshell_sc_amb,n_cds_shell,n_sshell,'S')
# plot_underc(Eex,sum_underc_cdshell_sc_amb,n_underc_cdshell_sc_amb,n_cds_shell,n_cdshell,'Cd')
#

#AMBIGUOUS
# SHELL--including core as NN
# plot_underc_compare(nex,sum_underc_sshell_sc,sum_underc_sshell_sc_amb,n_underc_sshell_sc,n_underc_sshell_sc_amb,n_cds_shell,n_sshell,'S',w=1)#,savefig=['sshell_underc_sc_h_comparecut_2p8_3.pdf','sshell_underc_sc_e_comparecut_2p8_3.pdf'])
# plot_underc_compare(nex,sum_underc_cdshell_sc,sum_underc_cdshell_sc_amb,n_underc_cdshell_sc,n_underc_cdshell_sc_amb,n_cds_shell,n_cdshell,'Cd',w=1)#,w=1,savefig=['cdshell_underc_sc_h_comparecut_2p8_3.pdf','cdshell_underc_sc_e_comparecut_2p8_3.pdf'])

plt.show()

# print(np.count_nonzero(sum_underc_sshell_sc_amb[:,0:2] > sum_underc_sshell_sc[:,0:2],axis=0))

#
# ####
# #
# # PRINTS INFO ABOUT SPECIFIC EXCITATIONS
# #
# ####
# print_indiv_ex(chargefrac_tot,ind_S,sshell_underc_ind,0,'S')
