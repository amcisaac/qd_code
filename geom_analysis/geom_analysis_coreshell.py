import numpy as np
import sys
import matplotlib.pyplot as plt
from qd_helper import *
from geom_helper import *
import copy

'''
Script to do geometry analysis of CdSe QD's and determine if any surface atoms are
undercoordinated.

Usage: python3 geom_analysis_opt_clean.py [xyz file of crystal structure] [xyz of core crystal structure] [xyz file of optimized structure] [lowdin charge file]
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
save_flag = False
save_flag=True
cutoff = 3.1 # nearest neighbor cutoff distance (lowest)
print('UC cutoff: ',cutoff)
cutoff2 = 3 # nearest neighbor cutoff distance (highest)
print('clash cutoff:',cutoff2)
nncutoff = 3  # number of nearest neighbors to be considered "unpassivated" (incl. ligands)
lig_atom= False
lig_atom_2 = False
lig_atom = "N" # atom that attaches to the Cd in the ligand
# lig_atom_2='H'

QD_file_start=sys.argv[1] # QD crystal xyz file
core_xyz=sys.argv[2]
QD_file_end=sys.argv[3]   # QD optimized xyz file
# charges_input = sys.argv[4]
# spectrum = sys.argv[5]
# savename=sys.argv[4]

QD_xyz_start,atom_name_start = read_input_xyz(QD_file_start)
core_xyz_start,core_atom_name= read_input_xyz(core_xyz)
QD_xyz_end,atom_name_end = read_input_xyz(QD_file_end)

# Eex = np.loadtxt(spectrum,delimiter=',',usecols=2,skiprows=1,dtype=float)

# UGH SO JANKY
ind_core=np.full(atom_name_start.shape,False)
for i,coord in enumerate(QD_xyz_start):
    # print(i)
    for coord2 in core_xyz_start:
        if np.all(np.isclose(coord2,coord,rtol=1e-03)):
            # print(coord)
            ind_core[i]=True

print('Number of core atoms:',np.count_nonzero(ind_core))
ind_Cd = (atom_name_start == 'Cd')
ind_Se = (atom_name_start == 'Se')
ind_S  = (atom_name_start == 'S')
ind_attach = (atom_name_start == lig_atom) # Cd ligand
ind_lig = np.logical_or(np.logical_or(np.logical_or(np.logical_or((atom_name_start=='N'), (atom_name_start == 'C')),(atom_name_start == 'H')),(atom_name_start=='Cl')),(atom_name_start=='F'))
ind_attach2 = (atom_name_start==lig_atom_2) # S ligand
# print(ind_lig)
# ind_lig = np.logical_and(np.logical_and(np.logical_not(ind_Cd),np.logical_not(ind_Se)),np.logical_not(ind_S))
# print('N lig atoms',np.count_nonzero(ind_lig))
ind_shell = np.logical_and(np.logical_not(ind_core),np.logical_not(ind_lig))
ind_chal = np.logical_or(ind_S,ind_Se)
ind_chal_lig = np.logical_or(ind_chal,ind_attach)
ind_S_lig = np.logical_or(ind_S,ind_attach)    # S + lig that binds to Cd
ind_shell_Cd = np.logical_and(ind_shell,ind_Cd)
ind_core_Cd = np.logical_and(ind_core,ind_Cd)
ind_shell_Cd_lig = np.logical_and(ind_shell_Cd,ind_attach2) # Cd + lig that binds to S

# print(ind_shell.shape)
# print(ind_core.shape)
# print(ind_chal.shape)
# print(ind_chal_lig.shape)
# print(ind_S_lig.shape)
# print(ind_shell_Cd.shape)
# print(ind_core_Cd.shape)
# print(ind_Se.shape)
# print(ind_attach.shape)

# print(np.count_nonzero(ind_shell))
# print(np.count_nonzero(ind_core))
# print(np.count_nonzero(ind_chal))
# print(np.count_nonzero(ind_chal_lig))
# print(np.count_nonzero(ind_S_lig))
# print(np.count_nonzero(ind_shell_Cd))
# print(np.count_nonzero(ind_core_Cd))
# print(np.count_nonzero(ind_Se))
# print(np.count_nonzero(ind_attach))

# print("N core atoms", np.count_nonzero(ind_core))
# print('N shell atoms',np.count_nonzero(ind_shell))
# print('N S atoms',np.count_nonzero(ind_S))
# print('N Se atoms',np.count_nonzero(ind_Se))
# ind_lig =
# ind_attach = False


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
# core only
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
# nn_histogram(QD_xyz_start,ind_S,ind_S,label1='crystal (s-s distances)',xyz2=QD_xyz_end,label2='optimized (s-s)')

# plt.show()

# np.savetxt('hist.csv',cdse_dists.flatten())

# nn_histogram(QD_xyz_start,ind_Cd,ind_Cd,label1='crystal (core/shell)',xyz2=QD_xyz_end,label2='optimized (core/shell)')
# # plt.savefig(cd_cd_dist.pdf)
# plt.ylim(0,20)
# plt.title('Cd-Cd distance')
# plt.savefig('cd_cd_dist.pdf')
#
# nn_histogram(QD_xyz_start,ind_chal,ind_chal,label1='crystal (core/shell)',xyz2=QD_xyz_end,label2='optimized (core/shell)')
# # plt.savefig(s_s)
# plt.ylim(0,20)
# plt.title('Chal-Chal distance')
# plt.savefig('chal_chal_dist.pdf')

# plt.show()

####
#
# ANALYZING GEOMETRY
#   Note: to switch between structures (crystal/opt), change dist_list
#
####

# # core/shell
dist_list = get_dists_cs(QD_xyz_end,ind_core_Cd,ind_Se,ind_shell_Cd,ind_S,ind_attach,ind_attach2)
cdcore_underc_ind,secore_underc_ind,cdcore_wshell_underc_ind,secore_wshell_underc_ind,cdshell_underc_ind,sshell_underc_ind,attach_underc_ind,attach_underc_ind2=get_underc_index_cs(ind_core_Cd,ind_Se,ind_shell_Cd,ind_S,cutoff,nncutoff,dist_list,ind_attach,ind_attach2)
cd_core_bond_ind,se_core_bond_ind,cd_cs_bond_ind,ses_cs_bond_ind,cd_shell_bond_ind,s_shell_bond_ind=get_bonded_index_cs(dist_list[0],ind_core_Cd,ind_Se,ind_shell_Cd,ind_S,cutoff2)

# separating clashing/UC
# clashing AND undercoordinated
cd_clash_uc = np.logical_and(cdshell_underc_ind,cd_shell_bond_ind)
s_clash_uc = np.logical_and(sshell_underc_ind,s_shell_bond_ind)

cd_uc_only = np.logical_xor(cd_clash_uc,cdshell_underc_ind)
s_uc_only = np.logical_xor(s_clash_uc,sshell_underc_ind)
cd_clash_only = np.logical_xor(cd_clash_uc,cd_shell_bond_ind)
s_clash_only = np.logical_xor(s_clash_uc,s_shell_bond_ind)



n_underc_cdcore_co = float(np.count_nonzero(cdcore_underc_ind))
n_underc_secore_co = float(np.count_nonzero(secore_underc_ind))
n_underc_cdshell_sc = float(np.count_nonzero(cdshell_underc_ind))
n_underc_sshell_sc = float(np.count_nonzero(sshell_underc_ind))

n_bond_cdcore = float(np.count_nonzero(cd_core_bond_ind))
n_bond_secore = float(np.count_nonzero(se_core_bond_ind))

n_bond_cd_cs = float(np.count_nonzero(cd_cs_bond_ind))
n_bond_ses_cs = float(np.count_nonzero(ses_cs_bond_ind))

n_bond_cdshell = float(np.count_nonzero(cd_shell_bond_ind))
n_bond_sshell = float(np.count_nonzero(s_shell_bond_ind))


print('Optimized geometry, bare core:')
print('Undercoordinated Cd:',np.count_nonzero(cdcore_underc_ind))
print('Undercoordinated Se:',np.count_nonzero(secore_underc_ind))
print('Clashing Cd:',n_bond_cdcore)
print('Clashing Se:',n_bond_secore)

print('')
print('Optimized geometry, core with shell:')
print('Undercoordinated Cd:',np.count_nonzero(cdcore_wshell_underc_ind))
print('Undercoordinated Se:',np.count_nonzero(secore_wshell_underc_ind))
print('Clashing Cd:',n_bond_cd_cs-n_bond_cdshell-n_bond_cdcore)
print('Clashing Se/S:',n_bond_ses_cs-n_bond_sshell-n_bond_secore)

print('')
print('Optimized geometry, shell with core:')
print('Undercoordinated Cd:',np.count_nonzero(cdshell_underc_ind))
print('Undercoordinated S :',np.count_nonzero(sshell_underc_ind))
print('Ligand 1 fallen off: ',np.count_nonzero(attach_underc_ind))
print('Ligand 2 fallen off: ',np.count_nonzero(attach_underc_ind2))
print('Clashing Cd:',n_bond_cdshell)
print('Clashing S:',n_bond_sshell)
print('Number of clashing Cd not UC', np.count_nonzero(cd_clash_only))
print('Number of clashing S not UC', np.count_nonzero(s_clash_only))
print('Number of UC Cd not clashing', np.count_nonzero(cd_uc_only))
print('Number of UC S not clashing', np.count_nonzero(s_uc_only))
print('Number of clashing & UC Cd', np.count_nonzero(cd_clash_uc))
print('Number of clashing & UC S', np.count_nonzero(s_clash_uc))



# print(attach_underc_ind.shape)
if np.any(ind_attach):
    lig_underc_ind = get_underc_ind_large(ind_attach,attach_underc_ind)
    ind_attach_lg = get_underc_ind_large(ind_attach,attach_underc_ind)
    # print(lig_underc_ind.shape)
    for i in range(0,len(ind_attach_lg)):
        if ind_attach_lg[i] == True:
            lig_underc_ind[i-1:i+6] = True

if save_flag:
    np.save('cdshell_underc_ind_3p1',get_underc_ind_large(ind_shell_Cd,cdshell_underc_ind))
    np.save('sshell_underc_ind_3p1',get_underc_ind_large(ind_S,sshell_underc_ind))

    np.save('cdshell_clash_ind_3p0',get_underc_ind_large(ind_shell_Cd,cd_shell_bond_ind))
    np.save('sshell_clash_ind_3p0',get_underc_ind_large(ind_S,s_shell_bond_ind))

    # print(np.count_nonzero(lig_underc_ind))

    # write_underc_xyz(QD_xyz_start,atom_name_end,ind_shell_Cd,ind_S,cdshell_underc_ind,sshell_underc_ind,'underc_'+str(cutoff)+'_startgeom','Undercoordinated atoms from '+QD_file_end + 'in starting geom cutoff '+str(cutoff))
    # write_underc_xyz(QD_xyz_start,atom_name_end,ind_shell_Cd,ind_S,cd_shell_bond_ind,s_shell_bond_ind,'bonded_'+str(cutoff2)+'_startgeom','Clashing atoms from '+QD_file_end + 'in starting geom cutoff '+str(cutoff2))

    write_underc_xyz(QD_xyz_end,atom_name_end,ind_shell_Cd,ind_S,cdshell_underc_ind,sshell_underc_ind,'underc_'+str(cutoff),'Undercoordinated atoms from '+QD_file_end + ' cutoff '+str(cutoff))
    write_underc_xyz(QD_xyz_end,atom_name_end,ind_shell_Cd,ind_S,cd_shell_bond_ind,s_shell_bond_ind,'bonded_'+str(cutoff2),'Clashing atoms from '+QD_file_end + ' cutoff '+str(cutoff2))
    if np.any(ind_attach):
        np.save('lig_fall_ind_3p1',lig_underc_ind)
        lig_fall_xyz = QD_xyz_end[lig_underc_ind]
        lig_fall_name = atom_name_end[lig_underc_ind]
        write_xyz('ligand_detach_{}.xyz'.format(str(cutoff)),lig_fall_name,lig_fall_xyz,'xyz coordinates of ligands that detached from '+QD_file_end+' with cutoff '+str(cutoff))

# print(QD_xyz_end[get_underc_ind_large(ind_shell_Cd,cdshell_underc_ind)])
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
# cdcore_underc_ind2,secore_underc_ind2,cdcore_wshell_underc_ind2,secore_wshell_underc_ind2,cdshell_underc_ind2,sshell_underc_ind2=get_underc_index_cs(ind_core_Cd,ind_Se,ind_shell_Cd,ind_S,cutoff2,nncutoff,dist_list)
#
# cdcore_underc_ind_amb=np.logical_xor(cdcore_underc_ind,cdcore_underc_ind2)
# secore_underc_ind_amb=np.logical_xor(secore_underc_ind,secore_underc_ind2)
# cdcore_wshell_underc_ind_amb=np.logical_xor(cdcore_wshell_underc_ind,cdcore_wshell_underc_ind2)
# secore_wshell_underc_ind_amb=np.logical_xor(secore_wshell_underc_ind,secore_wshell_underc_ind2)
# cdshell_underc_ind_amb=np.logical_xor(cdshell_underc_ind,cdshell_underc_ind2)
# sshell_underc_ind_amb=np.logical_xor(sshell_underc_ind,sshell_underc_ind2)
#
# n_underc_cdcore_co_amb = float(np.count_nonzero(cdcore_underc_ind_amb))
# n_underc_secore_co_amb = float(np.count_nonzero(secore_underc_ind_amb))
# n_underc_cdshell_sc_amb = float(np.count_nonzero(cdshell_underc_ind_amb))
# n_underc_sshell_sc_amb = float(np.count_nonzero(sshell_underc_ind_amb))
#
# # np.save('cdshell_underc_ind_2p8',get_underc_ind_large(ind_shell_Cd,cdshell_underc_ind_amb))
# # np.save('sshell_underc_ind_2p8',get_underc_ind_large(ind_S,sshell_underc_ind_amb))
#
# print('')
# print('Cutoff between ',cutoff2, 'and ',cutoff)
# print('Optimized geometry, bare core:')
# print('Undercoordinated Cd:',np.count_nonzero(cdcore_underc_ind_amb))
# print('Undercoordinated Se:',np.count_nonzero(secore_underc_ind_amb))
#
# print('')
# print('Optimized geometry, core with shell:')
# print('Undercoordinated Cd:',np.count_nonzero(cdcore_wshell_underc_ind_amb))
# print('Undercoordinated Se:',np.count_nonzero(secore_wshell_underc_ind_amb))
#
# print('')
# print('Optimized geometry, shell with core:')
# print('Undercoordinated Cd:',np.count_nonzero(cdshell_underc_ind_amb))
# print('Undercoordinated S :',np.count_nonzero(sshell_underc_ind_amb))


####
#
# CHARGE ANALYSIS
#
####

'''
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


####
#
# PLOTTING CHARGE FRACTIONS FOR ALL EXCITATIONS
# if not using undercoordination as your metric, change axis titles
#
####

nex = range(0,len(Eex))
# BARE CORE--if there were no shell
# plot_underc(nex,sum_underc_secore_co,n_underc_secore_co,n_cdse_core,n_secore,'Se',w=1,savefig=['secore_underc_co_h_3cut.pdf','secore_underc_co_e_3cut.pdf'])
# plot_underc(nex,sum_underc_cdcore_co,n_underc_cdcore_co,n_cdse_core,n_cdcore,'Cd',w=1,savefig=['cdcore_underc_co_h_3cut.pdf','cdcore_underc_co_e_3cut.pdf'])

# SHELL--including core as NN
# larger cutoff
plot_underc(Eex,sum_underc_sshell_sc,n_underc_sshell_sc,n_cds_shell,n_sshell,'S')
# plot_underc(Eex,sum_underc_cdshell_sc,n_underc_cdshell_sc,n_cds_shell,n_cdshell,'Cd')

# smaller cutoff
# plot_underc(Eex,sum_underc_sshell_sc_amb,n_underc_sshell_sc_amb,n_cds_shell,n_sshell,'S')
# plot_underc(Eex,sum_underc_cdshell_sc_amb,n_underc_cdshell_sc_amb,n_cds_shell,n_cdshell,'Cd')

#AMBIGUOUS comparison
# SHELL--including core as NN
# plot_underc_compare(nex,sum_underc_sshell_sc,sum_underc_sshell_sc_amb,n_underc_sshell_sc,n_underc_sshell_sc_amb,n_cds_shell,n_sshell,'S',w=1)#,savefig=['sshell_underc_sc_h_comparecut_2p8_3.pdf','sshell_underc_sc_e_comparecut_2p8_3.pdf'])
# plot_underc_compare(nex,sum_underc_cdshell_sc,sum_underc_cdshell_sc_amb,n_underc_cdshell_sc,n_underc_cdshell_sc_amb,n_cds_shell,n_cdshell,'Cd',w=1)#,w=1,savefig=['cdshell_underc_sc_h_comparecut_2p8_3.pdf','cdshell_underc_sc_e_comparecut_2p8_3.pdf'])

plt.show()
'''
####
#
# PRINTS INFO ABOUT SPECIFIC EXCITATIONS
#
####
# print_indiv_ex(chargefrac_tot,ind_S,sshell_underc_ind,0,'S')
