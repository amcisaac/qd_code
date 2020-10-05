import numpy as np
import sys
import matplotlib.pyplot as plt
from qd_helper import *
import copy

'''
Script to do geometry analysis of CdSe QD's and determine if any surface atoms are
undercoordinated.

Usage: python3 geom_analysis_opt_clean.py [xyz file of crystal structure] [xyz file of optimized structure]
'''

def parse_ind(atom_name,lig_attach="N"):
    '''
    Function to parse the indices for a quantum dot.

    Inputs:
        atom_name: np array with the atom names for the QD
        lig_attach: atom in the ligand that attaches to the Cd

    Returns:
        ind_Cd: boolean array of shape Natoms, indexing the Cd atoms
        ind_Se: boolean array of shape Natoms, indexing the Se atoms
        ind_CdSe: boolean array of shape Natoms, indexing both Cd and Se atoms
        ind_lig: boolean array of shape Natoms, indexing the ligand atoms (or, atoms that aren't Cd or Se)
        ind_selig: boolean array of shape Natoms, indexing the Se atoms and attach atoms
    '''
    ind_Cd = (atom_name == "Cd")
    ind_Se = (atom_name == "Se")
    ind_CdSe = np.logical_or(ind_Cd, ind_Se)
    ind_lig = np.logical_not(ind_CdSe)  # ligand atoms are defined as anything that isn't cd or se (!)
    ind_selig = np.logical_or(ind_Se,(atom_name == lig_attach))  # NOTE: not robust! only uses N, change for other ligands
    return ind_Cd, ind_Se, ind_CdSe, ind_lig, ind_selig

def get_underc_index(xyz,ind_Cd,ind_Se,ind_lig,ind_attach,cutoff,nncutoff,verbose=False):
    '''
    Function that finds undercoordinated Cd and Se atoms in a QD.

    Inputs:
        xyz: np array of xyz coordinates for the QD. shape (Natoms,3)
        ind_Cd: boolean array of shape Natoms, indexing the Cd atoms
        ind_Se: boolean array of shape Natoms, indexing the Se atoms
        ind_lig: boolean array of shape Natoms, indexing the ligand atoms
        ind_attach: boolean array indexing the ligand atoms that bind to Cd
        cutoff: cutoff for a nearest neighbor distance
        nncutoff: number of nearest neighbors to be considered "fully coordinated"
                  (< this classified as "undercoordinated")
        verbose: if True, prints the number of nearest neighbors for
                 "undercoordinated" atoms
    '''
    all_dists,cdse_dists,cdlig_dists,cdselig_dists,secd_dists = get_dists(xyz,ind_Cd,ind_Se,ind_attach)
    Natoms = len(ind_Cd)
    print(Natoms)
    all_nn,cd_nn_selig,se_nn_cd = get_nn(cdselig_dists,secd_dists,ind_Cd,ind_Se,cutoff,Natoms,ind_lig)

    print(cd_nn_selig)
    print(se_nn_cd)

    cd_underc_ind = cd_nn_selig<nncutoff
    se_underc_ind = se_nn_cd<nncutoff

    if verbose:
        print('Undercoordinated Cd:',cd_nn_selig[cd_underc_ind])
        print('Undercoordinated Se:',se_nn_cd[se_underc_ind])
    return cd_underc_ind,se_underc_ind


def get_underc_index_cs(xyz,ind_Cd_core,ind_Se,ind_Cd_shell,ind_S,cutoff,nncutoff,dist_list,verbose=False):
    '''
    Function that finds undercoordinated Cd and Se atoms in a QD.

    Inputs:
        xyz: np array of xyz coordinates for the QD. shape (Natoms,3)
        ind_Cd: boolean array of shape Natoms, indexing the Cd atoms
        ind_Se: boolean array of shape Natoms, indexing the Se atoms
        ind_lig: boolean array of shape Natoms, indexing the ligand atoms
        ind_attach: boolean array indexing the ligand atoms that bind to Cd
        cutoff: cutoff for a nearest neighbor distance
        nncutoff: number of nearest neighbors to be considered "fully coordinated"
                  (< this classified as "undercoordinated")
        verbose: if True, prints the number of nearest neighbors for
                 "undercoordinated" atoms
    '''
    # all_dists,cdse_dists,cdlig_dists,cdselig_dists,secd_dists = get_dists(xyz,ind_Cd,ind_Se,ind_attach)
    all_dists,cdse_core_dists,cdcore_ses_dist,secore_cd_dist,cdshell_ses_dist,sshell_cd_dist = dist_list #get_dists_cs(xyz,ind_Cd_core,ind_Se,ind_Cd_shell,ind_S)
    Natoms = len(ind_Cd)
    all_nn,cd_se_nn,se_cd_nn = get_nn(cdse_core_dists,cdse_core_dists,ind_Cd_core,ind_Se,cutoff,Natoms) # bare core
    all_nn,cdcore_nn_ses,se_nn_cdcd = get_nn(cdcore_ses_dist,secore_cd_dist,ind_Cd_core,ind_Se,cutoff,Natoms)    # core coord. when considering shell too
    all_nn,cdshell_nn_ses,s_nn_cdcd = get_nn(cdshell_ses_dist,sshell_cd_dist,ind_Cd_shell,ind_S,cutoff,Natoms) # undercoordinated shell atoms (includes core-shell bonds)


    # core-core
    cd_underc_ind = cd_se_nn < nncutoff
    se_underc_ind = se_cd_nn < nncutoff

    # core-core&shell
    cd_underc_inclshell_ind = cdcore_nn_ses < nncutoff
    se_underc_inclshell_ind = se_nn_cdcd < nncutoff

    # shell - core&shell
    cdshell_underc_inclcore_ind = cdshell_nn_ses < nncutoff
    s_underc_inclcore_ind = s_nn_cdcd < nncutoff

    if verbose:
        print('Undercoordinated Cd (core only):',cd_se_nn[cd_underc_ind])
        print('Undercoordinated Se (core only):',se_cd_nn[se_underc_ind])
        print('Undercoordinated Cd (core with shell):',cdcore_nn_ses[cd_underc_inclshell_ind])
        print('Undercoordinated Se (core with shell):',se_nn_cdcd[se_underc_inclshell_ind])
        print('Undercoordinated Cd (shell with core):',cdshell_nn_ses[cdshell_underc_inclcore_ind])
        print('Undercoordinated Se (shell with core):',s_nn_cdcd[s_underc_inclcore_ind])
    return cd_underc_ind,se_underc_ind,cd_underc_inclshell_ind,se_underc_inclshell_ind,cdshell_underc_inclcore_ind,s_underc_inclcore_ind

def get_ind_dif(xyz1,ind_Cd,ind_Se,ind_lig,ind_attach,cutoff,nncutoff,cutoff2=None,xyz2=None):
    '''
    Function to get indices of atoms that changed number of nearest neighbors.
    Can be in response to a different cutoff (in which case, supply cutoff2) or
    over an optimization (in which case, supply xyz2)

    Inputs:
        xyz1: np array of xyz coordinates for the QD. shape (Natoms,3)
        ind_Cd: boolean array of shape Natoms, indexing the Cd atoms
        ind_Se: boolean array of shape Natoms, indexing the Se atoms
        ind_lig: boolean array of shape Natoms, indexing the ligand atoms
        ind_attach: boolean array indexing the ligand atoms that bind to Cd
        cutoff: cutoff for a nearest neighbor distance
        nncutoff: number of nearest neighbors to be considered "fully coordinated"
                  (< this classified as "undercoordinated")
        cutoff2: (optional) second cutoff to compare
        xyz2: (optional) second set of xyz coordinates to compare

    Outputs:
        ind_change_cd_pos: boolean array indexing the cd's that gain nearest neighbors
        ind_change_cd_neg: boolean array indexing the cd's that lose nearest neighbors
        ind_change_se_pos: boolean array indexing the se's that gain nearest neighbors
        ind_change_se_neg: boolean array indexing the se's that lose nearest neighbors
    '''
    all_dists1,cdse_dists1,cdlig_dists1,cdselig_dists1,secd_dists1 = get_dists(xyz1,ind_Cd,ind_Se,ind_attach)
    Natoms = len(ind_Cd)
    all_nn1,cd_nn_selig1,se_nn_cd1 = get_nn(cdselig_dists1,secd_dists1,ind_Cd,ind_Se,cutoff,Natoms,ind_lig)
    if cutoff2:
        # distances all the same, just a different cutoff
        all_nn2,cd_nn_selig2,se_nn_cd2 = get_nn(cdselig_dists1,secd_dists1,ind_Cd,ind_Se,cutoff2,Natoms,ind_lig)
    elif np.any(xyz2):
        # different xyz, so different distances, but same cutoff
        all_dists2,cdse_dists2,cdlig_dists2,cdselig_dists2,secd_dists2 = get_dists(xyz2,ind_Cd,ind_Se,ind_attach)
        all_nn2,cd_nn_selig2,se_nn_cd2 = get_nn(cdselig_dists2,secd_dists2,ind_Cd,ind_Se,cutoff,Natoms,ind_lig)

    cd_underc_ind1 = cd_nn_selig1<nncutoff
    se_underc_ind1 = se_nn_cd1<nncutoff
    cd_underc_ind2 = cd_nn_selig2<nncutoff
    se_underc_ind2 = se_nn_cd2<nncutoff

    nn_change_cd = cd_nn_selig2 - cd_nn_selig1
    nn_change_se = se_nn_cd2 - se_nn_cd1

    ind_change_cd_pos = nn_change_cd > 0
    ind_change_cd_neg = nn_change_cd < 0
    ind_change_se_pos = nn_change_se > 0
    ind_change_se_neg = nn_change_se < 0

    return ind_change_cd_pos,ind_change_cd_neg,ind_change_se_pos,ind_change_se_neg

def write_underc_xyz(xyz,atom_name,ind_Cd,ind_Se,cd_underc_ind,se_underc_ind,filestart,comment):
    '''
    Function to write the coordinates of undercoordinated atoms.

    Inputs:
        xyz: np array of xyz coordinates for the QD. shape (Natoms,3)
        atom_name: array of atom names that correspond to xyz
        ind_Cd: boolean array of shape Natoms, indexing the Cd atoms
        ind_Se: boolean array of shape Natoms, indexing the Se atoms
        cd_underc_ind: boolean array corresponding to atom_name, indexing
                       undercoordinated Cd's
        se_underc_ind: boolean array corresponding to atom_name, indexing
                       undercoordinated Se's
        filestart: most of the descriptive file name for the coordinates.
                       will have '_se.xyz' or '_cd.xyz' appended to it
        comment: comment for the xyz files
    Outputs:
        writes two xyz files: {filestart}_se.xyz with the coordinates of
        undercoordinated se's, and {filestart}_cd.xyz, with undercoordinated cd's
    '''
    cd_underc_name = atom_name[ind_Cd][cd_underc_ind]
    se_underc_name = atom_name[ind_Se][se_underc_ind]
    cd_underc_xyz = xyz[ind_Cd][cd_underc_ind]
    se_underc_xyz = xyz[ind_Se][se_underc_ind]

    write_xyz(filestart+'_se.xyz', se_underc_name, se_underc_xyz,comment)
    write_xyz(filestart+'_cd.xyz', cd_underc_name, cd_underc_xyz,comment)
    return

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

def get_underc_ind_large(ind_orig,ind_underc):
    '''
    Returns index for undercoordinated atom type, with the dimensions of the original number of atoms
    e.g. under coordinated Se index for Cd33Se33 will be len 33, this will return len 66 for use with other properties
    '''
    large_underc_ind = copy.deepcopy(ind_orig)
    large_underc_ind[ind_orig] = ind_underc # USE INDICES FROM WHATEVER METHOD YOU PREFER
                                            # this is the undercoordinated at the end of the optimization
    return large_underc_ind

def sum_chargefrac(chargefrac_tot,ind_orig,ind_underc):
    large_underc_ind = get_underc_ind_large(ind_orig,ind_underc)
    chargefrac_underc = chargefrac_tot[large_underc_ind]
    sum_chargefrac_underc= np.sum(chargefrac_underc,axis=0)
    sum_underc_reshape = np.reshape(sum_chargefrac_underc,(-1,3))
    return sum_underc_reshape

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
#
# ####
# #
# # ANALYZING STARTING GEOMETRY
# #
# ####
# # '''
# cd_underc_ind_s,se_underc_ind_s = get_underc_index(QD_xyz_start,ind_Cd,ind_Se,ind_lig,ind_attach,cutoff,nncutoff,verbose=False)
#
# print('Starting geometry')
# print('Undercoordinated Cd:',np.count_nonzero(cd_underc_ind_s))
# print('Undercoordinated Se:',np.count_nonzero(se_underc_ind_s))
# '''
# beg_s = '.'.join(QD_file_start.split('.')[0:-1])
# comment_s='Undercoordinated atoms from '+QD_file_start + ' cutoff '+str(cutoff)
# write_underc_xyz(QD_xyz_start,atom_name_start,ind_Cd,ind_Se,cd_underc_ind_s,se_underc_ind_s,beg_s,comment_s)
# '''
# ####
# #
# # ANALYZING FINAL GEOMETRY
# #
# ####

# # core/shell
dist_list = get_dists_cs(QD_xyz_end,ind_core_Cd,ind_Se,ind_shell_Cd,ind_S)
cdcore_underc_ind,secore_underc_ind,cdcore_wshell_underc_ind,secore_wshell_underc_ind,cdshell_underc_ind,sshell_underc_ind=get_underc_index_cs(QD_xyz_end,ind_core_Cd,ind_Se,ind_shell_Cd,ind_S,cutoff,nncutoff,dist_list)


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
# ####
# # ambiguous zone:
# # looks at atoms that change their number of nearest neighbors
# # based on cutoff distance
# ####
# cdcore_underc_ind1,secore_underc_ind1,cdcore_wshell_underc_ind1,secore_wshell_underc_ind1,cdshell_underc_ind1,sshell_underc_ind1=get_underc_index_cs(QD_xyz_end,ind_core_Cd,ind_Se,ind_shell_Cd,ind_S,cutoff,nncutoff,dist_list)
cdcore_underc_ind2,secore_underc_ind2,cdcore_wshell_underc_ind2,secore_wshell_underc_ind2,cdshell_underc_ind2,sshell_underc_ind2=get_underc_index_cs(QD_xyz_end,ind_core_Cd,ind_Se,ind_shell_Cd,ind_S,cutoff2,nncutoff,dist_list)

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

np.save('cdshell_underc_ind_2p8',get_underc_ind_large(ind_shell_Cd,cdshell_underc_ind_amb))
np.save('sshell_underc_ind_2p8',get_underc_ind_large(ind_S,sshell_underc_ind_amb))

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

# '''
# ind_amb_cd_pos,ind_amb_cd_neg,ind_amb_se_pos,ind_amb_se_neg=get_ind_dif(QD_xyz_end,ind_Cd,ind_Se,ind_lig,ind_attach,cutoff,nncutoff,cutoff2=cutoff2)
# ind_amb_cd = np.logical_or(ind_amb_cd_pos,ind_amb_cd_neg)
# ind_amb_se = np.logical_or(ind_amb_se_pos,ind_amb_se_neg)
# beg_amb = '.'.join(QD_file_end.split('.')[0:-1])+"_amb"
# comment_amb='Ambiguous atoms from '+QD_file_end + ' cutoff1 '+str(cutoff)+' cutoff2 '+str(cutoff2)
# write_underc_xyz(QD_xyz_end,atom_name_end,ind_Cd,ind_Se,ind_amb_cd,ind_amb_se,beg_amb,comment_amb)
# '''
# ####
# #
# # Comparing number of nearest neighbors between starting and optimized structure:
# #
# ####
# '''
# ind_opt_cd_pos,ind_opt_cd_neg,ind_opt_se_pos,ind_opt_se_neg=get_ind_dif(QD_xyz_start,ind_Cd,ind_Se,ind_lig,ind_attach,cutoff,nncutoff,xyz2=QD_xyz_end)
# beg_pos = '.'.join(QD_file_end.split('.')[0:-1])+"_changeopt_pos"
# comment_pos='Atoms that gain NN after optimization. cutoff '+str(cutoff)
# write_underc_xyz(QD_xyz_end,atom_name_end,ind_Cd,ind_Se,ind_opt_cd_pos,ind_opt_se_pos,beg_pos,comment_pos)
#
# beg_neg = '.'.join(QD_file_end.split('.')[0:-1])+"_changeopt_neg"
# comment_neg='Atoms that lose NN after optimization. cutoff '+str(cutoff)
# write_underc_xyz(QD_xyz_end,atom_name_end,ind_Cd,ind_Se,ind_opt_cd_neg,ind_opt_se_neg,beg_pos,comment_neg)
# '''
#
# ####
# #
# # CHARGE ANALYSIS
# #
# ####
#

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

print(np.all(np.isclose(sum_underc_sshell_sc_amb,sum_underc_sshell_sc)))

#
# underc_eh_charge_cdcoresecore_co=np.concatenate((sum_underc_secore_co_frac_reshape[:,:2],sum_underc_cdcore_co_frac_reshape[:,:2]),axis=1)
# write_underc_charge=np.concatenate((np.array([[n_underc_se/n_se,n_underc_se/n_cdse,n_underc_cd/n_cd,n_underc_cd/n_cdse]]),underc_eh_charge))
# # np.savetxt(savename,write_underc_charge,header='se_e,se_h,cd_e,cd_h',delimiter=',')
#
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
# plot_underc(Eex,sum_underc_sshell_sc,n_underc_sshell_sc,n_cds_shell,n_sshell,'S')
# plot_underc(Eex,sum_underc_cdshell_sc,n_underc_cdshell_sc,n_cds_shell,n_cdshell,'Cd')

# plot_underc(Eex,sum_underc_sshell_sc_amb,n_underc_sshell_sc_amb,n_cds_shell,n_sshell,'S')
# plot_underc(Eex,sum_underc_cdshell_sc_amb,n_underc_cdshell_sc_amb,n_cds_shell,n_cdshell,'Cd')
#

#AMBIGUOUS
# SHELL--including core as NN
# plot_underc_compare(nex,sum_underc_sshell_sc,sum_underc_sshell_sc_amb,n_underc_sshell_sc,n_underc_sshell_sc_amb,n_cds_shell,n_sshell,'S',w=1)#,savefig=['sshell_underc_sc_h_comparecut_2p8_3.pdf','sshell_underc_sc_e_comparecut_2p8_3.pdf'])
# plot_underc_compare(nex,sum_underc_cdshell_sc,sum_underc_cdshell_sc_amb,n_underc_cdshell_sc,n_underc_cdshell_sc_amb,n_cds_shell,n_cdshell,'Cd',w=1)#,w=1,savefig=['cdshell_underc_sc_h_comparecut_2p8_3.pdf','cdshell_underc_sc_e_comparecut_2p8_3.pdf'])

# plt.show()

# print(np.count_nonzero(sum_underc_sshell_sc_amb[:,0:2] > sum_underc_sshell_sc[:,0:2],axis=0))

#
# ####
# #
# # PRINTS INFO ABOUT SPECIFIC EXCITATIONS
# #
# ####
# '''
# n=0
#
# print('')
# print('Fraction of charge on each undercoordinated Se for excitation {}:'.format(n))
# print('   e           h')
# print(chargefrac_underc_se[:,3*n:3*n+2])
# print('')
# print('Sum of charge on undercoordinated Se for excitation {}:'.format(n))
# print('   e           h')
# print(sum_chargefrac_underc_se[3*n:3*n+2])
#
# max_ind = np.argmax(chargefrac_tot,axis=0) # index of the largest charge fraction on any atom
# max_charge=np.max(chargefrac_tot,axis=0)   # largest charge fraction on any atom
# print('')
# print('Largest charge fraction on any atom for excitation {}:'.format(n))
# print('   e           h')
# print(max_charge[3*n:3*n+2])
# print('')
# print('Is the largest charge fraction on an undercoordinated Se?')
# print('   e     h')
# print(np.any(chargefrac_underc_se[:,3*n:3*n+2]==max_charge[3*n:3*n+2],axis=0))
# # print(atom_name_start[max_ind][3*n:3*n+3]) # atom name with largest charge fraction
#
# # creates an array (Nex, 3) where each entry is whether the max charge fraction is on an undercoordinated se
# # found this wasn't useful because it's almost always on it, even for bulk excitations
# max_is_underc_long = np.any(chargefrac_underc_se==max_charge,axis=0)
# max_is_underc= np.reshape(max_is_underc_long,(-1,3))
# # print(max_is_underc[100:120])
#
# # finds the top 5 highest charge fractions on any atom
# top5_ind = np.argpartition(-chargefrac_tot,5,axis=0)[:5] # index of top 5
# top5 = np.take_along_axis(chargefrac_tot,top5_ind,axis=0) # value of top 5
# print('')
# print('Top 5 largest charge fractions on any atom for excitation {}:'.format(n))
# print('   e           h')
# print(top5[:,3*n:3*n+2])
#
# # charge fraction on undercordinated se as a ratio of the max
# # print(chargefrac_underc_se[:,3*n:3*n+3]/np.max(chargefrac_tot,axis=0)[3*n:3*n+3])
# '''
#
# # potential interesting analyses:
# # -some way of measuring if max atom is near undercoordinated
#
