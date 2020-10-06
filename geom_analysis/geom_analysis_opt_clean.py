import numpy as np
import sys
import matplotlib.pyplot as plt
from qd_helper import *
import copy
from geom_helper import *

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

###
### USER SPECIFIED INFO
###

cutoff = 2.8  # nearest neighbor cutoff distance (lowest)
print('cutoff: ',cutoff)
cutoff2 = 3.3 # nearest neighbor cutoff distance (highest)
nncutoff = 3  # number of nearest neighbors to be considered "unpassivated" (incl. ligands)
lig_atom = "O" # atom that attaches to the Cd in the ligand

QD_file_start=sys.argv[1] # QD crystal xyz file
QD_file_end=sys.argv[2]   # QD optimized xyz file
charges_input = sys.argv[3]
# savename=sys.argv[4]

QD_xyz_start,atom_name_start = read_input_xyz(QD_file_start)
QD_xyz_end,atom_name_end = read_input_xyz(QD_file_end)
# Natoms = len(atom_name_start)

if not np.all(atom_name_start==atom_name_end):
    print("WARNING: atom ordering changed in optimization!")

# getting indices of different types of atoms
# atom order shouldn't change in optimization, so just need one set
ind_Cd, ind_Se, ind_CdSe, ind_lig, ind_selig=parse_ind(atom_name_start,lig_atom)
ind_attach = (atom_name_start == lig_atom)

####
#
# PLOT A HISTOGRAM OF NEAREST NEIGHBOR DISTANCES--to determine cutoff
#
####

nn_histogram(QD_xyz_start,ind_Cd,ind_Se,label1='crystal',ind_attach=ind_attach,xyz2=QD_xyz_end,label2='optimized')
# plt.show()
# all_dists,cdse_dists,cdlig_dists,cdselig_dists,secd_dists = get_dists(QD_xyz_end,ind_Cd,ind_Se,ind_attach)
# np.savetxt('hist.csv',cdse_dists.flatten())

####
#
# ANALYZING STARTING GEOMETRY
#
####
# '''
cd_underc_ind_s,se_underc_ind_s = get_underc_index(QD_xyz_start,ind_Cd,ind_Se,ind_lig,ind_attach,cutoff,nncutoff,verbose=False)

print('Starting geometry')
print('Undercoordinated Cd:',np.count_nonzero(cd_underc_ind_s))
print('Undercoordinated Se:',np.count_nonzero(se_underc_ind_s))
'''
beg_s = '.'.join(QD_file_start.split('.')[0:-1])
comment_s='Undercoordinated atoms from '+QD_file_start + ' cutoff '+str(cutoff)
write_underc_xyz(QD_xyz_start,atom_name_start,ind_Cd,ind_Se,cd_underc_ind_s,se_underc_ind_s,beg_s,comment_s)
'''
####
#
# ANALYZING FINAL GEOMETRY
#
####

cd_underc_ind_e,se_underc_ind_e = get_underc_index(QD_xyz_end,ind_Cd,ind_Se,ind_lig,ind_attach,cutoff,nncutoff,verbose=False)

print('Optimized geometry')
print('Undercoordinated Cd:',np.count_nonzero(cd_underc_ind_e))
print('Undercoordinated Se:',np.count_nonzero(se_underc_ind_e))
'''
beg_e = '.'.join(QD_file_end.split('.')[0:-1])
comment_e='Undercoordinated atoms from '+QD_file_end + ' cutoff '+str(cutoff)
write_underc_xyz(QD_xyz_end,atom_name_end,ind_Cd,ind_Se,cd_underc_ind_e,se_underc_ind_e,beg_e,comment_e)

beg_es = '.'.join(QD_file_end.split('.')[0:-1])+"_start"
comment_es='Undercoordinated atoms from '+QD_file_start + ' in positions from end. cutoff '+str(cutoff)
write_underc_xyz(QD_xyz_end,atom_name_end,ind_Cd,ind_Se,cd_underc_ind_s,se_underc_ind_s,beg_es,comment_es)

'''
####
# ambiguous zone:
# looks at atoms that change their number of nearest neighbors
# based on cutoff distance
####
'''
ind_amb_cd_pos,ind_amb_cd_neg,ind_amb_se_pos,ind_amb_se_neg=get_ind_dif(QD_xyz_end,ind_Cd,ind_Se,ind_lig,ind_attach,cutoff,nncutoff,cutoff2=cutoff2)
ind_amb_cd = np.logical_or(ind_amb_cd_pos,ind_amb_cd_neg)
ind_amb_se = np.logical_or(ind_amb_se_pos,ind_amb_se_neg)
beg_amb = '.'.join(QD_file_end.split('.')[0:-1])+"_amb"
comment_amb='Ambiguous atoms from '+QD_file_end + ' cutoff1 '+str(cutoff)+' cutoff2 '+str(cutoff2)
write_underc_xyz(QD_xyz_end,atom_name_end,ind_Cd,ind_Se,ind_amb_cd,ind_amb_se,beg_amb,comment_amb)
'''
####
#
# Comparing number of nearest neighbors between starting and optimized structure:
#
####
'''
ind_opt_cd_pos,ind_opt_cd_neg,ind_opt_se_pos,ind_opt_se_neg=get_ind_dif(QD_xyz_start,ind_Cd,ind_Se,ind_lig,ind_attach,cutoff,nncutoff,xyz2=QD_xyz_end)
beg_pos = '.'.join(QD_file_end.split('.')[0:-1])+"_changeopt_pos"
comment_pos='Atoms that gain NN after optimization. cutoff '+str(cutoff)
write_underc_xyz(QD_xyz_end,atom_name_end,ind_Cd,ind_Se,ind_opt_cd_pos,ind_opt_se_pos,beg_pos,comment_pos)

beg_neg = '.'.join(QD_file_end.split('.')[0:-1])+"_changeopt_neg"
comment_neg='Atoms that lose NN after optimization. cutoff '+str(cutoff)
write_underc_xyz(QD_xyz_end,atom_name_end,ind_Cd,ind_Se,ind_opt_cd_neg,ind_opt_se_neg,beg_pos,comment_neg)
'''

####
#
# CHARGE ANALYSIS
#
####

# reading in charges (same as surf vs bulk)
Charges_full=np.loadtxt(charges_input,delimiter=',',skiprows=1,dtype=str)
Charges = Charges_full[:-1,1:].astype(float)

# sum over charges
sum_charge = np.sum(Charges,axis=0)
sum_charge[np.nonzero(np.abs(sum_charge)<=1e-15)] = 1e-8 # sometimes delta is too small, replace with 1e-8
                                                         # we never use delta so shouldn't matter
# calculate charge fractions
chargefrac_tot = Charges/sum_charge

sum_underc_se_frac_reshape=sum_chargefrac(chargefrac_tot,ind_Se,se_underc_ind_e)
sum_underc_cd_frac_reshape=sum_chargefrac(chargefrac_tot,ind_Cd,cd_underc_ind_e)


# charge fraction sum for hole on undercoordinated atoms
hole_sum_frac_underc_se = sum_underc_se_frac_reshape[:,1]
hole_sum_frac_underc_cd = sum_underc_cd_frac_reshape[:,1]
# charge fraction sum for electron on undercoordinated atoms
electron_sum_frac_underc_se = sum_underc_se_frac_reshape[:,0]
electron_sum_frac_underc_cd = sum_underc_cd_frac_reshape[:,0]

# number of each type of atom
n_underc_cd = float(np.count_nonzero(cd_underc_ind_e))
n_underc_se = float(np.count_nonzero(se_underc_ind_e))
n_cdse = float(np.count_nonzero(ind_CdSe))
n_se = float(np.count_nonzero(ind_Se))
n_cd = float(np.count_nonzero(ind_Cd))
nex = int(Charges.shape[1]/3)

underc_eh_charge=np.concatenate((sum_underc_se_frac_reshape[:,:2],sum_underc_cd_frac_reshape[:,:2]),axis=1)
write_underc_charge=np.concatenate((np.array([[n_underc_se/n_se,n_underc_se/n_cdse,n_underc_cd/n_cd,n_underc_cd/n_cdse]]),underc_eh_charge))
# np.savetxt(savename,write_underc_charge,header='se_e,se_h,cd_e,cd_h',delimiter=',')

####
#
# PLOTTING CHARGE FRACTIONS FOR ALL EXCITATIONS
# if not using undercoordination as your metric, change axis titles
#
####

Eex = range(0,nex)
plot_underc(Eex,sum_underc_se_frac_reshape,n_underc_se,n_cdse,n_se,'Se',w=.5,savefig=False)
plot_underc(Eex,sum_underc_cd_frac_reshape,n_underc_cd,n_cdse,n_cd,'Cd',w=.5,savefig=False)

plt.show()


####
#
# PRINTS INFO ABOUT SPECIFIC EXCITATIONS
#
####

print_indiv_ex(chargefrac_tot,ind_Se,se_underc_ind_e,0,'Se')
