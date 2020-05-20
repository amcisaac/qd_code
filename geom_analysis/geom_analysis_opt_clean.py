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
    ind_Cd = (atom_name == "Cd")
    ind_Se = (atom_name == "Se")
    ind_CdSe = np.logical_or(ind_Cd, ind_Se)
    ind_lig = np.logical_not(ind_CdSe)  # ligand atoms are defined as anything that isn't cd or se (!)
    ind_selig = np.logical_or(ind_Se,(atom_name == lig_attach))  # NOTE: not robust! only uses N, change for other ligands
    return ind_Cd, ind_Se, ind_CdSe, ind_lig, ind_selig

def get_underc_index(xyz,ind_Cd,ind_Se,ind_lig,ind_attach,cutoff,nncutoff,verbose=False):
    all_dists,cdse_dists,cdlig_dists,cdselig_dists,secd_dists = get_dists(xyz,ind_Cd,ind_Se,ind_attach)
    Natoms = len(ind_Cd)
    all_nn,cd_nn_selig,se_nn_cd = get_nn(cdselig_dists,secd_dists,ind_Cd,ind_Se,cutoff,Natoms,ind_lig)

    cd_underc_ind = cd_nn_selig<nncutoff
    se_underc_ind = se_nn_cd<nncutoff

    if verbose:
        print('Undercoordinated Cd:',cd_nn_selig[cd_underc_ind])
        print('Undercoordinated Se:',se_nn_cd[se_underc_ind])
    return cd_underc_ind,se_underc_ind

def write_underc_xyz(xyz,atom_name,ind_Cd,ind_Se,cd_underc_ind,se_underc_ind,filestart,comment):
    cd_underc_name = atom_name[ind_Cd][cd_underc_ind]
    se_underc_name = atom_name[ind_Se][se_underc_ind]
    cd_underc_xyz = xyz[ind_Cd][cd_underc_ind]
    se_underc_xyz = xyz[ind_Se][se_underc_ind]

    write_xyz(filestart+'_se.xyz', se_underc_name, se_underc_xyz,comment)
    write_xyz(filestart+'_cd.xyz', cd_underc_name, cd_underc_xyz,comment)
    return

def get_ind_dif(xyz1,ind_Cd,ind_Se,ind_lig,ind_attach,cutoff,nncutoff,cutoff2=None,xyz2=None):
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


###
### USER SPECIFIED INFO
###

cutoff = 3.0  # nearest neighbor cutoff distance (lowest)
cutoff2 = 3.3 # nearest neighbor cutoff distance (highest)
nncutoff = 3  # number of nearest neighbors to be considered "unpassivated" (incl. ligands)
lig_atom = "N" # atom that attaches to the Cd in the ligand


QD_file_start=sys.argv[1] # QD crystal xyz file
QD_file_end=sys.argv[2]   # QD optimized xyz file

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
'''
nn_histogram(QD_xyz_start,ind_Cd,ind_Se,label1='crystal',ind_attach=ind_attach,xyz2=QD_xyz_end,label2='optimized')
'''

####
#
# ANALYZING STARTING GEOMETRY
#
####
'''
cd_underc_ind_s,se_underc_ind_s = get_underc_index(QD_xyz_start,ind_Cd,ind_Se,ind_lig,ind_attach,cutoff,nncutoff,verbose=False)

print('Starting geometry')
print('Undercoordinated Cd:',np.count_nonzero(cd_underc_ind_s))
print('Undercoordinated Se:',np.count_nonzero(se_underc_ind_s))

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
charges_input = sys.argv[3]
Charges_full=np.loadtxt(charges_input,delimiter=',',skiprows=1,dtype=str)
Charges = Charges_full[:-1,1:].astype(float)

# reshape indices of undercoordinated Se and Cd so they can index the charges
se_underc_ind_e_lg = copy.deepcopy(ind_Se)
se_underc_ind_e_lg[ind_Se] = se_underc_ind_e # USE INDICES FROM WHATEVER METHOD YOU PREFER
                                             # this is the undercoordinated at the end of the optimization

cd_underc_ind_e_lg = copy.deepcopy(ind_Cd)
cd_underc_ind_e_lg[ind_Cd] =cd_underc_ind_e # USE INDICES FROM WHATEVER METHOD YOU PREFER
                                            # this is the undercoordinated at the end of the optimization

# sum over charges
sum_charge = np.sum(Charges,axis=0)
sum_charge[np.nonzero(np.abs(sum_charge)<=1e-15)] = 1e-8 # sometimes delta is too small, replace with 1e-8
                                                         # we never use delta so shouldn't matter
# calculate charge fractions
chargefrac_tot = Charges/sum_charge
chargefrac_underc_se = chargefrac_tot[se_underc_ind_e_lg]
chargefrac_underc_cd = chargefrac_tot[cd_underc_ind_e_lg]

# sum charge fractions on undercoordinated atoms
sum_chargefrac_underc_se = np.sum(chargefrac_underc_se,axis=0)
sum_chargefrac_underc_cd = np.sum(chargefrac_underc_cd,axis=0)

# reshape so that we have an array of shape (Nex, 3) where column 0 is electron
# charge sum, column 1 is hole charge sum, and column 2 is delta (ignored)
sum_underc_se_frac_reshape = np.reshape(sum_chargefrac_underc_se,(-1,3))
sum_underc_cd_frac_reshape = np.reshape(sum_chargefrac_underc_cd,(-1,3))

# charge fraction sum for hole on undercoordinated atoms
hole_sum_frac_underc_se = sum_underc_se_frac_reshape[:,1]
hole_sum_frac_underc_cd = sum_underc_cd_frac_reshape[:,1]
# charge fraction sum for electron on undercoordinated atoms
electron_sum_frac_underc_se = sum_underc_se_frac_reshape[:,0]
electron_sum_frac_underc_cd = sum_underc_cd_frac_reshape[:,0]

# number of each type of atom
n_underc_cd = float(np.count_nonzero(cd_underc_ind_e_lg))
n_underc_se = float(np.count_nonzero(se_underc_ind_e_lg))
n_cdse = float(np.count_nonzero(ind_CdSe))
n_se = float(np.count_nonzero(ind_Se))
n_cd = float(np.count_nonzero(ind_Cd))
nex = int(Charges.shape[1]/3)

####
#
# PLOTTING CHARGE FRACTIONS FOR ALL EXCITATIONS
# if not using undercoordination as your metric, change axis titles
#
####
'''
# hole, se
plt.figure()
plt.bar(range(0,nex),hole_sum_frac_underc_se)
plt.plot([0,nex],[n_underc_se/(n_cdse),n_underc_se/(n_cdse)],'k--',label='hole evenly distributed on all cd,se')
plt.plot([0,nex],[n_underc_se/(n_se),n_underc_se/(n_se)],'r--',label='hole evenly distributed on all se')
plt.legend()
plt.xlabel('Excitation number')
plt.ylabel('Fraction of charge on {} undercoordinated Se'.format(n_underc_se))
# plt.show()


# hole, cd
plt.figure()
plt.bar(range(0,nex),hole_sum_frac_underc_cd)
plt.plot([0,nex],[n_underc_cd/(n_cdse),n_underc_cd/(n_cdse)],'k--',label='hole evenly distributed on all cd,se')
plt.plot([0,nex],[n_underc_cd/(n_cd),n_underc_cd/(n_cd)],'r--',label='hole evenly distributed on all cd')
plt.legend()
plt.xlabel('Excitation number')
plt.ylabel('Fraction of charge on {} undercoordinated Cd'.format(n_underc_cd))
# plt.show()

# electron, se
plt.figure()
plt.bar(range(0,nex),electron_sum_frac_underc_se)
plt.plot([0,nex],[n_underc_se/(n_cdse),n_underc_se/(n_cdse)],'k--',label='e evenly distributed on all cd,se')
plt.plot([0,nex],[n_underc_se/(n_se),n_underc_se/(n_se)],'r--',label='e evenly distributed on all se')
plt.legend()
plt.xlabel('Excitation number')
plt.ylabel('Fraction of charge on {} undercoordinated Se'.format(n_underc_se))
# plt.show()

# electron, cd
plt.figure()
plt.bar(range(0,nex),electron_sum_frac_underc_cd)
plt.plot([0,nex],[n_underc_cd/(n_cdse),n_underc_cd/(n_cdse)],'k--',label='e evenly distributed on all cd,se')
plt.plot([0,nex],[n_underc_cd/(n_cd),n_underc_cd/(n_cd)],'r--',label='e evenly distributed on all cd')
plt.legend()
plt.xlabel('Excitation number')
plt.ylabel('Fraction of charge on {} undercoordinated Cd'.format(n_underc_cd))
plt.show()
'''

####
#
# PRINTS INFO ABOUT SPECIFIC EXCITATIONS
#
####
'''
n=0

print('')
print('Fraction of charge on each undercoordinated Se for excitation {}:'.format(n))
print('   e           h')
print(chargefrac_underc_se[:,3*n:3*n+2])
print('')
print('Sum of charge on undercoordinated Se for excitation {}:'.format(n))
print('   e           h')
print(sum_chargefrac_underc_se[3*n:3*n+2])

max_ind = np.argmax(chargefrac_tot,axis=0) # index of the largest charge fraction on any atom
max_charge=np.max(chargefrac_tot,axis=0)   # largest charge fraction on any atom
print('')
print('Largest charge fraction on any atom for excitation {}:'.format(n))
print('   e           h')
print(max_charge[3*n:3*n+2])
print('')
print('Is the largest charge fraction on an undercoordinated Se?')
print('   e     h')
print(np.any(chargefrac_underc_se[:,3*n:3*n+2]==max_charge[3*n:3*n+2],axis=0))
# print(atom_name_start[max_ind][3*n:3*n+3]) # atom name with largest charge fraction

# creates an array (Nex, 3) where each entry is whether the max charge fraction is on an undercoordinated se
# found this wasn't useful because it's almost always on it, even for bulk excitations
max_is_underc_long = np.any(chargefrac_underc_se==max_charge,axis=0)
max_is_underc= np.reshape(max_is_underc_long,(-1,3))
# print(max_is_underc[100:120])

# finds the top 5 highest charge fractions on any atom
top5_ind = np.argpartition(-chargefrac_tot,5,axis=0)[:5] # index of top 5
top5 = np.take_along_axis(chargefrac_tot,top5_ind,axis=0) # value of top 5
print('')
print('Top 5 largest charge fractions on any atom for excitation {}:'.format(n))
print('   e           h')
print(top5[:,3*n:3*n+2])

# charge fraction on undercordinated se as a ratio of the max
# print(chargefrac_underc_se[:,3*n:3*n+3]/np.max(chargefrac_tot,axis=0)[3*n:3*n+3])
'''

# potential interesting analyses:
# -some way of measuring if max atom is near undercoordinated
