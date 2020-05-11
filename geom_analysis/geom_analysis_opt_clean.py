import numpy as np
import sys
import matplotlib.pyplot as plt

'''
Script to do geometry analysis of CdSe QD's and determine if any surface atoms are
undercoordinated.

Usage: python3 geom_analysis_opt_clean.py [xyz file of crystal structure] [xyz file of optimized structure]
'''


def read_input_xyz(input_xyz):
    '''
    Function that reads xyz file into arrays

    Inputs: input_xyz -- .xyz file with the QD coordinates
    Outputs: xyz_coords -- np array with the coordinates of all the atoms (float)
             atom_names -- np array with the atom names (str)
    '''
    xyz_coords = np.loadtxt(input_xyz,skiprows=2,usecols=(1,2,3))
    atom_names = np.loadtxt(input_xyz,skiprows=2,usecols=(0,),dtype=str)
    return xyz_coords, atom_names

def write_xyz(out_file, atom_names, atom_xyz, comment=''):
    '''
    Function that writes xyz coordinate arrays to .xyz file

    Inputs: out_file   -- name of the file to write coordinates to
            atom_names -- np array or list of atom names (str)
            atom_xyz   -- np array or list of atom xyz coordinates (float)
            comment    -- comment line to write in the xyz file

    Outputs: Writes to xyz_file
    '''
    with open(out_file,'w') as xyz_file:
        xyz_file.write(str(len(atom_names))+'\n')
        xyz_file.write(comment+'\n')
        for i, atom in enumerate(atom_names):
            xyz_file.write(atom +'    '+ str(atom_xyz[i][0])+'    '+str(atom_xyz[i][1])+'    '+str(atom_xyz[i][2])+'\n')
    return

def dist_all_points(xyz):
     # return np array of all distances for each atom
     dists = [] # list to return with format above
     for atom in xyz: # xyz = for each cd
         dist = np.sqrt(np.sum((atom - xyz)**2,axis=1)) # calc dist between cd(i) and all se's
         dists.append(dist) # add dist to list

     return np.array(dists)#, np.array(N_nns)

def dist_atom12(all_dists,ind_1,ind_2):
    return all_dists[ind_1].T[ind_2].T

def num_nn(dist_list,cutoff):
    # returns list of number of NN for each atom
    # could get xyz coordinates from the cd_se_dists_all < cutoff indices i think
    return np.sum(dist_list < cutoff,axis=1)

def parse_ind(atom_name,lig_attach="N"):
    ind_Cd = (atom_name == "Cd")
    ind_Se = (atom_name == "Se")
    ind_CdSe = np.logical_or(ind_Cd, ind_Se)
    ind_lig = np.logical_not(ind_CdSe)  # ligand atoms are defined as anything that isn't cd or se (!)
    ind_selig = np.logical_or(ind_Se,(atom_name == lig_attach))  # NOTE: not robust! only uses N, change for other ligands
    return ind_Cd, ind_Se, ind_CdSe, ind_lig, ind_selig

def nn_analysis(QD_xyz,atom_name,ind1,ind2,cutoff,nn_cutoff):
    dists,nn=nearest_neighbor_cdse(QD_xyz[ind1], QD_xyz[ind2], cutoff)
    undercoord_ind = nn < nn_cutoff
    undercoord_xyz = QD_xyz[ind1][undercoord_ind]
    undercoord_name = atom_name[ind1][undercoord_ind]
    return dists,nn,undercoord_ind,undercoord_xyz,undercoord_name

def get_dists(QD_xyz,ind_Cd,ind_Se,ind_attach):
    all_dists = dist_all_points(QD_xyz)
    ind_selig = np.logical_or(ind_Se,ind_attach)
    # distances separated into different atoms
    # want these for histograms
    cd_se_dists_all = dist_atom12(all_dists,ind_Cd,ind_Se)
    cd_lig_dists_all = dist_atom12(all_dists,ind_Cd,ind_attach)
    cd_se_lig_dists_all = dist_atom12(all_dists,ind_Cd,ind_selig)
    se_cd_dists_all = dist_atom12(all_dists,ind_Se,ind_Cd)

    return all_dists,cd_se_dists_all,cd_lig_dists_all,cd_se_lig_dists_all,se_cd_dists_all

def get_nn(cdselig_dists,secd_dists,ind_Cd,ind_Se,ind_lig,cutoff,Natoms):
    cd_nn_selig = num_nn(cdselig_dists,cutoff)
    se_nn_cdonly = num_nn(secd_dists,cutoff)

    all_nn = np.zeros(Natoms)
    all_nn[ind_Cd]=cd_nn_selig # using all nn for cd
    all_nn[ind_Se]=se_nn_cdonly # using just cd for se
    all_nn[ind_lig]=100 # set these to very high to avoid any weirdness

    return all_nn,cd_nn_selig,se_nn_cdonly
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
Natoms = len(atom_name_start)

if not np.all(atom_name_start==atom_name_end):
    print("WARNING: atom ordering changed in optimization!")

# getting indices of different types of atoms
# atom order shouldn't change in optimization, so just need one set
ind_Cd, ind_Se, ind_CdSe, ind_lig, ind_selig=parse_ind(atom_name_start,lig_atom)
ind_attach = (atom_name_start == lig_atom)

# ANALYZING STARTING GEOMETRY
all_dists_s,cdse_dists_s,cdlig_dists_s,cdselig_dists_s,secd_dists_s = get_dists(QD_xyz_start,ind_Cd,ind_Se,ind_attach)

all_nn_s,cd_nn_selig_s,se_nn_cd_s = get_nn(cdselig_dists_s,secd_dists_s,ind_Cd,ind_Se,ind_lig,cutoff,Natoms)

# print(cd_nn_selig_s)
cd_underc_ind_s = cd_nn_selig_s<nncutoff
se_underc_ind_s = se_nn_cd_s<nncutoff

cd_underc_name_s = atom_name_start[ind_Cd][cd_underc_ind_s]
se_underc_name_s = atom_name_start[ind_Se][se_underc_ind_s]
cd_underc_xyz_s = QD_xyz_start[ind_Cd][cd_underc_ind_s]
se_underc_xyz_s = QD_xyz_start[ind_Se][se_underc_ind_s]

print('Starting geometry')
print('Undercoordinated Cd:',cd_nn_selig_s[cd_underc_ind_s])
print('Undercoordinated Se:',se_nn_cd_s[se_underc_ind_s])

# ENDING GEOMETRY
all_dists_e,cdse_dists_e,cdlig_dists_e,cdselig_dists_e,secd_dists_e = get_dists(QD_xyz_end,ind_Cd,ind_Se,ind_attach)

all_nn_e,cd_nn_selig_e,se_nn_cd_e = get_nn(cdselig_dists_e,secd_dists_e,ind_Cd,ind_Se,ind_lig,cutoff,Natoms)
cd_underc_ind_e = cd_nn_selig_e<nncutoff
se_underc_ind_e = se_nn_cd_e<nncutoff

print('Optimized Geometry')
print('Undercoordinated Cd:',cd_nn_selig_e[cd_underc_ind_e])
print('Undercoordinated Se:',se_nn_cd_e[se_underc_ind_e])

cd_underc_name_e = atom_name_end[ind_Cd][cd_underc_ind_e]
se_underc_name_e = atom_name_end[ind_Se][se_underc_ind_e]
cd_underc_xyz_e = QD_xyz_end[ind_Cd][cd_underc_ind_e]
se_underc_xyz_e = QD_xyz_end[ind_Se][se_underc_ind_e]

# atoms in ending structure that started width 2
cd_underc_name_es = atom_name_end[ind_Cd][cd_underc_ind_s]
se_underc_name_es = atom_name_end[ind_Se][se_underc_ind_s]
cd_underc_xyz_es = QD_xyz_end[ind_Cd][cd_underc_ind_s]
se_underc_xyz_es = QD_xyz_end[ind_Se][se_underc_ind_s]

beg_s = '.'.join(QD_file_start.split('.')[0:-1])
beg_e = '.'.join(QD_file_end.split('.')[0:-1])
# write xyz files to look at the undercoordinated atoms
# write_xyz(beg_s+'_se_underc.xyz', se_underc_name_s, se_underc_xyz_s,'Undercoordinated Se atoms from '+QD_file_start + ' cutoff '+str(cutoff))
# write_xyz(beg_s+'_cd_underc.xyz', cd_underc_name_s, cd_underc_xyz_s,'Undercoordinated Cd atoms from '+QD_file_start + ' cutoff '+str(cutoff))
# write_xyz(beg_e+'_se_underc.xyz', se_underc_name_e, se_underc_xyz_e,'Undercoordinated Se atoms from '+QD_file_end + ' cutoff '+str(cutoff))
# write_xyz(beg_e+'_cd_underc.xyz', cd_underc_name_e, cd_underc_xyz_e,'Undercoordinated Cd atoms from '+QD_file_end + ' cutoff '+str(cutoff))
# write_xyz(beg_e+'_start_se_underc.xyz', se_underc_name_es, se_underc_xyz_es,'Undercoordinated Se atoms from '+QD_file_start + 'positions from end. cutoff '+str(cutoff))
# write_xyz(beg_e+'_start_cd_underc.xyz', cd_underc_name_es, cd_underc_xyz_es,'Undercoordinated Cd atoms from '+QD_file_start + 'positions from end. cutoff '+str(cutoff))


# print(se_underc_ind_s==se_underc_ind_e)

# ambiguous zone:
# looks at atoms that change their number of nearest neighbors based on cutoff distance
all_nn_e2,cd_nn_selig_e2,se_nn_cd_e2 = get_nn(cdselig_dists_e,secd_dists_e,ind_Cd,ind_Se,ind_lig,cutoff2,Natoms)
cd_underc_ind_e2 = cd_nn_selig_e2<nncutoff
se_underc_ind_e2 = se_nn_cd_e2<nncutoff

nn_change_cd_cut = cd_nn_selig_e2 - cd_nn_selig_e
nn_change_se_cut = se_nn_cd_e2 - se_nn_cd_e

nn_change_cd_cut_end_pos = nn_change_cd_cut > 0
nn_change_cd_cut_end_neg = nn_change_cd_cut < 0
nn_change_se_cut_end_pos = nn_change_se_cut > 0
nn_change_se_cut_end_neg = nn_change_se_cut < 0
nn_change_cd_cut_ind = nn_change_cd_cut != 0
nn_change_se_cut_ind = nn_change_se_cut != 0

print('ambiguous zone')
# prints change in # NN
print('Cd that change nearest neighbor',nn_change_cd_cut)
print('Se that change nearest neighbor',nn_change_se_cut)

cd_amb_name_e = atom_name_end[ind_Cd][nn_change_cd_cut_ind]
se_amb_name_e = atom_name_end[ind_Se][nn_change_se_cut_ind]
cd_amb_xyz_e = QD_xyz_end[ind_Cd][nn_change_cd_cut_ind]
se_amb_xyz_e = QD_xyz_end[ind_Se][nn_change_se_cut_ind]

# writes xyz files for just ones that change, doesn't distinguish gaining/losing
write_xyz(beg_e+'_se_amb.xyz', se_amb_name_e, se_amb_xyz_e,'Ambiguous Se atoms from '+QD_file_end+'cutoff 1 '+str(cutoff)+'cutoff2 '+str(cutoff2) )
write_xyz(beg_e+'_cd_amb.xyz', cd_amb_name_e, cd_amb_xyz_e,'Ambiguous Cd atoms from '+QD_file_end+'cutoff 1 '+str(cutoff)+'cutoff2 '+str(cutoff2) )


# Comparing number of nearest neighbors between starting and optimized structure:
nn_change_cd = cd_nn_selig_e - cd_nn_selig_s
nn_change_se = se_nn_cd_e - se_nn_cd_s
print('Optimization:')
print('Cd that change nearestneighbor',nn_change_cd)
print('Se that change nearest neighbor',nn_change_se)

nn_change_cd_pos = nn_change_cd > 0
nn_change_cd_neg = nn_change_cd < 0
nn_change_se_pos = nn_change_se > 0
nn_change_se_neg = nn_change_se < 0

cd_pos_name = atom_name_start[ind_Cd][nn_change_cd_pos]
se_pos_name = atom_name_start[ind_Se][nn_change_se_pos]
cd_pos_xyz = QD_xyz_end[ind_Cd][nn_change_cd_pos]
se_pos_xyz = QD_xyz_end[ind_Se][nn_change_se_pos]

cd_neg_name = atom_name_start[ind_Cd][nn_change_cd_neg]
se_neg_name = atom_name_start[ind_Se][nn_change_se_neg]
cd_neg_xyz = QD_xyz_end[ind_Cd][nn_change_cd_neg]
se_neg_xyz = QD_xyz_end[ind_Se][nn_change_se_neg]


# writes xyz files for ones that gain nearest neighbors (pos) and lose nearest neighbors (neg)
# write_xyz(beg_e+'_se_pos_change.xyz', se_pos_name, se_pos_xyz,'Se atoms with more NN after opt '+QD_file_end + ' cutoff '+str(cutoff))
# write_xyz(beg_e+'_cd_pos_change.xyz', cd_pos_name, cd_pos_xyz,'Cd atoms with more NN after opt '+QD_file_end + ' cutoff '+str(cutoff))
# write_xyz(beg_e+'_se_neg_change.xyz', se_neg_name, se_neg_xyz,'Se atoms with less NN after opt '+QD_file_end + ' cutoff '+str(cutoff))
# write_xyz(beg_e+'_cd_neg_change.xyz', cd_neg_name, cd_neg_xyz,'Cd atoms with less NN after opt '+QD_file_end + ' cutoff '+str(cutoff))


# UNCOMMENT TO PLOT A HISTOGRAM OF NEAREST NEIGHBOR DISTANCES

## Cd-Se distance histogram
plt.figure()
plt.title("Cd-Se distance")
plt.hist(cdse_dists_s.flatten(),bins=800) # crystal
plt.hist(cdse_dists_e.flatten(),bins=800) # optimized
plt.xlim(0,4)
plt.show()

# Cd-ligand distance histogram
plt.figure()
plt.title("Cd-ligand distance")
plt.hist(cdlig_dists_s.flatten(),bins=800)
plt.hist(cdlig_dists_e.flatten(),bins=800)
plt.xlim(0,4)
plt.show()

# #
# # # plt.figure()
# # # plt.title("Cd-Se and Cd-ligands")
# # # plt.hist(cd_lig_dists_s.flatten(),bins=800)
# # # plt.hist(cd_lig_dists_e.flatten(),bins=800)
# # # plt.xlim(0,4)
# #

#
# plt.figure()
# plt.hist(all_dists[ind_Se].flatten(),bins=800)
# plt.show()
