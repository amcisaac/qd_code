# from qd_helper import *
import copy
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from matplotlib import pyplot as plt

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
            xyz_file.write('{:<2s}    {:< 14.8f}{:< 14.8f}{:< 14.8f}\n'.format(atom,atom_xyz[i][0],atom_xyz[i][1],atom_xyz[i][2]))
    return

def dist_all_points(xyz):
    '''
    Function that returns the distances between all atoms in an xyz file.

    Inputs:
        xyz: numpy array of xyz coordinates of atoms to calculate the
             distances between. Size (Natoms,3)

    Outputs:
        dist_array: array of distances between all atoms. Size (Natoms,Natoms).
                    dist_array[i][j] is the distance between atom i and atom j
    '''
    dists = [] # list to help build array
    for atom in xyz: # xyz = for each atom
        dist = np.sqrt(np.sum((atom - xyz)**2,axis=1)) # calc dist between atom(i) and all others
        dists.append(dist)
    dist_array = np.array(dists)
    return dist_array

def dist_atom12(all_dists,ind_1,ind_2):
    '''
    Function that returns an array of distances between two types of atoms.

    Inputs:
        all_dists: array of distances between all atoms, size (Natoms,Natoms)
        ind_1: array of boolean indices for first atom type (e.g. all Cd's)
        ind_2: array of boolean indices for second atom type (e.g. all Se's)

    Outputs:
        Returns a subset of all_dists that are the distances between atom
        type 1 and 2. Array of size (Natom1,Natom2)
    '''
    return all_dists[ind_1].T[ind_2].T

def get_dists_cs(QD_xyz,ind_Cd,ind_Se,ind_shell_cd,ind_shell_chal,ind_attach=False,ind_attach2=False):
    all_dists = dist_all_points(QD_xyz)
    cd_se_dists_all = dist_atom12(all_dists,ind_Cd,ind_Se) # cd (core) - se (core)
    se_cd_dists_all = dist_atom12(all_dists,ind_Se,ind_Cd) # se (core) - cd (core)
    # print(np.all(cd_se_dists_all==se_cd_dists_all.T))

    ind_ses =np.logical_or(ind_Se,ind_shell_chal) # index of se and s atoms
    ind_cdcd = np.logical_or(ind_Cd,ind_shell_cd) # index of all cd

    # print(ind_ses)

    cdcore_ses_dist = dist_atom12(all_dists,ind_Cd,ind_ses) # cd (core) - se and s
    secore_cd_dist  = dist_atom12(all_dists,ind_Se,ind_cdcd) # se (core) - cd (core) and cd (shell)
    cdshell_ses_dist = dist_atom12(all_dists,ind_shell_cd,ind_ses) # cd (shell) - se and s
    sshell_cd_dist = dist_atom12(all_dists, ind_shell_chal,ind_cdcd) # s (shell) - cd (core) and cd (shell)
    # print(all_dists,cd_se_dists_all,cdcore_ses_dist,secore_cd_dist,cdshell_ses_dist,sshell_cd_dist)

    # print(cdcore_ses_dist.shape)
    # print(sshell_cd_dist.shape)
    if np.any(ind_attach): # if ligands present
        ind_challig = np.logical_or(ind_ses,ind_attach)
        cd_lig_dists_all = dist_atom12(all_dists,ind_shell_cd,ind_attach)
        cd_chal_lig_dists_all = dist_atom12(all_dists,ind_shell_cd,ind_challig)
        # print(cd_lig_dists_all.shape)

    else: # may need to fix this so it works with non-lig
        cd_lig_dists_all = []
        cd_chal_lig_dists_all = cdshell_ses_dist # this should mean that anything that uses this will default to just cd-ses

    if np.any(ind_attach2):
        ind_cdlig = np.logical_or(ind_cdcd,ind_attach2)
        s_lig_dists_all = dist_atom12(all_dists,ind_shell_chal,ind_attach2)
        s_cd_lig_dists_all = dist_atom12(all_dists,ind_shell_chal,ind_cdlig)
    else:
        s_lig_dists_all = []
        s_cd_lig_dists_all = sshell_cd_dist # if no ligand 2, just return cd-s dists

    return all_dists,cd_se_dists_all,cdcore_ses_dist,secore_cd_dist,cdshell_ses_dist,sshell_cd_dist,cd_lig_dists_all,cd_chal_lig_dists_all,s_lig_dists_all,s_cd_lig_dists_all


def get_dists(QD_xyz,ind_Cd,ind_Se,ind_attach=False,ind_shell_cd=False,ind_shell_chal=False,cs=False):
    '''
    Function that calculates the distance between all atoms, as well as
    the distance between two types of atoms.

    Inputs:
        QD_xyz: xyz coordinates of all atoms in the QD (array size (Natoms,3))
        ind_Cd: indices of atom type 1 (e.g. Cd)
        ind_Se: indices of atom type 2 (e.g. Se)
        ind_attacH: (optional) indices of the ligand atoms that attach to Cd
                    (e.g. indices of N for MeNH2)

    Outputs:
        all_dists: np array with distances between all atoms, size (Natoms, Natoms)
        cd_se_dists_all: np array with distances between atom type 1 and atom
                         type 2 (e.g. Cd-Se distances only)
        se_cd_dists_all: np array with distances between atom type 2 and atom type 1
                         (e.g. Se-Cd distances) -- same as cd_se_dists_all but indexed
                         differently
        cd_lig_dists_all: only returned if ind_attach provided. distances between
                          cd atoms and ligand attach atoms
        cd_se_lig_dists_all: only returned if ind_attach provided. distances between
                             cd atoms and ligand attach atoms AND cd atoms and se atoms
    '''
    all_dists = dist_all_points(QD_xyz)
    cd_se_dists_all = dist_atom12(all_dists,ind_Cd,ind_Se)
    se_cd_dists_all = dist_atom12(all_dists,ind_Se,ind_Cd)

    if np.any(ind_attach): # if ligands present
        ind_selig = np.logical_or(ind_Se,ind_attach)
        cd_lig_dists_all = dist_atom12(all_dists,ind_Cd,ind_attach)
        cd_se_lig_dists_all = dist_atom12(all_dists,ind_Cd,ind_selig)

        return all_dists,cd_se_dists_all,cd_lig_dists_all,cd_se_lig_dists_all,se_cd_dists_all


    else:
        return all_dists,cd_se_dists_all,[],cd_se_dists_all,se_cd_dists_all

def get_dists_bonded(all_dists,ind_Cd,ind_Se):
    cdcd_dist=dist_atom12(all_dists,ind_Cd,ind_Cd)
    sese_dist=dist_atom12(all_dists,ind_Se,ind_Se)
    return cdcd_dist,sese_dist

def num_nn(dist_list,cutoff):
    '''
    Function that calculates the number of nearest neighbors that each atom has,
    based on a cutoff.

    Inputs:
        dist_list: array of distances between all atoms, size (Natoms,Natoms)
        cutoff: distance cutoff (in A) below which atoms are considered nearest
                neighbors/bonded

    Outputs:
        nn_list: an array of the number of nearest neighbors for each atom. Size (Natoms,).
                 nn_list[i] = # of nearest neighbors for atom i

    '''

    nn_list = np.sum(dist_list < cutoff,axis=1)
    return nn_list

def get_nn(cdselig_dists,secd_dists,ind_Cd,ind_Se,cutoff,Natoms,ind_lig=False,ind_lig2=False):
    '''
    Function that calculates the number of nearest neighbors for each atom,
    based on atom type. E.g. can restrict such that Cd only has Se NN's

    Inputs:
        cdselig_dists: distances between cd and se (or cd, and se + ligs)
        secd_dists: distances between se and cd
        ind_Cd: indices of cd atoms
        ind_Se: indices of se atoms
        cutoff: cutoff for NN interaction
        Natoms: number of atoms in the system

    Outputs:
        all_nn: an array of the number of nearest neighbors for each atom. Size (Natoms,).
                nn_list[i] = # of nearest neighbors for atom i
        cd_nn_selig: an array of the number of nearest neighbors for cd (size (Ncd,))
        se_nn_cdonly: an array of the number of nearest neighbors for se (size (Nse,))
    '''
    cd_nn_selig = num_nn(cdselig_dists,cutoff)
    se_nn_cdonly = num_nn(secd_dists,cutoff)

    all_nn = np.zeros(Natoms)
    all_nn[ind_Cd]=cd_nn_selig # using all nn for cd
    all_nn[ind_Se]=se_nn_cdonly # using just cd for se
    if np.any(ind_lig):
        all_nn[ind_lig]=100 # set these to very high to avoid any weirdness
    if np.any(ind_lig2):
        all_nn[ind_lig2]=100 # set these to very high to avoid any weirdness
    return all_nn,cd_nn_selig,se_nn_cdonly

def nn_histogram(xyz,ind_Cd,ind_Se,label1='',ind_attach=False,xyz2=False,label2=''):
    '''
    Function that makes a histogram of all nearest-neighbor distances in a QD (or comparing multiple!).

    Inputs:
        xyz: np array of xyz coordinates for the QD. shape (Natoms,3)
        ind_Cd: boolean array with the indices of Cd atoms in xyz
        ind_Se: boolean array with the indices of Se atoms in xyz
        label1: (optional) label for the legend of the histogram for xyz
        ind_attach: (optional) boolean array with the indices of attaching
                    ligand atoms in xyz (e.g. N for MeNH2)
        xyz2: (optional) np array of xyz coordinates for another QD to compare
        label2: (optional) label for the legend of the histogram for xyz2

    Outputs:
        plots a histogram of Cd-Se distances. If ind_attach is true, also
        plots histogram of Cd-ligand distances.
    '''

    all_dists,cdse_dists,cdlig_dists,cdselig_dists,secd_dists = get_dists(xyz,ind_Cd,ind_Se,ind_attach)

    if np.any(xyz2):
        all_dists2,cdse_dists2,cdlig_dists2,cdselig_dists2,secd_dists2 = get_dists(xyz2,ind_Cd,ind_Se,ind_attach)

    # Cd-Se distance histogram
    plt.figure()
    plt.title("Cd-Se distance")
    plt.hist(cdse_dists.flatten(),bins=800,label=label1) # crystal
    if np.any(xyz2): plt.hist(cdse_dists2.flatten(),bins=800,label=label2) # optimized
    if label2 !='': plt.legend()
    plt.xlim(2.25,4)
    # plt.show()
    #
    if np.any(ind_attach):
        # # Cd-ligand distance histogram
        plt.figure()
        plt.title("Cd-ligand distance")
        plt.hist(cdlig_dists.flatten(),bins=800,label=label1)
        if np.any(xyz2): plt.hist(cdlig_dists2.flatten(),bins=800,label=label2)
        if label2 !='': plt.legend()
        plt.xlim(2.25,4)

    # plt.show()

    return


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
    all_nn,cd_nn_selig,se_nn_cd = get_nn(cdselig_dists,secd_dists,ind_Cd,ind_Se,cutoff,Natoms,ind_lig)

    cd_underc_ind = cd_nn_selig<nncutoff
    se_underc_ind = se_nn_cd<nncutoff

    if verbose:
        print('Undercoordinated Cd:',cd_nn_selig[cd_underc_ind])
        print('Undercoordinated Se:',se_nn_cd[se_underc_ind])
    return cd_underc_ind,se_underc_ind

def get_bonded_index(xyz,ind_Cd,ind_Se,ind_lig,ind_attach,cutoff,nncutoff,verbose=False):
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
    cdcd_dists,sese_dists=get_dists_bonded(all_dists,ind_Cd,ind_Se)
    Natoms = len(ind_Cd)
    all_nn,cd_nn_cd,se_nn_se = get_nn(cdcd_dists,sese_dists,ind_Cd,ind_Se,cutoff,Natoms,ind_lig)

    cd_bond_ind = cd_nn_cd>1
    se_bond_ind = se_nn_se>1

    if verbose:
        print('Bonded Cd:',cd_nn_cd[cd_bond_ind])
        print('Bonded Se:',se_nn_se[se_bond_ind])
    return cd_bond_ind,se_bond_ind



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



def get_underc_ind_large(ind_orig,ind_underc):
    '''
    Returns index for undercoordinated atom type, with the dimensions of the
    original number of atoms e.g. under coordinated Se index for Cd33Se33 will
    be len 33, this will return len 66 for use with other properties

    Inputs:
        ind_orig: index array for all atoms of X type (size: Natoms(total))
        ind_underc: index array for all undercoordinated atoms of X type
            (size: number of atoms of X type)

    Returns:
        large_underc_ind: index array for undercoordinated atoms of type X,
            mapped back to size of ind_orig (size: Natoms (total))
    '''
    large_underc_ind = copy.deepcopy(ind_orig)
    large_underc_ind[ind_orig] = ind_underc # USE INDICES FROM WHATEVER METHOD YOU PREFER
                                            # this is the undercoordinated at the end of the optimization
    return large_underc_ind

def sum_chargefrac(chargefrac_tot,ind_orig,ind_underc):
    '''
    Sums the charge fraction on the undercoordinated atoms and shapes array into (Nex,3)

    Inputs:
        chargefrac_tot: array of normalized charges on each atom
        ind_orig: index array for all atoms of X type (size: Natoms(total))
        ind_underc: index array for all undercoordinated atoms of X type
            (size: number of atoms of X type)

    Returns:
        sum_underc_reshape: array with summed charges on the undercoordinated
            atom for each excitation. Size (Nex,3) where col. 0 is electron,
            col. 1 is hole, col. 2 is delta (ignore)
    '''
    large_underc_ind = get_underc_ind_large(ind_orig,ind_underc)
    chargefrac_underc = chargefrac_tot[large_underc_ind]
    sum_chargefrac_underc= np.sum(chargefrac_underc,axis=0)
    # reshape so that we have an array of shape (Nex, 3) where column 0 is electron
    # charge sum, column 1 is hole charge sum, and column 2 is delta (ignored)
    sum_underc_reshape = np.reshape(sum_chargefrac_underc,(-1,3))
    return sum_underc_reshape


def print_indiv_ex(chargefrac_tot,ind_orig,ind_underc,n,atomname):
    '''
    Prints charge info about specific excitations and atom types

    Inputs:
        chargefrac_tot: array of normalized charges on each atom
        ind_orig: index array for all atoms of X type (size: Natoms(total))
        ind_underc: index array for all undercoordinated atoms of X type
            (size: number of atoms of X type)
        n: excitation number
        atomname: name of the atom (just for printing)
    '''
    large_underc_ind = get_underc_ind_large(ind_orig,ind_underc)
    chargefrac_underc = chargefrac_tot[large_underc_ind]
    sum_chargefrac_underc= np.sum(chargefrac_underc,axis=0)

    print('')
    print('Fraction of charge on each undercoordinated {} for excitation {}:'.format(atomname,n))
    print('   e           h')
    print(chargefrac_underc[:,3*n:3*n+2])
    print('')
    print('Sum of charge on undercoordinated {} for excitation {}:'.format(atomname,n))
    print('   e           h')
    print(sum_chargefrac_underc[3*n:3*n+2])

    max_ind = np.argmax(chargefrac_tot,axis=0) # index of the largest charge fraction on any atom
    max_charge=np.max(chargefrac_tot,axis=0)   # largest charge fraction on any atom
    print('')
    print('Largest charge fraction on any atom for excitation {}:'.format(n))
    print('   e           h')
    print(max_charge[3*n:3*n+2])
    print('')
    print('Is the largest charge fraction on an undercoordinated {}?'.format(atomname))
    print('   e     h')
    print(np.any(chargefrac_underc[:,3*n:3*n+2]==max_charge[3*n:3*n+2],axis=0))
    # print(atom_name_start[max_ind][3*n:3*n+3]) # atom name with largest charge fraction

    # creates an array (Nex, 3) where each entry is whether the max charge fraction is on an undercoordinated se
    # found this wasn't useful because it's almost always on it, even for bulk excitations
    max_is_underc_long = np.any(chargefrac_underc==max_charge,axis=0)
    max_is_underc= np.reshape(max_is_underc_long,(-1,3))
    # print(max_is_underc[100:120])

    # finds the top 5 highest charge fractions on any atom
    top5_ind = np.argpartition(-chargefrac_tot,5,axis=0)[:5] # index of top 5
    top5 = np.take_along_axis(chargefrac_tot,top5_ind,axis=0) # value of top 5
    print('')
    print('Top 5 largest charge fractions on any atom for excitation {}:'.format(n))
    print('   e           h')
    print(top5[:,3*n:3*n+2])

    return

#######
#
# CORE-SHELL SPECIFIC FUNCTIONS
#
#######
def get_underc_index_cs(ind_Cd_core,ind_Se,ind_Cd_shell,ind_S,cutoff,nncutoff,dist_list,ind_attach=False,ind_attach2=False,verbose=False):
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
    all_dists,cdse_core_dists,cdcore_ses_dist,secore_cd_dist,cdshell_ses_dist,sshell_cd_dist,cdshell_lig_dist,cdshell_chal_lig_dist,sshell_lig_dist,sshell_cd_lig_dist = dist_list #get_dists_cs(xyz,ind_Cd_core,ind_Se,ind_Cd_shell,ind_S)
    # print(cdse_core_dists)
    # print(cdse_core_dists.T)
    Natoms = len(ind_Cd_core)
    # print(Natoms)
    all_nn,cd_se_nn,se_cd_nn = get_nn(cdse_core_dists,cdse_core_dists.T,ind_Cd_core,ind_Se,cutoff,Natoms) # bare core
    all_nn,cdcore_nn_ses,se_nn_cdcd = get_nn(cdcore_ses_dist,secore_cd_dist,ind_Cd_core,ind_Se,cutoff,Natoms)    # core coord. when considering shell too
    all_nn,cdshell_nn_ses,s_nn_cdcd = get_nn(cdshell_chal_lig_dist,sshell_cd_lig_dist,ind_Cd_shell,ind_S,cutoff,Natoms,ind_attach,ind_attach2) # undercoordinated shell atoms (includes core-shell bonds)

    # core-core
    cd_underc_ind = cd_se_nn < nncutoff
    se_underc_ind = se_cd_nn < nncutoff

    # core-core&shell
    cd_underc_inclshell_ind = cdcore_nn_ses < nncutoff
    se_underc_inclshell_ind = se_nn_cdcd < nncutoff

    # shell - core&shell
    cdshell_underc_inclcore_ind = cdshell_nn_ses < nncutoff
    s_underc_inclcore_ind = s_nn_cdcd < nncutoff

    if len(cdshell_lig_dist) > 0:
        all_nn,cdshell_nn_lig,lig_nn_cdshell = get_nn(cdshell_lig_dist,cdshell_lig_dist.T,ind_Cd_shell,ind_attach,cutoff,Natoms)

        # ligand-Cd
        attach_underc_ind = lig_nn_cdshell < 1 # each N should be bound to one cd
    else:
        attach_underc_ind = []
    if len(sshell_lig_dist) > 0:
        all_nn,sshell_nn_lig,lig_nn_sshell = get_nn(sshell_lig_dist,sshell_lig_dist.T,ind_S,ind_attach2,cutoff,Natoms)

        # ligand-Cd
        attach_underc_ind2 = lig_nn_sshell < 1 # each N should be bound to one cd
    else:
        attach_underc_ind2 = []
    if verbose:
        print('Undercoordinated Cd (core only):',cd_se_nn[cd_underc_ind])
        print('Undercoordinated Se (core only):',se_cd_nn[se_underc_ind])
        print('Undercoordinated Cd (core with shell):',cdcore_nn_ses[cd_underc_inclshell_ind])
        print('Undercoordinated Se (core with shell):',se_nn_cdcd[se_underc_inclshell_ind])
        print('Undercoordinated Cd (shell with core):',cdshell_nn_ses[cdshell_underc_inclcore_ind])
        print('Undercoordinated Se (shell with core):',s_nn_cdcd[s_underc_inclcore_ind])
    return cd_underc_ind,se_underc_ind,cd_underc_inclshell_ind,se_underc_inclshell_ind,cdshell_underc_inclcore_ind,s_underc_inclcore_ind,attach_underc_ind,attach_underc_ind2

# def get_bonded_index_cs(xyz,ind_Cd,ind_Se,ind_lig,ind_attach,cutoff,nncutoff,verbose=False):
def get_bonded_index_cs(all_dists,ind_Cd,ind_Se,ind_cd_shell,ind_s_shell,cutoff,verbose=False):
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
    ind_cd_all = np.logical_or(ind_Cd,ind_cd_shell)
    ind_chal = np.logical_or(ind_Se, ind_s_shell)
    cdcd_core_dists,sese_core_dists=get_dists_bonded(all_dists,ind_Cd,ind_Se)
    cdcd_cs_dists,ses_cs_dists=get_dists_bonded(all_dists,ind_cd_all,ind_chal)
    cdcd_shell_dists,ss_shell_dists=get_dists_bonded(all_dists,ind_cd_shell,ind_s_shell)
    Natoms = len(ind_Cd)

    all_nn,cd_nn_cd_core,se_nn_se_core = get_nn(cdcd_core_dists,sese_core_dists,ind_Cd,ind_Se,cutoff,Natoms)
    all_nn,cd_nn_cd_cs,se_nn_s_cs = get_nn(cdcd_cs_dists,ses_cs_dists,ind_cd_all,ind_chal,cutoff,Natoms)
    all_nn,cd_nn_cd_shell,s_nn_s_shell = get_nn(cdcd_shell_dists,ss_shell_dists,ind_cd_shell,ind_s_shell,cutoff,Natoms)

    cd_core_bond_ind = cd_nn_cd_core>1
    se_core_bond_ind = se_nn_se_core>1

    cd_cs_bond_ind = cd_nn_cd_cs>1
    ses_cs_bond_ind = se_nn_s_cs>1

    cd_shell_bond_ind = cd_nn_cd_shell>1
    s_shell_bond_ind = s_nn_s_shell>1


    if verbose:
        print('Bonded Cd:',cd_nn_cd[cd_bond_ind])
        print('Bonded Se:',se_nn_se[se_bond_ind])
    return cd_core_bond_ind,se_core_bond_ind,cd_cs_bond_ind,ses_cs_bond_ind,cd_shell_bond_ind,s_shell_bond_ind
