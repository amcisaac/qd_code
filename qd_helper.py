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

def get_dists_cs(QD_xyz,ind_Cd,ind_Se,ind_shell_cd,ind_shell_chal):
    all_dists = dist_all_points(QD_xyz)
    cd_se_dists_all = dist_atom12(all_dists,ind_Cd,ind_Se) # cd (core) - se (core)
    se_cd_dists_all = dist_atom12(all_dists,ind_Se,ind_Cd) # se (core) - cd (core)

    ind_ses =np.logical_or(ind_Se,ind_shell_chal) # index of se and s atoms
    ind_cdcd = np.logical_or(ind_Cd,ind_shell_cd) # index of all cd

    # print(ind_ses)

    cdcore_ses_dist = dist_atom12(all_dists,ind_Cd,ind_ses) # cd (core) - se and s
    secore_cd_dist  = dist_atom12(all_dists,ind_Se,ind_cdcd) # se (core) - cd (core) and cd (shell)
    cdshell_ses_dist = dist_atom12(all_dists,ind_shell_cd,ind_ses) # cd (shell) - se and s
    sshell_cd_dist = dist_atom12(all_dists, ind_shell_chal,ind_cdcd) # s (shell) - cd (core) and cd (shell)
    # print(all_dists,cd_se_dists_all,cdcore_ses_dist,secore_cd_dist,cdshell_ses_dist,sshell_cd_dist)

    return all_dists,cd_se_dists_all,cdcore_ses_dist,secore_cd_dist,cdshell_ses_dist,sshell_cd_dist


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

def get_nn(cdselig_dists,secd_dists,ind_Cd,ind_Se,cutoff,Natoms,ind_lig=False):
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
