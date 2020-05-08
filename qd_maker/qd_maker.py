import numpy as np
import sys
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
            xyz_file.write(atom +'    '+ str(atom_xyz[i][0])+'    '+str(atom_xyz[i][1])+'    '+str(atom_xyz[i][2])+'\n')
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
    # return np array of all distances for each atom
    dists = [] # list to return with format above
    for atom in xyz: # xyz = for each cd
        dist = np.sqrt(np.sum((atom - xyz)**2,axis=1)) # calc dist between cd(i) and all se's
        dists.append(dist) # add dist to list
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


def get_dists(QD_xyz,ind_Cd,ind_Se,ind_attach=''):
    '''
    Function that calculates the distance between all atoms, as well as
    the distance between two types of atoms.

    Inputs:
        QD_xyz: xyz coordinates of all atoms in the QD (array size (Natoms,3))
        ind_Cd: indices of atom type 1 (e.g. Cd)
        ind_Se: indices of atom type 2 (e.g. Se)

    Outputs:
        all_dists: np array with distances between all atoms, size (Natoms, Natoms)
        cd_se_dists_all: np array with distances between atom type 1 and atom
                         type 2 (e.g. Cd-Se distances only)
        se_cd_dists_all: np array with distances between atom type 2 and atom type 1
                         (e.g. Se-Cd distances) -- same as cd_se_dists_all but indexed
                         differently
    '''
    all_dists = dist_all_points(QD_xyz)
    cd_se_dists_all = dist_atom12(all_dists,ind_Cd,ind_Se)
    se_cd_dists_all = dist_atom12(all_dists,ind_Se,ind_Cd)

    return all_dists,cd_se_dists_all,se_cd_dists_all

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
    # returns list of number of NN for each atom
    # could get xyz coordinates from the cd_se_dists_all < cutoff indices i think
    nn_list = np.sum(dist_list < cutoff,axis=1)
    return nn_list

def get_nn(cdselig_dists,secd_dists,ind_Cd,ind_Se,cutoff,Natoms):
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
    # all_nn[ind_lig]=100 # set these to very high to avoid any weirdness

    return all_nn,cd_nn_selig,se_nn_cdonly

def build_dot(xyz,atoms,ctr,radius,nncutoff=2):
    '''
    Function to carve dot from bulk and remove surface atoms
    with 1 nearest neighbor.

    Inputs:
        xyz: np array of xyz coordinates from which to carve dot.
             e.g. a large crystal slab. Size (Natoms,3)
        atoms: np array of all atom names that correspond to the
               xyz coordinates. Size (Natoms)
        ctr: where to center the xyz coordinates.
        radius: desired radius of the dot (float)
        nncutoff: cutoff for number of nearest neighbors. any
                  atom with fewer nearest neighbors will be removed
                  from the surface.

    Outputs:
        coord_ind_all: boolean numpy array to index xyz and atoms. True elements
                       are atoms that are within the radius and have at
                       least 2 nearest neighbors.
                       e.g. xyz[coord_ind_all] gives xyz coordinates for the
                       resulting dot
    '''
    xyz_ctr = xyz - ctr
    dist_from_ctr = np.linalg.norm(xyz_ctr,axis=1)
    in_r = dist_from_ctr <= radius
    atoms_in_r = atoms[in_r]
    xyz_in_r = xyz_ctr[in_r]

    ind_Cd_r = (atoms_in_r == "Cd")
    ind_Se_r = (atoms_in_r == "Se")

    all_dists,cd_se_dists,se_cd_dists = get_dists(xyz_in_r,ind_Cd_r,ind_Se_r)
    all_nn,cd_nn,se_nn=get_nn(cd_se_dists,se_cd_dists,ind_Cd_r,ind_Se_r,3.0,len(atoms_in_r))

    coord_ind = all_nn >= nncutoff

    coord_ind_all = dist_from_ctr <= radius
    coord_ind_all[in_r] = coord_ind

    return coord_ind_all

def get_coreonly(xyz,atoms,radius):
    '''
    Function to build a stoichiometric core-only dot.

    Inputs:
        xyz: np array of xyz coordinates from which to carve dot.
             e.g. a large crystal slab. Size (Natoms,3)
        atoms: np array of all atom names that correspond to the
               xyz coordinates. Size (Natoms)
        radius: desired radius of the dot (float)

    Outputs:
        coord_ind_all: boolean numpy array to index xyz and atoms, yielding
                       a core-only QD that is stoichiometric. True elements
                       are atoms that are within the radius and have at
                       least 2 nearest neighbors.
                       e.g. xyz[coord_ind_all] gives xyz coordinates for the
                       resulting dot
        ctr: the coordinates used to center xyz, chosen to produce a stoichiometric dot
    '''
    ctr0 = np.mean(xyz,axis=0)

    Ncd = 100
    Nse = 101
    ctr = ctr0
    nit = 0
    while Ncd != Nse and nit < 500:
        coord_ind_all=build_dot(xyz,atoms,ctr,radius)

        Ncd = np.count_nonzero(atoms[coord_ind_all]=='Cd')
        Nse = np.count_nonzero(atoms[coord_ind_all]=='Se')

        print(Nse,' Se atoms')
        print(Ncd,' Cd atoms')

        if Ncd != Nse:
            ctr = ctr0 + np.random.random_sample((1,3))

        nit += 1

    if nit == 500:
        print('Core build did not converge!')

    return coord_ind_all,ctr


def get_coreshell(xyz,atoms,core_rad,shell_rad):
    '''
    Function to build a stoichiometric core-shell dot, which is
    stoichiometric in the core and the shell.

    Inputs:
        xyz: np array of xyz coordinates from which to carve dot.
             e.g. a large crystal slab. Size (Natoms,3)
        atoms: np array of all atom names that correspond to the
               xyz coordinates. Size (Natoms)
        core_rad: desired radius of the dot core (float)
        shell_rad: desired radius of the overall dot

    Outputs:
        coreshell_ind_all: boolean numpy array to index xyz and atoms, yielding
                       a core-shell QD that is stoichiometric in both the core
                       and the shell. True elements are atoms that are within
                       shell_rad and have at least 2 nearest neighbors.
                       e.g. xyz[coreshell_ind_all] gives xyz coordinates for the
                       final core-shell dot
        core_ind_all:  boolean numpy array to index xyz and atoms, yielding
                       the stoichiometric core of the core-shell QD. True elements
                       are atoms that are within core_rad and have at
                       least 2 nearest neighbors.
                       e.g. xyz[core_ind_all] gives xyz coordinates for the
                       core of the final core-shell dot
        shell_ind_all:  boolean numpy array to index xyz and atoms, yielding
                       the stoichiometric shell of the core-shell QD. True elements
                       are atoms that are not in the core but within shell_rad,
                       and have at least 2 nearest neighbors.
                       e.g. xyz[shell_ind_all] gives xyz coordinates for the
                       shell of the final core-shell dot
    '''
    Ncdshell=0
    Nseshell=1
    nit2 = 0
    while Ncdshell != Nseshell and nit2 < 500:
        core_ind_all,core_ctr=get_coreonly(xyz,atoms,core_rad)
        Ncore = np.count_nonzero(core_ind_all)/2
        coreshell_ind_all=build_dot(xyz,atoms,core_ctr,shell_rad)

        Ncdshell = np.count_nonzero(atoms[coreshell_ind_all] == 'Cd') - Ncore
        Nseshell = np.count_nonzero(atoms[coreshell_ind_all] == 'Se') - Ncore

        print(Ncdshell,' Cd in shell only')
        print(Nseshell,' Se in shell only')

        nit2 += 1

    if nit2 == 500:
        print('Shell build did not converge!')

    shell_ind_all = np.logical_xor(coreshell_ind_all,core_ind_all)
    return coreshell_ind_all,core_ind_all,shell_ind_all

def avg_radius(xyz,atoms,atom1,atom2,cutoff=3.0):
    '''
    Function to calculate the average radius of a QD
    (defined as the average distance from the center to
    the surface atoms), as well as the min and max radius,
    and standard deviation of distances.

    Inputs:
        xyz: np array with xyz coordinates of the QD, size (Natoms,3)
        atoms: np array with the atom names of the QD atoms, size (Natoms,)
        atom1: name of the first type of atom (str) (e.g. 'Cd')
        atom2: name of the second type of atom (str) (e.g. 'Se')
        cutoff: cutoff distance for nearest-neighbor, used to determine surface atoms

    Outputs:
        avg_dist_surf: average distance from the surface atoms to the center
        std_dist_surf: standard deviation of surface-center distances
        max_dist_surf: furthest distance from surface atom to center
        min_dist_surf: smallest distance from surface atom to center

    '''
    xyz_ctr = xyz - np.mean(xyz,axis=0)
    ind1= (atoms==atom1)
    ind2= (atoms==atom2)
    Nat = len(atoms)
    all_dists,dist12,dist21 = get_dists(xyz_ctr,ind1,ind2)
    all_nn,nn1,nn2=get_nn(dist12,dist21,ind1,ind2,cutoff,Nat)
    surf_xyz = xyz_ctr[all_nn < 4]
    # print(surf_xyz.shape)
    surf_dist = np.linalg.norm(surf_xyz,axis=1)
    # print(surf_dist.shape)
    avg_dist_surf = np.mean(surf_dist)
    std_dist_surf = np.std(surf_dist)
    max_dist_surf = np.max(surf_dist)
    min_dist_surf = np.min(surf_dist)
    return avg_dist_surf,std_dist_surf, max_dist_surf, min_dist_surf


input_file = sys.argv[1]
rad = 9. # in Angstroms
rad2 = 15.

xyzcoords,atom_names = read_input_xyz(input_file)
coreshell_ind,core_ind,shell_ind = get_coreshell(xyzcoords,atom_names,rad,rad2)

Ncore=np.count_nonzero(core_ind)
print(Ncore," core atoms")
Nshell = np.count_nonzero(shell_ind)
print(Nshell,"shell atoms")
Ntot = np.count_nonzero(coreshell_ind)
print(Ntot, 'total atoms')

# replace shell Se's with S:
shell_se_ind = np.logical_and(atom_names == 'Se', shell_ind)
atom_names[shell_se_ind] = 'S'

atom_names_coreshell = atom_names[coreshell_ind]
xyz_coreshell = xyzcoords[coreshell_ind]
ctr_coreshell = np.mean(xyz_coreshell,axis=0)
xyz_coreshell = xyz_coreshell - ctr_coreshell # center

atom_names_coreonly = atom_names[core_ind]
xyz_coreonly = xyzcoords[core_ind] - ctr_coreshell

atom_names_shellonly = atom_names[shell_ind]
xyz_shellonly = xyzcoords[shell_ind] - ctr_coreshell

dot_r,dot_std,dot_max,dot_min=avg_radius(xyz_coreshell,atom_names_coreshell,'Cd','S')
core_r,core_std,core_max,core_min=avg_radius(xyz_coreonly,atom_names_coreonly,'Cd','Se')
print('Core radius: ',core_r)
print('Core standard dev: ', core_std)
print('Shell thickness:',dot_r-core_r)
print('Dot radius: ', dot_r)
print('Dot standard dev: ', dot_std)

write_xyz('core_only.xyz',atom_names_coreonly,xyz_coreonly)
write_xyz('shell_only.xyz',atom_names_shellonly,xyz_shellonly)
write_xyz('coreshell.xyz',atom_names_coreshell,xyz_coreshell)

# TO DO: check to make sure no unpassivated se's, comment everything, separate functions into different file


## Cd-Se distance histogram
# plt.figure()
# plt.title("Cd-Se distance")
# plt.hist(cd_se_dists.flatten(),bins=800) # crystal
# # plt.hist(cdse_dists_e.flatten(),bins=800) # optimized
# plt.xlim(0,4)
# plt.show()
