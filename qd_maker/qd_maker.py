import numpy as np
import sys
from matplotlib import pyplot as plt
from qd_helper import *

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
    xyz_ctr = xyz - ctr # center the xyz coordinates
    dist_from_ctr = np.linalg.norm(xyz_ctr,axis=1)
    in_r = dist_from_ctr <= radius # boolean indices for atoms within given radius
    atoms_in_r = atoms[in_r]
    xyz_in_r = xyz_ctr[in_r]

    ind_Cd_r = (atoms_in_r == "Cd") #boolean indices for cd's within radius
    ind_Se_r = (atoms_in_r == "Se")

    # getting distances between all atoms
    all_dists,cd_se_dists,se_cd_dists = get_dists(xyz_in_r,ind_Cd_r,ind_Se_r)
    # getting nearest neighbor info
    all_nn,cd_nn,se_nn=get_nn(cd_se_dists,se_cd_dists,ind_Cd_r,ind_Se_r,3.0,len(atoms_in_r))

    # boolean indices of atoms that are correctly coordinated
    coord_ind = all_nn >= nncutoff

    # reshaping coord_ind to be the shape of the input xyz
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
    ctr0 = np.mean(xyz,axis=0) # center at the avg coordinate of the slab

    Ncd = 100
    Nse = 101
    ctr = ctr0
    nit = 0
    while Ncd != Nse and nit < 500:
        # try building a dot at the actual center
        coord_ind_all=build_dot(xyz,atoms,ctr,radius)

        # count # Cd, Se
        Ncd = np.count_nonzero(atoms[coord_ind_all]=='Cd')
        Nse = np.count_nonzero(atoms[coord_ind_all]=='Se')

        print(Nse,' Se atoms in core')
        print(Ncd,' Cd atoms in core')

        # if not stoichiometric, move the center and try again
        if Ncd != Nse:
            ctr = ctr0 + np.random.random_sample((1,3))

        nit += 1

    if nit == 500:
        print('WARNING: Core build did not converge!')

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
        # build the core
        core_ind_all,core_ctr=get_coreonly(xyz,atoms,core_rad)
        Ncore = np.count_nonzero(core_ind_all)/2
        # try building a shell at the same center as the core
        coreshell_ind_all=build_dot(xyz,atoms,core_ctr,shell_rad)

        #count # cd, se in the shell. if they aren't equal,
        #start over, rebuilding the core & getting a new center for the shell
        Ncdshell = np.count_nonzero(atoms[coreshell_ind_all] == 'Cd') - Ncore
        Nseshell = np.count_nonzero(atoms[coreshell_ind_all] == 'Se') - Ncore

        print(Ncdshell,' Cd in shell only')
        print(Nseshell,' Se in shell only')

        nit2 += 1

    if nit2 == 500:
        print('WARNING: Shell build did not converge!')

    # boolean indices of shell atoms only
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
    xyz_ctr = xyz - np.mean(xyz,axis=0) # center the dot

    # get indices for the 2 atom types
    ind1= (atoms==atom1)
    ind2= (atoms==atom2)
    Nat = len(atoms)

    # get nearest neighbors
    all_dists,dist12,dist21 = get_dists(xyz_ctr,ind1,ind2)
    all_nn,nn1,nn2=get_nn(dist12,dist21,ind1,ind2,cutoff,Nat)

    # determine surface atoms by # nearest neighbors
    surf_xyz = xyz_ctr[all_nn < 4]

    # distance of surface atoms to center
    surf_dist = np.linalg.norm(surf_xyz,axis=1)

    # statistics on surface distances
    avg_dist_surf = np.mean(surf_dist)
    std_dist_surf = np.std(surf_dist)
    max_dist_surf = np.max(surf_dist)
    min_dist_surf = np.min(surf_dist)
    return avg_dist_surf,std_dist_surf, max_dist_surf, min_dist_surf

###
# USER SPECIFIED INFO
###
input_file = sys.argv[1] # xyz file of crystal slab
rad = 9. # core radius, in Angstroms
rad2 = 11. # shell radius, in Angstroms

xyzcoords,atom_names = read_input_xyz(input_file)
coreshell_ind,core_ind,shell_ind = get_coreshell(xyzcoords,atom_names,rad,rad2)

Ncore=np.count_nonzero(core_ind)
Nshell = np.count_nonzero(shell_ind)
Ntot = np.count_nonzero(coreshell_ind)

print('')
print('DOT INFO:')
print(Ncore," core atoms")
print(Nshell,"shell atoms")
print(Ntot, 'total atoms')

# REPLACE SHELL SE'S WITH S
shell_se_ind = np.logical_and(atom_names == 'Se', shell_ind)
atom_names[shell_se_ind] = 'S'

# GETTING DATA FOR CORE, SHELL, BOTH
atom_names_coreshell = atom_names[coreshell_ind]
xyz_coreshell = xyzcoords[coreshell_ind]
ctr_coreshell = np.mean(xyz_coreshell,axis=0)
xyz_coreshell = xyz_coreshell - ctr_coreshell # center at total dot center

atom_names_coreonly = atom_names[core_ind]
xyz_coreonly = xyzcoords[core_ind] - ctr_coreshell # center at total dot center

atom_names_shellonly = atom_names[shell_ind]
xyz_shellonly = xyzcoords[shell_ind] - ctr_coreshell # center at total dot center


# CALCULATING AVG RADIUS
dot_r,dot_std,dot_max,dot_min=avg_radius(xyz_coreshell,atom_names_coreshell,'Cd','S')
core_r,core_std,core_max,core_min=avg_radius(xyz_coreonly,atom_names_coreonly,'Cd','Se')
print('Core radius: ',core_r,'; requested ',rad)
print('Core standard dev: ', core_std)
print('Shell thickness:',dot_r-core_r)
print('Dot radius: ', dot_r, '; reqeuested, ',rad2)
print('Dot standard dev: ', dot_std)

# CHECKING FOR UNPASSIVATED CORE ATOMS
all_dists,cd_se_dists,se_cd_dists = get_dists(xyz_coreshell,atom_names_coreshell=='Cd',atom_names_coreshell != 'Cd')
all_nn,cd_nn,se_nn=get_nn(cd_se_dists,se_cd_dists,atom_names_coreshell=='Cd',atom_names_coreshell!='Cd',3.0,len(atom_names_coreshell))

core_cd = np.logical_and(core_ind,atom_names == 'Cd') # boolean indices for core cd's, shape of crystal slab
core_cd_dot=core_cd[coreshell_ind] # boolean indices for core cd's, shape of core-shell dot
core_se_dot = atom_names_coreshell == 'Se' # boolean indices for core se's (easier b/c shell is S)

core_se_nn = all_nn[core_se_dot]
core_cd_nn = all_nn[core_cd_dot]

unpass_core_cd = np.count_nonzero(core_cd_nn < 4)
unpass_core_se = np.count_nonzero(core_se_nn < 4)

if unpass_core_cd+unpass_core_se > 1:
    print('WARNING: not all surface atoms are passivated! Choose a larger shell')
    print(unpass_core_cd, 'unpassivated Cd')
    print(unpass_core_se, 'unpassivated Se')


# MAKING XYZ
write_xyz('core_only.xyz',atom_names_coreonly,xyz_coreonly)
write_xyz('shell_only.xyz',atom_names_shellonly,xyz_shellonly)
write_xyz('coreshell.xyz',atom_names_coreshell,xyz_coreshell)

# TO DO: check to make sure no unpassivated se's, comment everything, separate functions into different file
