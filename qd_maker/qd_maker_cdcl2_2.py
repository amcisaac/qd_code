import numpy as np
import sys
from matplotlib import pyplot as plt
from qd_helper import *
from geom_helper import *

'''
Script to build a core-shell QD from a slab of bulk crystal material.

Usage: python3 qd_maker.py [bulk crystal xyz coordinates]

Please specify a radius and method below.
'''

############
#
# USER SPECIFIED INFO
#
############

input_file = sys.argv[1] # xyz file of crystal slab
rad = 8   # core radius, in Angstroms
rad2 = 12 #15.1  # shell radius, in Angstroms
save=False
save=True
n_ctr =6  # specifies method of choosing the center -- 6 is middle of cage,
              # 3 is middle of plane, 1 is middle of bond, 'cd' is on cd, 'se' is on se
              # 0 is center of xyz, 'random' will determine the center randomly to find a stoichiometric dot

end='ctr'+str(n_ctr)

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
    all_dists,cd_se_dists,garb,garb2,se_cd_dists = get_dists(xyz_in_r,ind_Cd_r,ind_Se_r)
    # getting nearest neighbor info
    all_nn,cd_nn,se_nn=get_nn(cd_se_dists,se_cd_dists,ind_Cd_r,ind_Se_r,3.0,len(atoms_in_r))

    # boolean indices of atoms that are correctly coordinated
    coord_ind = all_nn >= nncutoff

    # reshaping coord_ind to be the shape of the input xyz
    coord_ind_all = dist_from_ctr <= radius
    coord_ind_all[in_r] = coord_ind

    return coord_ind_all

def ctr_qd(xyz,atoms,n_ctr):
    '''
    Function that determines the center of the QD.

    Inputs:
        xyz: xyz coordinates for the crystal slab (np array, size (Natoms,3))
        atoms: list of atom names that correspond to xyz
        n_ctr: method of centering.
               n_ctr = 0 or 'random' returns the average coordinate from xyz
               n_ctr = 'Cd' or 'Se' centers on one atom
               n_ctr = 1 centers the atom in the middle of a cd-se bond
               n_ctr = 3 centers the atom in the middle of a cd3-se3 plane
               n_ctr = 6 centers the atom in the middle of the 3d hexagonal cage
    '''

    xtal_ctr = np.mean(xyz,axis=0) # center at the avg coordinate of the slab
    if n_ctr == 0 or n_ctr=='random':
        return xtal_ctr
    elif n_ctr == 'Cd' or n_ctr == 'Se':
       dist_from_ctr=np.linalg.norm(xyz-xtal_ctr,axis=1)
       ctr_cdi = np.argpartition(dist_from_ctr[atoms==n_ctr],1)[:1]
       ctr_cd = xyz[atoms==n_ctr][ctr_cdi]
       return ctr_cd
    else:
        dist_from_ctr=np.linalg.norm(xyz-xtal_ctr,axis=1)
        ctr_cdi = np.argpartition(dist_from_ctr[atoms=='Cd'],n_ctr)[:n_ctr]
        ctr_sei = np.argpartition(dist_from_ctr[atoms=='Se'],n_ctr)[:n_ctr]
        # print(ctr_ind)
        ctr_cd = xyz[atoms=='Cd'][ctr_cdi]
        ctr_se = xyz[atoms=='Se'][ctr_sei]
        ctr_cdse_xyz = np.concatenate((ctr_cd,ctr_se),axis=0)
        ctr_cdse = np.mean(ctr_cdse_xyz,axis=0)
        return ctr_cdse

def get_coreonly(xyz,atoms,radius,n_ctr):
    '''
    Function to build a stoichiometric core-only dot.

    Inputs:
        xyz: np array of xyz coordinates from which to carve dot.
             e.g. a large crystal slab. Size (Natoms,3)
        atoms: np array of all atom names that correspond to the
               xyz coordinates. Size (Natoms)
        radius: desired radius of the dot (float)
        n_ctr: determines how to calculate the center of the dot & how to find a stoichiometric dot.
               n_ctr = 'random' uses random centers to build a stoichiometric dot
               n_ctr = 0 centers the dot at the center of the xyz coordinates,
                        and finds a stoichiometric dot by increasing the radius
               n_ctr = 'Cd' or 'Se' centers on one atom,
                        and finds a stoichiometric dot by increasing the radius
               n_ctr = 1 centers the atom in the middle of a cd-se bond,
                        and finds a stoichiometric dot by increasing the radius
               n_ctr = 3 centers the atom in the middle of a cd3se3 plane,
                        and finds a stoichiometric dot by increasing the radius
               n_ctr = 6 centers the atom in the middle of the 3d hexagonal cage,
                        and finds a stoichiometric dot by increasing the radius
    Outputs:
        coord_ind_all: boolean numpy array to index xyz and atoms, yielding
                       a core-only QD that is stoichiometric. True elements
                       are atoms that are within the radius and have at
                       least 2 nearest neighbors.
                       e.g. xyz[coord_ind_all] gives xyz coordinates for the
                       resulting dot
        ctr: the coordinates used to center xyz, chosen to produce a stoichiometric dot
    '''

    ctr0=ctr_qd(xyz,atoms,n_ctr)

    Ncd = 100
    Nse = 101
    nit = 0
    it_max = 500
    ctr = ctr0
    while Ncd != Nse and nit < it_max:
        # try building a dot at the actual center
        coord_ind_all=build_dot(xyz,atoms,ctr,radius)

        # count # Cd, Se
        Ncd = np.count_nonzero(atoms[coord_ind_all]=='Cd')
        Nse = np.count_nonzero(atoms[coord_ind_all]=='Se')

        print(Nse,' Se atoms in core')
        print(Ncd,' Cd atoms in core')

        # if not stoichiometric, change the center (if n_ctr = random), or increase the radius (else)
        if Ncd != Nse:
            if n_ctr == 'random':
                ctr = ctr0 + np.random.random_sample((1,3))
            else:
                radius = radius + 0.05
                print('new radius '+radius)
        nit += 1

    if nit == it_max:
        print('WARNING: Core build did not converge!')

    return coord_ind_all,ctr


def get_coreshell(xyz,atoms,core_rad,shell_rad,n_ctr):
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
        core_ind_all,core_ctr=get_coreonly(xyz,atoms,core_rad,n_ctr)
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
        if Ncdshell < 0:
            print('WARNING: core radius exceeds shell radius')
            break

    if nit2 == 500 or Ncdshell < 0:
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
    all_dists,dist12,garb,garb2,dist21 = get_dists(xyz_ctr,ind1,ind2)
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


# read in big lattice
xyzcoords,atom_names = read_input_xyz(input_file)
# get core/shell dot
coreshell_ind,core_ind,shell_ind = get_coreshell(xyzcoords,atom_names,rad,rad2,n_ctr)
# indices of cd, se
coreshell_ind_cd = np.logical_and(coreshell_ind,atom_names=='Cd')
coreshell_ind_se = np.logical_and(coreshell_ind,atom_names=="Se")

Ncore=np.count_nonzero(core_ind)
Nshell = np.count_nonzero(shell_ind)
Ntot = np.count_nonzero(coreshell_ind)

print('')
print('DOT INFO, before modification:')
print(Ncore," core atoms")
print(Nshell,"shell atoms")
print(Ntot, 'total atoms')

xyz_coreshell = xyzcoords[coreshell_ind]
ctr_coreshell = np.mean(xyz_coreshell,axis=0)
xyz_coreshell = xyz_coreshell - ctr_coreshell # center at total dot center
# GETTING DATA FOR CORE, SHELL, BOTH
atom_names_coreshell = atom_names[coreshell_ind]

# REPLACE SHELL SE'S WITH S
shell_se_ind = np.logical_and(atom_names == 'Se', shell_ind)
shell_cd_ind = np.logical_and(atom_names=='Cd', shell_ind)
print('Total number of Cd in shell:',np.count_nonzero(shell_cd_ind) )
# atom_names[shell_se_ind] = 'S'
atom_names_coreshell = atom_names[coreshell_ind]

atom_names_coreonly = atom_names[core_ind]
xyz_coreonly = xyzcoords[core_ind] - ctr_coreshell # center at total dot center

atom_names_shellonly = atom_names[shell_ind]
xyz_shellonly = xyzcoords[shell_ind] - ctr_coreshell # center at total dot center

# get undercoordinated S indices
underc_cd_ind,underc_se_ind=get_underc_index(xyz_coreshell,atom_names_coreshell=='Cd',atom_names_coreshell=='Se',atom_names_coreshell=='N',atom_names_coreshell=='N',3.0,4)
underc_se_ind_lg1 = get_underc_ind_large(atom_names_coreshell=='Se',underc_se_ind)
underc_se_ind_lg = get_underc_ind_large(coreshell_ind,underc_se_ind_lg1)
# find Cd atoms attached to undercoordinated S
all_dists,cdse_dists,cdlig_dists,cdselig_dists,secd_dists = get_dists(xyzcoords,atom_names=='Cd',atom_names=="Se",atom_names=='N')
dist_ucse_tocd=dist_atom12(all_dists, underc_se_ind_lg,atom_names=='Cd') # distances of UC Se atoms to all Cd's
cd_nn_ind = dist_ucse_tocd<3.0 # find Cd that are NN to UC Se
test=np.any(cd_nn_ind,axis=0)  # flatten, and remove duplicates. smaller than cd_nn_ind b/c some Cd are nn of two Se
cd_nn_ind1 = get_underc_ind_large(atom_names=='Cd',test)
print("Number of Cd that are NN to UC S:", np.count_nonzero(cd_nn_ind1))
shell_xor_cd_ind = np.logical_xor(cd_nn_ind1,coreshell_ind_cd ) # indices that are either extra or in dot but not both
extra_cd_ind =np.logical_and(shell_xor_cd_ind,cd_nn_ind1) # indices of *only* extra Cd
print('Number of extra Cd:', np.count_nonzero(extra_cd_ind))
extra_cd_name = atom_names[extra_cd_ind] # just extra
extra_cd_xyz = xyzcoords[extra_cd_ind]
extra_cd_xyz = extra_cd_xyz - ctr_coreshell

# xyz_cscd=xyzcoords[extra_cd_ind]
# atomname_cscd=np.array(atomname_extracd)
ind_extracd = extra_cd_ind
ind_cscd = np.logical_or(extra_cd_ind,coreshell_ind) # indices of core/shell atoms + extra Cd
xyz_cscd=xyzcoords[ind_cscd]
atomname_cscd=atom_names[ind_cscd]
atom_names_cscd=atomname_cscd

print('Number of Cd that are NN to UC S:', np.count_nonzero(ind_extracd))

# remove Cd with 1 NN
underc_cd_ind2,underc_se_ind2=get_underc_index(xyz_cscd,atom_names_cscd=='Cd',atom_names_cscd=='Se',atom_names_cscd=='N',atom_names_cscd=='N',3.0,2)
# print(underc_cd_ind2.shape)
underc_cd_ind2_intermediate = get_underc_ind_large(atom_names_cscd=='Cd',underc_cd_ind2)
# print(underc_cd_ind2_intermediate.shape)
underc_cd_ind2_lg = get_underc_ind_large(ind_cscd,underc_cd_ind2_intermediate)
print('Number of 1C Cd to trim:',np.count_nonzero(underc_cd_ind2_lg))
print(ind_cscd[np.where(underc_cd_ind2_lg)])
ind_cscd[np.where(underc_cd_ind2_lg)]=False # directly modify the extra cd + c/s indices ( don't modify c/s because 1C extras aren't in there anyway)
ind_extracd[np.where(underc_cd_ind2_lg)]=False
atom_names_cscd_trim = atom_names[ind_cscd]
xyz_cscd_trim = xyzcoords[ind_cscd]
print('Number of extra Cd after trimming 1C Cd:',np.count_nonzero(ind_extracd))
print('Total number of Cd after trimming 1C Cd:',np.count_nonzero(atom_names_cscd_trim=='Cd'))


# remove 2C S leftover
print('Total number of S in shell:', np.count_nonzero(shell_se_ind))
underc_cd_ind3,underc_se_ind3=get_underc_index(xyz_cscd_trim,atom_names_cscd_trim=='Cd',atom_names_cscd_trim=='Se',atom_names_cscd_trim=='N',atom_names_cscd_trim=='N',3.0,3)
# print(underc_cd_ind2.shape)
underc_se_ind3_intermediate = get_underc_ind_large(atom_names_cscd_trim=='Se',underc_se_ind3)
# print(underc_cd_ind2_intermediate.shape)
underc_se_ind3_lg = get_underc_ind_large(ind_cscd,underc_se_ind3_intermediate)
print('Number of 2C S to trim:',np.count_nonzero(underc_se_ind3_lg))
print(ind_cscd[np.where(underc_se_ind3_lg)])
ind_cscd[np.where(underc_se_ind3_lg)]=False
coreshell_ind[np.where(underc_se_ind3_lg)]=False
shell_se_ind[np.where(underc_se_ind3_lg)]=False
coreshell_ind_se[np.where(underc_se_ind3_lg)]=False

atom_names_cscd_trim2 = atom_names[ind_cscd]
xyz_cscd_trim2 = xyzcoords[ind_cscd]
# ind_extracd[np.where(underc_se_ind3_lg)]=False
# print('Number of extra Se after trimming 2C Se:',np.count_nonzero(ind_extracd))
print('Total number of Se after trimming 2C S:',np.count_nonzero(shell_se_ind))


# get S coordinated to UC Cd
# TO DO: need to find a way to replace bridging S first, then add terminal...as is it's not charge balanced

underc_cd_ind4,underc_se_ind4=get_underc_index(xyz_cscd_trim2,atom_names_cscd_trim2=='Cd',atom_names_cscd_trim2=='Se',atom_names_cscd_trim2=='N',atom_names_cscd_trim2=='N',3.0,3)
underc_cd_ind_lg4_temp = get_underc_ind_large(atom_names_cscd_trim2=='Cd',underc_cd_ind4)
underc_cd_ind_lg4 = get_underc_ind_large(ind_cscd,underc_cd_ind_lg4_temp)
# find S atoms attached to undercoordinated Cd
# all_dists,cdse_dists,cdlig_dists,cdselig_dists,secd_dists = get_dists(xyzcoords,atom_names=='Cd',atom_names=="Se",atom_names=='N')
dist_uccd_tos=dist_atom12(all_dists, underc_cd_ind_lg4,atom_names=='Se')
s_nn_ind = dist_uccd_tos<3.0
test2=np.any(s_nn_ind,axis=0) # smaller than cd_nn_ind b/c some Cd are nn of two Se
s_nn_ind1 = get_underc_ind_large(atom_names=='Se',test2)
print("Number of S that are NN to UC Cd:", np.count_nonzero(s_nn_ind1))
shell_xor_s_ind = np.logical_xor(s_nn_ind1,coreshell_ind_se)
extra_se_ind =np.logical_and(shell_xor_s_ind,s_nn_ind1) # indices for extra Se ONLY, excludes NN that were already in shell
print('Number of extra S:', np.count_nonzero(extra_se_ind))
extra_s_name = atom_names[extra_se_ind] # just extra
extra_s_xyz = xyzcoords[extra_se_ind]
extra_s_xyz = extra_s_xyz - ctr_coreshell
ind_cscdcl = np.logical_or(extra_se_ind,ind_cscd)
#replace UC S with Cl
atom_names[extra_se_ind]='Cl'

atom_names[shell_se_ind] = 'S'
atom_names_coreshell=atom_names[coreshell_ind]



atom_names_all = atom_names[ind_cscdcl]
xyz_all = xyzcoords[ind_cscdcl]


# atom_names_all[shell_se_ind]= 'S'
# CALCULATING AVG RADIUS
dot_r,dot_std,dot_max,dot_min=avg_radius(xyz_all,atom_names_all,'Cd','S')
core_r,core_std,core_max,core_min=avg_radius(xyz_coreonly,atom_names_coreonly,'Cd','Se')
print('Core radius: ',core_r,'; requested ',rad)
print('Core standard dev: ', core_std)
print('Shell thickness:',dot_r-core_r)
print('Dot radius: ', dot_r, '; reqeuested, ',rad2)
print('Dot standard dev: ', dot_std)
comment = 'r1 = {}A, r2 = {}A, r_c = {}A, std_c = {}, r_tot = {}A, std_tot = {}'.format(rad,rad2,np.around(core_r,3),np.around(core_std,3),np.around(dot_r,3),np.around(dot_std,3))

# CHECKING FOR UNPASSIVATED CORE ATOMS
all_dists,cd_se_dists,garb,garb2,se_cd_dists = get_dists(xyz_all,atom_names_all=='Cd',atom_names_all != 'Cd')
all_nn,cd_nn,se_nn=get_nn(cd_se_dists,se_cd_dists,atom_names_all=='Cd',atom_names_all!='Cd',3.0,len(atom_names_all))

core_cd = np.logical_and(core_ind,atom_names == 'Cd') # boolean indices for core cd's, shape of crystal slab
core_cd_dot=core_cd[ind_cscdcl] # boolean indices for core cd's, shape of core-shell dot
core_se_dot = atom_names_all == 'Se' # boolean indices for core se's (easier b/c shell is S)

core_se_nn = all_nn[core_se_dot]
core_cd_nn = all_nn[core_cd_dot]

unpass_core_cd = np.count_nonzero(core_cd_nn < 4)
unpass_core_se = np.count_nonzero(core_se_nn < 4)

if unpass_core_cd+unpass_core_se > 1:
    print('WARNING: not all surface atoms are passivated! Choose a larger shell')
    print(unpass_core_cd, 'unpassivated Cd')
    print(unpass_core_se, 'unpassivated Se')




if save:
    # MAKING XYZ
    write_xyz('{}_{}_core_only_{}.xyz'.format(rad,rad2,end),atom_names_coreonly,xyz_coreonly,comment)
    write_xyz('{}_{}_shell_only_{}.xyz'.format(rad,rad2,end),atom_names_shellonly,xyz_shellonly,comment)
    write_xyz('{}_{}_coreshell_{}.xyz'.format(rad,rad2,end),atom_names_coreshell,xyz_coreshell,comment)
    # write_xyz('{}_{}_extracd_{}.xyz'.format(rad,rad2,end),extra_cd_name,extra_cd_xyz,comment)
    write_xyz('{}_{}_all_{}.xyz'.format(rad,rad2,end),atom_names_all,xyz_all,comment)
