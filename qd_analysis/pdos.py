import numpy as np
import sys
from qd_helper import read_input_xyz,write_xyz
from matplotlib import pyplot as plt

def dos_grid(E_grid, sigma, E_orb, c):
    '''
    Function to calculate the weighted DOS based on the orbital energies
    and fraction of the orbital on the core/shell.

    Inputs:
        E_grid: energy grid to calculate the DOS over
        sigma: broadening parameter for the gaussians
        E_orb: array of all the orbital energies
        c: array of the fraction of each orbital on the core (or shell) atoms

    Returns:
        dos_grid: the gaussian braodened DOS, weighted by c
    '''
    dos_grid=np.zeros(E_grid.shape)
    for i in range(0,len(c)):
        dos_grid += (c[i]/np.sqrt(2*np.pi*sigma**2))*np.exp(-(E_grid-E_orb[i])**2/(2*sigma**2))
    return dos_grid

def make_mo_coeff_mat(mo_file_raw,nbas):
    '''
    Function to build the MO coefficient matrix from a QChem file

    Inputs:
        mo_file_raw: MO coefficient file from QChem (53.0), with each line having a new coefficient.
        nbas: number of basis functions

    Returns:
        mo_coeff_mat: MO coefficient matrix with MO's as columns and AO's as rows
                      (same order as atoms in xyz). Nbas x Nbas
        mo_Es: array of MO energies
    '''
    mo_coeff_mat = np.zeros((nbas,nbas))
    mo_Es = np.zeros(nbas)
    with open(mo_file_raw,'r') as mo_file:
        mo_lines=mo_file.readlines()
        for i in range(0,nbas): # row
            e_ind = 2 * nbas * nbas + i
            mo_Es[i] = float(mo_lines[e_ind]) # energy
            for j in range(0,nbas): # column
                mo_ind = i * nbas + j
                mo_coeff_mat[j,i] = float(mo_lines[mo_ind]) # coeff
    return mo_coeff_mat,mo_Es


coeff_file = sys.argv[1] # hex version of qchem 53.0
nbas=int(sys.argv[2])    # number of basis functions
xyz_file=sys.argv[3]        # whole dot xyz file (must correspond to core coordinates)
core_xyz_file = sys.argv[4] # core xyz file

# number of orbitals per atom for each atom: 8 Se, 8 S, 18 Cd
orb_per_atom_lanl2dz = {'Cd': 18, 'Se': 8, 'S': 8,'A':2, 'B': 3,'C':4}

# get MO matrix and energies
mo_mat,mo_e = make_mo_coeff_mat(coeff_file,nbas)
# print(mo_mat)
# print(mo_e)

# read xyz files
xyz,atoms=read_input_xyz(xyz_file)
core_xyz,core_atoms=read_input_xyz(core_xyz_file)

# get core/shell indices, and AO indices
ind_core=np.full(atoms.shape,False)
ind_core_ao = np.full(nbas,False)
j=0
n_ao = 0
for i,coord in enumerate(xyz):
    # print(i)
    j += n_ao
    atom = atoms[i]
    n_ao = orb_per_atom_lanl2dz[atom]
    print(j,atom,n_ao)
    for coord2 in core_xyz:
        if np.all(coord2==coord):
            # print(coord,j)
            ind_core[i]=True
            ind_core_ao[j:j+n_ao]=True

ind_shell = np.logical_not(ind_core)
ind_shell_ao = np.logical_not(ind_core_ao)

# separate the MO matrix into core and shell contributions
mo_mat_coreonly = mo_mat[ind_core_ao] # N MO (col) x N core AO's (row)
mo_mat_shellonly = mo_mat[ind_shell_ao] # N MO (col) x N shell AO's (row)

# sum down the columns to get the total fraction of the orbital on core/shell
alpha_core = np.sum(mo_mat_coreonly,axis=0)
alpha_shell = np.sum(mo_mat_shellonly,axis=0)
# check it adds to 1
print(alpha_core+alpha_shell)

# energy grid to evaluate the DOS over
E_grid = np.arange(0.9*mo_e[0],1.1*mo_e[-1],0.01)
# broadening parameter
sigma=0.01

# calculate projected DOS
core_dos = dos_grid(E_grid,sigma,mo_e,alpha_core)
shell_dos=dos_grid(E_grid,sigma,mo_e,alpha_shell)

# plot PDOS
plt.figure()
plt.plot(E_grid,core_dos)
plt.plot(E_grid,shell_dos)
plt.stem(mo_e,alpha_core)
plt.stem(mo_e, alpha_shell)
plt.show()
