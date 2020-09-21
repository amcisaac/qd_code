import numpy as np
import numpy.linalg as npl
import sys
from qd_helper import read_input_xyz,write_xyz
from matplotlib import pyplot as plt

def transformation(S):
    '''
    Find the transformation vector, X, that orthogonalizes the basis set
    Uses symmetric orthogonalization, as described in Szabo and Ostlund p 143

    Inputs:
        S = overlap matrix
    Returns:
        X = transformation matrix (S^-.5)
        X_dagger = conjugate transpose of X
    '''
    S_eval, S_evec = npl.eigh(S)                     # extract eigenvalues and eigenvectors from overlap matrix

    s_sqrt = np.diag(np.power(S_eval,-0.5))    # initialize s^-0.5, the diagonalized overlap matrix to the -1/2 power


    U = S_evec                                   # find the unitary transform matrix
    U_dagger = np.transpose(U)                        # find the conjugate transpose of the unitary matrix
    X = np.dot(U, np.dot(s_sqrt, U_dagger))      # form X = S^-0.5, the transform matrix to orthogonalize the basis set
    X_dagger = np.transpose(X)                   # conjugate transpose is just transpose since all values are real
    return X, X_dagger

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

def make_s_matrix(s_file_raw,nbas):
    s_mat = np.zeros((nbas,nbas))
    with open(s_file_raw,'r') as s_file:
        s_lines=s_file.readlines()
        for i in range(0,nbas):
            for j in range(0,nbas):
                s_ind = i*nbas+j
                s_mat[j,i]=float(s_lines[s_ind])

    return s_mat

def get_mul_low(P,S,X_inv,atoms,orb_per_atom,z):
    PS = P@S
    mul_perorb=np.diag(PS)
    SPS = X_inv@P@X_inv
    low_perorb=np.diag(SPS)

    mul=[]
    low=[]
    j = 0
    for atom in atoms:
        nbas_i = orb_per_atom[atom]
        mul_AO = mul_perorb[j:j+nbas_i]
        low_AO = low_perorb[j:j+nbas_i]

        mul.append(np.sum(mul_AO,axis=0))
        low.append(np.sum(low_AO,axis=0))
        j += nbas_i

    mul_charge=z-np.array(mul)
    low_charge=z-np.array(low)
    return mul_charge,low_charge

def make_P_from_MO(mo_mat_unnorm,nocc):
    P=2*mo_mat_unnorm[:,0:nocc]@mo_mat_unnorm[:,0:nocc].T # reproduces qchem
    return P

coeff_file = sys.argv[1] # hex version of qchem 53.0
nbas=int(sys.argv[2])    # number of basis functions
# xyz_file=sys.argv[3]        # whole dot xyz file (must correspond to core coordinates)
# core_xyz_file = sys.argv[4] # core xyz file
s_file=sys.argv[5] # 320
p_file=sys.argv[6] # 54
x_file=sys.argv[7] # 51
atoms=np.array(['H','C','H','H','H'])

# number of orbitals per atom for each atom: 8 Se, 8 S, 18 Cd
orb_per_atom_lanl2dz = {'Cd': 18, 'Se': 8, 'S': 8}
orb_per_atom_sto3g={'C':5,'H':1,'He':1}
orb_per_atom=orb_per_atom_sto3g

# get MO matrix and energies
z=np.array([1,6,1,1,1])

S = make_s_matrix(s_file,nbas)  # 320.0
P = 2*make_s_matrix(p_file,nbas)  # 54.0
mo_mat_unnorm,mo_e = make_mo_coeff_mat(coeff_file,nbas) # 53.0
X,X_dagger=transformation(S)
X_inv = np.linalg.inv(X)

mo_mat= X_inv@mo_mat_unnorm
mo_mat_sum=np.sum(np.power(mo_mat,2),axis=0)
print('MO normalization:',mo_mat_sum)
if not np.all(np.isclose(mo_mat_sum,1)): print("WARNING: MO's not normalized!")
mulliken,lowdin=get_mul_low(P,S,X_inv,atoms,orb_per_atom,z)
print('Mulliken charges for ',atoms,':',mulliken)
print('Lowdin charges for ',atoms,':',lowdin)


'''
# read xyz files
xyz,atoms=read_input_xyz(xyz_file)
# core_xyz,core_atoms=read_input_xyz(core_xyz_file)

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
    # print(j,atom,n_ao)
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
alpha_core = np.sum(np.power(mo_mat_coreonly,2),axis=0)
alpha_shell = np.sum(np.power(mo_mat_shellonly,2),axis=0)

# print(np.sum(np.power(mo_mat,2),axis=1))
# check it adds to 1
print(alpha_core+alpha_shell)
'''
'''
mo_e = mo_e * 27.1
# energy grid to evaluate the DOS over
E_grid = np.arange(0.9*mo_e[0],1.1*mo_e[-1],0.01)
# broadening parameter
sigma=0.1

# alpha_core=np.ones(alpha_core.shape)
# calculate projected DOS
core_dos = dos_grid(E_grid,sigma,mo_e,alpha_core)
shell_dos=dos_grid(E_grid,sigma,mo_e,alpha_shell)

# plot PDOS
plt.figure()
plt.plot(E_grid,core_dos)
plt.plot(E_grid,shell_dos)
# plt.stem(mo_e,alpha_core)
# plt.stem(mo_e, alpha_shell)
plt.xlim(-8,-2)
plt.show()
'''
