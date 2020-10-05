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
    # print(S_eval)
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
        # print(c[i])
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

def get_z(atoms):
    z_dict = {'Cd':48-36,'Se':34-28,'S':16-10,'H':1,'C':6,'He':2}
    z = np.zeros(len(atoms))
    for i,atom in enumerate(atoms):
        z[i]=z_dict[atom]
    return z


def get_mo_partition(mo_mat,ao_inds):
    mo_mat_list = []
    for ind in ao_inds:
        mo_mat_list.append(mo_mat[ind])

    return mo_mat_list

def get_alpha(mo_mat,ao_inds):
    mo_mats=get_mo_partition(mo_mat,ao_inds)
    alpha_list = []
    for mo_mat in mo_mats:
        alpha_x = np.sum(np.power(mo_mat,2),axis=0)
        alpha_list.append(alpha_x)

    return alpha_list


xyz_file=sys.argv[1]        # whole dot xyz file (must correspond to core coordinates)
core_xyz_file = sys.argv[2] # core xyz file
nbas=int(sys.argv[3])       # number of basis functions
nocc=int(sys.argv[4])       # number of occupied basis functions
coeff_file = sys.argv[5]    # hex version of qchem 53.0
s_file=sys.argv[6]          # hex version of qchem 320.0
charge_analysis=False       # make True for mulliken/lowdin analysis (just standard atomic charge)
underc=False

if len(sys.argv)>7:
    underc=True
    cdshell_underc_ind=np.load(sys.argv[7])
    sshell_underc_ind=np.load(sys.argv[8])
    cdshell_underc_ind_amb=np.load(sys.argv[9])
    sshell_underc_ind_amb=np.load(sys.argv[10])


# read xyz files
xyz,atoms=read_input_xyz(xyz_file)
core_xyz,core_atoms=read_input_xyz(core_xyz_file)

# number of orbitals per atom for each atom: 8 Se, 8 S, 18 Cd
orb_per_atom_lanl2dz = {'Cd': 18, 'Se': 8, 'S': 8}
# orb_per_atom_sto3g={'C':5,'H':1,'He':1} # for testing purposes
# orb_per_atom_ccpvdz={'C':14,'H':5,'He':5} # for testing purposes
orb_per_atom=orb_per_atom_lanl2dz

S = make_s_matrix(s_file,nbas)  # 320.0
print('Done building S')

mo_mat_unnorm,mo_e = make_mo_coeff_mat(coeff_file,nbas) # 53.0
print('Done building C')

X,X_dagger=transformation(S)
X_inv = np.linalg.inv(X)
print('Done building X')

P=make_P_from_MO(mo_mat_unnorm,nocc) # to read in, use file 54.0
print('Done building P')

mo_mat= X_inv@mo_mat_unnorm
print('Done with MO normalization')

mo_mat_sum=np.sum(np.power(mo_mat,2),axis=0)
print('MO normalization:',np.all(np.isclose(mo_mat_sum,1)))
if not np.all(np.isclose(mo_mat_sum,1)): print("WARNING: MO's not normalized!")

if charge_analysis:
    mulliken,lowdin=get_mul_low(P,S,X_inv,atoms,orb_per_atom,z)
    print('Mulliken charges for ',atoms,':',mulliken)
    print('Lowdin charges for ',atoms,':',lowdin)


# get core/shell indices, and AO indices
ind_core=np.full(atoms.shape,False)
ind_core_ao = np.full(nbas,False)
ind_cd_ao = np.full(nbas,False)
ind_se_ao = np.full(nbas,False)
ind_s_ao = np.full(nbas,False)
if underc:
    cdshell_underc_ind_ao=np.full(nbas,False)
    sshell_underc_ind_ao=np.full(nbas,False)
    cdshell_underc_ind_amb_ao=np.full(nbas,False)
    sshell_underc_ind_amb_ao=np.full(nbas,False)

j=0
n_ao = 0
for i,coord in enumerate(xyz):
    # print(i)
    j += n_ao
    atom = atoms[i]
    n_ao = orb_per_atom[atom]
    if atom == 'Cd': ind_cd_ao[j:j+n_ao]=True
    if atom == 'Se': ind_se_ao[j:j+n_ao]=True
    if atom == 'S': ind_s_ao[j:j+n_ao]=True

    # print(j,atom,n_ao)
    for coord2 in core_xyz:
        if np.all(coord2==coord):
            # print(coord,j)
            ind_core[i]=True
            ind_core_ao[j:j+n_ao]=True
        else:
            if underc:
                if cdshell_underc_ind[i]==True: cdshell_underc_ind_ao[j:j+n_ao]=True
                if sshell_underc_ind[i]==True: sshell_underc_ind_ao[j:j+n_ao]=True
                if cdshell_underc_ind_amb[i]==True: cdshell_underc_ind_amb_ao[j:j+n_ao]=True
                if sshell_underc_ind_amb[i]==True: sshell_underc_ind_amb_ao[j:j+n_ao]=True

ind_shell = np.logical_not(ind_core)
ind_shell_ao = np.logical_not(ind_core_ao)

ind_Cd = (atoms == 'Cd')
ind_Se = (atoms == 'Se')
ind_S = (atoms=='S')

ind_cd_core_ao=np.logical_and(ind_cd_ao,ind_core_ao)
ind_cd_shell_ao=np.logical_and(ind_cd_ao,ind_shell_ao)

'''
get squared coefficients on core,shell
'''
alpha_cd_core,alpha_cd_shell,alpha_se,alpha_s = get_alpha(mo_mat,[ind_cd_core_ao,ind_cd_shell_ao,ind_se_ao,ind_s_ao])
alpha_cd = alpha_cd_core + alpha_cd_shell
alpha_core = alpha_cd_core + alpha_se
alpha_shell = alpha_cd_shell + alpha_s
print('Alphas add to 1?:',np.all(np.isclose(alpha_cd_core+alpha_cd_shell+alpha_se+alpha_s,1)))


'''
calculate projected DOS
'''
mo_e = mo_e * 27.2114 # MO energy, in eV
E_grid = np.arange(1.5*mo_e[0],.5*mo_e[-1],0.001) # energy grid to evaluate the DOS over
sigma=0.1 # broadening parameter

se_dos=dos_grid(E_grid,sigma,mo_e,alpha_se)
s_dos=dos_grid(E_grid,sigma,mo_e,alpha_s)
cd_core_dos = dos_grid(E_grid,sigma,mo_e,alpha_cd_core)
cd_shell_dos = dos_grid(E_grid,sigma,mo_e,alpha_cd_shell)
cd_dos = cd_core_dos + cd_shell_dos
core_dos = se_dos + cd_core_dos
shell_dos = cd_shell_dos + s_dos
#
#
# if underc:
#     sshell_underc = sshell_underc_ind_ao #np.logical_or(sshell_underc_ind_ao,sshell_underc_ind_amb_ao)
#     cdshell_underc = cdshell_underc_ind_ao #np.logical_or(cdshell_underc_ind_ao,cdshell_underc_ind_amb_ao)
#
#     sshell_nouc = np.logical_xor(sshell_underc,ind_s_ao)
#     cdshell_nouc = np.logical_xor(cdshell_underc,ind_cd_shell)
#
#     mo_mat_sshell_uc=mo_mat[sshell_underc]
#     mo_mat_cdshell_uc=mo_mat[cdshell_underc]
#
#     mo_mat_sshell_fc=mo_mat[sshell_nouc]
#     mo_mat_cdshell_fc=mo_mat[cdshell_nouc]
#
#     alpha_cd_uc = np.sum(np.power(mo_mat_cdshell_uc,2),axis=0)
#     alpha_s_uc = np.sum(np.power(mo_mat_sshell_uc,2),axis=0)
#
#
#     alpha_cd_fc = np.sum(np.power(mo_mat_cdshell_fc,2),axis=0)
#     alpha_s_fc = np.sum(np.power(mo_mat_sshell_fc,2),axis=0)
#     alpha_shell_fc=alpha_cd_fc+alpha_s_fc
#
#     print('Alphas add to 1?:',np.all(np.isclose(alpha_shell_fc+alpha_cd_uc+alpha_s_uc+alpha_core,1)))
#     print(alpha_cd_fc+alpha_cd_uc+alpha_s_fc+alpha_s_uc+alpha_core)
#
#     cd_uc_dos = dos_grid(E_grid,sigma,mo_e,alpha_cd_uc)
#     s_uc_dos = dos_grid(E_grid,sigma,mo_e,alpha_s_uc)
#     shell_fc_dos = dos_grid(E_grid,sigma,mo_e,alpha_shell_fc)
#     # s_fc_dos = dos_grid(E_grid,sigma,mo_e,alpha_s_fc)
#
# plot PDOS
# plt.figure()
# plt.plot(E_grid,core_dos,'b',label='Core')
# plt.plot(E_grid,shell_dos,'r',label='Shell')
# plt.plot(E_grid,core_dos+shell_dos,'k',label='Total')
# plt.legend()
# plt.xlim(-8,-2)
# plt.ylim(0,100)
# plt.ylabel('Density of States')
# plt.xlabel('Orbital Energy (eV)')
# plt.show()
# #
# # # plot PDOS
# plt.figure()
# plt.plot(E_grid,cd_dos,'c',label='Cd')
# plt.plot(E_grid,se_dos,color='orange',label='Se')
# plt.plot(E_grid,s_dos,'y',label='S')
# plt.plot(E_grid,cd_dos+se_dos+s_dos,'k',label='Total')
# plt.legend()
# plt.xlim(-8,-2)
# plt.ylim(0,100)
# plt.ylabel('Density of States')
# plt.xlabel('Orbital Energy (eV)')
# plt.show()
# #
plt.figure()
plt.plot(E_grid,cd_core_dos,'c',label='Cd (core)')
plt.plot(E_grid,cd_shell_dos,'m',label='Cd (shell)')
plt.plot(E_grid,se_dos,color='orange',label='Se')
plt.plot(E_grid,s_dos,'y',label='S')
plt.plot(E_grid,cd_core_dos+cd_shell_dos+se_dos+s_dos,'k',label='Total')
plt.legend()
plt.xlim(-8,-2)
plt.ylim(0,100)
plt.ylabel('Density of States')
plt.xlabel('Orbital Energy (eV)')
plt.show()
# #
# # # plot PDOS
# # plt.figure()
# # plt.plot(E_grid,se_dos,color='orange',label='Se')
# # plt.plot(E_grid,core_dos,'b',label='Core')
# # plt.plot(E_grid,cd_dos+se_dos+s_dos,'k',label='Total')
# # plt.legend()
# # plt.xlim(-8,-2)
# # plt.ylim(0,100)
# # plt.ylabel('Density of States')
# # plt.xlabel('Orbital Energy (eV)')
# # plt.show()
#
# # plot PDOS
# plt.figure()
# plt.plot(E_grid,cd_uc_dos,color='cyan',label='2-c Cd')
# plt.plot(E_grid,s_uc_dos,color='gold',label='2-c S')
# plt.plot(E_grid,shell_fc_dos,color='r',label='3/4-c shell')
# # plt.plot(E_grid,s_fc_dos,'b',label='3/4-c S')
# plt.plot(E_grid,core_dos,'b',label='Core')
# plt.plot(E_grid,cd_uc_dos+s_uc_dos+shell_fc_dos+core_dos,'k',label='Total')
# plt.legend()
# plt.xlim(-8,-2)
# plt.ylim(0,100)
# plt.ylabel('Density of States')
# plt.xlabel('Orbital Energy (eV)')
# plt.show()
