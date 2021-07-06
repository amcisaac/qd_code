###
# from pdos.py
###
import numpy as np
import numpy.linalg as npl

#######
#
# Building/normalizing MOs
#
#######

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
    U_dagger = np.transpose(U)                   # find the conjugate transpose of the unitary matrix
    X = np.dot(U, np.dot(s_sqrt, U_dagger))      # form X = S^-0.5, the transform matrix to orthogonalize the basis set
    X_dagger = np.transpose(X)                   # conjugate transpose is just transpose since all values are real
    return X, X_dagger

def make_mo_coeff_mat(mo_file_raw,nbas):
    '''
    Function to build the MO coefficient matrix from a QChem file.

    Inputs:
        mo_file_raw: MO coefficient file from QChem (53.0), with each line having a new coefficient.
        nbas: number of basis functions

    Returns:
        mo_coeff_mat: MO coefficient matrix with MO's as columns and AO's as rows
                      (same order as atoms in xyz). Nbas x Nbas. Not normalized
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
    '''
    Function to build S matrix from txt version of 320.0

    Inputs:
        s_file_raw: file name for ascii version of 320.0, obtained via:
                    hexdump -v -e '1/8 "% .16e " "\n"' 320.0 > 320.txt
        nbas: (int) number of basis functions
    Returns:
        s_mat: numpy array of overlap matrix (S). size nbas x nbas
    '''
    s_mat = np.zeros((nbas,nbas))
    with open(s_file_raw,'r') as s_file:
        s_lines=s_file.readlines()
        for i in range(0,nbas):
            for j in range(0,nbas):
                s_ind = i*nbas+j
                s_mat[j,i]=float(s_lines[s_ind])

    return s_mat


def make_P_from_MO(mo_mat_unnorm,nocc):
    '''
    Function to calculate the density matrix from the MO matrix.

    Inputs:
        mo_mat_unnorm: numpy array with unnormalized MO coeff matrix (from 53.0)
        nocc: (int) number of occupied orbitals
    '''
    P=2*mo_mat_unnorm[:,0:nocc]@mo_mat_unnorm[:,0:nocc].T # reproduces qchem
    return P


def build_S_mo(s_file,coeff_file,nbas,nocc):
    '''
    Integrates above functions to build the overlap matrix and normalized MO
    coefficient matrices

    Inputs:
        s_file: file name for ASCII version of QChem 320.0, obtained via:
                hexdump -v -e '1/8 "% .16e " "\n"' 320.0 > 320.txt
        coeff_file: file name for ASCII version of QChem 53.0, obtained via:
                hexdump -v -e '1/8 "% .16e " "\n"' 53.0 > 53.txt
        nbas: total number of basis functions (int)
        nocc: total number of occupied basis functions (int)
    '''
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

    mo_mat_sum=np.sum(np.power(mo_mat,2),axis=0)
    if not np.all(np.isclose(mo_mat_sum,1)): raise ValueError("MO's not normalized!")
    print('Done with MO normalization')

    return mo_mat,mo_e,S

###########
#
# Functions for calculting the PDOS
#
###########

def get_ao_ind(ind_list,atoms,nbas,orb_per_atom):
    '''
    Function that takes a list of indices of different atoms (Natoms) and returns the
    indices of those atom's AOs (Nbas).

    Inputs:
        ind_list: list of Boolean index arrays for each atom or partition. Each
                array should be size (Natoms).
        atoms: list or array of all the atoms in the molecule
        nbas: number of basis functions
        orb_per_atom: dictionary of the number of AO's for each atom in the system.

    Returns:
        ao_ind_list: list of Boolean index arrays indicating the AOs for each
                     atom/partition. In the same order as ind_list. Each array is
                     size (Nbas) and is True where a given atom's AOs are.
    '''
    # initialize arrays for AO indices
    ao_ind_list = [] # pass anything that you want to partition as a list of indices
    for ind in ind_list: # ind = indices for each type of atom
        ao_ind_list.append(np.full(nbas,False))

    j=0
    n_ao = 0
    # loop through all atoms in the molecule
    for i,atom in enumerate(atoms):
        # bookkeeping on number of AO's, etc
        j += n_ao
        n_ao = orb_per_atom[atom]
        # this loop broadcasts the Boolean indexing array from each
        # type of atom to index the AOs
        for k,ind in enumerate(ind_list): # loops through different atom types via indices
            # if atom i is the type of atom we're making the AO list for,
            # set the AO list to true for all AO's for that atom.
            if ind[i]: ao_ind_list[k][j:j+n_ao]=True

    if len(ao_ind_list)==1:
        return ao_ind_list[0] # if there's just one, don't return a list
    else:
        return ao_ind_list

def get_mo_partition(mo_mat,ao_inds):
    '''
    Function that takes a list of ao_inds for various atom partitions, and
    returns the subset of the MO matrix that includes only these AOs.
    '''
    mo_mat_list = []
    for ind in ao_inds:
        mo_mat_list.append(mo_mat[ind])
    return mo_mat_list

def get_alpha(mo_mat,ao_inds):
    '''
    Function to calculate the fraction of each orbital (c, alpha) on a given
    atom or set of atoms.

    Inputs:
        mo_mat: MO coefficient matrix, symmetrically orthonormalized.
        ao_inds: list of AO index arrays for different partitions of atoms.

    Returns:
        alpha_list: list of alpha arrays, containing the fraction of each orbital
                    on a partition of atoms. in the same order as ao_ind list
    '''
    mo_mats=get_mo_partition(mo_mat,ao_inds)
    alpha_list = []
    for mo_mat in mo_mats:
        alpha_x = np.sum(np.power(mo_mat,2),axis=0)
        alpha_list.append(alpha_x)
    return alpha_list


def dos_grid_1(E_grid, sigma, E_orb, c):
    '''
    NOTE: very slow, don't use
    Function to calculate the weighted DOS based on the orbital energies
    and fraction of the orbital on the core/shell. NOTE: only does one partition!
    e.g. you specify only the c/alpha for one partition (say Cd) and have to run
    again for each other partition (e.g. S, Se, ...), and it doesn't make use of
    matrix math. This makes it very slow/clunky.

    Inputs:
        E_grid: energy grid to calculate the DOS over
        sigma: broadening parameter for the gaussians
        E_orb: array of all the orbital energies
        c: array of the fraction of each orbital on a specific partition of
           atoms (also called alpha)

    Returns:
        dos_grid: the gaussian braodened DOS, weighted by c
    '''
    dos_grid=np.zeros(E_grid.shape)
    for i in range(0,len(c)):
        dos_grid += (c[i]/np.sqrt(2*np.pi*sigma**2))*np.exp(-(E_grid-E_orb[i])**2/(2*sigma**2))
    return dos_grid

def dos_grid_general(E_grid, sigma, E_orb, c_list):
    '''
    FAST AND GENERAL TO ANY SYSTEM!
    Function to calculate the weighted DOS based on the orbital energies
    and fraction of the orbital on the core/shell. Much faster than dos_grid
    and general to any system/set of partitions specified in c_list.

    Inputs:
        E_grid: energy grid to calculate the DOS over
        sigma: broadening parameter for the gaussians
        E_orb: array of all the orbital energies
        c_list: list of arrays of the fraction of each orbital (alpha) on
                each partition of atoms.

    Returns:
        dos_list: list of the gaussian braodened DOS, weighted by c (alpha), for
                  each partition of atoms. returned in the same order as c_list.
    '''
    # reshapes c matrix from (Nbas) to (Nbas,1) for each partition
    c_rs = []
    for c_i in c_list:
        c_rs.append(np.reshape(c_i,(c_i.shape[0],1)))

    # broadcast the energy grid so that it's repeated Norb times
    # Shape = Norb rows x len(E_grid) columns--each row is one copy of E_grid
    E_grid_reshape=np.broadcast_to(E_grid,(len(E_orb),len(E_grid)))

    # broadcast the orbital energies so that it's repeated len(E_grid) times
    # then transpose so that it's Norb rows x len(E_grid) columns
    E_orb_reshape=np.broadcast_to(E_orb,(len(E_grid),len(E_orb))).T

    # energy difference between the orbital energies and the grid
    # row i is the delta E grid for orbital i
    deltaE=E_grid_reshape-E_orb_reshape
    # delta E squared
    deltaE2 = np.power(deltaE,2)
    # calculate gaussians
    exp_eorb = np.exp(-(deltaE2)/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)

    # weight gaussians by c (alpha) and sum gaussians for each partition
    dos_list = []
    for i,c_i_rs in enumerate(c_rs):
        dos_grid_i = np.sum(c_i_rs * exp_eorb,axis=0)
        dos_list.append(dos_grid_i)

    return dos_list



##########
#
# Core/shell specific
#
#########

def get_cs_ind_ao_underc(atoms,core_ind,nbas,orb_per_atom,cdshell_underc_ind,sshell_underc_ind,cdshell_underc_amb,sshell_underc_amb):
    # get core/shell indices, and AO indices
    cdshell_underc_ind_ao=np.full(nbas,False)
    sshell_underc_ind_ao=np.full(nbas,False)
    cdshell_underc_ind_amb_ao=np.full(nbas,False)
    sshell_underc_ind_amb_ao=np.full(nbas,False)

    j=0
    n_ao = 0
    for i,atom in enumerate(atoms):
        # print(i)
        j += n_ao
        atom = atoms[i]
        n_ao = orb_per_atom[atom]

        if not core_ind[i]:
            if cdshell_underc_ind[i]==True: cdshell_underc_ind_ao[j:j+n_ao]=True
            if sshell_underc_ind[i]==True: sshell_underc_ind_ao[j:j+n_ao]=True
            if cdshell_underc_amb[i]==True: cdshell_underc_ind_amb_ao[j:j+n_ao]=True
            if sshell_underc_amb[i]==True: sshell_underc_ind_amb_ao[j:j+n_ao]=True

    return cdshell_underc_ind_ao,sshell_underc_ind_ao,cdshell_underc_ind_amb_ao,sshell_underc_ind_amb_ao

def get_cs_ind(full_xyz,core_xyz,atoms,ind_lig=False):
    ind_core=np.full(atoms.shape,False)
    for i,coord in enumerate(full_xyz):
        for coord2 in core_xyz:
            if np.all(coord2==coord):
                ind_core[i]=True
    ind_shell = np.logical_not(ind_core)
    if np.any(ind_lig):
        ind_shell = np.logical_and(np.logical_not(ind_core),np.logical_not(ind_lig))
    return ind_core,ind_shell


def get_cs_ind_ao(full_xyz,core_xyz,atoms,nbas,orb_per_atom,lig_atom=False):
    ind_Cd = (atoms == 'Cd')
    ind_Se = (atoms == 'Se')
    ind_S = (atoms=='S')
    # get core/shell indices, and AO indices
    ind_core=np.full(atoms.shape,False)
    ind_core_ao = np.full(nbas,False)
    ind_cd_ao = np.full(nbas,False)
    ind_se_ao = np.full(nbas,False)
    ind_s_ao = np.full(nbas,False)

    if lig_atom:
        ind_lig = np.logical_or(np.logical_or(np.logical_or((atoms=='N'), (atoms == 'C')),(atoms == 'H')),(atoms=='Cl'))
        ind_lig_ao = np.full(nbas,False)
    else:
        ind_lig = np.full(atoms.shape,False)
        ind_lig_ao = np.full(nbas,False)

    j=0
    n_ao = 0
    for i,coord in enumerate(full_xyz):
        # print(i)
        j += n_ao
        atom = atoms[i]
        n_ao = orb_per_atom[atom]
        if atom == 'Cd': ind_cd_ao[j:j+n_ao]=True
        if atom == 'Se': ind_se_ao[j:j+n_ao]=True
        if atom == 'S': ind_s_ao[j:j+n_ao]=True
        if atom in ['N','C','H','Cl']: ind_lig_ao[j:j+n_ao]=True

        for coord2 in core_xyz:
            if np.all(coord2==coord):
                ind_core[i]=True
                ind_core_ao[j:j+n_ao]=True

    ind_shell = np.logical_not(ind_core)
    ind_shell_ao = np.logical_not(ind_core_ao)

    return ind_Cd,ind_Se,ind_S,ind_core,ind_shell,ind_lig,ind_cd_ao,ind_se_ao,ind_s_ao,ind_core_ao,ind_shell_ao,ind_lig_ao

def dos_grid_cdses(E_grid, sigma, E_orb, c_cd_core,c_cd_shell,c_se,c_s,c_lig):
    '''
    SPECIFIC TO CORE/SHELL CDSE/CDS!
    Function to calculate the weighted DOS based on the orbital energies
    and fraction of the orbital on the core, shell, ligands, etc for core/shell
    CdSe/CdS. Matrix math makes it much faster than dos_grid_1.

    Inputs:
        E_grid: energy grid to calculate the DOS over
        sigma: broadening parameter for the gaussians
        E_orb: array of all the orbital energies
        c_cd_core: array of the fraction of each orbital (alpha) on the core Cd atoms
        c_cd_shell: array of the fraction of each orbital (alpha) on the shell Cd atoms
        c_se: array of the fraction of each orbital (alpha) on Se atoms
        c_s: array of the fraction of each orbital (alpha) on S atoms
        c_lig: array of the fraction of each orbital (alpha) on the ligand atoms

    Returns:
        dos_grid_x: the gaussian braodened DOS, weighted by c (alpha) for cd_core,
                    cd_shell, se, s, and ligand atoms (in that order)
    '''
    # reshapes c matrix from (Nbas) to (Nbas,1)
    c_cd_core_rs = np.reshape(c_cd_core,(c_cd_core.shape[0],1))
    c_cd_shell_rs = np.reshape(c_cd_shell,(c_cd_shell.shape[0],1))
    c_se_rs = np.reshape(c_se,(c_se.shape[0],1))
    c_s_rs = np.reshape(c_s,(c_s.shape[0],1))
    c_lig_rs = np.reshape(c_lig,(c_lig.shape[0],1))

    # broadcast the energy grid so that it's repeated Norb times
    # Shape = Norb rows x len(E_grid) columns--each row is one copy of E_grid
    E_grid_reshape=np.broadcast_to(E_grid,(len(E_orb),len(E_grid)))

    # broadcast the orbital energies so that it's repeated len(E_grid) times
    # then transpose so that it's Norb rows x len(E_grid) columns
    E_orb_reshape=np.broadcast_to(E_orb,(len(E_grid),len(E_orb))).T

    # energy difference between the orbital energies and the grid
    # row i is the delta E grid for orbital i
    deltaE=E_grid_reshape-E_orb_reshape
    # energy dif squared
    deltaE2 = np.power(deltaE,2)
    # calculate the gaussians
    exp_eorb = np.exp(-(deltaE2)/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)

    # weight gaussians by c (alpha) and sum gaussians
    dos_grid_cd_core=np.sum(c_cd_core_rs*exp_eorb,axis=0)
    dos_grid_cd_shell=np.sum(c_cd_shell_rs*exp_eorb,axis=0)
    dos_grid_se=np.sum(c_se_rs*exp_eorb,axis=0)
    dos_grid_s=np.sum(c_s_rs*exp_eorb,axis=0)
    dos_grid_lig = np.sum(c_lig_rs*exp_eorb,axis=0)

    return dos_grid_cd_core,dos_grid_cd_shell,dos_grid_se,dos_grid_s,dos_grid_lig

###########
#
# InP specific
#
###########

def get_inp_ind_ao(full_xyz,atoms,nbas,orb_per_atom,lig,lig2=False,lig3=False):
    ind_In = (atoms == 'In')
    ind_P = (atoms == 'P')
    ind_Ga = (atoms=='Ga')
    ind_lig = (atoms==lig)

    # get core/shell indices, and AO indices
    ind_in_ao = np.full(nbas,False)
    ind_p_ao = np.full(nbas,False)
    ind_ga_ao = np.full(nbas,False)
    ind_lig_ao = np.full(nbas,False)
    ind_lig2_ao = np.full(nbas,False)

    j=0
    n_ao = 0
    for i,coord in enumerate(full_xyz):
        # print(i)
        j += n_ao
        atom = atoms[i]
        n_ao = orb_per_atom[atom]
        if atom == 'In': ind_in_ao[j:j+n_ao]=True
        if atom == 'P': ind_p_ao[j:j+n_ao]=True
        if atom == 'Ga': ind_ga_ao[j:j+n_ao]=True
        if atom == lig: ind_lig_ao[j:j+n_ao]=True
        if atom == lig2: ind_lig2_ao[j:j+n_ao]=True

        # for OH--just add in to O/lig2
        if atom == lig3: ind_lig2_ao[j:j+n_ao]=True

    return ind_In,ind_P,ind_Ga,ind_lig,ind_in_ao,ind_p_ao,ind_ga_ao,ind_lig_ao,ind_lig2_ao


def dos_grid_inp(E_grid, sigma, E_orb, c_in,c_ga,c_p,c_lig,c_lig2):
    '''
    SPECIFIC TO INP AND INGAP!
    Function to calculate the weighted DOS based on the orbital energies
    and fraction of the orbital on the core/shell. Faster than dos_grid_1

    Inputs:
        E_grid: energy grid to calculate the DOS over
        sigma: broadening parameter for the gaussians
        E_orb: array of all the orbital energies
        c_in: array of the fraction of each orbital on In
        c_ga: array of fraction of each orbital on Ga
        c_p: array of fraction of each orbital on p
        c_lig: array of fraction of each orbital on ligands
        c_lig2: array of fraction of each orbital on lig2 + lig3

    Returns:
        dos_grid_x: the gaussian braodened DOS, weighted by c, for different atoms
    '''
    # reshapes c matrix from (Nbas) to (Nbas,1)
    c_in_rs = np.reshape(c_in,(c_in.shape[0],1))
    c_ga_rs = np.reshape(c_ga,(c_ga.shape[0],1))
    c_p_rs = np.reshape(c_p,(c_p.shape[0],1))
    c_lig_rs = np.reshape(c_lig,(c_lig.shape[0],1))
    c_lig2_rs = np.reshape(c_lig2,(c_lig2.shape[0],1))

    # broadcast the energy grid so that it's repeated Norb times
    # Shape = Norb rows x len(E_grid) columns--each row is one copy of E_grid
    E_grid_reshape=np.broadcast_to(E_grid,(len(E_orb),len(E_grid)))

    # broadcast the orbital energies so that it's repeated len(E_grid) times
    # then transpose so that it's Norb rows x len(E_grid) columns
    E_orb_reshape=np.broadcast_to(E_orb,(len(E_grid),len(E_orb))).T

    # energy difference between the orbital energies and the grid
    # row i is the delta E grid for orbital i
    deltaE=E_grid_reshape-E_orb_reshape
    # energy dif squared
    deltaE2 = np.power(deltaE,2)
    # calculate the gaussians
    exp_eorb = np.exp(-(deltaE2)/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)

    # weight gaussians by c (alpha) and sum gaussians
    dos_grid_in=np.sum(c_in_rs*exp_eorb,axis=0)
    dos_grid_ga=np.sum(c_ga_rs*exp_eorb,axis=0)
    dos_grid_p=np.sum(c_p_rs*exp_eorb,axis=0)
    dos_grid_lig=np.sum(c_lig_rs*exp_eorb,axis=0)
    dos_grid_lig2=np.sum(c_lig2_rs*exp_eorb,axis=0)

    return dos_grid_in,dos_grid_ga,dos_grid_p,dos_grid_lig,dos_grid_lig2

###########
#
# Functions for Mul/Low charges--for debugging MO normalization mostly
#
###########
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
        mul_AO = mul_perorb[j:j+nbas_i] # mulliken charge for atom i, per orbital
        low_AO = low_perorb[j:j+nbas_i] # lowdin charge for atom i, per orbital

        mul.append(np.sum(mul_AO,axis=0)) # total mul = sum over orbitals for atom i, add this to list of charges
        low.append(np.sum(low_AO,axis=0))
        j += nbas_i

    mul_charge=z-np.array(mul)
    low_charge=z-np.array(low)
    return mul_charge,low_charge

def get_z(atoms):
    z_dict = {'Cd':48-36,'Se':34-28,'S':16-10,'H':1,'C':6,'He':2}
    z = np.zeros(len(atoms))
    for i,atom in enumerate(atoms):
        z[i]=z_dict[atom]
    return z
