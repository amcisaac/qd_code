# import matplotlib as mpl
# mpl.use('Agg')
import numpy as np
import numpy.linalg as npl
import sys
from qd_helper import read_input_xyz,write_xyz
from matplotlib import pyplot as plt
from pdos_helper import *
import time
import copy

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


'''
parsing input
'''
xyz_file=sys.argv[1]        # whole dot xyz file (must correspond to core coordinates)
core_xyz_file = sys.argv[2] # core xyz file
bas_file=sys.argv[3]        # file with n_basis,n_occupied
coeff_file = sys.argv[4]    # numpy version of qchem 53.0
# s_file=sys.argv[5]          # numpy version of qchem 320.0
underc=False
attach_atom=False
attach_atom2=False
attach_atom='N'
# attach_atom2='F'
clash=False
lig_fall = False

if len(sys.argv)>5:
    underc=True
    cdshell_underc_ind=np.load(sys.argv[5])
    sshell_underc_ind=np.load(sys.argv[6])

if len(sys.argv)>7:
    clash=True
    cdshell_clash_ind = np.load(sys.argv[7])
    sshell_clash_ind = np.load(sys.argv[8])

if len(sys.argv)>9:
    lig_fall = True
    lig_fall_ind = np.load(sys.argv[9])


# read xyz files
xyz,atoms=read_input_xyz(xyz_file)
core_xyz,core_atoms=read_input_xyz(core_xyz_file)

#read basis set numbers
nbas,nocc = np.loadtxt(bas_file,dtype=int,unpack=True)
print(nbas,nocc)
# number of orbitals per atom for each atom: 8 Se, 8 S, 18 Cd
orb_per_atom_lanl2dz = {'Cd': 18, 'Se': 8, 'S': 8, 'N':9,'C':9,'H':2,'Cl':8,'F':9}
orb_per_atom=orb_per_atom_lanl2dz

# reading in matrices
if coeff_file.split('.')[-1]=='npz':
    # S = np.load(s_file)
    coeff_file_expand=np.load(coeff_file)
    mo_mat=coeff_file_expand['arr_0'] # normalized
    mo_e = coeff_file_expand['arr_1']

else:
    mo_mat,mo_e,S=build_S_mo(s_file,coeff_file,nbas,nocc)

    np.savez('53.npz',mo_mat,mo_e)
    np.save('320.npy',S)
    raise NameError('Saved 53.txt and 320.txt files as .npz and .npy. Rerun with these .npy files')
#
# check for normalization
mo_mat_sum=np.sum(np.power(mo_mat,2),axis=0)
print('MO normalization:',np.all(np.isclose(mo_mat_sum,1)))
if not np.all(np.isclose(mo_mat_sum,1)): print("WARNING: MO's not normalized!")

'''
get indices for different partitions
'''
# print(atoms[270])
ind_Cd = (atoms=='Cd')
ind_S = (atoms=='S')
ind_Se = (atoms=='Se')
ind_lig = np.logical_or(np.logical_or(np.logical_or(np.logical_or((atoms=='N'), (atoms == 'C')),(atoms == 'H')),(atoms=='Cl')),(atoms=='F'))
ind_core,ind_shell = get_cs_ind(xyz,core_xyz,atoms,ind_lig)
ind_cd_core = np.logical_and(ind_core, ind_Cd)
ind_cd_shell = np.logical_and(ind_shell,ind_Cd)

print(np.count_nonzero(ind_core))

ind_cd_ao,ind_se_ao,ind_s_ao,ind_core_ao,ind_shell_ao,ind_lig_ao \
    =get_ao_ind([ind_Cd,ind_Se,ind_S,ind_core,ind_shell,ind_lig],atoms,nbas,orb_per_atom)
ind_cd_core_ao = np.logical_and(ind_core_ao,ind_cd_ao)
ind_cd_shell_ao = np.logical_and(ind_shell_ao,ind_cd_ao)

'''
get squared coefficients on core,shell
'''
alpha_cd_core,alpha_cd_shell,alpha_se,alpha_s,alpha_lig \
    = get_alpha(mo_mat,[ind_cd_core_ao,ind_cd_shell_ao,ind_se_ao,ind_s_ao,ind_lig_ao])
alpha_cd = alpha_cd_core + alpha_cd_shell
alpha_core = alpha_cd_core + alpha_se
alpha_shell = alpha_cd_shell + alpha_s
print('Atom breakdown: Alphas add to 1?:',np.all(np.isclose(alpha_cd_core+alpha_cd_shell+alpha_se+alpha_s+alpha_lig,1)))
print(alpha_shell.shape)
'''
calculate projected DOS
'''
mo_e = mo_e * 27.2114 # MO energy, in eV
E_grid = np.arange(-500,50,0.01) # energy grid to evaluate the DOS over
sigma=0.1 # broadening parameter

cd_core_dos,cd_shell_dos,se_dos,s_dos,lig_dos \
    = dos_grid_general(E_grid, sigma, mo_e, [alpha_cd_core,alpha_cd_shell,alpha_se,alpha_s,alpha_lig])

cd_dos = cd_core_dos + cd_shell_dos
core_dos = se_dos + cd_core_dos
shell_dos = cd_shell_dos + s_dos


'''
repeat for other partitions
'''
if underc:
    # get AO indices
    cdshell_underc_ind_ao,sshell_underc_ind_ao \
        =get_ao_ind([cdshell_underc_ind,sshell_underc_ind],atoms,nbas,orb_per_atom)

    sshell_nouc_ind_ao = np.logical_xor(sshell_underc_ind_ao,ind_s_ao)
    cdshell_nouc_ind_ao = np.logical_xor(cdshell_underc_ind_ao,ind_cd_shell_ao)

    # get alphas
    alpha_cd_uc,alpha_s_uc,alpha_cd_fc,alpha_s_fc \
        = get_alpha(mo_mat,[cdshell_underc_ind_ao,sshell_underc_ind_ao,cdshell_nouc_ind_ao,sshell_nouc_ind_ao])

    alpha_shell_fc=alpha_cd_fc+alpha_s_fc

    print('Undercoordinated: Alphas add to 1?:',np.all(np.isclose(alpha_shell_fc+alpha_cd_uc+alpha_s_uc+alpha_core+alpha_lig,1)))

    # get DOS
    cd_uc_dos,s_uc_dos,cd_fc_dos,s_fc_dos \
        = dos_grid_general(E_grid,sigma,mo_e,[alpha_cd_uc,alpha_s_uc,alpha_cd_fc,alpha_s_fc])
    shell_fc_dos = cd_fc_dos + s_fc_dos

if clash:
    # get AO indices
    cdshell_clash_ind_ao,sshell_clash_ind_ao \
        = get_ao_ind([cdshell_clash_ind,sshell_clash_ind],atoms,nbas,orb_per_atom)

    sshell_noclash_ind_ao = np.logical_xor(sshell_clash_ind_ao,ind_s_ao)
    cdshell_noclash_ind_ao = np.logical_xor(cdshell_clash_ind_ao,ind_cd_shell_ao)

    # get alphas
    alpha_cd_clash,alpha_s_clash,alpha_cd_noclash,alpha_s_noclash \
        = get_alpha(mo_mat,[cdshell_clash_ind_ao,sshell_clash_ind_ao,cdshell_noclash_ind_ao,sshell_noclash_ind_ao])

    alpha_shell_noclash=alpha_cd_noclash+alpha_s_noclash

    print('Clashing: Alphas add to 1?:',np.all(np.isclose(alpha_shell_noclash+alpha_cd_clash+alpha_s_clash+alpha_core+alpha_lig,1)))

    # get DOS
    cd_clash_dos,s_clash_dos,cd_noclash_dos,s_noclash_dos \
        = dos_grid_general(E_grid,sigma,mo_e,[alpha_cd_clash,alpha_s_clash,alpha_cd_noclash,alpha_s_noclash])

    shell_noclash_dos = cd_noclash_dos + s_noclash_dos

    # separate clash vs uc
    # separating clashing/UC
    # clashing AND undercoordinated
    cd_clash_uc_ind_ao = np.logical_and(cdshell_underc_ind_ao,cdshell_clash_ind_ao)
    s_clash_uc_ind_ao = np.logical_and(sshell_underc_ind_ao,sshell_clash_ind_ao)

    cd_uc_only_ind_ao = np.logical_xor(cd_clash_uc_ind_ao,cdshell_underc_ind_ao)
    s_uc_only_ind_ao = np.logical_xor(s_clash_uc_ind_ao,sshell_underc_ind_ao)

    cd_clash_only_ind_ao = np.logical_xor(cd_clash_uc_ind_ao,cdshell_clash_ind_ao)
    s_clash_only_ind_ao = np.logical_xor(s_clash_uc_ind_ao,sshell_clash_ind_ao)

    alpha_cd_clash_uc,alpha_s_clash_uc,alpha_cd_uc_only,alpha_s_uc_only,alpha_cd_clash_only,alpha_s_clash_only \
        = get_alpha(mo_mat,[cd_clash_uc_ind_ao,s_clash_uc_ind_ao,cd_uc_only_ind_ao,s_uc_only_ind_ao,cd_clash_only_ind_ao,s_clash_only_ind_ao])

    # removes atoms that are (undercoordinated AND clashing)
    alpha_shell_no_ucandclash = alpha_shell - alpha_cd_clash_uc - alpha_s_clash_uc
    # removes atoms that are clashing ONLY
    alpha_shell_noclash_only = alpha_shell - alpha_cd_clash_only - alpha_s_clash_only #- alpha_cd_clash_uc - alpha_s_clash_uc
    # removes atoms that are UC ONLY
    alpha_shell_nouc_only = alpha_shell - alpha_cd_uc_only - alpha_s_uc_only #- alpha_cd_clash_uc - alpha_s_clash_uc
    # removes all clashing and UC atoms
    alpha_shell_normal = alpha_shell- alpha_cd_clash_uc - alpha_s_clash_uc - alpha_cd_clash_only - alpha_s_clash_only- alpha_cd_uc_only - alpha_s_uc_only

    print('UC and Clashing: Alphas add to 1?:',np.all(np.isclose(alpha_cd_clash_uc+alpha_s_clash_uc+alpha_shell_no_ucandclash+alpha_core+alpha_lig,1)))
    print('UC only: Alphas add to 1?:',np.all(np.isclose(alpha_cd_uc_only+alpha_s_uc_only+alpha_shell_nouc_only+alpha_core+alpha_lig,1)))
    print('Clashing only: Alphas add to 1?:',np.all(np.isclose(alpha_cd_clash_only+alpha_s_clash_only+alpha_shell_noclash_only+alpha_core+alpha_lig,1)))
    print('All defect atoms: Alphas add to 1?:',np.all(np.isclose(alpha_cd_uc_only+alpha_s_uc_only+alpha_cd_clash_uc+alpha_s_clash_uc+alpha_cd_clash_only+alpha_s_clash_only+alpha_shell_normal+alpha_core+alpha_lig,1)))


    # get DOS
    cd_clash_uc_dos,s_clash_uc_dos,cd_uc_only_dos,s_uc_only_dos,cd_clash_only_dos,s_clash_only_dos \
        = dos_grid_general(E_grid,sigma,mo_e,[alpha_cd_clash_uc,alpha_s_clash_uc,alpha_cd_uc_only,alpha_s_uc_only,alpha_cd_clash_only,alpha_s_clash_only])

    shell_no_ucandclash_dos=shell_dos - cd_clash_uc_dos - s_clash_uc_dos
    shell_noclash_only_dos = shell_dos - cd_clash_only_dos - s_clash_only_dos
    shell_nouc_only_dos = shell_dos - cd_uc_only_dos - s_uc_only_dos
    shell_normal_dos = shell_dos - cd_clash_uc_dos - s_clash_uc_dos- cd_clash_only_dos - s_clash_only_dos- cd_uc_only_dos - s_uc_only_dos


if lig_fall:
    # get AO indices
    lig_fall_ind_ao = get_ao_ind([lig_fall_ind],atoms,nbas,orb_per_atom)
    ligand_regular_ind_ao = np.logical_xor(lig_fall_ind_ao,ind_lig_ao)

    # get alpha
    alpha_lig_fall,alpha_lig_regular = get_alpha(mo_mat,[lig_fall_ind_ao,ligand_regular_ind_ao])
    print('Detached ligand: Alphas add to 1?:',np.all(np.isclose(alpha_lig_fall+alpha_lig_regular+alpha_shell+alpha_core,1)))

    # get DOS
    lig_fall_dos,lig_regular_dos = dos_grid_general(E_grid,sigma,mo_e,[alpha_lig_fall,alpha_lig_regular])

if attach_atom2:
    # get indices
    lig2_ind = (atoms==attach_atom2)
    lig1_ind = (atoms==attach_atom)

    # get AO indices
    lig2_ind_ao= get_ao_ind([lig2_ind],atoms,nbas,orb_per_atom)
    lig1_ind_ao = np.logical_xor(ind_lig_ao,lig2_ind_ao)

    # get alpha
    alpha_lig1,alpha_lig2=get_alpha(mo_mat,[lig1_ind_ao,lig2_ind_ao])
    print('Ligand 2: Alphas add to 1?:',np.all(np.isclose(alpha_lig1+alpha_lig2+alpha_shell+alpha_core,1)))
    lig1_dos,lig2_dos = dos_grid_general(E_grid,sigma,mo_e,[alpha_lig1,alpha_lig2])

# to add any other partitions: just copy the above code but substitute those indices
#
homo = nocc - 1
print("Band gap:", mo_e[homo+1]-mo_e[homo])
print("HOMO energy:",mo_e[homo])
x_limit = (mo_e[homo]-3,mo_e[homo+1]+3)

please_help_me=False
if please_help_me:
    # trap 1
    # cl_indx = np.array([5,6,16,31,32,39])
    # h_indx = np.array([3,4,13,14,16,31,32,42,43])
    # cd_indx = np.array([20,22,42,64,66,78])
    # s_indx = np.array([5,6,9,10,19,20,23,40,41,49,54,55])
    cl_ind = (atoms=='Cl')
    h_ind = (atoms=='H')
    trap_indx1 = np.array([20,22,42,64,66,78,99,100,111,112,125,126,132,155,156,168,177,178,185, 186, 196, 211, 212, 219,228, 229, 238, 239, 241, 256, 257, 267, 268])-1

    #trap2
    trap_indx2 = np.array([89,133,134,180,224,242,243,270])-1
    # write_xyz('test2.xyz',atoms[trap_indx],xyz[trap_indx])
    # print(atoms[trap_indx])

    #trap 3
    trap_indx3 = np.array([1,11,27,91,117,146,181,191,195,204,214,255])-1
    # write_xyz('test3.xyz',atoms[trap_indx],xyz[trap_indx])

    trap_ind1 = np.full(atoms.shape,False)
    trap_ind1[trap_indx1] = True

    trap_ind2 = np.full(atoms.shape,False)
    trap_ind2[trap_indx2] = True

    trap_ind3 = np.full(atoms.shape,False)
    trap_ind3[trap_indx3] = True
    # print(np.count_nonzero(trap_ind))

    # ind_shell_trap=np.logical_and(ind_shell,trap_ind)
    # ind_cl_trap = np.logical_and(lig2_ind,trap_ind)
    # ind_h_trap = np.logical_and(lig1_ind,trap_ind)
    #
    # ind_shell_notrap = np.logical_xor(ind_shell,ind_shell_trap)
    # ind_cl_notrap = np.logical_xor(lig2_ind,ind_cl_trap)
    # ind_h_notrap = np.logical_xor(lig1_ind,ind_h_trap)

    # print(atoms[ind_shell])
    # print('shell:',np.count_nonzero(ind_shell))
    # print('shell trap:',np.count_nonzero(ind_shell_trap))
    # print('shell no trap',np.count_nonzero(ind_shell_notrap))
    #
    # print('Cl:',np.count_nonzero(lig2_ind))
    # print('Cl trap:',np.count_nonzero(ind_cl_trap))
    # print('Cl no trap',np.count_nonzero(ind_cl_notrap))
    #
    # print('H:',np.count_nonzero(lig1_ind))
    # print('H trap:',np.count_nonzero(ind_h_trap))
    # print('H no trap',np.count_nonzero(ind_h_notrap))

    trap1_ind_ao,trap2_ind_ao,trap3_ind_ao = get_ao_ind([trap_ind1,trap_ind2,trap_ind3],atoms,nbas,orb_per_atom)

    ind_shell_trap1_ao=np.logical_and(ind_shell_ao,trap1_ind_ao)
    ind_cl_trap1_ao = np.logical_and(lig2_ind_ao,trap1_ind_ao)
    ind_h_trap1_ao = np.logical_and(lig1_ind_ao,trap1_ind_ao)

    ind_shell_trap2_ao=np.logical_and(ind_shell_ao,trap2_ind_ao)
    ind_cl_trap2_ao = np.logical_and(lig2_ind_ao,trap2_ind_ao)
    ind_h_trap2_ao = np.logical_and(lig1_ind_ao,trap2_ind_ao)

    ind_shell_trap3_ao=np.logical_and(ind_shell_ao,trap3_ind_ao)
    ind_cl_trap3_ao = np.logical_and(lig2_ind_ao,trap3_ind_ao)
    ind_h_trap3_ao = np.logical_and(lig1_ind_ao,trap3_ind_ao)

    ind_shell_trap_ao = np.logical_or(np.logical_or(ind_shell_trap1_ao,ind_shell_trap2_ao),ind_shell_trap3_ao)
    ind_cl_trap_ao = np.logical_or(np.logical_or(ind_cl_trap1_ao,ind_cl_trap2_ao),ind_cl_trap3_ao)
    ind_h_trap_ao = np.logical_or(np.logical_or(ind_h_trap1_ao,ind_h_trap2_ao),ind_h_trap3_ao)


    ind_shell_notrap_ao = np.logical_xor(ind_shell_ao,ind_shell_trap_ao)
    ind_cl_notrap_ao = np.logical_xor(lig2_ind_ao,ind_cl_trap_ao)
    ind_h_notrap_ao = np.logical_xor(lig1_ind_ao,ind_h_trap_ao)

    # get alphas
    alpha_trap1,alpha_trap2,alpha_trap3,alpha_shell_notrap,alpha_cl_notrap,alpha_h_notrap \
        = get_alpha(mo_mat,[trap1_ind_ao,trap2_ind_ao,trap3_ind_ao,ind_shell_notrap_ao,ind_cl_notrap_ao,ind_h_notrap_ao])

    alpha_trap = alpha_trap1+alpha_trap2+alpha_trap3
    print('Custom: Alphas add to 1?:',np.all(np.isclose(alpha_trap+alpha_shell_notrap+alpha_cl_notrap+alpha_h_notrap+alpha_core,1)))

    # get DOS
    trap1_dos,trap2_dos,trap3_dos,shell_notrap_dos,cl_notrap_dos,h_notrap_dos \
        = dos_grid_general(E_grid,sigma,mo_e,[alpha_trap1,alpha_trap2,alpha_trap3,alpha_shell_notrap,alpha_cl_notrap,alpha_h_notrap])

    trap_dos = trap1_dos+trap2_dos + trap3_dos

    # plot PDOS
    plt.figure()
    plt.plot(E_grid,core_dos,'b',label='Core')
    plt.plot(E_grid,shell_notrap_dos,'r',label='Shell')
    plt.plot(E_grid,h_notrap_dos,'g',label='H')
    plt.plot(E_grid,cl_notrap_dos,color='lime',label='Cl')
    plt.plot(E_grid,core_dos+shell_notrap_dos+h_notrap_dos+cl_notrap_dos+trap_dos,'k',label='Total')
    plt.plot(E_grid,trap1_dos,color='cyan',label='Trap 1')
    plt.plot(E_grid,trap2_dos,color='magenta',label='Trap 2')
    plt.plot(E_grid,trap3_dos,color='purple',label='Trap 3')


    # plt.stem(mo_e[homo-9:homo+1],[1,1,1,1,1,1,1,1,1,1])
    plt.legend()
    plt.xlim(x_limit)
    plt.ylim(0,100)
    plt.ylabel('Density of States')
    plt.xlabel('Orbital Energy (eV)')
    plt.savefig('pdos_trap_all.pdf')
    # plt.show()


'''
plotting PDOS--core/shell
'''
# plot PDOS
plt.figure()
plt.plot(E_grid,core_dos,'b',label='Core')
plt.plot(E_grid,shell_dos,'r',label='Shell')
if attach_atom2:
    plt.plot(E_grid,lig1_dos,'g',label=attach_atom)
    plt.plot(E_grid,lig2_dos,color='lime',label=attach_atom2)
elif attach_atom:
    plt.plot(E_grid, lig_dos,'g',label='ligand')

plt.plot(E_grid,core_dos+shell_dos+lig_dos,'k',label='Total')
# plt.stem(mo_e[homo-9:homo+1],[1,1,1,1,1,1,1,1,1,1])
plt.legend()
plt.xlim(x_limit)
plt.ylim(0,100)
plt.ylabel('Density of States')
plt.xlabel('Orbital Energy (eV)')
plt.savefig('pdos_cs.pdf')
# plt.show()

# orbs=[1563,1558,1544,1530,1499]
# print(np.count_nonzero(ind_Se)/Natoms)
# for orb in orbs:
#     print('Fraction on core for orbital', orb,':',alpha_core[orb])
# plt.rcParams['font.size'] = 16
'''
plotting PDOS--cd, s, se
'''
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
# plt.savefig('pdos_atom_reduced.pdf')
# plt.show()

'''
plotting PDOS--core cd, shell cd, s, se
'''
plt.figure()
plt.plot(E_grid,cd_core_dos,'c',label='Cd (core)')
plt.plot(E_grid,cd_shell_dos,'m',label='Cd (shell)')
plt.plot(E_grid,se_dos,color='orange',label='Se')
plt.plot(E_grid,s_dos,color='gold',label='S')
if attach_atom2:
    plt.plot(E_grid,lig1_dos,'g',label=attach_atom)
    plt.plot(E_grid,lig2_dos,color='lime',label=attach_atom2)
elif attach_atom:
    plt.plot(E_grid, lig_dos,'g',label='ligand')
plt.plot(E_grid,cd_core_dos+cd_shell_dos+se_dos+s_dos+lig_dos,'k',label='Total')
plt.legend()
plt.xlim(x_limit)
plt.ylim(0,100)
plt.ylabel('Density of States')
plt.xlabel('Orbital Energy (eV)')
plt.savefig('pdos_atom.pdf')
# plt.show()


'''
plotting PDOS--undercoordinated atoms
'''
if underc:
    plt.figure()
    plt.plot(E_grid,shell_fc_dos,color='r',label='3/4-c shell')
    plt.plot(E_grid,core_dos,'b',label='Core')
    if attach_atom2:
        plt.plot(E_grid,lig1_dos,'g',label=attach_atom)
        plt.plot(E_grid,lig2_dos,color='lime',label=attach_atom2)
    elif attach_atom:
        plt.plot(E_grid, lig_dos,'g',label='ligand')
    plt.plot(E_grid,cd_uc_dos,color='cyan',label='2-c Cd')
    plt.plot(E_grid,s_uc_dos,color='gold',label='2-c S')
    plt.plot(E_grid,cd_uc_dos+s_uc_dos+shell_fc_dos+core_dos,'k',label='Total')
    plt.legend()
    plt.xlim(x_limit)
    plt.ylim(0,100)
    plt.ylabel('Density of States')
    plt.xlabel('Orbital Energy (eV)')
    plt.savefig('pdos_uc.pdf')

'''
plotting pdos--clashing atoms
'''
if clash:
    plt.figure()
    plt.plot(E_grid,shell_noclash_dos,color='r',label='Shell (no clash)')
    plt.plot(E_grid,core_dos,'b',label='Core')
    if attach_atom:
        plt.plot(E_grid, lig_dos,'g',label='ligand')
    plt.plot(E_grid,cd_clash_dos,color='cyan',label='Cd-Cd clash')
    plt.plot(E_grid,s_clash_dos,color='gold',label='S-S clash')
    plt.plot(E_grid,cd_clash_dos+s_clash_dos+shell_noclash_dos+core_dos+lig_dos,'k',label='Total')
    plt.legend()
    plt.xlim(x_limit)
    plt.ylim(0,100)
    plt.ylabel('Density of States')
    plt.xlabel('Orbital Energy (eV)')
    plt.savefig('pdos_clash_2.pdf')

    plt.figure()
    plt.plot(E_grid,shell_noclash_only_dos,color='r',label='Shell (no clash)')
    plt.plot(E_grid,core_dos,'b',label='Core')
    if attach_atom:
        plt.plot(E_grid, lig_dos,'g',label='ligand')
    plt.plot(E_grid,cd_clash_only_dos,color='cyan',label='Cd-Cd only clash')
    plt.plot(E_grid,s_clash_only_dos,color='gold',label='S-S only clash')
    plt.plot(E_grid,cd_clash_dos+s_clash_dos+shell_noclash_dos+core_dos+lig_dos,'k',label='Total')
    plt.legend()
    plt.xlim(x_limit)
    plt.ylim(0,100)
    plt.ylabel('Density of States')
    plt.xlabel('Orbital Energy (eV)')
    plt.savefig('pdos_clash_only_2.pdf')

    plt.figure()
    plt.plot(E_grid,shell_nouc_only_dos,color='r',label='Shell (no UC)')
    plt.plot(E_grid,core_dos,'b',label='Core')
    if attach_atom:
        plt.plot(E_grid, lig_dos,'g',label='ligand')
    plt.plot(E_grid,cd_uc_only_dos,color='cyan',label='2C Cd no clash')
    plt.plot(E_grid,s_uc_only_dos,color='gold',label='2C S no clash')
    plt.plot(E_grid,cd_clash_dos+s_clash_dos+shell_noclash_dos+core_dos+lig_dos,'k',label='Total')
    plt.legend()
    plt.xlim(x_limit)
    plt.ylim(0,100)
    plt.ylabel('Density of States')
    plt.xlabel('Orbital Energy (eV)')
    plt.savefig('pdos_uc_only_2.pdf')

    plt.figure()
    plt.plot(E_grid,shell_no_ucandclash_dos,color='r',label='Shell (no UC and clash)')
    plt.plot(E_grid,core_dos,'b',label='Core')
    if attach_atom:
        plt.plot(E_grid, lig_dos,'g',label='ligand')
    plt.plot(E_grid,cd_clash_uc_dos,color='cyan',label='Cd 2C and clashing')
    plt.plot(E_grid,s_clash_uc_dos,color='gold',label='S 2C and clashing')
    plt.plot(E_grid,cd_clash_dos+s_clash_dos+shell_noclash_dos+core_dos+lig_dos,'k',label='Total')
    plt.legend()
    plt.xlim(x_limit)
    plt.ylim(0,100)
    plt.ylabel('Density of States')
    plt.xlabel('Orbital Energy (eV)')
    plt.savefig('pdos_clash_and_uc_2.pdf')

    plt.figure()
    plt.plot(E_grid,shell_normal_dos,color='r',label='Shell (no defects)')
    plt.plot(E_grid,core_dos,'b',label='Core')
    if attach_atom:
        plt.plot(E_grid, lig_dos,'g',label='ligand')
    plt.plot(E_grid,cd_clash_uc_dos+cd_clash_only_dos+cd_uc_only_dos,color='cyan',label='Cd defect')
    plt.plot(E_grid,s_clash_uc_dos+s_clash_only_dos+s_uc_only_dos,color='gold',label='S defect')
    plt.plot(E_grid,cd_clash_dos+s_clash_dos+shell_noclash_dos+core_dos+lig_dos,'k',label='Total')
    plt.legend()
    plt.xlim(x_limit)
    plt.ylim(0,100)
    plt.ylabel('Density of States')
    plt.xlabel('Orbital Energy (eV)')
    plt.savefig('pdos_all_defect_2.pdf')

'''
plotting pdos--ligand fall off
'''
if lig_fall:
    plt.figure()
    plt.plot(E_grid,lig_regular_dos,color='green',label='Attached ligands')
    plt.plot(E_grid,shell_dos,color='r',label='Shell')
    # plt.plot(E_grid,s_fc_dos,'b',label='3/4-c S')
    plt.plot(E_grid,core_dos,'b',label='Core')
    plt.plot(E_grid,lig_fall_dos,color='lime',label='Detached ligands')
    plt.plot(E_grid,shell_dos+core_dos+lig_dos,'k',label='Total')
    plt.legend()
    plt.xlim(x_limit)
    plt.ylim(0,100)
    plt.ylabel('Density of States')
    plt.xlabel('Orbital Energy (eV)')
    plt.savefig('pdos_lig_fall.pdf')
plt.show()
