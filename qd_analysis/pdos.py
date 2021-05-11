# import matplotlib as mpl
# mpl.use('Agg')
import numpy as np
import numpy.linalg as npl
import sys
from qd_helper import read_input_xyz,write_xyz
from matplotlib import pyplot as plt
from pdos_helper import *
import time


'''
parsing input
'''
xyz_file=sys.argv[1]        # whole dot xyz file (must correspond to core coordinates)
core_xyz_file = sys.argv[2] # core xyz file
nbas=int(sys.argv[3])       # number of basis functions
nocc=int(sys.argv[4])       # number of occupied basis functions
coeff_file = sys.argv[5]    # hex version of qchem 53.0
s_file=sys.argv[6]          # hex version of qchem 320.0
charge_analysis=False       # make True for mulliken/lowdin analysis (just standard atomic charge)
underc=False
attach_atom=False
attach_atom2=False
attach_atom='H'
# attach_atom2='Cl'
clash=False
lig_fall = False


# read xyz files
xyz,atoms=read_input_xyz(xyz_file)
core_xyz,core_atoms=read_input_xyz(core_xyz_file)
cdshell_underc_ind_amb=np.full(atoms.shape,False)
sshell_underc_ind_amb=np.full(atoms.shape,False)

if len(sys.argv)>7:
    underc=True
    cdshell_underc_ind=np.load(sys.argv[7])
    sshell_underc_ind=np.load(sys.argv[8])
    # cdshell_underc_ind_amb=np.load(sys.argv[9])
    # sshell_underc_ind_amb=np.load(sys.argv[10])
    cdshell_underc_ind_amb=np.full(cdshell_underc_ind.shape,False)
    sshell_underc_ind_amb=np.full(sshell_underc_ind.shape,False)

if len(sys.argv)>9:
    clash=True
    cdshell_clash_ind = np.load(sys.argv[9])
    sshell_clash_ind = np.load(sys.argv[10])

if len(sys.argv)>11:
    lig_fall = True
    lig_fall_ind = np.load(sys.argv[11])



# number of orbitals per atom for each atom: 8 Se, 8 S, 18 Cd
orb_per_atom_lanl2dz = {'Cd': 18, 'Se': 8, 'S': 8, 'N':9,'C':9,'H':2,'Cl':8}
# orb_per_atom_sto3g={'C':5,'H':1,'He':1} # for testing purposes
# orb_per_atom_ccpvdz={'C':14,'H':5,'He':5} # for testing purposes
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

# check for normalization
mo_mat_sum=np.sum(np.power(mo_mat,2),axis=0)
print('MO normalization:',np.all(np.isclose(mo_mat_sum,1)))
if not np.all(np.isclose(mo_mat_sum,1)): print("WARNING: MO's not normalized!")

'''
get indices for different partitions
'''
ind_Cd,ind_Se,ind_S,ind_core,ind_shell,ind_lig,ind_cd_ao,ind_se_ao,ind_s_ao,ind_core_ao,ind_shell_ao,ind_lig_ao = get_cs_ind_ao(xyz,core_xyz,atoms,nbas,orb_per_atom,attach_atom)

ind_cd_core = np.logical_and(ind_core,ind_Cd)
ind_cd_shell = np.logical_and(ind_shell,ind_Cd)
ind_cd_core_ao = np.logical_and(ind_core_ao,ind_cd_ao)
ind_cd_shell_ao = np.logical_and(ind_shell_ao,ind_cd_ao)

if underc:
    cdshell_underc_ind_ao,sshell_underc_ind_ao,cdshell_underc_ind_amb_ao,sshell_underc_ind_amb_ao = get_cs_ind_ao_underc(atoms,ind_core,nbas,orb_per_atom,cdshell_underc_ind,sshell_underc_ind,cdshell_underc_ind_amb,sshell_underc_ind_amb)

if clash:
    cdshell_clash_ind_ao,sshell_clash_ind_ao,x,y = get_cs_ind_ao_underc(atoms,ind_core,nbas,orb_per_atom,cdshell_clash_ind,sshell_clash_ind,cdshell_underc_ind_amb,sshell_underc_ind_amb)

if lig_fall:
    lig_fall_ind_ao,x,y,z = get_cs_ind_ao_underc(atoms,ind_core,nbas,orb_per_atom,lig_fall_ind,cdshell_underc_ind_amb,cdshell_underc_ind_amb,sshell_underc_ind_amb)

if attach_atom2:
    lig2_ind = (atoms==attach_atom2)
    lig1_ind = (atoms==attach_atom)
    lig2_ind_ao,x,y,z = get_cs_ind_ao_underc(atoms,ind_core,nbas,orb_per_atom,lig2_ind,cdshell_underc_ind_amb,cdshell_underc_ind_amb,sshell_underc_ind_amb)
    lig1_ind_ao = np.logical_xor(ind_lig_ao,lig2_ind_ao)

'''
get squared coefficients on core,shell
'''
alpha_cd_core,alpha_cd_shell,alpha_se,alpha_s,alpha_lig = get_alpha(mo_mat,[ind_cd_core_ao,ind_cd_shell_ao,ind_se_ao,ind_s_ao,ind_lig_ao])
alpha_cd = alpha_cd_core + alpha_cd_shell
alpha_core = alpha_cd_core + alpha_se
alpha_shell = alpha_cd_shell + alpha_s
print('Alphas add to 1?:',np.all(np.isclose(alpha_cd_core+alpha_cd_shell+alpha_se+alpha_s+alpha_lig,1)))


'''
calculate projected DOS
'''
mo_e = mo_e * 27.2114 # MO energy, in eV
E_grid = np.arange(-500,50,0.01) # energy grid to evaluate the DOS over
sigma=0.1 # broadening parameter


cd_core_dos,cd_shell_dos,se_dos,s_dos,lig_dos = dos_grid_general(E_grid, sigma, mo_e, [alpha_cd_core,alpha_cd_shell,alpha_se,alpha_s,alpha_lig])

cd_dos = cd_core_dos + cd_shell_dos
core_dos = se_dos + cd_core_dos
shell_dos = cd_shell_dos + s_dos

#
#
if underc:
    sshell_underc = sshell_underc_ind_ao #np.logical_or(sshell_underc_ind_ao,sshell_underc_ind_amb_ao)
    cdshell_underc = cdshell_underc_ind_ao #np.logical_or(cdshell_underc_ind_ao,cdshell_underc_ind_amb_ao)

    sshell_nouc = np.logical_xor(sshell_underc,ind_s_ao)
    cdshell_nouc = np.logical_xor(cdshell_underc,ind_cd_shell_ao)

    alpha_cd_uc,alpha_s_uc,alpha_cd_fc,alpha_s_fc = get_alpha(mo_mat,[cdshell_underc,sshell_underc,cdshell_nouc,sshell_nouc])
    alpha_shell_fc=alpha_cd_fc+alpha_s_fc

    print('Alphas add to 1?:',np.all(np.isclose(alpha_shell_fc+alpha_cd_uc+alpha_s_uc+alpha_core+alpha_lig,1)))

    # cd_uc_dos = dos_grid(E_grid,sigma,mo_e,alpha_cd_uc)
    # s_uc_dos = dos_grid(E_grid,sigma,mo_e,alpha_s_uc)
    # shell_fc_dos = dos_grid(E_grid,sigma,mo_e,alpha_shell_fc)

    cd_uc_dos,s_uc_dos,cd_fc_dos,s_fc_dos,lig_dos_2 = dos_grid_cdses(E_grid,sigma,mo_e,alpha_cd_uc,alpha_s_uc,alpha_cd_fc,alpha_s_fc,alpha_lig)
    shell_fc_dos = cd_fc_dos + s_fc_dos
    # print(np.all(lig_dos==lig_dos_2))

if clash:
    # sshell_underc = sshell_underc_ind_ao #np.logical_or(sshell_underc_ind_ao,sshell_underc_ind_amb_ao)
    # cdshell_underc = cdshell_underc_ind_ao #np.logical_or(cdshell_underc_ind_ao,cdshell_underc_ind_amb_ao)

    sshell_noclash = np.logical_xor(sshell_clash_ind_ao,ind_s_ao)
    cdshell_noclash = np.logical_xor(cdshell_clash_ind_ao,ind_cd_shell_ao)

    alpha_cd_clash,alpha_s_clash,alpha_cd_noclash,alpha_s_noclash = get_alpha(mo_mat,[cdshell_clash_ind_ao,sshell_clash_ind_ao,cdshell_noclash,sshell_noclash])
    alpha_shell_noclash=alpha_cd_noclash+alpha_s_noclash

    print('Alphas add to 1?:',np.all(np.isclose(alpha_shell_noclash+alpha_cd_clash+alpha_s_clash+alpha_core+alpha_lig,1)))

    # cd_uc_dos = dos_grid(E_grid,sigma,mo_e,alpha_cd_uc)
    # s_uc_dos = dos_grid(E_grid,sigma,mo_e,alpha_s_uc)
    # shell_fc_dos = dos_grid(E_grid,sigma,mo_e,alpha_shell_fc)

    cd_clash_dos,s_clash_dos,cd_noclash_dos,s_noclash_dos,lig_dos_2 = dos_grid_cdses(E_grid,sigma,mo_e,alpha_cd_clash,alpha_s_clash,alpha_cd_noclash,alpha_s_noclash,alpha_lig)
    shell_noclash_dos = cd_noclash_dos + s_noclash_dos

if lig_fall:
    # sshell_underc = sshell_underc_ind_ao #np.logical_or(sshell_underc_ind_ao,sshell_underc_ind_amb_ao)
    # cdshell_underc = cdshell_underc_ind_ao #np.logical_or(cdshell_underc_ind_ao,cdshell_underc_ind_amb_ao)

    ligand_regular_ind_ao = np.logical_xor(lig_fall_ind_ao,ind_lig_ao)
    # cdshell_noclash = np.logical_xor(cdshell_clash_ind_ao,ind_cd_shell_ao)

    alpha_lig_fall,alpha_lig_regular = get_alpha(mo_mat,[lig_fall_ind_ao,ligand_regular_ind_ao])

    print('Alphas add to 1?:',np.all(np.isclose(alpha_lig_fall+alpha_lig_regular+alpha_shell+alpha_core,1)))

    # cd_uc_dos = dos_grid(E_grid,sigma,mo_e,alpha_cd_uc)
    # s_uc_dos = dos_grid(E_grid,sigma,mo_e,alpha_s_uc)
    # shell_fc_dos = dos_grid(E_grid,sigma,mo_e,alpha_shell_fc)

    lig_fall_dos,lig_regular_dos,cd_noclash_dos,s_noclash_dos,lig_dos_2 = dos_grid_cdses(E_grid,sigma,mo_e,alpha_lig_fall,alpha_lig_regular,alpha_cd_noclash,alpha_s_noclash,alpha_lig)

if attach_atom2:
    alpha_lig1,alpha_lig2=get_alpha(mo_mat,[lig1_ind_ao,lig2_ind_ao])
    print('Alphas add to 1?:',np.all(np.isclose(alpha_lig1+alpha_lig2+alpha_shell+alpha_core,1)))
    lig1_dos,lig2_dos,x,y,z = dos_grid_cdses(E_grid,sigma,mo_e,alpha_lig1,alpha_lig2,alpha_cd,alpha_s,alpha_lig)

homo = nocc - 1
print("Band gap:", mo_e[homo]-mo_e[homo+1])
print("HOMO energy:",mo_e[homo])
x_limit = (mo_e[homo]-3,mo_e[homo+1]+3)
# '''
# plotting PDOS--core/shell
# '''
# # plot PDOS
# plt.figure()
# plt.plot(E_grid,core_dos,'b',label='Core')
# plt.plot(E_grid,shell_dos,'r',label='Shell')
# if attach_atom2:
#     plt.plot(E_grid,lig1_dos,'g',label=attach_atom)
#     plt.plot(E_grid,lig2_dos,color='lime',label=attach_atom2)
# elif attach_atom:
#     plt.plot(E_grid, lig_dos,'g',label='ligand')
#
# plt.plot(E_grid,core_dos+shell_dos+lig_dos,'k',label='Total')
# # plt.stem(mo_e[homo-9:homo+1],[1,1,1,1,1,1,1,1,1,1])
# plt.legend()
# plt.xlim(x_limit)
# plt.ylim(0,100)
# plt.ylabel('Density of States')
# plt.xlabel('Orbital Energy (eV)')
# plt.savefig('pdos_cs.pdf')
# # plt.show()
#
#
# '''
# plotting PDOS--cd, s, se
# '''
# # plt.figure()
# # plt.plot(E_grid,cd_dos,'c',label='Cd')
# # plt.plot(E_grid,se_dos,color='orange',label='Se')
# # plt.plot(E_grid,s_dos,'y',label='S')
# # plt.plot(E_grid,cd_dos+se_dos+s_dos,'k',label='Total')
# # plt.legend()
# # plt.xlim(-8,-2)
# # plt.ylim(0,100)
# # plt.ylabel('Density of States')
# # plt.xlabel('Orbital Energy (eV)')
# # plt.savefig('pdos_atom_reduced.pdf')
# # plt.show()
#
# '''
# plotting PDOS--core cd, shell cd, s, se
# '''
# plt.figure()
# plt.plot(E_grid,cd_core_dos,'c',label='Cd (core)')
# plt.plot(E_grid,cd_shell_dos,'m',label='Cd (shell)')
# plt.plot(E_grid,se_dos,color='orange',label='Se')
# plt.plot(E_grid,s_dos,color='gold',label='S')
# if attach_atom2:
#     plt.plot(E_grid,lig1_dos,'g',label=attach_atom)
#     plt.plot(E_grid,lig2_dos,color='lime',label=attach_atom2)
# elif attach_atom:
#     plt.plot(E_grid, lig_dos,'g',label='ligand')
# plt.plot(E_grid,cd_core_dos+cd_shell_dos+se_dos+s_dos+lig_dos,'k',label='Total')
# plt.legend()
# plt.xlim(x_limit)
# plt.ylim(0,100)
# plt.ylabel('Density of States')
# plt.xlabel('Orbital Energy (eV)')
# plt.savefig('pdos_atom.pdf')
# # plt.show()
#
#
# '''
# plotting PDOS--undercoordinated atoms
# '''
# if underc:
#     plt.figure()
#     plt.plot(E_grid,shell_fc_dos,color='r',label='3/4-c shell')
#     plt.plot(E_grid,core_dos,'b',label='Core')
#     if attach_atom:
#         plt.plot(E_grid, lig_dos,'g',label='ligand')
#     # plt.plot(E_grid,s_fc_dos,'b',label='3/4-c S')
#     plt.plot(E_grid,cd_uc_dos,color='cyan',label='2-c Cd')
#     plt.plot(E_grid,s_uc_dos,color='gold',label='2-c S')
#     plt.plot(E_grid,cd_uc_dos+s_uc_dos+shell_fc_dos+core_dos,'k',label='Total')
#     plt.legend()
#     plt.xlim(x_limit)
#     plt.ylim(0,100)
#     plt.ylabel('Density of States')
#     plt.xlabel('Orbital Energy (eV)')
#     plt.savefig('pdos_uc.pdf')
#
# '''
# plotting pdos--clashing atoms
# '''
# if clash:
#     plt.figure()
#     plt.plot(E_grid,shell_noclash_dos,color='r',label='Shell (no clash)')
#     # plt.plot(E_grid,s_fc_dos,'b',label='3/4-c S')
#     plt.plot(E_grid,core_dos,'b',label='Core')
#     if attach_atom:
#         plt.plot(E_grid, lig_dos,'g',label='ligand')
#     plt.plot(E_grid,cd_clash_dos,color='cyan',label='Cd-Cd clash')
#     plt.plot(E_grid,s_clash_dos,color='gold',label='S-S clash')
#     plt.plot(E_grid,cd_clash_dos+s_clash_dos+shell_noclash_dos+core_dos+lig_dos,'k',label='Total')
#     plt.legend()
#     plt.xlim(x_limit)
#     plt.ylim(0,100)
#     plt.ylabel('Density of States')
#     plt.xlabel('Orbital Energy (eV)')
#     plt.savefig('pdos_clash.pdf')
#
#
# '''
# plotting pdos--ligand fall off
# '''
# if lig_fall:
#     plt.figure()
#     plt.plot(E_grid,lig_regular_dos,color='green',label='Attached ligands')
#     plt.plot(E_grid,shell_dos,color='r',label='Shell')
#     # plt.plot(E_grid,s_fc_dos,'b',label='3/4-c S')
#     plt.plot(E_grid,core_dos,'b',label='Core')
#     plt.plot(E_grid,lig_fall_dos,color='lime',label='Detached ligands')
#     plt.plot(E_grid,shell_dos+core_dos+lig_dos,'k',label='Total')
#     plt.legend()
#     plt.xlim(x_limit)
#     plt.ylim(0,100)
#     plt.ylabel('Density of States')
#     plt.xlabel('Orbital Energy (eV)')
#     plt.savefig('pdos_lig_fall.pdf')
# plt.show()
