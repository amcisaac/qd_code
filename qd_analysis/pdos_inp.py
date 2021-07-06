import numpy as np
import numpy.linalg as npl
import sys
from qd_helper import read_input_xyz
from matplotlib import pyplot as plt
from pdos_helper import dos_grid_general,get_alpha,get_ao_ind



'''
parsing input
'''
# USER-SPECIFIED!!!
lig='F'     # ligand
lig2=''
lig3=''
lig2_label=''
# lig2='O'  # uncomment/modify to include second ligand or defect atom (e.g. oxide)
# lig3='H'  # third ligand or defect atom; will be grouped with lig2 for PDOS
# lig2_label = 'OH' # label for extra ligands in pdos--change to O for oxide, OH for hydroxide, etc

# command line arguments
xyz_file=sys.argv[1]        # whole dot xyz file (must correspond to core coordinates)
bas_file = sys.argv[2]      # txt file with total number of orbitals and occupied orbitals
coeff_file = sys.argv[3]    # txt version of qchem 53.0 OR numpy version
try: s_file=sys.argv[4]     # optional: txt version of qchem 320.0, only if using txt 53
except IndexError: s_file = False # if numpy version of 53, don't need 320
# charge_analysis=False       # make True for mulliken/lowdin analysis (just standard atomic charge)
# underc=False                # analyze undercoordinated atoms--not implemented for InP

# parse orbital info
nbas,nocc=np.loadtxt(bas_file,dtype=int,unpack=True)
homo=nocc-1

# read xyz file
xyz,atoms=read_input_xyz(xyz_file)

# number of orbitals per atom for each atom: 8 Se, 8 S, 18 Cd
# orb_per_atom_lanl2dz = {'Cd': 18, 'Se': 8, 'S': 8}
# orb_per_atom_sto3g={'C':5,'H':1,'He':1} # for testing purposes
# orb_per_atom_ccpvdz={'C':14,'H':5,'He':5} # for testing purposes

# Dictionary specifying the number of AO's per atom in the given basis set
# 18 P, 18 Cl, 32 Br, 32 Ga, 26 In for def2-svp
orb_per_atom_def2svp={'In': 26, 'Ga': 32, 'P': 18, 'Cl': 18, 'Br': 32,'F':14,'H':5,'O':14}
orb_per_atom=orb_per_atom_def2svp # choose which dictionary to use

# build MO matrix if 53.txt provided
if s_file:
    mo_mat,mo_e,S=build_S_mo(s_file,coeff_file,nbas,nocc)

# get MO matrix from 53.npy if provided
else:
    coeff_file_expand=np.load(coeff_file)
    mo_mat=coeff_file_expand['arr_0'] # normalized
    mo_e = coeff_file_expand['arr_1']

# charge analysis (for testing mostly)
# if charge_analysis:
#     mulliken,lowdin=get_mul_low(P,S,X_inv,atoms,orb_per_atom,z)
#     print('Mulliken charges for ',atoms,':',mulliken)
#     print('Lowdin charges for ',atoms,':',lowdin)


'''
get indices for different partitions
'''
# lig2 includes lig3 also--may want to change but this works for OH
# ind_In,ind_P,ind_Ga,ind_lig,ind_in_ao,ind_p_ao,ind_ga_ao,ind_lig_ao,ind_lig2_ao = get_inp_ind_ao(xyz,atoms,nbas,orb_per_atom,lig,lig2,lig3)

ind_In = (atoms=='In')
ind_P = (atoms=='P')
ind_Ga = (atoms=='Ga')
ind_lig = (atoms == lig)
if lig2 and lig3: ind_lig2 = np.logical_or((atoms==lig2), (atoms==lig3))
else: ind_lig2 = (atoms==lig2)
ind_in_ao,ind_p_ao,ind_ga_ao,ind_lig_ao,ind_lig2_ao = get_ao_ind([ind_In,ind_P,ind_Ga,ind_lig,ind_lig2],atoms,nbas,orb_per_atom)

'''
get squared coefficients on core,shell
'''
alpha_list = get_alpha(mo_mat,[ind_in_ao,ind_p_ao,ind_ga_ao,ind_lig_ao,ind_lig2_ao])
alpha_in,alpha_p,alpha_ga,alpha_lig,alpha_lig2=alpha_list
alpha_inga=alpha_in+alpha_ga

# make sure alphas add to 1
test = np.all(np.isclose(alpha_in+alpha_p+alpha_ga+alpha_lig+alpha_lig2,1))
if test == False: raise ValueError('Alpha doesnt add to 1!')
print('Alphas add to 1?:',test)


'''
calculate projected DOS
'''
mo_e = mo_e * 27.2114 # MO energy, in eV
E_grid = np.arange(-50,50,0.01) # energy grid to evaluate the DOS over
sigma=0.1 # broadening parameter
print('HOMO energy',mo_e[homo])
print('Band gap:', mo_e[homo+1]-mo_e[homo])


dos_list = dos_grid_general(E_grid, sigma,mo_e, alpha_list)
in_dos,p_dos,ga_dos,lig_dos,lig2_dos=dos_list
inga_dos=in_dos+ga_dos


'''
plotting regular DOS
'''
# plt.figure()
# plt.plot(E_grid,in_dos+ga_dos+p_dos+lig_dos,'k')
# # plt.legend()
# plt.xlim(-16,-8)
# plt.ylim(0,110)
# plt.ylabel('Density of States')
# plt.xlabel('Orbital Energy (eV)')
# plt.savefig('dos.pdf')

'''
plotting PDOS--atom breakdown
'''
plt.figure()
plt.plot(E_grid,in_dos,'C0',label='In')
if np.any(ind_Ga): plt.plot(E_grid,ga_dos,'C1',label='Ga')
plt.plot(E_grid,p_dos,'C2',label='P')
plt.plot(E_grid,lig_dos,'C3',label=lig)
if np.any(ind_lig2): plt.plot(E_grid,lig2_dos,'xkcd:violet',label=lig2_label)
plt.plot(E_grid,in_dos+ga_dos+p_dos+lig_dos+lig2_dos,'k',label='Total')
# plt.stem([mo_e[homo]],[10]) # put a stem at the HOMO
plt.legend()
plt.xlim(mo_e[homo-3]-3,mo_e[homo+1]+5)
plt.ylim(0,30)
plt.ylabel('Density of States')
plt.xlabel('Orbital Energy (eV)')
plt.savefig('pdos_atom.pdf')
plt.show()
