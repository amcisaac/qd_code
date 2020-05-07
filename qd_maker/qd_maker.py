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
     # return np array of all distances for each atom
     dists = [] # list to return with format above
     for atom in xyz: # xyz = for each cd
         dist = np.sqrt(np.sum((atom - xyz)**2,axis=1)) # calc dist between cd(i) and all se's
         dists.append(dist) # add dist to list

     return np.array(dists)#, np.array(N_nns)

def dist_atom12(all_dists,ind_1,ind_2):
    return all_dists[ind_1].T[ind_2].T



def get_dists(QD_xyz,ind_Cd,ind_Se,ind_attach=''):
    all_dists = dist_all_points(QD_xyz)
    cd_se_dists_all = dist_atom12(all_dists,ind_Cd,ind_Se)
    se_cd_dists_all = dist_atom12(all_dists,ind_Se,ind_Cd)

    return all_dists,cd_se_dists_all,se_cd_dists_all

def num_nn(dist_list,cutoff):
    # returns list of number of NN for each atom
    # could get xyz coordinates from the cd_se_dists_all < cutoff indices i think
    return np.sum(dist_list < cutoff,axis=1)

def get_nn(cdselig_dists,secd_dists,ind_Cd,ind_Se,cutoff,Natoms):
    cd_nn_selig = num_nn(cdselig_dists,cutoff)
    se_nn_cdonly = num_nn(secd_dists,cutoff)

    all_nn = np.zeros(Natoms)
    all_nn[ind_Cd]=cd_nn_selig # using all nn for cd
    all_nn[ind_Se]=se_nn_cdonly # using just cd for se
    # all_nn[ind_lig]=100 # set these to very high to avoid any weirdness

    return all_nn,cd_nn_selig,se_nn_cdonly

def build_dot(xyz,atoms,ctr,radius):
    # print(ctr)
    xyz_ctr = xyz - ctr
    dist_from_ctr = np.linalg.norm(xyz_ctr,axis=1)
    in_r = dist_from_ctr <= radius
    atoms_in_r = atoms[in_r]
    # print(atoms_in_r.shape)
    # print((dist_from_ctr <= radius).shape)
    xyz_in_r = xyz_ctr[in_r]

    ind_Cd_r = (atoms_in_r == "Cd")
    ind_Se_r = (atoms_in_r == "Se")
    ind_CdSe_r = np.logical_or(ind_Cd_r, ind_Se_r)

    all_dists,cd_se_dists,se_cd_dists = get_dists(xyz_in_r,ind_Cd_r,ind_Se_r)
    all_nn,cd_nn,se_nn=get_nn(cd_se_dists,se_cd_dists,ind_Cd_r,ind_Se_r,3.0,len(atoms_in_r))

    nncutoff = 2

    cd_coord_ind = cd_nn>=nncutoff
    se_coord_ind = se_nn>=nncutoff
    coord_ind = all_nn >= nncutoff

    coord_ind_all = dist_from_ctr <= radius
    coord_ind_all[in_r] = coord_ind
    # print(coord_ind_all.shape)
    # print(np.all(xyz_in_r[coord_ind] == xyz_ctr[coord_ind_all]))

    return coord_ind,xyz_in_r[coord_ind],atoms_in_r[coord_ind],coord_ind_all

def get_coreonly(xyz,atoms,radius):
    ctr0 = np.mean(xyz,axis=0)

    i=0
    Ncd = 100
    Nse = 101
    ctr = ctr0
    nit = 0
    while Ncd != Nse and nit < 500:
        coord_ind,coord_xyz,coord_atoms,coord_ind_all=build_dot(xyz,atoms,ctr,radius)

        Ncd = np.count_nonzero(coord_atoms=='Cd')
        Nse = np.count_nonzero(coord_atoms=='Se')

        print(Nse,' Se atoms')
        print(Ncd,' Cd atoms')

        if Ncd != Nse:
            ctr = ctr0 + np.random.random_sample((1,3))
            # print(Ncd,Nse,ctr0)
        nit += 1

    if nit == 500:
        print('Did not converge!')
    # write_xyz('test_qd_allatoms.xyz',atoms_in_r,xyz_in_r)

    # cd_coord_name = atoms_in_r[ind_Cd_r][cd_coord_ind]
    # se_coord_name = atoms_in_r[ind_Se_r][se_coord_ind]

    # cd_coord_xyz = xyz_in_r[ind_Cd_r][cd_coord_ind]
    # se_coord_xyz = xyz_in_r[ind_Se_r][se_coord_ind]

    # cdse_atomname = np.concatenate((cd_coord_name,se_coord_name))
    # cdse_atomname=coord_atoms


    # print(cdse_atomname.shape)
    # cdse_xyz = np.concatenate((cd_coord_xyz,se_coord_xyz),axis=0)
    # cdse_xyz = coord_xyz
    # print(cdse_xyz.shape)


    return coord_ind,coord_xyz,coord_atoms,ctr,coord_ind_all


def get_shell(xyz,atoms,shell_rad,core_ctr,Ncore):
    shell_coord_ind,shell_coord_xyz,shell_coord_atoms,coord_ind_all=build_dot(xyz,atoms,core_ctr,shell_rad)
    print(np.count_nonzero(shell_coord_atoms=='Cd')-Ncore,'Cd atoms in shell')
    print(np.count_nonzero(shell_coord_atoms=='Se')-Ncore, 'Se atoms in shell')

    return shell_coord_ind,shell_coord_xyz,shell_coord_atoms,coord_ind_all

input_file = sys.argv[1]
rad = 9. # in Angstroms
rad2 = 10

xyzcoords,atom_names = read_input_xyz(input_file)

Ncdshell=0
Nseshell=1
nit2 = 0
while Ncdshell != Nseshell and nit2 < 500:
    core_ind,core_xyz,core_atomname,ctr,core_ind_all=get_coreonly(xyzcoords,atom_names,rad)
    shell_ind,shell_xyz,shell_atomname,shell_ind_all=get_shell(xyzcoords,atom_names,rad2,ctr,np.count_nonzero(core_atomname=='Cd'))
    Ncdshell = np.count_nonzero(shell_atomname == 'Cd')
    Nseshell = np.count_nonzero(shell_atomname == 'Se')
    nit2 += 1

if nit2 == 500:
    print('Did not converge!')

# shell_xyz_ctr = shell_xyz - np.mean(shell_xyz,axis=0)
# core_xyz_ctr = core_xyz - np.mean(shell_xyz,axis=0)
# print(core_ind.shape)
# print(core_ind_all.shape)

Ncore=np.count_nonzero(core_ind_all)
print(Ncore," core atoms")
shell_only_ind_all = np.logical_xor(core_ind_all,shell_ind_all)
Nshell = np.count_nonzero(shell_only_ind_all)
print(Nshell,"shell atoms")


# print(core_ind.shape)
# write_xyz('test_qd_fullycoor_t.xyz',core_atomname,core_xyz)

# just for checking symmetry
# ctr_again = cdse_xyz - np.mean(cdse_xyz,axis=0)
# ind_cd_final = cdse_atomname == 'Cd'
# ind_se_final = cdse_atomname == 'Se'
# all_dists,cd_se_dists,se_cd_dists = get_dists(ctr_again,ind_cd_final,ind_se_final)
# all_nn,cd_nn,se_nn=get_nn(cd_se_dists,se_cd_dists,ind_cd_final,ind_se_final,3.0,len(cdse_atomname))
# print('avg radius: ',np.mean(np.linalg.norm(ctr_again,axis=1)))
# print('avg radius (surf only): ',np.mean(np.linalg.norm(ctr_again[all_nn[np.logical_or(ind_cd_final,ind_se_final)] < 4],axis=1)))
# print('largest radius: ',np.max(np.linalg.norm(ctr_again,axis=1)))
# print('std radius (surf only): ',np.std(np.linalg.norm(ctr_again[all_nn[np.logical_or(ind_cd_final,ind_se_final)] < 4],axis=1)))


## Cd-Se distance histogram
# plt.figure()
# plt.title("Cd-Se distance")
# plt.hist(cd_se_dists.flatten(),bins=800) # crystal
# # plt.hist(cdse_dists_e.flatten(),bins=800) # optimized
# plt.xlim(0,4)
# plt.show()
