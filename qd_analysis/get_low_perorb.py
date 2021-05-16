import numpy as np
from qd_helper import read_input_xyz,write_xyz
import sys

outputfile=sys.argv[1]     # qchem output file
xyzfile=sys.argv[2]
Nshells = int(sys.argv[3]) # number of shells (e.g. length of the column)
Norbs = int(sys.argv[4])   # total number of orbitals


xyz,atoms=read_input_xyz(xyzfile)
Natom = len(atoms)
# print(Natom)
remainder=(Norbs % 6)
# print(remainder)

shell_per_atom_lanl2dz = {'Cd': 3, 'Se': 2, 'S': 2}
shell_per_atom_sto3g = {'C':2,'H':1}
shell_per_atom_ccpvdz = {'C':3,'H':2}
shell_per_atom_test = shell_per_atom_lanl2dz

partial_low = np.zeros((Natom,Norbs))
# print(partial_low)

with open(outputfile,'r') as outfile:
    outlines=outfile.readlines()

# extracting the charges into an array
flag=True
for i,line in enumerate(outlines):
    # if line.find("Partial Lowdin") != -1:
    #     break
    # if line.find("1    H 1   s") != -1:
    if line.find('1    Cd1   s') != -1:
        # j += 1
        # print(i,line)
        # print(outlines[i:i+Nshells])
        try:
            x=np.loadtxt(outputfile,skiprows=i,max_rows=Nshells,usecols=(-6,-5,-4,-3,-2,-1))
        except ValueError:
            cols=(-6,-5,-4,-3,-2,-1)[6-remainder:]
            # print(cols)
            x=np.loadtxt(outputfile,skiprows=i,max_rows=Nshells,usecols=cols)
            # print(x)
        if not flag:
            partial_low_lg= np.concatenate((partial_low_lg,x),axis=1)
            # print(partial_low)
        if flag:
            partial_low_lg = x
            flag=False

# print(partial_low_lg.shape)

# create an array that's Natoms x Norbs, not Nshells x N orbs
# contract shells into one per atom


def get_low_peratom(atoms,low_perorb,shell_per_atom):
    low=[]
    j = 0
    for atom in atoms:
        nshell_i = shell_per_atom[atom]
        low_atom = low_perorb[j:j+nshell_i] # lowdin charge for atom i, for each shell, per orbital

        low.append(np.sum(low_atom,axis=0)) # sum over the shells so that this array is the lowdin charge on atom i for each MO
        j += nshell_i

    return np.array(low)

low_peratom=get_low_peratom(atoms,partial_low_lg,shell_per_atom_test)
# print(test[0:3])
np.save('low_orb',low_peratom)

# now we need the IPR--should be relatively straightforward.
