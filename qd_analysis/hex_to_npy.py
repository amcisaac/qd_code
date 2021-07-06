import numpy as np
import sys
from pdos_helper import build_S_mo

bas_file=sys.argv[1]       # number of basis functions
coeff_file = sys.argv[2]    # hex version of qchem 53.0
s_file=sys.argv[3]          # hex version of qchem 320.0

nbas,nocc=np.loadtxt(bas_file,dtype=int,unpack=True)

mo_mat,mo_e,S=build_S_mo(s_file,coeff_file,nbas,nocc)

np.savez('53.npz',mo_mat,mo_e)
np.save('320.npy',S)
