import numpy as np
import sys

def make_mo_coeff_mat(mo_file_raw,nbas):
    mo_coeff_mat = np.zeros((nbas,nbas))
    mo_Es = np.zeros(nbas)
    with open(mo_file_raw,'r') as mo_file:
        mo_lines=mo_file.readlines()
        for i in range(0,nbas): # row
            e_ind = 2 * nbas * nbas + i
            mo_Es[i] = float(mo_lines[e_ind])
            for j in range(0,nbas): # column
                mo_ind = i * nbas + j
                mo_coeff_mat[j,i] = float(mo_lines[mo_ind])
    return mo_coeff_mat,mo_Es


file = sys.argv[1]
nbas=int(sys.argv[2])
mo_mat,mo_e = make_mo_coeff_mat(file,nbas)
print(mo_mat)
print(mo_e)
