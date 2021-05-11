import numpy as np
import sys
from qd_helper import read_input_xyz,write_xyz

'''
Script to analyze the mulliken/lowdin charges from tddft/tda excitations

USAGE:
python3 surf_vs_bulk_clean.py [QD xyz file] [charge file] [optional: number of excitation to print detailed info about]
'''

def inv_par_rat(index_CdSe, Charges):
    '''
    Function to calculate the inverse participation ratio for the e/h of each excitation.

    Inputs: index_Cdse -- np array of indices of the Cd and Se atoms, in numpy format where
                          index_CdSe[i]= True if atom i is Cd or Se and False if it's a ligand
                          atom (length of this array is equal to the total number of atoms in the dot)
            Charges    -- np array of the raw charges for each atom and excitation

    Outputs: ipr_write -- np array where the ith row is the ipr for the ith excitation and
                          the first column is the electron, second is the hole, third is delta
    '''
    cdse_sum=np.sum(Charges[index_CdSe],0) # total charge on cd and se
    cdse_sum[np.nonzero(np.abs(cdse_sum)<=1e-15)] = 1e-8
    Ch_cdse_n=Charges[index_CdSe]/cdse_sum # fraction of charge on each cdse
    #print(np.sum(np.power(Ch_cdse_n,2),0))
    ipr=1.0/np.sum(np.power(Ch_cdse_n,2),0)  # calculate ipr
    ipr_write=np.reshape(ipr,(-1,3)) /np.sum(index_CdSe) # get into correct shape to write
                                                         # the indices are in numpy format with [True, True, ...False, False]
                                                         # so we use sum instead of len to count the CdSe's

    #ipr_max=np.amax(ipr_write, axis=0)                   # biggest ipr (most delocalized)
    return ipr_write



###
### USER SPECIFIED INFO
###

QD_file_input=sys.argv[1] # QD xyz file
charges_input=sys.argv[2] # file with all the charges for all excitations
if len(sys.argv) > 3:
    n = int(sys.argv[3]) -1   # can optionally specify a specific excitation to print info for
    indiv = True
    verbose = True
else:
    indiv = False

###
### SEPARATING ATOMS INTO SURFACE VS BULK
###
QD_xyz,atom_name = read_input_xyz(QD_file_input)

# getting indices of different types of atoms
ind_Cd = (atom_name == "Cd")
ind_Se = (atom_name == "Se")
ind_CdSe = np.logical_or(ind_Cd, ind_Se)
ind_lig = np.logical_not(ind_CdSe)  # ligand atoms are defined as anything that isn't cd or se (!)
ind_all=np.logical_or(ind_CdSe,ind_lig) # for core/shell


###
### CHARGE ANALYSIS
###
#'''
# array of all charges for all excitations in same format as file
# columns: ex1e, ex1h, ex1D, ex2e, ex2h, ... rows: atom 1, atom 2...
Charges_full=np.loadtxt(charges_input,delimiter=',',skiprows=1,dtype=str)
Charges = Charges_full[:-1,1:].astype(float)

# calculate ipr
ipr_write = inv_par_rat(ind_CdSe, Charges)

###
### OUTPUT
###

# if individual excitation info requested, print it
if indiv:
    print('')
    print("Information for excitation #", n+1, "[electron, hole]")
    print("")
    print('IPR:', ipr_write[n][0:-1])

# if individual info not requested, save charges to file
if not indiv:
    # ipr has just the beginning of the charge file as the filename
    np.savetxt('.'.join(charges_input.split('.')[0:-1]) + '_ipr.csv',ipr_write,delimiter=',')
#'''

# # TODO: more functions/organization?, warnings for if alpha negative?
