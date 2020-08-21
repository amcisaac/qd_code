import sys
import numpy as np

# script to parse qchem CIS_ANAL output 
# usage: python3 get_mul_low.py Natom

# Creates 4 new csv files with data for TDA Mulliken, TDA Lowdin, TDDFT Mulliken, TDDFT Lowdin
# Files are saved as [file name] + mul_tda.csv, +low_tda.csv, +mul_tddft.csv, +low_tddft.csv


Natoms=int(sys.argv[2]) # number of atoms in the molecule
inputfile=sys.argv[1]  # output of TDDFT analysis calculation

inp_name = inputfile.split('.')[0]

mul_tda = []
low_tda = []


mul_tddft = []
low_tddft = []

with open(inputfile,'r') as inp:
    flag=0  # flag indicates whether you're in the excitation energy part of the file
    for i,line in enumerate(inp):
        if flag==1: # TDA analysis
            # parse excitation info
            if line.find('Mulliken analysis') != -1:
                e,h,D = np.loadtxt(inputfile,skiprows=i+4,max_rows=Natoms+1,unpack=True,usecols=(1,2,3))
                mul_tda.append(e)
                mul_tda.append(h)
                mul_tda.append(D)
            if line.find('Loewdin analysis') != -1:
                if line.find('Mulliken') == -1:
                    e,h,D = np.loadtxt(inputfile,skiprows=i+4,max_rows=Natoms+1,unpack=True,usecols=(1,2,3))
                    low_tda.append(e)
                    low_tda.append(h)
                    low_tda.append(D)
        if flag==2: # TDDFT analysis
            # parse excitation info
            if line.find('Mulliken analysis') != -1:
                e,h,D = np.loadtxt(inputfile,skiprows=i+4,max_rows=Natoms+1,unpack=True,usecols=(1,2,3))
                mul_tddft.append(e)
                mul_tddft.append(h)
                mul_tddft.append(D)
            
            if line.find('Loewdin analysis') != -1:
                e,h,D = np.loadtxt(inputfile,skiprows=i+4,max_rows=Natoms+1,unpack=True,usecols=(1,2,3))
                low_tddft.append(e)
                low_tddft.append(h)
                low_tddft.append(D)

        if line.find('CIS_N_ROOTS') != -1:
            Nex = 2*int(line.split()[-1])

        if line.find('Mulliken & Loewdin analysis of') != -1: # beginning of analysis. occurs twice, once for tda then for rpa
            flag +=1 

        if line.find('TDA excitation amplitudes in the NTO basis') != -1:
            # analysis ended
            break

# array rows have format (ex #1 e, ex #1 h, ex #1 D, ex #2 e, ex #2 h, ex #2 D...) 
# each row is one atom
mul_tda_array = np.array(mul_tda).T
low_tda_array = np.array(low_tda).T
mul_tddft_array = np.array(mul_tddft).T
low_tddft_array = np.array(low_tddft).T

# debugging info
# shapes should be Natoms+1 x 3*Nex
print('N atoms:', Natoms)
print('N ex:', Nex)
print('3*Nex:',3*Nex)

print('Shape of Mul. TDA array:',mul_tda_array.shape)
print('Shape of Low. TDA array:',low_tda_array.shape)
print('Shape of Mul. TDDFT array:', mul_tddft_array.shape)
print('Shape of Low. TDDFT array:',low_tddft_array.shape)


# creating a column for the atom numbers
atoms = np.array(range(0,Natoms)).reshape(Natoms,1)
atoms = np.concatenate((atoms+1,np.array([['Sum']])))

# adding the atom numbers to the charge arrays
mul_tda_array = np.concatenate((atoms,mul_tda_array),axis=1)
low_tda_array = np.concatenate((atoms,low_tda_array),axis=1)
mul_tddft_array = np.concatenate((atoms,mul_tddft_array),axis=1)
low_tddft_array = np.concatenate((atoms,low_tddft_array),axis=1)


# creating a header for the csv files
head = ['Atom']
exs = range(1,Nex+1)
for ex in exs:
    head.append(str(ex)+'e')
    head.append(str(ex)+'h')
    head.append(str(ex)+'D')

# saving the files
np.savetxt(inp_name+'_mul_tda.csv',mul_tda_array,delimiter=',',header=','.join(head),fmt='%.18s')
np.savetxt(inp_name+'_low_tda.csv',low_tda_array,delimiter=',',header=','.join(head),fmt='%.18s')
np.savetxt(inp_name+'_mul_tddft.csv',mul_tddft_array,delimiter=',',header=','.join(head),fmt='%.18s')
np.savetxt(inp_name+'_low_tddft.csv',low_tddft_array,delimiter=',',header=','.join(head),fmt='%.18s')
