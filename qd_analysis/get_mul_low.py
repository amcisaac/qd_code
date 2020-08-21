import argparse
import numpy as np

# script to parse qchem CIS_ANAL output
# usage: python3 get_mul_low.py Natom

# Creates new csv files with data for TDA Mulliken, TDA Lowdin, TDDFT Mulliken, TDDFT Lowdin (depending on options given)
# Files are saved as [file name] + _mul_tda.csv, +_low_tda.csv, +_mul_tddft.csv, +_low_tddft.csv , unless -o/--output is provided

my_parser = argparse.ArgumentParser(description='Get excited state Mulliken and Lowdin from a QChem TDA/TDDFT output file')
my_parser.add_argument('inputfile',metavar='qchem_file',help='The QChem output file')
my_parser.add_argument('Natom',metavar='number_of_atoms',help='Number of atoms in the molecule')
my_parser.add_argument('--tda',action='store_true',help='Get TDA spectrum')
my_parser.add_argument('--tddft',action='store_true',help='Get TDDFT spectrum')
my_parser.add_argument('-o','--output',action='store',help='Specify different name for output file')

args=my_parser.parse_args()

Natoms=int(args.Natom) # number of atoms in the molecule
inputfile=args.inputfile  # output of TDDFT analysis calculation

if args.output: inp_name= '/'.join(args.inputfile.split('/')[:-1])+ '/'+ args.output
else: inp_name = inputfile.split('.')[0]

mul_tda = []
low_tda = []

mul_tddft = []
low_tddft = []

with open(inputfile,'r') as inp:
    flag=0  # flag indicates whether you're in the excitation energy part of the file
    for i,line in enumerate(inp):
        if flag==1 and args.tda: # TDA analysis
            # print('flag1')
            # parse excitation info
            if line.find('Mulliken analysis of TDA') != -1:
                e,h,D = np.loadtxt(inputfile,skiprows=i+4,max_rows=Natoms+1,unpack=True,usecols=(1,2,3))
                mul_tda.append(e)
                mul_tda.append(h)
                mul_tda.append(D)
            if line.find('Loewdin analysis of TDA') != -1:
                if line.find('Mulliken') == -1:
                    e,h,D = np.loadtxt(inputfile,skiprows=i+4,max_rows=Natoms+1,unpack=True,usecols=(1,2,3))
                    low_tda.append(e)
                    low_tda.append(h)
                    low_tda.append(D)
        if flag==2: # TDDFT analysis
            # print('flag2')
            if not args.tddft:
                break
            # parse excitation info
            if line.find('Mulliken analysis of RPA') != -1:
                e,h,D = np.loadtxt(inputfile,skiprows=i+4,max_rows=Natoms+1,unpack=True,usecols=(1,2,3))
                mul_tddft.append(e)
                mul_tddft.append(h)
                mul_tddft.append(D)

            if line.find('Loewdin analysis of RPA') != -1:
                # print(i,line)
                if line.find('Mulliken') == -1:
                    e,h,D = np.loadtxt(inputfile,skiprows=i+4,max_rows=Natoms+1,unpack=True,usecols=(1,2,3))
                    low_tddft.append(e)
                    low_tddft.append(h)
                    low_tddft.append(D)

        if line.find('CIS_N_ROOTS') != -1:
            Nex = 2*int(line.split()[-1])

        if line.find('Mulliken & Loewdin analysis of') != -1: # beginning of analysis. occurs twice, once for tda then for rpa
            flag += 1
            if args.tddft and not args.tda: flag = 2

        if line.find('excitation amplitudes in the NTO basis') != -1:
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
if args.tda: mul_tda_array = np.concatenate((atoms,mul_tda_array),axis=1)
if args.tda: low_tda_array = np.concatenate((atoms,low_tda_array),axis=1)
if args.tddft: mul_tddft_array = np.concatenate((atoms,mul_tddft_array),axis=1)
if args.tddft: low_tddft_array = np.concatenate((atoms,low_tddft_array),axis=1)


# creating a header for the csv files
head = ['Atom']
exs = range(1,Nex+1)
for ex in exs:
    head.append(str(ex)+'e')
    head.append(str(ex)+'h')
    head.append(str(ex)+'D')

# saving the files
if args.tda: np.savetxt(inp_name+'_mul_tda_t.csv',mul_tda_array,delimiter=',',header=','.join(head),fmt='%.18s')
if args.tda: np.savetxt(inp_name+'_low_tda_t.csv',low_tda_array,delimiter=',',header=','.join(head),fmt='%.18s')
if args.tddft: np.savetxt(inp_name+'_mul_tddft_t.csv',mul_tddft_array,delimiter=',',header=','.join(head),fmt='%.18s')
if args.tddft: np.savetxt(inp_name+'_low_tddft_t.csv',low_tddft_array,delimiter=',',header=','.join(head),fmt='%.18s')
