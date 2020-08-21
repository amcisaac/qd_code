import argparse

# script to extract spectrum from TDDFT calculation
# usage: python3 get_spectrum [name of TDDFT output file]
# writes to [TDDFT output] + [tddft or tda]_spec.csv unless -o/--output specified

# tddft and tda refer to what's in the file--can't select only tda if tddft is also in the file :(
my_parser = argparse.ArgumentParser(description='Get spectrum from a QChem TDA/TDDFT output file')
my_parser.add_argument('inputfile',metavar='qchem file',help='The QChem output file')
my_parser.add_argument('--tda',action='store_true',help='Get TDA spectrum')
my_parser.add_argument('--tddft',action='store_true',help='Get TDDFT spectrum')
my_parser.add_argument('-t','--triplet',action='store_true',help='Specify if the system is a triplet') # only really matters for calcs with only tddft
my_parser.add_argument('-o','--output',action='store',help='Specify different name for output file') # NOTE: preserves path of original file

args=my_parser.parse_args()

# name files for excitation energies:
if args.output: beg_output_name = '/'.join(args.inputfile.split('/')[:-1])+ '/'+args.output
else: beg_output_name = '.'.join(args.inputfile.split('.')[:-1])

if args.tda:
    output1=beg_output_name+ '_tda_spec.csv'
if args.tddft:
    output2=beg_output_name+ '_tddft_spec.csv'

with open(args.inputfile,'r') as inp:
    flag=0  # flag indicates whether you're in the excitation energy part of the file
    tddft_flag=False # indicates whether you're in the TDDFT part of the spectrum (only matters for triplet)

    for i,line in enumerate(inp):
        if flag==1:
            # parse excitation info
            if line.find('Excited state ') != -1:
                splitline=line.split()
                Ex_N = splitline[2].strip(':')
                Ex_ev = splitline[-1]

            if line.find('Total energy for state') != -1:
                E_tot = line.split()[-2]

            if line.find('Multiplicity') != -1: # for TDA, TDDFT singlet
                mult = line.split()[-1]

            if line.find('S**2') != -1: # for TDA, triplet only
                mult=line.split()[-1]

            if line.find('Strength') != -1:
                dipole = line.split()[-1]

                if args.triplet and tddft_flag:
                    spect_line = ','.join([Ex_N,E_tot,Ex_ev,dipole])
                else:
                    spect_line = ','.join([Ex_N,E_tot,Ex_ev,dipole,mult])
                outfile.write(spect_line+'\n')  # once you hit dipole, compile everything for that ex + print

        if line.find('TDDFT/TDA Excitation Energies') != -1: # indicates beginning of TDA results
            # print file header when you get to the TDA part, change flag to 1
            outfile=open(output1,'w')
            outfile.write('Excitation number,Total energy (au),Excitation energy (eV),Dipole strength,'+args.inputfile+'\n')
            flag = 1

        if args.tda + args.tddft == 2 and line.find('Direct TDDFT calculation will be performed') != -1: # if both TDA and TDDFT are selected, this indicates the end of the TDA
            outfile.close() # close tDA file; tddft will be opened at tddft line
            flag=0 # reset but don't break

        if line.find('TDDFT Excitation Energies') != -1: # indicates beginning of TDDFT results
            # print file header when you get to the TDDFT part, change flag to 1
            outfile=open(output2,'w')
            outfile.write('Excitation number,Total energy (au),Excitation energy (eV),Dipole strength,'+args.inputfile+'\n')
            flag = 1
            tddft_flag=True

        if line.find('SETman timing summary') != -1: # if only TDA, this is the end; if only TDDFT or both TDDFT and TDA this is the end of the TDDFT
            outfile.close()
            break
