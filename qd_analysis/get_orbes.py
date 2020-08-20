import argparse

# script to extract orbital energies
# usage: python3 get_orbes [name of TDDFT output file] > [CSV file]
# (prints to terminal by default)

my_parser = argparse.ArgumentParser(description='Get orbital energies from a QChem output file')
my_parser.add_argument('inputfile',metavar='qchem file',help='The QChem output file')
my_parser.add_argument('-c','--opt_cycle',action='store',help='Specify an optimization cycle to get orbitals from')
my_parser.add_argument('-t','--triplet',action='store_true',help='Specify if the system is a triplet') # not implemented

args=my_parser.parse_args()

# inputfile=sys.argv[1]  # output of TDDFT calculation

with open(args.inputfile,'r') as inp:
    flag=0  # flag indicates whether you're in the excitation energy part of the file
    flag2=0
    if args.opt_cycle:
        flag3 = 0
    else:
        flag3=1
    occ = "Occupied"
    for i,line in enumerate(inp):
        if flag==1:
            # parse excitation info
            if flag2 == 1:
                if line.find(' --------------------------------------------------------------') != -1:
                    break

                if line.find('Virtual') != -1:
                    occ = "Virtual"
                else:
                    orbEs = line.split()
                    for E in orbEs:
                        print(E,occ)
            if line.find('Occupied') != -1:
                occ = "Occupied"
                flag2=1
        if flag3 ==1:
            if line.find('Alpha MOs') != -1:
                # print file header when you get to the orbital E's part, change flag to 1
                print('Orbital energy (au),Occupation,'+args.inputfile)
                flag = 1
        if line.find('Optimization Cycle') != -1:
            if line.split()[-1] == args.opt_cycle:
                flag3 = 1
