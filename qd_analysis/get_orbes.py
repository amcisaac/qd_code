import sys

# script to extract orbital energies
# usage: python3 get_orbes [name of TDDFT output file] > [CSV file]
# (prints to terminal by default)



inputfile=sys.argv[1]  # output of TDDFT calculation

with open(inputfile,'r') as inp:
    flag=0  # flag indicates whether you're in the excitation energy part of the file
    flag2=0
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
        if line.find('Alpha MOs') != -1:
            # print file header when you get to the orbital E's part, change flag to 1
            print('Orbital energy (au),Occupation,'+inputfile)
            flag = 1 
