import sys

# script to extract spectrum from TDDFT calculation
# usage: python3 get_spectrum [name of TDDFT output file] > [CSV file]
# (prints to terminal by default)



inputfile=sys.argv[1]  # output of TDDFT calculation

with open(inputfile,'r') as inp:
    flag=0  # flag indicates whether you're in the excitation energy part of the file
    for i,line in enumerate(inp):
        if flag==1:
            # parse excitation info
            if line.find('Excited state ') != -1:
                splitline=line.split()
                Ex_N = splitline[2].strip(':')
                Ex_ev = splitline[-1]
            if line.find('Total energy for state') != -1:
                E_tot = line.split()[-2]
            if line.find('Strength') != -1:
                dipole = line.split()[-1]
                spect_line = ','.join([Ex_N,E_tot,Ex_ev,dipole])
                print(spect_line)  # once you hit dipole, compile everything for that ex + print
        if line.find('TDDFT Excitation Energies') != -1:
            # print file header when you get to the TDDFT part, change flag to 1
            print('Excitation number,Total energy (au),Excitation energy (eV),Dipole strength,'+inputfile)
            flag = 1 
        if line.find('SETman timing summary') != -1:
            # TDDFT part ended
            break
