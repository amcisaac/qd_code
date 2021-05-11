import numpy as np
import matplotlib.pyplot as plt
# from qd_helper import *
import copy
from geom_helper import *
import argparse

'''
Script to do geometry analysis of CdSe QD's and determine if any surface atoms are
undercoordinated.

Usage: python3 get_hist.py [xyz file of structure] [ligand attach atom] [options]
'''

my_parser = argparse.ArgumentParser(description='Get histogram of Cd-Se and/or Cd-ligand distances from xyz file')
my_parser.add_argument('inputfile',metavar='xyz file',help='The XYZ file for the structure of interest')
my_parser.add_argument('lig_atom',metavar='ligand atom',help='Ligand atom that binds to Cd (e.g. N for MeNH2)')
my_parser.add_argument('-c','--cutoff',action='store',help='Cutoff where Cd-Se dist < cutoff means an atom is undercoordinated. Triggers saving the undercoordinated atom histogram.')
my_parser.add_argument('-x','--xtal',action='store',help='The xyz file for the crystal (optional). Triggers saving the surface histogram')
my_parser.add_argument('-d','--dont_save',action='store_true',help='Do not save full histogram')
my_parser.add_argument('-l','--save_ligand',action='store_true',help='Save Cd-ligand histogram')
my_parser.add_argument('-p','--plot',action='store_true',help='Plot histograms')
args=my_parser.parse_args()

###
### PARSING ARGUMENTS
###

QD_file_end=args.inputfile   # QD optimized xyz file
QD_xyz_end,atom_name_end = read_input_xyz(QD_file_end)
print('Analyzing file '+QD_file_end)
header = '#Analyzing file '+QD_file_end
qd_beg_filename = '.'.join(QD_file_end.split('.')[0:-1])
print(qd_beg_filename)

lig_atom = args.lig_atom # atom that attaches to the Cd in the ligand
print('Ligand attach atom: ',lig_atom)
header = header+ '   Ligand attach atom: '+lig_atom
# print(header)

save_surface = False

if args.cutoff:
    cutoff = float(args.cutoff)
    save_uc = True
    print('Cutoff: ',cutoff)
    nncutoff = 3  # number of nearest neighbors to be considered "unpassivated" (incl. ligands)
    uc_header = header+ '   UC cutoff: {} A'.format(cutoff)
    # print(uc_header)
if args.xtal:
    QD_file_start = args.xtal
    save_surface = True
    QD_xyz_start,atom_name_start = read_input_xyz(QD_file_start)
    surface_header = header+'   Crystal file: ' + QD_file_start
    # print(surface_header)

save_hist = np.logical_not(args.dont_save)
save_lig = args.save_ligand
plot = args.plot

# getting indices of different types of atoms
# atom order shouldn't change in optimization, so just need one set
# ind_Cd, ind_Se, ind_CdSe, ind_lig, ind_selig=parse_ind(atom_name_start,lig_atom)
ind_Cd = atom_name_end=='Cd'
ind_Se = atom_name_end=='Se'
ind_CdSe= np.logical_or(ind_Cd,ind_Se)
ind_attach = (atom_name_end == lig_atom)
ind_lig=ind_attach
ind_false = (atom_name_end=='') # array of "False" the size of ind_attach

####
#
# HISTOGRAM OF ALL NEAREST NEIGHBOR DISTANCES
#
####

all_dists,cdse_dists,cdlig_dists,cdselig_dists,secd_dists = get_dists(QD_xyz_end,ind_Cd,ind_Se,ind_attach)
cdse_hist = cdse_dists.flatten()
if save_hist:
    np.savetxt(qd_beg_filename+'_hist.csv',cdse_hist)

if plot:
    plt.figure()
    plt.hist(cdse_hist,bins=800,label='Cd-Se')

if save_lig:
    cdlig_hist = cdlig_dists.flatten()
    np.savetxt(qd_beg_filename+'_cdlig_hist.csv',cdlig_hist)
    if plot: plt.hist(cdlig_hist,bins=800,label='Cd-{}'.format(lig_atom))

####
#
# SURFACE HISTOGRAM
#
####

if save_surface:
    #get ind of surface cd/se from xtal structure
    cd_underc_ind_s,se_underc_ind_s = get_underc_index(QD_xyz_start,ind_Cd,ind_Se,ind_false,ind_false,cutoff,nncutoff,verbose=False)

    surf_se_hist = secd_dists[se_underc_ind_s].flatten()
    surf_cd_hist = cdse_dists[cd_underc_ind_s].flatten()
    surf_hist = np.concatenate((surf_se_hist,surf_cd_hist))

    np.savetxt(qd_beg_filename+'_surf_hist.csv',surf_hist) # i am pretty sure this is double counting surf-surf bonds

    if plot: plt.hist(surf_hist,bins=800,label='Surface')

####
#
# UNDERCOORDINATED ATOM HISTOGRAM
#
####

if save_uc:
    # get ind of underc cd/se from final structure
    cd_underc_ind_e,se_underc_ind_e = get_underc_index(QD_xyz_end,ind_Cd,ind_Se,ind_lig,ind_attach,cutoff,nncutoff,verbose=False)

    underc_se_hist = secd_dists[se_underc_ind_e].flatten()
    underc_cd_hist = cdse_dists[cd_underc_ind_e].flatten()

    if np.any(se_underc_ind_e):
        # np.savetxt(qd_beg_filename+'_undercse_hist.csv',underc_se_hist)
        if plot: plt.hist(underc_se_hist, bins=800,label='UC Se')
    if np.any(cd_underc_ind_e):
        # np.savetxt(qd_beg_filename+'_underccd_hist.csv',underc_cd_hist)
        if plot: plt.hist(underc_cd_hist,bins=800,label='UC Cd')

if plot:
    plt.xlim(2.4,6)
    plt.legend()
    plt.show()
