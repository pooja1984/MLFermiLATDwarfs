from __future__ import division
from optparse import OptionParser
import math
import numpy as np
import os
import os.path
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import rcParams, rc
from matplotlib import cm

rc('text', usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['cmr']})
rc('font',**{'family':'serif','serif':['cmr']})
rc('text.latex', preamble=r'\usepackage{soul}')

MDM = [0.1, 0.11, 0.121, 0.1331, 0.14641, 0.161051, 0.177156, 0.194872, 0.214359, 0.235795, 0.259374, 0.285312, 0.313843, 0.345227, 0.37975, 0.417725, 0.459497, 0.505447, 0.555992, 0.611591, 0.67275, 0.740025, 0.814027, 0.89543, 0.984973, 1.08347, 1.19182, 1.311, 1.4421, 1.58631, 1.74494, 1.91943, 2.11138, 2.32252, 2.55477, 2.81024, 3.09127, 3.40039, 3.74043, 4.11448, 4.52593, 4.97852, 5.47637, 6.02401, 6.62641, 7.28905, 8.01795, 8.81975, 9.70172, 10.6719, 11.7391, 12.913, 14.2043, 15.6247, 17.1872, 18.9059, 20.7965, 22.8762, 25.1638, 27.6801, 30.4482, 33.493, 36.8423, 40.5265, 44.5792, 49.0371, 53.9408, 59.3349, 65.2683, 71.7952, 78.9747, 86.8722, 95.5594, 105.115, 115.627, 127.19, 139.908, 153.899, 169.289, 186.218, 204.84, 225.324, 247.856, 272.642, 299.906, 329.897, 362.887, 399.175, 439.093, 483.002, 531.302, 584.432, 642.876, 707.163, 777.88, 855.668, 941.234, 1035.36, 1138.89, 1252.78, 1378.06, 1515.87, 1667.45, 1834.2, 2017.62, 2219.38, 2441.32, 2685.45, 2954.0, 3249.4, 3574.34, 3931.77, 4324.95, 4757.44, 5233.19, 5756.5, 6332.15, 6965.37, 7661.91, 8428.1, 9270.91, 10198.0, 11217.8, 12339.6, 13573.5, 14930.9, 16424.0, 18066.4, 19873.0, 21860.3, 24046.3, 26451.0, 29096.1, 32005.7, 35206.3, 38726.9, 42599.6, 46859.5, 51545.5, 56700.0, 62370.0, 68607.0, 75467.7, 83014.5, 91316.0, 100448.0, 110492.0, 121542.0, 133696.0, 147065.0, 161772.0, 177949.0, 200000.0]

steigmann_sigv = [4.3266e-26, 4.34965e-26, 4.37358e-26, 4.39682e-26, 4.42092e-26, 4.44366e-26, 4.46704e-26, 4.48935e-26, 4.50792e-26, 4.52736e-26, 4.54576e-26, 4.56065e-26, 4.57102e-26, 4.57519e-26, 4.57162e-26, 4.56204e-26, 4.54615e-26, 4.52498e-26, 4.50084e-26, 4.47446e-26, 4.44606e-26, 4.41781e-26, 4.38901e-26, 4.36041e-26, 4.33274e-26, 4.30628e-26, 4.2804e-26, 4.25448e-26, 4.22729e-26, 4.19732e-26, 4.16148e-26, 4.1183e-26, 4.06452e-26, 3.98976e-26, 3.87001e-26, 3.69277e-26, 3.46364e-26, 3.22315e-26, 3.02186e-26, 2.87315e-26, 2.75929e-26, 2.66764e-26, 2.59088e-26, 2.52509e-26, 2.46885e-26, 2.41957e-26, 2.37624e-26, 2.33809e-26, 2.30362e-26, 2.27274e-26, 2.24495e-26, 2.21975e-26, 2.19709e-26, 2.17706e-26, 2.1587e-26, 2.14231e-26, 2.12711e-26, 2.11346e-26, 2.10076e-26, 2.08939e-26, 2.07915e-26, 2.07031e-26, 2.0626e-26, 2.05596e-26, 2.05042e-26, 2.04628e-26, 2.04299e-26, 2.04075e-26, 2.03955e-26, 2.03872e-26, 2.03908e-26, 2.04033e-26, 2.04226e-26, 2.04423e-26, 2.04724e-26, 2.05063e-26, 2.05472e-26, 2.05853e-26, 2.06331e-26, 2.06815e-26, 2.07334e-26, 2.07856e-26, 2.08379e-26, 2.08903e-26, 2.09401e-26, 2.09865e-26, 2.10291e-26, 2.1065e-26, 2.11007e-26, 2.1125e-26, 2.11463e-26, 2.11569e-26, 2.1164e-26, 2.11642e-26, 2.11567e-26, 2.11424e-26, 2.11279e-26, 2.11066e-26, 2.10819e-26, 2.10596e-26, 2.10339e-26, 2.10159e-26, 2.09984e-26, 2.09874e-26, 2.09798e-26, 2.09761e-26, 2.09794e-26, 2.09901e-26, 2.10009e-26, 2.10229e-26, 2.10477e-26, 2.1076e-26, 2.1108e-26, 2.1147e-26, 2.11863e-26, 2.12327e-26, 2.12791e-26, 2.13294e-26, 2.13799e-26, 2.14342e-26, 2.14888e-26, 2.15479e-26, 2.16105e-26, 2.16728e-26, 2.17352e-26, 2.17978e-26, 2.18652e-26, 2.19324e-26, 2.19997e-26, 2.20671e-26, 2.21386e-26, 2.22066e-26, 2.22789e-26, 2.2351e-26, 2.24234e-26, 2.24962e-26, 2.25691e-26, 2.26423e-26, 2.27157e-26, 2.27933e-26, 2.28667e-26, 2.29409e-26, 2.30153e-26, 2.30899e-26, 2.31687e-26, 2.32434e-26, 2.33188e-26, 2.33944e-26, 2.34703e-26, 2.35504e-26, 2.36263e-26, 2.37029e-26, 2.37959e-26]

def plot_limits_scan(root= "./excl_limits_case", dwarf_list = ["Draco", ], case="J", plot_type="all"):
    
    x = 16.
    plt.figure(figsize=(x,x/((np.sqrt(5.) + 1.0)/2.0)))
    
    data = np.loadtxt(root + "{}/sv_limits_case{}_".format(case, case) + "+".join(dwarf_list) + ".dat")
    mDM = data[:, 0]
    svlim = data[:, -1]
    plt.loglog(mDM, svlim, color = 'black', linewidth = 3, label = 'stacked dSphs ({} profiling)'.format(case))
    
    if plot_type == "all":
        frac = np.linspace(0., 1., len(dwarf_list))
        for k, dSph_tag in enumerate(dwarf_list):
            c = cm.Dark2(frac[k], 1)
            data_single = np.loadtxt(root + "{}/sv_limits_case{}_".format(case, case) + dSph_tag + ".dat")
            plt.loglog(data_single[:, 0], data_single[:, -1], color = c, linewidth = 1, label = dSph_tag)
            
    plt.loglog(MDM, steigmann_sigv, linewidth = 1.0, color = 'black', linestyle = '--')

    plt.grid()
    plt.xlim([8., 1000])
    plt.ylim([1e-27, 1e-22])
    
    ax = plt.gca()
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_tick_params(labelsize = 40, width = 2, length = 10, which = 'major')
    ax.yaxis.set_tick_params(width = 1, length = 7, which = 'minor')
    ax.xaxis.set_tick_params(labelsize = 40, width = 2, length = 10, which = 'major')
    ax.xaxis.set_tick_params(width = 1, length = 7, which = 'minor')

    plt.xlabel("$M_{\\mathrm{DM}}$ [GeV]", fontsize=38)
    plt.ylabel("$\\langle \\sigma v \\rangle\\;\\left[\\mathrm{cm}^3\\,\\mathrm{s}^{-1}\\right]$", fontsize=38)
    plt.title(" + ".join(dwarf_list), fontsize = 38)
    
    plt.legend(loc= 0, fontsize = 27, handlelength=3, facecolor = 'None', edgecolor = 'None')

    if plot_type == "all":
        plt.savefig(root + "{}/FermiLAT_".format(case) + case + "_profiling_excl_limits_stacked_dSph_" + "+".join(dwarf_list) + "_w_single_dwarfs.pdf", dpi = 600, bbox_inches = 'tight')
    else:
        plt.savefig(root + "{}/FermiLAT_".format(case) + case + "_profiling_excl_limits_stacked_dSph_" + "+".join(dwarf_list) + "_wo_single_dwarfs.pdf", dpi = 600, bbox_inches = 'tight')

def interactive():
    parser = OptionParser()
    
    parser.add_option("-d", "--dwarf", dest="dwarf_list", help="dSph list, multiple dwarfs are joined with a '+'", metavar="NDWARF", type='str', default = "Draco")
    
    parser.add_option("-c", "--case", dest="case", help="Case: J - profiling only the J-factor; JB - combined profiling of dSph J-factor and background contribution", metavar="MODE", type='str', default = "J")
    
    # Sample dSph for either single case or combined limits
    parser.add_option("--type", dest="plot_type", help="Case: all -- plot stacked limit and all single dwarf limits which are part of the stacked limit; single -- plot only the limit from the input combination of dwarfs", metavar="CASE", type='str', default ="all")
    

    (options, args) = parser.parse_args()
    
    list_dwarfs = options.dwarf_list.split('+')
    
    plot_limits_scan(dwarf_list = list_dwarfs, case = options.case, plot_type=options.plot_type)

if __name__ == '__main__':
    interactive()
