import numpy as np
from astropy.io import fits 
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion
from optparse import OptionParser
from healpy.rotator import angdist
from gammapy.maps import Map, MapAxis, WcsGeom, HpxNDMap
from scipy.interpolate import interp1d
import multiprocessing
from joblib import Parallel, delayed
from scipy import interpolate
import os
import os.path
from scipy import stats
import itertools
import healpy as hp
from copy import deepcopy
from scipy.signal import savgol_filter
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import fileinput
from multiprocessing import Pool
from functools import partial
from iminuit import Minuit

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++ some global declarations +++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#source_path = "/Users/christopher/Documents/ESCAPE_project/mlfermidwarfs/data/"
source_path = "./"

#load 3FGL source catalogue to exclude positions in the sky with bright sources
plsfile = source_path + './gll_psc_v16.fit'

# global parameters
rad = 0.5 # deg
Area = np.pi*rad**2
DOmega = Area*(np.pi/180.)**2 # Delta Omega of regions
NbinsMAX = 24
Elist0 = np.logspace(np.log10(0.5),np.log10(500.),NbinsMAX+1)
Elist_rebinning_constraints = [0.66, 0.88, 1.18, 1.58, 2.81] ## energies in GeV

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++ declaration of functions        ++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def create_energy_macro_bins(ebins_data, path):
    rebin_edges = [0]
    initial_edges = deepcopy(ebins_data)
    for E_thr in Elist_rebinning_constraints:
        if E_thr < min(ebins_data):
            continue
        pass_E = list(filter(lambda x: x < E_thr, ebins_data))
        rebin_edges += [max(rebin_edges) + len(pass_E)]
        ebins_data = sorted(set(ebins_data) - set(pass_E))
    os.system('rm ' + path + 'Newbins_edges.dat')
    fileout = open(path + 'Newbins_edges.dat', "a")
    for id in (rebin_edges + [-1]):
        fileout.write("{} {}\n".format(int(id), initial_edges[int(id)]))
    fileout.close()
    return rebin_edges

def getdSphs_data_from_table(dSphs_table, N_data_cols):
    dsph_data = np.loadtxt(dSphs_table, usecols = np.arange(1, N_data_cols))
    dshp_names = np.loadtxt(dSphs_table, dtype = "S", usecols = (0, ))
    dshp_names = list(map(lambda x: x.decode('UTF-8'), dshp_names))
    dshp_dict = {}
    for name, data_array in zip(dshp_names, dsph_data):
        dsph_properties = {
            "POS": (data_array[0], data_array[1]),    ###Galactic coordinates: LON/LAT [deg/deg]
            "JFACTOR": (data_array[2], data_array[3]),###J_factors as log_10(J) and scatter
            "EXP": data_array[4:]                     ###Fermi-LAT exposure at given dwarf position (uses the original data binning, rebinned during the profiling)
        }
        dshp_dict[name] = dsph_properties
    return dshp_dict

def getdSphs(dSph_data_table, exceptions = None):
    # data from  1611.03184
    # all = 'no' : remove ultrafaint dSphs
    blist = []
    llist = []
    for dSph, data in dSph_data_table.items():
        if not exceptions is None and dSph in exceptions:
            continue
        else:
            blist.append(data["POS"][1])
            llist.append(data["POS"][0])
    return blist, llist

#bdwth = width of the kernel
def kde_skl(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x)
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

def optbandwidth(lista, bwmin, bwmax, Np):
    grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(bwmin, bwmax,Np)},cv=10) # cross validation = CV
    grid.fit(lista)
    string=str(grid.best_params_)
    trash,val = string.split()
    return val[:-1]

def buildPDF(lista,grid,bw):
    # lista: array from which PDF will be built
    # bw: optimal bandwidth
    pdf = kde_skl((lista),grid,bandwidth=bw)
    pdf_norm = pdf/sum(pdf)
    return pdf_norm

def buildPDFlon(lista,grid,bw):
    # lista: array from which PDF will be built
    # bw: optimal bandwidth
    pdf = np.array([1/(2*np.pi)]*len(grid))
    pdf_norm = pdf/sum(pdf)
    return pdf_norm

def buildPDFlat(lista,grid,bw):
    # lista: array from which PDF will be built
    # bw: optimal bandwidth
    pdf = np.cos(grid*np.pi/180)/2
    pdf_norm = pdf/sum(pdf)
    return pdf_norm

def ppf(name, lista, grid, bw, y, kind = 'opt'): 
    lista = np.array(lista).reshape((-1,1))
    if(name=='lon'):
        if kind == 'opt':
            RVdiscrete_l = stats.rv_discrete(values=(grid,buildPDF(lista,grid,bw)),
                                            name='RVdiscrete')
            return RVdiscrete_l.ppf(y)
        else:
            RVdiscrete_l = stats.rv_discrete(values=(grid,buildPDFlon(lista,grid,bw)),
                                            name='RVdiscrete')
            return RVdiscrete_l.ppf(y)            
    if(name=='lat'):
        if kind == 'opt':
            RVdiscrete_b = stats.rv_discrete(values=(grid,buildPDF(lista,grid,bw)),
                                         name='RVdiscrete')
            return RVdiscrete_b.ppf(y)
        else:
            RVdiscrete_b = stats.rv_discrete(values=(grid,buildPDFlat(lista,grid,bw)),
                                         name='RVdiscrete')
            return RVdiscrete_b.ppf(y)
    if(name=='bckg'):
        RVdiscrete_bckg = stats.rv_discrete(values=(grid,buildPDF(lista,grid,bw)),
                                            name='RVdiscrete')
        return RVdiscrete_bckg.ppf(y)
    
def euclidean_norm(v1, v2):
    return np.linalg.norm(v1 - v2, ord = 2)

def voidgen(Nvoid, dSphs_LON, dSphs_LAT, bwb, bwl, plslat, plslon, kind = 'opt'):
    listb, listl = dSphs_LAT, dSphs_LON
    gridl = np.linspace(1.,359.,1000)
    gridb = np.linspace(-89.,89.,1000)
    
    # including extended sources to the lists of dwarfs
    l_ext1 = [302.14498901]
    l_ext2 = [278.83999634]
    b_ext1 = [-44.41999817]
    b_ext2 = [-32.84999847]
    listb += b_ext1
    listb += b_ext2
    listl += l_ext1
    listl += l_ext2
    
    # generating a preliminary list of candidates
    np.random.seed() # (1111), (1982) # (123) random seed for reproducibility
    lpdfval=np.random.random(Nvoid)
    bpdfval=np.random.random(Nvoid)
    l_voidpre = ppf('lon', listl, gridl, bwl, lpdfval, kind = kind) #pre--> preliminar
    b_voidpre = ppf('lat', listb, gridb, bwb, bpdfval, kind = kind) 

    # excluding those overlapping with dwarfs, extended sources, disk,
    # and point-like sources
    rad = 1.0 # impose this angular separation between centroids
    rdisk = 20.0 # angular extension of galactic disk in latitude
    l_void = None
    b_void = None
    print('Nvoids = ',Nvoid)

    for i, (tmp_LON, tmp_LAT) in enumerate(zip(l_voidpre, b_voidpre)):
        if abs(tmp_LAT) <= rdisk:
            continue
        if not l_void is None:
            dist_check = list(map(lambda x: euclidean_norm(np.array([tmp_LON, tmp_LAT]), np.array([x[0], x[1]])), list(zip(l_void, b_void))))
            if min(dist_check) <= rad:
                continue
            
        # array of delta_r between
        #every empty region and all the dwarfs
        delta_r = list(map(lambda x: euclidean_norm(np.array([tmp_LON, tmp_LAT]), np.array([x[0], x[1]])), list(zip(listl, listb))))
            
        if min(delta_r) > rad:
            # removing point-like sources
            #delta_rp = list(map(lambda x: (angdist([tmp_LON, tmp_LAT], [x[0], x[1]], lonlat = True) / np.pi * 180)[0], list(zip(plslon, plslat))))
            delta_rp = list(map(lambda x: euclidean_norm(np.array([tmp_LON, tmp_LAT]), np.array([x[0], x[1]])), list(zip(plslon, plslat))))

            if min(delta_rp) > 2.5:
                l_void = [tmp_LON] if l_void is None else l_void + [tmp_LON]
                b_void = [tmp_LAT] if b_void is None else b_void + [tmp_LAT]
        if i % 1000 == 0 and i != 0:
            print(i,'-th sample, len(array) = ',len(b_void))
    return l_void, b_void

def voidstofile(path, llist, blist):
    os.system('rm ' + path + 'control_voids.dat')
    fileout = open(path + 'control_voids.dat', "a")
    palline=np.zeros((len(llist),2))
    for i in range(len(llist)):
        palline[i] = np.array([llist[i],blist[i]])
        fileout.write(" ".join([str(palline[i,0]),str(palline[i,1]),"\n"]))
    fileout.close()

def get_counts_from_data(llist, blist, LAT_counts, LAT_exposure, path = './', kind = 'dSphs'):
    counts_data = Map.read(LAT_counts)
    exposure_data = Map.read(LAT_exposure)
        
    energy_unit = counts_data.geom.axes[0].edges.unit
    Ebins_data = counts_data.geom.axes[0].edges.to(u.GeV).value
    
    Nsample = len(llist)
    Nen = len(counts_data.geom.axes[0].center.value)
    count_sample = np.zeros((Nsample,Nen))
    exp_sample = np.zeros((Nsample,Nen))
    rae = 5 #extendend radius of region, in units of 0.1 deg
    
    for i, (tmp_LON, tmp_LAT) in enumerate(zip(llist, blist)):
        if type(counts_data) == type(HpxNDMap.create(nside=4)):
            #all_pix = hp.query_disc(counts_data.geom.nside[0], hp.ang2vec(tmp_LON, tmp_LAT, lonlat = True), radius = np.radians(1.0))
            on_source_pix = hp.query_disc(counts_data.geom.nside[0], hp.ang2vec(tmp_LON, tmp_LAT, lonlat = True), radius = np.radians(0.5))
            #diff_pix = list(set(all_pix) -set(on_source_pix))
            
            valid_pixels = []
            for j in on_source_pix:
                valid_pixels.append(counts_data.data[:, j])
            valid_pixels = np.array(valid_pixels)
            summed_counts = (valid_pixels.T).sum(axis = -1)
            count_sample[i] += np.array(summed_counts).astype('float')
            
            #all_pix = hp.query_disc(exposure_data.geom.nside[0], hp.ang2vec(tmp_LON, tmp_LAT, lonlat = True), radius = np.radians(1.0))
            on_source_pix = hp.query_disc(exposure_data.geom.nside[0], hp.ang2vec(tmp_LON, tmp_LAT, lonlat = True), radius = np.radians(0.5))
            #diff_pix = list(set(all_pix) -set(on_source_pix))
            
            valid_pixels = []
            for j in on_source_pix:
                valid_pixels.append(exposure_data.data[:, j])
            valid_pixels = np.array(valid_pixels)
            mean_exposure = (valid_pixels.T).mean(axis = -1)
            
            if on_source_pix.size == 0:
                id = hp.ang2pix(exposure_data.geom.nside[0], tmp_LON, tmp_LAT, lonlat = True)
                mean_exposure = np.array(exposure_data.data[:, id]).astype('float')
            f_exp = interp1d(np.log10((exposure_data.geom.axes[0].center).to(u.GeV).value), np.log10(mean_exposure), fill_value = 'extrapolate')
            exp_E = (exposure_data.geom.axes[0].center).to(u.GeV).value
            exp_sample[i] += np.array([np.power(10., f_exp(np.log10(np.sqrt(E1 * E2)))) for E1, E2 in zip(exp_E, exp_E[1:])])
            
        else:
            position = SkyCoord(tmp_LON, tmp_LAT, frame="galactic", unit="deg")
            counts_cutout = counts_data.cutout(position = position, width=(1 * u.deg, 1 * u.deg))
            exposure_cutout = exposure_data.cutout(position = position, width=(1 * u.deg, 1 * u.deg))
            
            position_mask = CircleSkyRegion(position, Angle("0.5 deg"))
            masked_cutout = counts_cutout.geom.region_mask([position_mask])
            counts_in_region = (masked_cutout * counts_cutout.data).sum(axis = (1,2))
            count_sample[i] += counts_in_region
            
            masked_cutout = exposure_cutout.geom.region_mask([position_mask])
            exposure_in_region = (masked_cutout * exposure_cutout.data)
            exposure_in_region = np.array(list(map(lambda x: x[x > 0].mean(), exposure_in_region)))
            f_exp = interp1d(np.log10((exposure_data.geom.axes[0].center).to(u.GeV).value), np.log10(exposure_in_region), fill_value = 'extrapolate')
            exp_E = (exposure_data.geom.axes[0].center).to(u.GeV).value
            #exp_sample[i] += np.power(10., f_exp(np.log10((exposure_data.geom.axes[0].center).to(u.GeV).value)))
            exp_sample[i] += np.array([np.power(10., f_exp(np.log10(np.sqrt(E1 * E2)))) for E1, E2 in zip(exp_E, exp_E[1:])])
        
        if(i % 500 == 0):
            print('filling region # ',i,' out of ',Nsample)
    os.system('rm ' + path + 'counts_' + kind + '.dat')
    os.system('rm ' + path + 'exp_' + kind + '.dat')
    fileout0 = open(path + 'counts_' + kind + '.dat', "a")
    fileout1 = open(path + 'exp_' + kind + '.dat', "a")
    for i in range(Nsample):
        string0 = str(count_sample[i])
        substr0 = string0[1:-1].split()
        string1 = str(exp_sample[i])
        substr1 = string1[1:-1].split()
        fileout0.write(" ".join([substr0[j] for j in range(len(substr0))]))
        fileout0.write("\n")
        fileout1.write(" ".join([substr1[j] for j in range(len(substr1))]))
        fileout1.write("\n")
    fileout0.close()
    fileout1.close()
    return Ebins_data
    
def update_dwarf_data_table(dwarf_table, dwarf_exposure, out_path):
    N_E = len(dwarf_exposure[0])
    f_out = open(out_path + "current_dwarf_data_table.dat", "+w")
    with open(dwarf_table, 'r') as infl:
        num_dwarf = 0
        for j, line in enumerate(infl):
            if "#" in line:
                f_out.write(line)
            else:
                tmp = line.split()
                f_out.write(" ".join(tmp[:5]) + " " + " ".join(list(map(str, dwarf_exposure[num_dwarf]))) + "\n")
                num_dwarf += 1
    f_out.close()
    return True

def PDFvect2(id, yi, xi, sigma, varsigma):
    lnL = 0.0
    n = len(xi)
    for i in range(n):
        Dx = xi[i] - xi
        Dx_del = np.delete(Dx, i, axis = 0)
        DX = Dx_del[:,0]**2 + Dx_del[:,1]**2
        expX = np.longdouble(np.exp(-DX/(2*sigma**2)))
        Dy = (np.log(yi[i]) - np.log(yi))**2
        Dy_del = np.delete(Dy, i, axis = 0)
        expY = np.longdouble(np.exp(-Dy_del/(2*varsigma**2))) #exponential in (4.4)
        lnL += np.log((expX * expY / yi[i]).sum() / ((2*np.pi)**(3/2)*(sigma**2) * varsigma * (n - 1)))
    return [id, varsigma, sigma, lnL]

def yhat_arr(d,xD,yi,xi,sigma):
    # first removing points with zero counts
    argdel=np.where(yi==0.)[0]
    yi=np.delete(yi,argdel)
    xi=np.delete(xi,argdel,axis=0)
    x=xD[d]
    n = len(xi)
    X = np.array([x]*n).reshape((-1,2))
    AuxD = X-xi
    D =  AuxD[:,0]**2 + AuxD[:,1]**2
    expX = np.float128(np.exp(-D/(2*sigma**2)))
    return np.sum(expX*np.log(yi))/ np.sum(expX)

def yhat2_arr(d,xD,yi,xi,sigma):
    x=xD[d]
    n = len(xi)
    X = np.array([x]*n).reshape((-1,2))
    AuxD = X-xi
    D =  AuxD[:,0]**2 + AuxD[:,1]**2
    expX = np.float128(np.exp(-D/(2*sigma**2)))
    return np.sum(expX*(np.log(yi))**2)/ np.sum(expX)

def get_admissible_void_regions(LAT_source_catalog, dwarf_data, path, kind = 'opt'):
    source_catalog = fits.getdata(LAT_source_catalog, 1)
    plslat=source_catalog.field('GLAT')
    plslon=source_catalog.field('GLON')
    
    #+++++++++++++++++++++++++++++++++++++++++
    print('++++++++++++ generating void centroids ++++++++++++++')
    #++++++++++++++++++++++++++++++++++++++++++
    
    # +++++ getting dwarfs coordinates ++++++++++++
    dSphs_LAT, dSphs_LON = getdSphs(dwarf_data) #'no'
    llist = np.array(dSphs_LON).reshape(-1,1)
    blist = np.array(dSphs_LAT).reshape(-1,1)
    
    # +++++ computing optimum bandwidth for KDE
    bandwlon=float(optbandwidth(llist,1,50,10))/2 
    bandwlat=float(optbandwidth(blist,1,50,10))/2
    print('optimum bandwidth for lon and lat: ',bandwlon,bandwlat)
    
    # +++ max number of void regions of area = pi deg^2 (r=1deg)
    rad = 0.5 # deg
    Area = np.pi*rad**2 #area of a single region
    Nvoid = int(4*np.pi/(Area*(np.pi/180)**2))
    print('generating ',Nvoid,' regions')
    l_void, b_void = voidgen(Nvoid, dSphs_LON, dSphs_LAT, bandwlat, bandwlon, plslat, plslon, kind = kind)
    print('final number of void regions = ',len(l_void))
    print('+++++++exporting void coordinates to data file +++++')
    voidstofile(path, l_void, b_void)
    return True

def fill_analysis_regions(dwarf_table, path, LAT_counts, LAT_exposure):
    dwarf_data = getdSphs_data_from_table(dwarf_table, 29)
    for component in ['dSphs', 'control_voids']:
        if component == 'dSphs':
            print('+++++ filling dSphs positions with counts ++++')
            D_blist, D_llist = getdSphs(dwarf_data)
            D_blist=np.array(D_blist).flatten()
            D_llist=np.array(D_llist).flatten()    
            get_counts_from_data(D_llist, D_blist, LAT_counts, LAT_exposure, path = path, kind = component)
            dwarf_exposure = np.loadtxt(path + 'exp_' + component + '.dat')
            update_dwarf_data_table(dwarf_table, dwarf_exposure, path)
        if component == 'control_voids':
            print('+++++ filling voids positions with counts ++++')
            l_void = []
            b_void = []
            vfile = fileinput.input(path + component + '.dat')
            for linea in vfile:
                lon, lat = linea.split()
                l_void.append(float(lon))
                b_void.append(float(lat))
            vfile.close()
            Ebins_data = get_counts_from_data(l_void, b_void, LAT_counts, LAT_exposure, path = path, kind = component)
    return Ebins_data

def create_counts_macro_bins(path, dwarf_data, rebinE_edges):
    x_voids = np.loadtxt(path + 'control_voids.dat')
    lat_dSphs, lon_dSphs = getdSphs(dwarf_data)
    lat_dSphs, lon_dSphs = np.array(lat_dSphs), np.array(lon_dSphs)
    count_voids = np.loadtxt(path + 'counts_control_voids.dat')
    counttotV = np.sum(count_voids, axis = 1)
    argdel = np.where(counttotV==0.)[0]
    counttotV = np.delete(counttotV, argdel)
    count_voids = np.delete(count_voids, argdel, axis = 0)
    x_voids = np.delete(x_voids, argdel, axis = 0)
    count_dSphs = np.loadtxt(path + 'counts_dSphs.dat')
    counttotD = np.sum(count_dSphs, axis=1)
    x_dSphs = np.hstack((lon_dSphs.reshape((-1,1)),lat_dSphs.reshape((-1,1))))
    
    # ++++ re-grouping in len(rebinE_edges) bins; can't have c=0 for log reasons ++++++++
    b = rebinE_edges + [len(count_voids[0])]
    bincountV = count_voids[:,b[0]].reshape((-1,1))
    bincountD = count_dSphs[:,b[0]].reshape((-1,1))
        
    for i in range(1,len(b)-1):
        if(b[i]==b[i+1]-1):
            bincountV=np.concatenate((bincountV,count_voids[:,b[i]].reshape((-1,1))),axis=1)
            bincountD=np.concatenate((bincountD,count_dSphs[:,b[i]].reshape((-1,1))),axis=1)
        else:
            BinV=np.sum(count_voids[:,b[i]:b[i+1]],axis=1)
            BinD=np.sum(count_dSphs[:,b[i]:b[i+1]],axis=1)
            bincountV=np.concatenate((bincountV,BinV.reshape((-1,1))), axis=1)
            bincountD=np.concatenate((bincountD,BinD.reshape((-1,1))), axis=1)
    
    os.system('rm ' + path + 'Newbins_counts_control_voids.dat')
    fileout = open(path + 'Newbins_counts_control_voids.dat', "a")
    for j in range(len(bincountV)):
        fileout.write(" ".join(list(map(str, bincountV[j].flatten())) + ["\n"]))
    fileout.close()
    
    os.system('rm ' + path + 'Newbins_counts_dSphs.dat')
    fileout = open(path + 'Newbins_counts_dSphs.dat', "a")
    for j in range(len(bincountD)):
        fileout.write(" ".join(list(map(str, bincountD[j].flatten())) + ["\n"]))
    fileout.close()
    

def optimize_smoothing_parameters(path, sX = (1.5, 3.0), sY = (0.05, 0.5), samples = 20):
    print('++++++ importing counts and coordinates from voids +++++')
    x_voids = np.loadtxt(path + 'control_voids.dat')
    lon_voids, lat_voids = x_voids[:,0], x_voids[:,1]
    count_voids = np.loadtxt(path + 'Newbins_counts_control_voids.dat')
    counts0 = count_voids[:,0]
    print('original len=',len(x_voids))
    # removing void regions with zero counts in 1st bin
    argdel = np.where(counts0 == 0.)[0]
    counts0 = np.delete(counts0, argdel)
    x_voids = np.delete(x_voids, argdel, axis=0)
    print('new len=',len(x_voids))

    # ++++ scan  ++++
    sX_min, sX_max = sX
    sY_min, sY_max = sY
    sigma_list = np.logspace(np.log10(sX_min),np.log10(sX_max), samples)
    sigmaY_list = np.logspace(np.log10(sY_min),np.log10(sY_max), samples)
    os.system('rm ' + path + 'pdf_grid.dat')
    test_sX, test_sY = np.meshgrid(sigmaY_list, sigma_list)
    parallel_input = zip(test_sX.flatten(), test_sY.flatten())
    num_cores = int(multiprocessing.cpu_count())
    res = Parallel(n_jobs = num_cores)(delayed(PDFvect2)(k, counts0, x_voids, sigX, sigY) for k, (sigY, sigX) in enumerate(parallel_input))
    res = np.array(list(sorted(res)))
    
    fileout = open(path + 'pdf_grid.dat',"a")
    for j, (x, y) in enumerate(list(zip(test_sX.flatten(), test_sY.flatten()))):
        fileout.write(" ".join([str(x), str(y), str(res[j, -1]), "\n"]))
    fileout.close()
    
    sigma, varsigma, lnPDF = np.loadtxt(path + 'pdf_grid.dat', unpack = True)
    sigma = np.array(sigma).reshape((-1,1))
    varsigma = np.array(varsigma).reshape((-1,1))
    points = np.hstack((sigma,varsigma))
    lnPDF = np.array(lnPDF)
    X = np.logspace(np.log10(min(sigma)),np.log10(max(sigma)),100)
    Y = np.logspace(np.log10(min(varsigma)),np.log10(max(varsigma)),100)
    func = lambda x,y: interpolate.griddata(points, lnPDF, (x,y), method = 'cubic', rescale = True)
    S, V = np.meshgrid(X, Y)
    Z = func(S, V)
    Xstar = np.round(X[np.argmax(func(X, Y))], 2)
    Ystar = np.round(Y[np.argmax(func(X, Y))], 2)
    Zmax = np.max(func(X, Y))
    Xstar, Ystar = np.round(S.flatten()[np.argmax(Z.flatten())], 2), np.round(V.flatten()[np.argmax(Z.flatten())], 2)
    print(Xstar, Ystar)
    return Xstar, Ystar

def optimize_smoothing_parameters_perE(path, NE, sX = (1.5, 3.0), sY = (0.05, 0.5), samples = 20):
    print('++++++ importing counts and coordinates from voids +++++')
    x_voids = np.loadtxt(path + 'control_voids.dat')
    lon_voids, lat_voids = x_voids[:,0], x_voids[:,1]
    count_voids = np.loadtxt(path + 'Newbins_counts_control_voids.dat')
    counts0 = count_voids[:, NE]
    print('original len=',len(x_voids))
    # removing void regions with zero counts in 1st bin
    argdel = np.where(counts0 == 0.)[0]
    counts0 = np.delete(counts0, argdel)
    x_voids = np.delete(x_voids, argdel, axis=0)
    print('new len=',len(x_voids))

    # ++++ scan  ++++
    sX_min, sX_max = sX
    sY_min, sY_max = sY
    sigma_list = np.logspace(np.log10(sX_min),np.log10(sX_max), samples)
    sigmaY_list = np.logspace(np.log10(sY_min),np.log10(sY_max), samples)
    os.system('rm ' + path + 'pdf_grid_per_NE_{}.dat'.format(NE))
    test_sX, test_sY = np.meshgrid(sigmaY_list, sigma_list)
    parallel_input = zip(test_sX.flatten(), test_sY.flatten())
    num_cores = int(multiprocessing.cpu_count())
    res = Parallel(n_jobs = num_cores)(delayed(PDFvect2)(k, counts0, x_voids, sigX, sigY) for k, (sigY, sigX) in enumerate(parallel_input))
    res = np.array(list(sorted(res)))
    
    fileout = open(path + 'pdf_grid_per_NE_{}.dat'.format(NE), 'a')
    for j, (x, y) in enumerate(list(zip(test_sX.flatten(), test_sY.flatten()))):
        fileout.write(" ".join([str(x), str(y), str(res[j, -1]), "\n"]))
    fileout.close()
    
    sigma, varsigma, lnPDF = np.loadtxt(path + 'pdf_grid_per_NE_{}.dat'.format(NE), unpack = True)
    sigma = np.array(sigma).reshape((-1,1))
    varsigma = np.array(varsigma).reshape((-1,1))
    points = np.hstack((sigma,varsigma))
    lnPDF = np.array(lnPDF)
    X = np.logspace(np.log10(min(sigma)),np.log10(max(sigma)),100)
    Y = np.logspace(np.log10(min(varsigma)),np.log10(max(varsigma)),100)
    func = lambda x,y: interpolate.griddata(points, lnPDF, (x,y), method = 'cubic', rescale = True)
    S, V = np.meshgrid(X, Y)
    Z = func(S, V)
    Xstar = np.round(X[np.argmax(func(X, Y))], 2)
    Ystar = np.round(Y[np.argmax(func(X, Y))], 2)
    Zmax = np.max(func(X, Y))
    Xstar, Ystar = np.round(S.flatten()[np.argmax(Z.flatten())], 2), np.round(V.flatten()[np.argmax(Z.flatten())], 2)
    print(Xstar, Ystar)
    return Xstar, Ystar

def estimate_bkg_dSphs(path, dwarf_data, sigma, varsigma):
    bincountD = np.loadtxt(path + 'Newbins_counts_dSphs.dat')
    bincountV = np.loadtxt(path + 'Newbins_counts_control_voids.dat')
    
    lat_dSphs, lon_dSphs = getdSphs(dwarf_data)
    lat_dSphs, lon_dSphs = np.array(lat_dSphs), np.array(lon_dSphs)
    x_dSphs = np.hstack((lon_dSphs.reshape((-1,1)),lat_dSphs.reshape((-1,1))))
    x_voids = np.loadtxt(path + 'control_voids.dat')
    
    counttotD = np.sum(bincountD, axis=1)
    counttotV = np.sum(bincountV, axis=1)
    # ++++++++++ computing prediction ++++++++++++++++++++++++++++++++++
    Nbins=len(bincountV[0])
    lnbckgest=[]
    for i in range(Nbins):
        print('+++++++++ this is bin # ',i)
        countV = bincountV[:,i]
        countD = bincountD[:,i]
        yhat_1arg=partial(yhat_arr,xD=x_dSphs,yi=countV,xi=x_voids,sigma=sigma)
        p=Pool(4)
        yhat_list=p.map(yhat_1arg,range(len(x_dSphs)))
        lnbckgest.append(yhat_list)
        os.system('rm ' + path + 'lnbckg_dSphs_bin_'+str(i)+'.dat')
        fileout = open(path + 'lnbckg_dSphs_bin_'+str(i)+'.dat',"a")
        for j in range(len(x_dSphs)):
            fileout.write(" ".join([str(np.log(countD[j])),str(yhat_list[j]),"\n"]))
        fileout.close()
    bckgest =  np.sum(np.exp(lnbckgest),axis=0)
    print('sum bckg_est = ',bckgest)
    print(bckgest[5],bckgest[3],bckgest[6],bckgest[10],bckgest[11],bckgest[16],
            bckgest[18],bckgest[22])
    print('ln(sum bckg est) = ',np.log(bckgest))
    print('Poisson = ',np.sqrt(counttotD))
    print('++++++++++++++++++computing variance++++++++++++++++++++++++++')
    yhat_1arg=partial(yhat_arr,xD=x_dSphs,yi=counttotV,xi=x_voids,sigma=sigma)
    p=Pool(4)
    yhat_list=p.map(yhat_1arg,range(len(x_dSphs)))
    w2_2=np.array(yhat_list)**2
    yhat2_1arg=partial(yhat2_arr,xD=x_dSphs,yi=counttotV,xi=x_voids,sigma=sigma)
    p=Pool(4)
    w2_1=p.map(yhat2_1arg,range(len(x_dSphs)))
    Var = varsigma**2 + np.array(w2_1) - w2_2
    DeltaB = bckgest*np.sqrt(Var)
    print('Delta ln = ',np.sqrt(Var))
    print('DeltaB = ',DeltaB)
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for i in range(len(bckgest)):
        print(counttotD[i],' $\pm$ ',np.round(np.sqrt(counttotD[i])),' & ',
                np.round(bckgest[i],2),' & ',
                np.round(np.log(bckgest[i]),2),' $\pm$ ',np.round(np.sqrt(Var[i]),2))
    
def perform_background_optimisation(path, LATdata, LATexposure, source_catalog, dwarf_data_table, kind = 'iso'):
    ###################################################################################################################
    #### generating centroids for voids
    dwarf_data = getdSphs_data_from_table(dwarf_data_table, 29)
    get_admissible_void_regions(source_catalog, dwarf_data, path, kind = kind)
    
    ###################################################################################################################
    #### filling voids and dwarfs with count data
    Ebins_data = fill_analysis_regions(dwarf_data_table, path, LATdata, LATexposure)
    
    ###################################################################################################################
    #### finding optimal smoothing parameters (based on first energy bin)
    selected_macro_Ebin_edges = create_energy_macro_bins(Ebins_data, path)
    print(selected_macro_Ebin_edges)
    create_counts_macro_bins(path, dwarf_data, selected_macro_Ebin_edges)
    varsigma, sigma = optimize_smoothing_parameters(path, sX = (1.5, 3.0), sY = (0.05, 0.5), samples = 20)
    
    ###################################################################################################################
    #### estimation of bckg at dSph positions
    estimate_bkg_dSphs(path, dwarf_data, sigma, varsigma)
    return True

    
def main():
    parser = OptionParser()
    
    parser.add_option("-d", "--data", dest="LAT_data", help="LAT data file", metavar="DATA", type='str', default = './data/CCUBE_P8R2_SOURCE_V6_w9to522_evtype3_nxpix3600_nypix1800_binsz0.1_Elo500_Ehi500000.fits')
    parser.add_option("-e", "--exposure", dest="LAT_exposure", help="LAT exposure file", metavar="EXP", type='str', default = './data/expmap_CCUBE_P8R2_SOURCE_V6_w9to522_evtype3_nxpix3600_nypix1800_binsz0.1_Elo500_Ehi500000.fits')
        
    parser.add_option("-l", "--dictionary", dest="dwarf_dict", help="dSph dict", metavar="DICT", type='str', default = "default_dwarf_summary_table.dat")
    parser.add_option("-p", "--data_path", dest="data_path", help="path to folder where created data is to be stored", metavar="PATH", type='str', default = source_path + "../analysis_data/")
    parser.add_option("--kind", "--void_dist", dest="void_dist", help="distribution of control voids; either optimized 'opt' w.r.t. position of dwarfs or 'iso', i.e. isotropic", metavar="KIND", type='str', default = 'iso')
    
    (options, args) = parser.parse_args()
    dwarf_table = options.dwarf_dict
    #### The second argument is the total number of columns in the dwarf data file. It needs to be updated by hand.
    dwarf_data = getdSphs_data_from_table(dwarf_table, 29)
    
    perform_background_optimisation(options.data_path, options.LAT_data, options.LAT_exposure, plsfile, options.dwarf_dict, kind = options.void_dist)
                    
   
if __name__ == "__main__":
    main()
