from sps4lat import model as mod
from sps4lat import preprocessing as exp
from sps4lat import optimize as opti
import healpy as hp 
import numpy as np 
import matplotlib.pyplot as plt

def flatten_cls(cls):
    s = cls.shape
    n = s[0]
    cl_list = []
    for j in range(n):
        cl_list += [np.diagonal(cls, offset=j, axis1=0, axis2=1)[:,i] for i in range(n-j)]
    return cl_list
    
freqs = np.array([50.,100.,150.,200.])
N = 4 
nside = 256
# nwhite = freqs 
lmax = int(3.*nside-1.)
ells = np.linspace(0,lmax,lmax+1)

bin_size=20
test = exp.Experiment('test',10,lmax,freqs,bin_size,beams=None)
print(test.beams)
test.read_map_file('../data/test/maps/test_pl')

mat = test.empirical_covmat
ell_plot = test.ell_mean
factor = ell_plot*(ell_plot+1)/2/np.pi
plt.figure()
plt.scatter(ell_plot, mat[2,2,:])
plt.show()

def start(emp_cov, dimension):
    emp_tot = emp_cov.sum(axis = -1)
    _,eigvects = np.linalg.eigh(emp_tot)
    noise = np.diag(emp_tot)
    A = eigvects[::-1,::-1]
    p_l = np.einsum('ba,bcl,cd->acl',A,emp_cov,A)
    return noise, A[:dimension,:dimension], p_l[:dimension,:dimension,:]
    
ell_read, cmb_tt_read = np.loadtxt("../data/cmb.dat", usecols=(0, 1),unpack=True)
cmb_tt = np.concatenate((np.zeros(2), cmb_tt_read /(ell_read * (ell_read + 1) /2. / np.pi)))
ells = test.ell_mean
N_l = len(ells)
nl,a,pl = start(test.empirical_covmat, 4)
cmb = mod.FactorizedCrossSpectrum(sed=mod.FreeSED(nu=freqs), cl=mod.FreeCls(ell=ells))
fgs = mod.CorrelatedFactorizedCrossSpectrum(sed=mod.Join(mod.FreeSED(nu=freqs),mod.FreeSED(nu=freqs)),
                                            cl=mod.PowerSpectraAndCovariance(mod.FreeCls(ell=ells),mod.FreeCls(ell=ells),mod.FreeCls(ell=ells)))
white_noise = mod.WhiteNoise(nu=freqs, ell = ells)
mod_opt = mod.Sum(cmb,fgs,white_noise)

dict_start = {'kwseq': [{'sed_kwargs': {'sed': a[0]}, 'cl_kwargs': { 'cls': pl[0,0,:]}}, {'sed_kwargs': {'kwseq': [{'sed': a[1]}, {'sed': a[2]}]}, 'cl_kwargs': 
             {'kwseq': [{'cls': pl[1,1,:]},{'cls': pl[2,2,:]}, {'cls': pl[1,2,:]}]}},{'nwhite':nl}]}
#mod_opt.set_defaults(**dict_start)

param_optimised = opti.optimiser_test(experiment=test, model=mod_opt, param_start=dict_start)
cov_opt = mod_opt.eval(**param_optimised)
cov_start = mod_opt.eval(**dict_start)


f,axs = plt.subplots(4,4,sharex = True,sharey = True,figsize=(14,8))
plt.subplots_adjust(wspace=0, hspace=0,left=0.1,right=0.98,top = 0.98,bottom=0.1)
for i in range(4):
    for j in range(4):
        axs[i,j].set_xlim(0,800)
        axs[i,j].set_ylim(0,8000)
        axs[i,j].scatter(ell_plot, factor*cov_opt[i,j,:], label = 'Best fit')
        axs[i,j].plot(ell_plot, factor*mat[i,j,:], label = 'Empirical')
        axs[i,j].scatter(ell_plot, factor*cov_start[i,j,:], label = 'Start')
        axs[i,j].text(0.9,0.5,'{:d}x{:d}'.format(int(test.freqs[i]),int(test.freqs[j])), horizontalalignment = 'center', verticalalignment='center', transform=axs[i,j].transAxes)
        axs[i,j].legend(loc = 'best')
plt.show()