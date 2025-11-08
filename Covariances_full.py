import sys
import numpy as np
import mpytools as mpy
from mockfactory import Catalog
import mockfactory.utils
import thecov.geometry
import thecov.covariance
from cosmoprimo.fiducial import DESI
from pypower import CatalogFFTPower, CatalogSmoothWindow,setup_logging, mpi
from pathlib import Path
from functools import partial
from astropy.table import Table, vstack, hstack
import fitsio
import healpy as hp
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc= comm.Get_size()

setup_logging()  # for logging messages
cosmo = DESI()  # fiducial cosmology


class full_covariance:
    def __init__(self,tracer,DR,data_fns,rand_fns,load_cats = ['data','randoms']):
        self.tracer = tracer
        self.DR = DR
        self.data_fns = data_fns
        self.rand_fns = rand_fns
        self.load_cats = load_cats
        
        if 'data' in self.load_cats:
            if rank ==0:
                print('Loading data catalog...')
                data = vstack([Table(fitsio.read(fn)) for fn in self.data_fns])
            else:
                data = None
            self.data = comm.bcast(data,root=0)
            print('data catalog loaded')
        if 'randoms' in self.load_cats:
            if rank ==0:
                print('Loading randoms catalog...')
                randoms = vstack([Table(fitsio.read(fn)) for fn in self.rand_fns])
            else:
                randoms = None
            self.randoms = comm.bcast(randoms,root=0)
            print('randoms catalog loaded')


    def get_positions_weights(self,catalog,weight_nms, zmin,zmax):
        zcut = (catalog['Z'] <= zmax) & (catalog['Z'] >= zmin)
        cat = catalog[zcut]
        weights= np.prod(np.array([cat[w] for w in weight_nms]),axis=0)
        return [np.radians(90-cat['DEC']),np.radians(cat['RA'])], weights
    
    def measure_pk_pypower(self,zrange,weight_nms,save_path):
        zmin,zmax = zrange
        kedges = np.linspace(0, 0.4, 80)
        ells = [0,2,4]
        if rank ==0:
            print('getting positions and weights for data and randoms...')
            data_positions,data_weights = self.get_positions_weights(self.data,weight_nms, zmin,zmax)
            rand_positions,rand_weights = self.get_positions_weights(self.randoms,weight_nms, zmin,zmax)
        else:
            data_positions, data_weights = None, None
            rand_positions, rand_weights = None, None
        data_positions = comm.bcast(data_positions,root=0)
        data_weights = comm.bcast(data_weights,root=0)
        rand_positions = comm.bcast(rand_positions,root=0)
        rand_weights = comm.bcast(rand_weights,root=0)

        if rank ==0: print('measuring Pk with pypower...')

        result = CatalogFFTPower(data_positions1=data_positions, data_weights1=data_weights,
                         randoms_positions1=rand_positions, randoms_weights1=rand_weights,
                         edges=kedges, ells=ells, interlacing=3, cellsize= 6., boxsize = None, resampler='tsc',
                         los='firstpoint', position_type= 'rdd',  dtype='f8', mpicomm=comm, mpiroot=0)
        
        if rank ==0: 
            print('Done, saving...')
            result.save(save_path + '.npy')
            print(f'Saved to {save_path + '.npy'}')
        return result
    
    def measure_windows(self,zrange,weight_nms,pk_poles,save_path):
        zmin,zmax = zrange
        if rank ==0:
            print('getting positions and weights for randoms...')
            rand_positions,rand_weights = self.get_positions_weights(self.randoms,weight_nms, zmin,zmax)
        else:
            rand_positions,rand_weights = None, None
        rand_positions = comm.bcast(rand_positions,root=0)
        rand_weights = comm.bcast(rand_weights,root=0)

        direct_attrs = {'nthreads': 128}
        win_direct_selection_attrs = direct_selection_attrs = None #{'theta': (0., 1.)}
        direct_edges = None
        boxsize = pk_poles.attrs['boxsize'][0]

        boxscales = [1., 5., 20.]

        boxsizes = boxsize*np.array(boxscales)
        edges = {'step': 2. * np.pi / np.max(boxsizes)}
        print(boxsizes)

        windows = []

        if rank ==0: print('measuring windows with pypower...')
        
        for iboxsize, boxsize in enumerate(boxsizes):
            boxsize = boxsizes[iboxsize]
            if rank == 0:
                print(iboxsize,boxsize)

            wind = CatalogSmoothWindow(randoms_positions1=rand_positions, randoms_weights1=rand_weights,
                                            power_ref=pk_poles, edges=edges, boxsize=boxsize, position_type='rdd',
                                            direct_attrs=direct_attrs,
                                            direct_selection_attrs=win_direct_selection_attrs if iboxsize == 0 else None,
                                            direct_edges=direct_edges if iboxsize == 0 else None,
                                            mpicomm=comm, mpiroot=0)
            if rank == 0:
                print(iboxsize,'th window finished')
                wind.save(save_path + f'_box{int(boxscales[iboxsize])}.npy')
                windows.append(wind)
        return windows
    
    def compute_pk_covariance(self,component,zrange,zeff,pk_fid,save_path,bias,krange = (0,0.4,0.005)):
        kmin, kmax, dk = krange
        kmid = np.arange(kmin + dk/2, kmax, dk)
        P0 = pk_fid[:,1]
        P2 = pk_fid[:,2]
        # P4 = pk_fid[:,3]

        zcut_rand = (self.randoms['Z'] <= zrange[1]) & (self.randoms['Z'] >= zrange[0])
        randoms = self.randoms[zcut_rand]
        zcut_dat = (self.data['Z'] <= zrange[1]) & (self.data['Z'] >= zrange[0])
        data = data[zcut_dat]
        Ngal,Nrands = len(data),len(randoms)
        alpha = Ngal/Nrands

        # Convert sky coordinates to cartesian using fiducial cosmology
        randoms['POSITION'] = mockfactory.utils.sky_to_cartesian(
                        cosmo.comoving_radial_distance(randoms['Z']),
                        randoms['RA'],
                        randoms['DEC'],
                        degree=True)
        
        
        if component == 'gaussian':
            print('Creating Geometry object...')
            # Create geometry object to be used in covariance calculation
            geometry = thecov.geometry.SurveyGeometry(
                                        randoms,
                                        nmesh=32,
                                        # kmax_window=0.02, # Nyquist wavelength of window FFTs
                                        boxpad=1.4, # multiplies the box size inferred from catalog
                                        alpha=alpha, # N_galaxies / N_randoms
                                        # kmodes_sampled=2000, # max N samples used in integ
                                    )
            print('done')
            print('Computing Gaussian Covariance...')

            gaussian = thecov.covariance.GaussianCovariance(geometry)
            gaussian.set_kbins(kmin, kmax, dk)

            # Load input power spectra (P0, P2, P4) for the Gaussian covariance

            gaussian.set_galaxy_pk_multipole(P0, 0, has_shotnoise=False)
            gaussian.set_galaxy_pk_multipole(P2, 2)
            # gaussian.set_galaxy_pk_multipole(P4, 4)

            gaussian.compute_covariance()
            gaussian.symmetrize()
            print('done')

            # fn = save_path + f'cov_gaussian_{self.tr_nm}_z{zrange[0]}-{zrange[1]}_DR{self.DR}'
            gaussian.savetxt(save_path.format('.txt'))
            print('saved gaussian covariance: ', save_path)
            covariance = gaussian
            
        if component == 't0':
            print('Creating Geometry object...')
            # Create geometry object to be used in covariance calculation
            geometry = thecov.geometry.SurveyGeometry(
                                        randoms,
                                        nmesh=32,
                                        # kmax_window=0.02, # Nyquist wavelength of window FFTs
                                        boxpad=1.4, # multiplies the box size inferred from catalog
                                        alpha=alpha, # N_galaxies / N_randoms
                                        # kmodes_sampled=2000, # max N samples used in integ
                                    )
            print('done')

            print('Computing trispectrum contribution...')
            t0 = thecov.covariance.RegularTrispectrumCovariance(geometry)
            t0.set_kbins(kmin, kmax, dk)

            plin = cosmo.get_fourier()

            t0.set_linear_matter_pk(np.vectorize(lambda k: plin.pk_kz(k, zeff)))  
            # assuming local lagrangian approximation if not given
            t0.set_params(fgrowth=cosmo.growth_rate(zeff), b1=bias[0], b2=bias[1], g2=bias[2])
            t0.compute_covariance()
            print('done')
            # fn = save_path + f'cov_t0_{self.tr_nm}_z{zrange[0]}-{zrange[1]}_DR{self.DR}'
            t0.savetxt(save_path.format('.txt'))
            print('saved t0 covariance: ', save_path)

            covariance = t0

        if component == 'SSC':
            print('Creating Geometry object...')
            geometry = thecov.geometry.SurveyGeometry(
                            randoms,
                            nmesh=64,
                            # kmax_window=0.02, # Nyquist wavelength of window FFTs
                            boxpad=4, # multiplies the box size inferred from catalog
                            alpha=alpha, # N_galaxies / N_randoms
                            # kmodes_sampled=2000, # max N samples used in integ
                           )
            print('done')
            print('Computing SSC contribution...')
            ssc = thecov.covariance.SuperSampleCovariance(geometry)
            ssc.set_kbins(kmin, kmax, dk)
            pk = partial(cosmo.get_fourier().pk_kz, z=zeff)

            def central_diff(f, x, dx=1e-4):
                return (f(x + dx) - f(x - dx)) / (2 * dx)

            dPk = central_diff(pk, kmid, dx=1e-4)

            ssc.set_linear_matter_pk(pk, dPk=dPk)

            ssc.set_params(fgrowth=cosmo.growth_rate(zeff), b1=bias[0], b2=bias[1], g2=bias[2])

            # ssc.set_shotnoise(shotnoises[0])
            ssc.compute_covariance()

            # fn = save_path + f'cov_ssc_{self.tr_nm}_z{zrange[0]}-{zrange[1]}_DR{self.DR}'
            ssc.savetxt(save_path.format('.txt'))

            print('saved ssc covariance: ', save_path)
            covariance = ssc
        return covariance
            
    def compute_PkXi_covariance(self,kvec,rvec,Veff,save_path):
        from scipy.special import hyp2f1

        from classy import Class
        from multipoles import LegendreMultipoleExtractor
        from covariances_pk_xi import PkXiCovariance
        # We are going to need the linear matter power spectrum.
        # We will use CLASS to compute it.

        def D_of_a(a,OmegaM=1):
            # From Stephen Chen
            return a * hyp2f1(1./3,1,11./6,-a**3/OmegaM*(1-OmegaM)) / hyp2f1(1./3,1,11./6,-1/OmegaM*(1-OmegaM))

        def f_of_a(a, OmegaM=1):
            # From Stephen Chen
            Da = D_of_a(a,OmegaM=OmegaM)
            ret = Da/a - a*(6*a**2 * (1 - OmegaM) * hyp2f1(4./3, 2, 17./6, -a**3 *  (1 - OmegaM)/OmegaM))/(11*OmegaM)/hyp2f1(1./3,1,11./6,-1/OmegaM*(1-OmegaM))
            return a * ret / Da

        z = 0.8
        pkparams = {
            'output': 'mPk',
            'P_k_max_h/Mpc': 20.,
            'z_pk': '0.0,10',
            'A_s': 2.1e-9,
            'n_s': 0.9649,
            'h': 70 / 100.,
            'N_ur': 2.033,
            'N_ncdm': 1,
            'm_ncdm': 0.06,
            'tau_reio': 0.0568,
            'omega_b': 0.022,
            'omega_cdm': 0.12}

        OmegaM = (pkparams['omega_cdm'] + pkparams['omega_b'] + 0.0106 *pkparams['m_ncdm']) / pkparams['h'] ** 2

        pkclass = Class()
        pkclass.set(pkparams)
        pkclass.compute()

        # Calculate growth rate
        fnu = pkclass.Omega_nu / pkclass.Omega_m()
        f = f_of_a(1 / (1. + z), OmegaM=OmegaM) * (1 - 0.6 * fnu)

        # Calculate and renormalize power spectrum
        plin = np.array([pkclass.pk_cb(k * pkparams['h'], z) * pkparams['h'] ** 3 for k in kvec])

        def D_of_mu(mu):
            return 1 + (1+f)**2 * mu**2

        def W(k, R_width=15):
            # A fourier-space gaussian smoothing kernel with a real-space width of 15 Mpc/h
            return np.exp(-0.5 * (k * R_width)**2)

        def sigma_squared(mu):
            sigmasq_integrand = plin* W(konh)**2 / (6*np.pi**2)  
            return D_of_mu(mu) * np.trapz(sigmasq_integrand, konh)

        def P_cross_of_mu(mu, b=1.95):
            return (b+f*mu**2)**2 * plin * np.exp(-konh**2 * sigma_squared(mu) / 2)
        
        # Compute multipoles of P_cross using Gauss-Legendre quadrature
        extractor = LegendreMultipoleExtractor(ngauss=8)
        Pcross_ells = extractor.project(P_cross_of_mu, ells=(0, 2, 4))

        P0, P2, P4 = Pcross_ells[0], Pcross_ells[2], Pcross_ells[4]

        # Use the Pk–Xi covariance class to compute the disconnected block
        V = Veff  # (Mpc/h)^3
        poles = [0, 2,4]

        # Initialize the covariance builder
        cov_builder = PkXiCovariance(k=kvec, r=rvec, V=V, poles_Pk=poles, poles_Xi=poles,
                                    ngauss_leg_product=48, r_avg_nsamp=100)
        # Provide the P_cross(k,mu) 
        cov_builder.set_Pcross_from_callable(P_cross_of_mu, ells=[0,2,4], ngauss_eval=8)
        # Note that we could also have provided the multipoles directly as
        #cov_builder.set_Pcross_from_dict(Pcross_ells)

        cov_PkXi = cov_builder.compute_covariance()  # shape (3, 3, nk, nr)
        print('cov_PkXi shape:', cov_PkXi.shape)

        return cov_PkXi
            

        
