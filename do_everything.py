from __future__ import division, print_function
import sys
import numpy as np
# sys.path.append('./desi-y1-kp/')
# from desi_y1_files import get_data_file_manager, is_baseline_2pt_setup
import matplotlib.pyplot as plt
import mpytools as mpy
import pandas as pd
import pickle
import argparse
import json

from copy import deepcopy

from mockfactory import Catalog
from cosmoprimo.fiducial import DESI
from pypower import CatalogFFTPower, CatalogSmoothWindow,setup_logging, mpi
from pathlib import Path

from astropy.table import Table, hstack
import fitsio
import healpy as hp

from Covariances_full import full_covariance

setup_logging()  # for logging messages
cosmo = DESI()  # fiducial cosmology


import sys, os, glob, time, warnings, gc
import numpy as np
from astropy.table import Table, vstack, hstack
import fitsio

import yaml

parser = argparse.ArgumentParser()

parser.add_argument('--tracer', type=str,   required=True)
parser.add_argument('--zrange', type=float, nargs=2, required=True,
                    help='Redshift range: ZMIN ZMAX')
parser.add_argument('--outpath', type=str,   required=True)
parser.add_argument('--DR', type=int,   required=True)
parser.add_argument('--version', type=str,   required=True)
parser.add_argument('--region', type=str,   required=False, choices=['NGC', 'SGC', 'GCcomb'],default='GCcomb')

parser.add_argument('--task', type=str,   required=True, choices=['measure_pk','compute_window','compute_pk_cov','compute_xi_cov','compute_pkxi_cov'])
parser.add_argument('--pkcov_component', required=False, type=str, choices=['gaussian','t0','ssc'])
parser.add_argument('--pk_poles_path', required=False, type=str)
parser.add_argument('--pk_fid_path', required=False, type=str)
parser.add_argument('--zeff',type=tuple,   required=True)
# parser.add_argument('--wbox', required=False, type=int)

args = parser.parse_args()

if args.DR == 1:
    catalog_dir = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/{args.version}/'
    if args.tracer == 'BGS':
        tr_nm = 'BGS_BRIGHT-21.5'
    elif args.tracer == 'ELG':
        tr_nm = 'ELG_LOPnotqso'
    else:
        tr_nm = args.tracer
if args.DR ==2:
    catalog_dir = f'/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/{args.version}/nonKP/'
    if args.tracer == 'BGS':
        tr_nm = 'BGS_BRIGHT-21.35'
    elif args.tracer == 'ELG':
        tr_nm = 'ELGnotqso'
    else:
        tr_nm = args.tracer

if args.region == 'NGC': region = ['NGC']
if args.region == 'SGC': region = ['SGC']
if args.region == 'GCcomb': region = ['NGC','SGC']

data_fnms = [catalog_dir + f'{tr_nm}_{reg}_clustering.dat.fits' for reg in region]
rand_fnms = [[catalog_dir + f'{tr_nm}_{reg}_{i}_clustering.ran.fits' for i in range(18)] for reg in region]
rand_fnms = np.array(rand_fnms).ravel()

if args.task == 'measure_pk':
    if args.tracer == 'ELG': 
        pypower_opts = (6,9000)
    elif args.tracer == 'QSO':
        pypower_opts = (6,10000)
    else:
        pypower_opts = (6,None)
    compute_class = full_covariance(tr_nm,args.DR,data_fnms,rand_fnms,load_cats = ['data','randoms'])
    
    pk_pypower = compute_class.measure_pk_pypower(zrange=args.zrange,weight_nms = ['WEIGHT','WEIGHT_FKP'],\
                                                  save_path = args.outpath, options = pypower_opts)

if args.task == 'compute_window':
    compute_class = full_covariance(tr_nm,args.DR,data_fnms,rand_fnms,load_cats = ['randoms'])
    
    pk_pypower = CatalogFFTPower.load(args.pk_poles_path)
    pk_poles = pk_pypower.poles
    window = compute_class.measure_windows(zrange=args.zrange,weight_nms = ['WEIGHT','WEIGHT_FKP'],pk_poles=pk_poles,save_path = args.outpath)

if args.task == 'compute_pk_cov':
    compute_class = full_covariance(tr_nm,args.DR,data_fnms,rand_fnms,load_cats = ['data','randoms'])
    
    with open(args.pk_fid_path, 'rb') as handle:
        pk_fid = json.load(handle)

    covariance = compute_class.compute_pk_covariance(component=args.pkcov_component,zrange=args.zrange,zeff=pk_fid['zeff'],\
                                                     pk_fid=np.array(pk_fid['spectra']),save_path= args.outpath,bias = pk_fid['bias'],krange = (0,0.4,0.005))

