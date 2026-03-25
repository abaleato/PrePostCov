DATA DIRECTORY README
=====================

This directory contains data vectors and covariance matrices for a joint
power spectrum (Pk) + correlation function (Xi) BAO analysis of DESI tracers.

Tracers: BGS, ELG, LRG1, LRG2, LRG3, QSO
Multipoles: monopole (ell=0) and quadrupole (ell=2), denoted "ell02" or "ell0"

------------------------------------------------------------------------------
LRG2_errors_jackknife.txt
  Jackknife error estimates for the LRG2 data vector (square matrix).

------------------------------------------------------------------------------
pkxi_dat/  -- Data vectors
  Pk_<tracer>_ell02.txt
    Power spectrum multipoles. Columns: k [h/Mpc], P0(k), P2(k).

  Pk_LRG2_ell02_EZmock.txt
    LRG2 Pk estimated from EZmock simulations.

  Pk_LRG2_ell02_rot.txt
    LRG2 Pk after window matrix rotation.

  Xipost_<tracer>_ell02.txt / Xipost_<tracer>_ell0.txt
    Post-reconstruction correlation function multipoles.
    Columns: s [Mpc/h], xi0(s), [xi2(s)].

  Xipost_<tracer>_*_smin20.txt
    Same as above but with a minimum scale cut applied (s_min = 20 Mpc/h).

------------------------------------------------------------------------------
pkxi_cov/  -- Covariance matrices for the joint Pk+Xi data vector
  PkXi_joint_<tracer>_ell02_cov.txt
    Baseline joint covariance matrix for each tracer.

  PkXi_joint_LRG2_ell02_cov_analyticPkXi.txt
    Covariance with the Pk-Xi cross-block replaced by an analytic estimate.

  PkXi_joint_LRG2_ell02_cov_noPkXi.txt
    Covariance with the Pk-Xi cross-block set to zero.

  PkXi_joint_LRG2_ell02_cov_noPkXi_DR1fid.txt
    Same as above but computed at the DR1 fiducial cosmology.

  PkXi_joint_LRG2_ell02_cov_nothetacut.txt
    Covariance without the small-angle (theta) cut applied.

  PkXi_joint_LRG2_ell02_cov_rotated.txt
    Baseline covariance after window matrix rotation.

  PkXi_joint_LRG2_ell02_cov_rotated_hack.txt
    Rotated covariance using a shortcut/approximate rotation scheme.

  TheCov/
    Pre-computed Gaussian covariance matrices from the TheCov code,
    one file per tracer and redshift bin:
      cov_gaussian_pre_<tracer>_GCcomb_<zmin>_<zmax>.txt

------------------------------------------------------------------------------
analytics/  -- Analytic covariance variants (LRG2 only)
  PkXi_joint_LRG2_ell02_cov.txt
    Fully analytic covariance matrix for LRG2.

  PkXi_joint_LRG2_ell02_cov_analyticPkXi_rot.txt
    Analytic covariance with window matrix rotation applied.

  PkXi_joint_LRG2_ell02_cov_analyticPkXi_V2p9_0p6SN.txt
    Analytic covariance with shot noise scaled by 0.6.

  PkXi_joint_LRG2_ell02_cov_rotated_hack.txt
    Rotated analytic covariance using the approximate rotation scheme.

  PkXi_joint_LRG2_ell02_cov_analytic_cross_rotated.txt
    The (160, 100) Pk–Xi cross-block of the analytic covariance after window
    matrix rotation. Pre-computed so that PkXi_correlation_plots.ipynb can be
    run with rotate_window=True without requiring the lsstypes package.
