# PrePostCov
Compute Pk(pre)-Xi(post) analytical covariances. Uses TheCov for the pre-recon Pk auto piece, RascalC for the post-recon Xi auto piece, and a gaussian disconnected approximation for the Pk x Xi pieces.

Overall steps are:
1. Compute P(k),Xi(s) from catalogs
2. Compute P(k) window matrices
3. Obtain fiducial spectra for covariances
(Can skip steps 1-3 if you already have fiducial spectra)
4. Compute gaussian, t0, and ssc pieces of P(k) covariance
5. Compute Xi(s) covariance
6. Compute PkxXi covariance and combine everyting

These are all subroutines in the full_covariance() class in Covariances_full.py. The front end script is do_everything.py to which you add arguments related to the desired task:

1. srun -n 64 python do_everything.py --task measure_pk --tracer LRG --zrange 0.4 0.6 --zeff 0.51 --DR 2 --version v2 --region GCcomb --outpath ./data/pk_measured/Pk_LRG_z0.4-0.6_DR2_GCcomb
2. srun -n 64 python do_everything.py --task compute_window --pk_poles_path ./data/pk_measured/Pk_LRG_z0.4-0.6_DR2_GCcomb.npy --tracer LRG --zrange 0.4 0.6 --zeff 0.51 --DR 2 --version v2 --region GCcomb --outpath ./data/pk_window/wmat_LRG_z0.4-0.6_DR2_GCcomb


