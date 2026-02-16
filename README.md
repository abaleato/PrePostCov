# PrePostCov
Code to compute analytical covariances between multipoles of the pre-reconstruction power spectrum and the post-reconstruction two-point correlation function.

## $P_{\ell}(k)^{\rm{pre-recon}}$ - $\xi_{\ell}(r)^{\rm{post-recon}}$ cross-covariance
The main novelty presented in this repo is an analytic calculation of the cross-covariance between $P_{\ell}(k)^{\rm{pre-recon}}$ and $\xi_{\ell}(r)^{\rm{post-recon}}$. The required functionality lives in `covariances_pk_xi.py`. Running this requires only very basic dependencies like `numpy` and `scipy`.

### Basic usage

For the basic cross-covariance calculation, see the worked example in [preprost_covariance.ipynb](preprost_covariance.ipynb). In brief:

```python
from covariances_pk_xi import PkXiCovariance

# Initialize the covariance builder
cov_builder = PkXiCovariance(k=k_array, r=r_array, V=survey_volume, 
                             poles_Pk=[0, 2], poles_Xi=[0, 2])

# Provide the fiducial P_cross(k, mu) (see paper for details)
cov_builder.set_Pcross_from_callable(P_cross_of_mu, ells=[0, 2, 4])

# Compute the Pk-Xi block of the covariance matrix
cov_PkXi = cov_builder.compute_covariance()  # shape (n_poles_Pk, n_poles_Xi, nk, nr)
```

## Full $P_{\ell}(k)^{\rm{pre-recon}}$ - $\xi_{\ell}(r)^{\rm{post-recon}}$ covariance

Additionally, we include code to calculate the remaining blocks of the covariance matrix involving $P_{\ell}(k)^{\rm{pre-recon}}$ and $\xi_{\ell}(r)^{\rm{post-recon}}$. To do this, we use [TheCov](https://github.com/cosmodesi/thecov) for the $P_{\ell}(k)^{\rm{pre-recon}}$ auto piece, [RascalC](https://github.com/mikhailov/rascalc) for $\xi_{\ell}(r)^{\rm{post-recon}}$.

Overall steps are:
1. Compute P(k),Xi(s) from catalogs
2. Compute P(k) window matrices
3. Obtain fiducial spectra for covariances
(Can skip steps 1-3 if you already have fiducial spectra)
4. Compute gaussian, t0, and ssc pieces of P(k) covariance
5. Compute Xi(s) covariance
6. Compute PkxXi covariance and combine everyting

These are all subroutines in the `full_covariance()` class in [Covariances_full.py](Covariances_full.py). The front end script is [do_everything.py](do_everything.py) to which you add arguments related to the desired task:

1. **Measure P(k):**
   ```bash
   srun -n 64 python do_everything.py --task measure_pk \
       --tracer LRG --zrange 0.4 0.6 --zeff 0.51 --DR 2 --version v2 \
       --region GCcomb --outpath ./data/pk_measured/Pk_LRG_z0.4-0.6_DR2_GCcomb
   ```

2. **Compute window:**
   ```bash
   srun -n 64 python do_everything.py --task compute_window \
       --pk_poles_path ./data/pk_measured/Pk_LRG_z0.4-0.6_DR2_GCcomb.npy \
       --tracer LRG --zrange 0.4 0.6 --zeff 0.51 --DR 2 --version v2 \
       --region GCcomb --outpath ./data/pk_window/wmat_LRG_z0.4-0.6_DR2_GCcomb
   ```

## Citation

If you use this code in your research, please cite:

[**Maus, Baleato Lizancos, White, de Mattia & Chen (2026)**](https://arxiv.org/abs/2602.12343)

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE) (CC BY-NC 4.0). You are free to use, share, and adapt this code for non-commercial purposes with appropriate attribution.

