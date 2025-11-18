#!/bin/bash -l
#SBATCH -J wmat_meas
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -o output_logs/comp_wmat%j.out
#SBATCH -e output_logs/comp_wmat%j.err
#SBATCH -q debug
# SBATCH -q regular
#SBATCH -C cpu
#SBATCH -A desi

date
#

module load python
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# export PYTHONPATH=${PYTHONPATH}:/pscratch/sd/m/mmaus/DESI_velocileptors_redef
# export PYTHONPATH=${PYTHONPATH}:/pscratch/sd/m/mmaus/DESI_velocileptors_redef/rsd_likelihood
# export PYTHONPATH=${PYTHONPATH}:/global/homes/m/mmaus/Python/velocileptors
# # export PYTHONPATH=${PYTHONPATH}:/global/homes/m/mmaus/Python/velocileptors
# # export PYTHONPATH=${PYTHONPATH}:/global/cscratch1/sd/mmaus/new_template/Cobaya_template/emulator/template/emu
# export OMP_NUM_THREADS=8

echo "Setup done.  Starting to run code ..."

# srun -n 64 python do_everything.py --task measure_pk --tracer BGS --zrange 0.1 0.4 --zeff 0.30 --DR 2 --version v2 --region GCcomb --outpath ./data/pk_measured/Pk_BGS_z0.1-0.4_DR2_GCcomb
# srun -n 64 python do_everything.py --task measure_pk --tracer LRG --zrange 0.4 0.6 --zeff 0.51 --DR 2 --version v2 --region GCcomb --outpath ./data/pk_measured/Pk_LRG_z0.4-0.6_DR2_GCcomb
# srun -n 64 python do_everything.py --task measure_pk --tracer LRG --zrange 0.6 0.8 --zeff 0.71 --DR 2 --version v2 --region GCcomb --outpath ./data/pk_measured/Pk_LRG_z0.6-0.8_DR2_GCcomb
# srun -n 64 python do_everything.py --task measure_pk --tracer LRG --zrange 0.8 1.1 --zeff 0.92 --DR 2 --version v2 --region GCcomb --outpath ./data/pk_measured/Pk_LRG_z0.8-1.1_DR2_GCcomb
# srun -n 128 python do_everything.py --task measure_pk --tracer ELG --zrange 1.1 1.6 --zeff 1.32 --DR 2 --version v2 --region GCcomb --outpath ./data/pk_measured/Pk_ELG_z1.1-1.6_DR2_GCcomb
srun -n 64 python do_everything.py --task measure_pk --tracer QSO --zrange 0.8 2.1 --zeff 1.49 --DR 2 --version v2 --region GCcomb --outpath ./data/pk_measured/Pk_QSO_z0.8-2.1_DR2_GCcomb

# srun -n 64 python do_everything.py --task compute_window --pk_poles_path ./data/pk_measured/Pk_BGS_z0.1-0.4_DR2_GCcomb.npy --tracer BGS --zrange 0.1 0.4 --zeff 0.30 --DR 2 --version v2 --region GCcomb --outpath ./data/pk_window/wmat_BGS_z0.1-0.4_DR2_GCcomb
# srun -n 64 python do_everything.py --task compute_window --pk_poles_path ./data/pk_measured/Pk_LRG_z0.4-0.6_DR2_GCcomb.npy --tracer LRG --zrange 0.4 0.6 --zeff 0.51 --DR 2 --version v2 --region GCcomb --outpath ./data/pk_window/wmat_LRG_z0.4-0.6_DR2_GCcomb
# srun -n 64 python do_everything.py --task compute_window --pk_poles_path ./data/pk_measured/Pk_LRG_z0.6-0.8_DR2_GCcomb.npy --tracer LRG --zrange 0.6 0.8 --zeff 0.71 --DR 2 --version v2 --region GCcomb --outpath ./data/pk_window/wmat_LRG_z0.6-0.8_DR2_GCcomb
# srun -n 128 python do_everything.py --task compute_window --pk_poles_path ./data/pk_measured/Pk_LRG_z0.8-1.1_DR2_GCcomb.npy --tracer LRG --zrange 0.8 1.1 --zeff 0.92 --DR 2 --version v2 --region GCcomb --outpath ./data/pk_window/wmat_LRG_z0.8-1.1_DR2_GCcomb
# srun -n 64 python do_everything.py --task compute_window --pk_poles_path ./data/pk_measured/Pk_ELG_z1.1-1.6_DR2_GCcomb.npy --tracer ELG --zrange 1.1 1.6 --zeff 1.32 --DR 2 --version v2 --region GCcomb --outpath ./data/pk_window/wmat_ELG_z1.1-1.6_DR2_GCcomb
# srun -n 64 python do_everything.py --task compute_window --pk_poles_path ./data/pk_measured/Pk_QSO_z0.8-2.1_DR2_GCcomb.npy --tracer QSO --zrange 0.8 2.1 --zeff 1.49 --DR 2 --version v2 --region GCcomb --outpath ./data/pk_window/wmat_QSO_z0.8-2.1_DR2_GCcomb


# srun -n 64 python y1_test.py
# srun -n 64 python measure_pk_pypower.py
# srun -n 64 python compute_windows.py 3