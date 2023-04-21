module load nvhpc/21.7-nompi
module load cuda/11.1
export LD_LIBRARY_PATH=/appl/SuiteSparse/5.1.2-sl73/lib:/appl/OpenBLAS/0.2.20/XeonGold6226R/gcc-6.4.0/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
export OMP_TARGET_OFFLOAD=MANDATORY
