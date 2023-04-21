module load cuda/11.5
module load gcc/11.3.0-binutils-2.38
#LLVMPATH=/work3/s174515/LLVM20230106
LLVMPATH=/work3/s174515/LLVM20230221
export PATH=$LLVMPATH/bin:$PATH
export LD_LIBRARY_PATH=$LLVMPATH/runtimes/runtimes-bins/openmp/runtime/src:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/appl/SuiteSparse/5.1.2-sl73/lib:/appl/OpenBLAS/0.2.20/XeonGold6226R/gcc-6.4.0/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
export OMP_TARGET_OFFLOAD=MANDATORY
echo "PATH is"
echo $PATH
echo "LD_LIBRARY_PATH is"
echo $LD_LIBRARY_PATH

