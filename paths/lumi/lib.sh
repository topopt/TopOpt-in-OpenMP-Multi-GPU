export LD_LIBRARY_PATH=/pfs/lustrep2/users/rydahlan/suitesparse/SuiteSparse-5.1.2/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/pfs/lustrep2/users/rydahlan/blas/BLAS-3.11.0:$LD_LIBRARY_PATH
module load rocm
module load cray-libsci
echo $LD_LIBRARY_PATH
echo $PATH
