export LUMI=1
module load rocm
export PATH=/users/rydahlan/LLVM/16.0/bin:$PATH
export LD_LIBRARY_PATH=/users/rydahlan/LLVM/16.0/runtimes/runtimes-bins/openmp/runtime/src:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=8
export ROCR_VISIBLE_DEVICES=0
export OMP_TARGET_OFFLOAD=MANDATORY
echo "PATH is"
echo $PATH
echo "LD_LIBRARY_PATH is"
echo $LD_LIBRARY_PATH

