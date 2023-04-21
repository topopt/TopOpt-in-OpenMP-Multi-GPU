export LUMI=1
module load rocm
export LD_LIBRARY_PATH=/project/project_465000434/GCC/13.0/gcc-offload/install/lib64:$LD_LIBRARY_PATH
export PATH=/project/project_465000434/GCC/13.0/gcc-offload/install/bin:$PATH
export ROCR_VISIBLE_DEVICES=0
export OMP_TARGET_OFFLOAD=MANDATORY
echo $PATH
echo $LD_LIBRARY_PATH
