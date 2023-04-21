#!/bin/sh 
#BSUB -q gpuv100
#BSUB -J top3d_volta
#BSUB -n 16
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=50GB]"
###BSUB -R "select[gpu80gb]"
#BSUB -W 0:40 -- 
#BSUB -o Output_%J.out 
#BSUB -e Error_%J.err
#BSUB -u s1745152student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N

print=0
verbose=0
iterations=1
threads=16
computecap=70
max_levels=6

PATH_=$PATH
LD_LIBRARY_PATH_=$LD_LIBRARY_PATH

source nvcpaths.sh

make -f Makefile_nvc clean

make -f Makefile_nvc USE_CHOLMOD=1 GPU_ARCH=cc$computecap

export OMP_NUM_THREADS=$threads
export OMP_TARGET_OFFLOAD=MANDATORY

export CUDA_VISIBLE_DEVICES=1
levels=4
for size in 64 128 256 288
do
    declare -i sizex=2*$size
    declare -i sizey=1*$size
    filename=./profilings/nvc_sm_$computecap.$size.txt
    echo "Running: nvprof ./top3d $sizex $sizey $sizey $verbose $print $iterations $levels"
	nvprof ./top3d $sizex $sizey $sizey $verbose $print $iterations $levels &> $filename
	declare -i levels=1+$levels
	declare -i levels=$(( $levels < $max_levels ? $levels : $max_levels ))
done

module unload nvhpc
module unload cuda

export PATH=$PATH_
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH_

source gccpaths.sh

make -f Makefile_gcc clean

make -f Makefile_gcc USE_CHOLMOD=1

export OMP_NUM_THREADS=$threads
export OMP_TARGET_OFFLOAD=MANDATORY

export CUDA_VISIBLE_DEVICES=1
levels=4
for size in 64 128 256 288
do
    declare -i sizex=2*$size
    declare -i sizey=1*$size
    filename=./profilings/gcc_sm_$computecap.$size.txt
    echo "Running: nvprof ./top3d $sizex $sizey $sizey $verbose $print $iterations $levels"
	nvprof ./top3d $sizex $sizey $sizey $verbose $print $iterations $levels &> $filename
	declare -i levels=1+$levels
	declare -i levels=$(( $levels < $max_levels ? $levels : $max_levels ))
done
