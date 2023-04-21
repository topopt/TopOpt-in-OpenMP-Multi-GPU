#!/bin/bash
set -o nounset -o errexit

BASEDIR=`pwd`/gcc-offload
NEWLIBDIR=$BASEDIR/newlib
LLVMDIR=$BASEDIR/llvm
AMDDIR=$BASEDIR/install/amdgcn-amdhsa
INSTALLDIR=$BASEDIR/install
AMDBUILDDIR=$BASEDIR/build-amdgcn-amdhsa-gcc
GCCDIR=$BASEDIR/gcc

mkdir -p $NEWLIBDIR
mkdir -p $LLVMDIR
mkdir -p $AMDDIR
mkdir -p $INSTALLDIR

cd $LLVMDIR
git clone --depth 1 https://github.com/llvm/llvm-project.git
cmake -DLLVM_TARGETS_TO_BUILD='X86;AMDGPU' \
	-DLLVM_ENABLE_PROJECTS=lld \
	-DCMAKE_BUILD_TYPE=Release \
	$LLVMDIR/llvm-project/llvm
make -j32
cd $BASEDIR

# don't install anything yet

mkdir -p $AMDDIR/bin
cp -a $LLVMDIR/bin/llvm-ar $AMDDIR/bin/ar
cp -a $LLVMDIR/bin/llvm-ar $AMDDIR/bin/ranlib
cp -a $LLVMDIR/bin/llvm-mc $AMDDIR/bin/as
cp -a $LLVMDIR/bin/llvm-nm $AMDDIR/bin/nm
cp -a $LLVMDIR/bin/lld $AMDDIR/bin/ld

## Installing Newlib

cd $NEWLIBDIR
git clone git://sourceware.org/git/newlib-cygwin.git
cd $BASEDIR

## Cloning GCC sources
git clone --depth 1 https://github.com/gcc-mirror/gcc.git

## Guessing on target
cd $GCCDIR
contrib/download_prerequisites
TARG=$(./config.guess)

## Linking newlib into gcc
cd $GCCDIR
ln -s $NEWLIBDIR/newlib-cygwin/newlib newlib

## Installing target compiler
mkdir -p $AMDBUILDDIR
cd $AMDBUILDDIR
$GCCDIR/configure --target=amdgcn-amdhsa --enable-languages="c,c++,lto,fortran"	--disable-sjlj-exceptions --with-newlib	--enable-as-accelerator-for=$TARG --with-build-time-tools=$AMDDIR/bin --disable-libquadmath --prefix=$INSTALLDIR
make -j16
make install


# Build host GCC
HOSTBUILDDIR=$BASEDIR/build-host-gcc
mkdir -p $HOSTBUILDDIR
cd $HOSTBUILDDIR
echo `pwd`
echo $TARG
$GCCDIR/configure --target=$TARG --build=$TARG --host=$TARG --disable-multilib --enable-offload-targets=amdgcn-amdhsa=$AMDDIR --disable-bootstrap --enable-languages="c,c++,fortran,lto" --prefix=$INSTALLDIR
make -j 32
make install
cd .. 
