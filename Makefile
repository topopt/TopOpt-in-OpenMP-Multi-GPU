# User must provide a desired compiler
ifndef COMPILER
$(error "Please set COMPILER=<nvc,gcc>")
else
ifeq ("$(wildcard config/$(COMPILER).mkf)","")
$(error "config/$(COMPILER).mkf does not exist")
endif
include config/$(COMPILER).mkf
endif

ifdef GPU
ifeq ("$(wildcard config/gpu/$(GPU).mkf)","")
$(error "config/gpu/$(GPU).mkf does not exist")
endif
include config/gpu/$(GPU).mkf
endif

# Definitions
ifndef USE_CHOLMOD
USE_CHOLMOD=1
endif

ifdef STENCIL_SIZE_Y
DEFS += -DSTENCIL_SIZE_Y=$(STENCIL_SIZE_Y)
endif

ifdef SIMD
ifeq ("$(SIMD)","1")
DEFS += -DSIMD=$(SIMD)
endif
endif

ifdef USE_CHOLMOD
ifeq ("$(USE_CHOLMOD)","1")
DEFS += -DUSE_CHOLMOD=$(USE_CHOLMOD)
endif
endif

ifndef MFLEVELS
MFLEVELS=2
endif

DEFS+= -DMFLEVELS=$(MFLEVELS)

# Standard settings which are identical for all compilers
ifdef LUMI
#CPPFLAGS = -I/pfs/lustrep2/users/rydahlan/suitesparse/SuiteSparse-5.1.2/include -I/pfs/lustrep2/users/rydahlan/blas/BLAS-3.11.0/include
LDFLAGS += -L/pfs/lustrep2/users/rydahlan/suitesparse/SuiteSparse-5.1.2/lib 
#LDFLAGS += -L/pfs/lustrep2/users/rydahlan/blas/BLAS-3.11.0/lib
#LDFLAGS += -L/pfs/lustrep2/users/rydahlan/suitesparse/SuiteSparse-5.1.2/lib -L/pfs/lustrep2/users/rydahlan/cblas/CBLAS/lib -lcblas
#LDFLAGS += -L/pfs/lustrep2/users/rydahlan/blas/BLAS-3.11.0 -lblas
CPPFLAGS += -I/pfs/lustrep2/users/rydahlan/suitesparse/SuiteSparse-5.1.2/include
#CPPFLAGS += -I/pfs/lustrep2/users/rydahlan/cblas/CBLAS/include
#CPPFLAGS += -I/pfs/lustrep2/users/rydahlan/blas/BLAS-3.11.0
CPPFLAGS += -I/opt/cray/pe/libsci/22.08.1.1/CRAY/9.0/x86_64/include
LDFLAGS += -L/opt/cray/pe/libsci/22.08.1.1/CRAY/9.0/x86_64/lib -lsci_cray
else
CPPFLAGS += -I/appl/SuiteSparse/5.1.2-sl73/include -I/appl/OpenBLAS/0.2.20/XeonGold6226R/gcc-6.4.0/include
LDFLAGS += -L/appl/SuiteSparse/5.1.2-sl73/lib -L/appl/OpenBLAS/0.2.20/XeonGold6226R/gcc-6.4.0/lib
endif

LDLIBS = -lm -lcholmod -lsuitesparseconfig -lcxsparse
ifndef LUMI
LDLIBS += -lopenblas
else
LDLIBS += -lgfortran
endif

OBJFILES = stencil_methods.o stencil_assembly.o stencil_solvers.o stencil_grid_utility.o stencil_optimization.o local_matrix.o eightColor_methods.o gpu_grid.o gpu_methods.o
OBJ = $(addprefix lib/,$(OBJFILES))

all: top3d


top3d: top3d.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

top3d.o: top3d.c include/definitions.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ -c $< $(DEFS)

lib/%.o: src/%.c include/definitions.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ -c $< $(DEFS)

.PHONY: clean
clean:
	-rm -f top3d *.o lib/*.o

.PHONY: info
info:
	$(info "COMPILER=$(COMPILER)")
	$(info "CC=$(CC)")
	$(info "CFLAGS=$(CFLAGS)")
	$(info "LDFLAGS=$(LDFLAGS)")
	$(info "LDLIBS=$(LDLIBS)")
	$(info "DEFS=$(DEFS)")
