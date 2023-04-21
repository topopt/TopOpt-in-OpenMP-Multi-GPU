#pragma once

#include <math.h>
#include <stdalign.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "cholmod.h"

#define __force_inline __attribute__((always_inline))
#define __force_unroll __attribute__((optimize("unroll-loops")))
#define __alignBound 64

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) > (b) ? (b) : (a))

// define stencil sizes at compile time
#define STENCIL_SIZE_X 1
#ifndef STENCIL_SIZE_Y
#define STENCIL_SIZE_Y 8
#endif
#define STENCIL_SIZE_Z 1

#ifndef MFLEVELS
#define number_of_matrix_free_levels 2
#else
#define number_of_matrix_free_levels MFLEVELS
#endif

// Set 1 for debugging mode
#define DEBUGGING_MODE 0

#define TASKING_THREADS 16

typedef double MTYPE;
typedef double STYPE;
typedef float CTYPE;

typedef float DTYPE; // design type, for element denseties, gradients and such.

struct FixedDofs {
  uint_fast32_t n;
  uint_fast32_t *idx;
};

struct gridContext {
  double E0;
  double Emin;
  double nu;
  double elementSizeX;
  double elementSizeY;
  double elementSizeZ;
  uint_fast32_t nelx;
  uint_fast32_t nely;
  uint_fast32_t nelz;
  uint_fast32_t wrapx;
  uint_fast32_t wrapy;
  uint_fast32_t wrapz;
  double penal;
  MTYPE **precomputedKE;

  struct FixedDofs *fixedDofs;
};

struct CSRMatrix {
  uint64_t nnz;
  int32_t nrows;

  int *rowOffsets;
  int *colIndex;
  MTYPE *vals;
};

struct CoarseSolverData {
  cholmod_common *cholmodCommon;
  cholmod_sparse *sparseMatrix;
  cholmod_factor *factoredMatrix;

  cholmod_dense *rhs;
  cholmod_dense *solution;

  cholmod_dense *Y_workspace;
  cholmod_dense *E_workspace;
};

// Gonjugate gradient data structure
// Note that the conjugate gradient method
// also uses the coarsest level pointers from 
// mg_ptrs. Namely mg.r[0] and mg.z[0]
typedef struct cg_ptrs {
  //CTYPE *r;
  CTYPE *p;
  CTYPE *q;
  //CTYPE *z;
} cg_ptrs;

// Multigrid data structure
typedef struct mg_ptrs {
  MTYPE *invD;
  CTYPE *d;
  CTYPE *r;
  CTYPE *z;
} mg_ptrs;

typedef struct mg_dims {
  uint_fast32_t * nelx;
  uint_fast32_t * nely;
  uint_fast32_t * nelz;
  uint_fast32_t * wrapx;
  uint_fast32_t * wrapy;
  uint_fast32_t * wrapz;
  uint_fast32_t * offset;
  uint_fast32_t * length;
  uint_fast32_t * ndof;
  uint_fast32_t * pKEsize;
} mg_dims;

typedef struct solver_ptrs {

  // cg data
  cg_ptrs cg;

  // jacobi + mg data
  mg_ptrs * mg;

  // explicitly assembled matrices
  struct CSRMatrix *coarseMatrices;

  #ifdef USE_CHOLMOD
  struct CoarseSolverData bottomSolver;
  #endif
} solver_ptrs;

typedef struct gpuNode {
	int id;
	struct gpuNode * prev;
	struct gpuNode * next;
    int design_size;
    struct gridContext gc;
    solver_ptrs solver;
    mg_dims dims;
    DTYPE * x;
    DTYPE * xtmp;
    DTYPE * xnew;
    DTYPE * dv;
    DTYPE * dc;
    CTYPE * F;
    STYPE * U;
    CTYPE * CTYPE_buffer;
    MTYPE * MTYPE_buffer;
    double * rhs;
    double * sol;
} gpuNode;

typedef struct gpuGrid{
  struct gridContext * gc;
  solver_ptrs * solver;
	int num_targets;
	gpuNode * targets;
} gpuGrid;
