#pragma once

#include "definitions.h"

#include "stencil_assembly.h"

void smoothDampedJacobi_halo(gpuGrid * gpu_grid,const struct gridContext gc, const DTYPE *x,
                             const uint_fast32_t nswp, const CTYPE omega,
                             mg_ptrs mg);

void smoothDampedJacobiSubspaceMatrix_halo(
    const struct gridContext gc, const struct CSRMatrix M, const int l,
    const uint_fast32_t nswp, const CTYPE omega,mg_ptrs * mg);

void smoothDampedJacobiSubspace_halo(const struct gridContext gc,
                                     const DTYPE *x, const int l,
                                     const uint_fast32_t nswp,
                                     const CTYPE omega,mg_ptrs * mg);

void solveStateMG_halo(gpuGrid * gpu_grid,const struct gridContext gc, DTYPE *x, const int nswp,
                       const int nl, const CTYPE tol,
                       solver_ptrs *data, int *finalIter,
                       float *finalRes, CTYPE *b, STYPE *u, const int verbose, double * cpu_time);

void allocateSolverData(const struct gridContext gc, const int nl,
                        solver_ptrs *data);

void freeSolverData(solver_ptrs *data, const int nl);

// compute the inner product of two vectors
// temperature: cold-medium, called 2 x number of cg iterations
__force_inline inline CTYPE innerProduct(CTYPE *a, CTYPE *b,
                                         uint_fast32_t size) {
  CTYPE val = 0.0;
  // long for loop, as ndof is typically 300.000 or more, but also trivially
// parallel.
  //#pragma omp target teams distribute parallel for reduction(+ : val) map(tofrom: val) //firstprivate(size)
  #pragma omp parallel for reduction(+:val)
  for (uint_fast32_t i = 0; i < size; i++)
    val += a[i] * b[i];
  return val;
}
