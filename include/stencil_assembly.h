#pragma once

#include "definitions.h"

#ifdef USE_CHOLMOD
#include "cholmod.h" //<suitesparse/cholmod.h>
#endif


void allocateSubspaceMatrix(const struct gridContext gc, const int l,
                            struct CSRMatrix *M);

void freeSubspaceMatrix(struct CSRMatrix *M);

void assembleSubspaceMatrix(const struct gridContext gc, const int l,
                            const DTYPE *x, struct CSRMatrix M, MTYPE *tmp);

void applyStateOperatorSubspaceMatrix(const struct gridContext gc, const int l,
                                      const struct CSRMatrix M, const CTYPE *in,
                                      CTYPE *out);
                                      
#ifdef USE_CHOLMOD
void initializeCholmod(const struct gridContext gc, const int l,
                       struct CoarseSolverData *ssolverData,
                       const struct CSRMatrix M);

void finishCholmod(const struct gridContext gc, const int l,
                   struct CoarseSolverData *solverData,
                   const struct CSRMatrix M);

void factorizeSubspaceMatrix(const struct gridContext gc, const int l,
                             struct CoarseSolverData solverData,
                             const struct CSRMatrix M);

void solveSubspaceMatrix(const struct gridContext gc, const int l,
                         struct CoarseSolverData solverData, const CTYPE *in,
                         CTYPE *out);
#endif
