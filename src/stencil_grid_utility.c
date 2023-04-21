#include "../include/stencil_grid_utility.h"

#include "../include/local_matrix.h"

void compute_num_cells(const struct gridContext *gc,const unsigned int l,uint_fast32_t * nelx,uint_fast32_t * nely,uint_fast32_t * nelz){
  const int ncell = pow(2, l);
  *nelx = (*gc).nelx / ncell;
  *nely = (*gc).nely / ncell;
  *nelz = (*gc).nelz / ncell;
}

void compute_padding(const struct gridContext *gc,const unsigned int l, uint_fast32_t * paddingx,uint_fast32_t * paddingy, uint_fast32_t * paddingz){
  uint_fast32_t nelx,nely,nelz;
  compute_num_cells(gc,l,&nelx,&nely,&nelz);
  *paddingx = (STENCIL_SIZE_X - ((nelx + 1) % STENCIL_SIZE_X)) % STENCIL_SIZE_X;
  *paddingy = (STENCIL_SIZE_Y - ((nely + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
  *paddingz = (STENCIL_SIZE_Z - ((nelz + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;
}

void compute_wrapping(const struct gridContext *gc,const unsigned int l,uint_fast32_t * wrapx,uint_fast32_t * wrapy,uint_fast32_t * wrapz){
  uint_fast32_t nelx,nely,nelz;
  compute_num_cells(gc,l,&nelx,&nely,&nelz);
  uint_fast32_t paddingx,paddingy,paddingz;
  compute_padding(gc,l,&paddingx,&paddingy,&paddingz);
  *wrapx = nelx + paddingx + 3;
  *wrapy = nely + paddingy + 3;
  *wrapz = nelz + paddingz + 3;
}

uint_fast32_t compute_ndof(const struct gridContext *gc,const unsigned int l){
  uint_fast32_t wrapx,wrapy,wrapz;
  compute_wrapping(gc,l,&wrapx,&wrapy,&wrapz);
  return 3 * wrapx * wrapy * wrapz;
}

void setFixedDof_halo(struct gridContext *gc, const int l) {
  uint_fast32_t nelxc,nelyc,nelzc;
  compute_num_cells(gc,l,&nelxc,&nelyc,&nelzc);

  uint_fast32_t wrapxc,wrapyc,wrapzc;
  compute_wrapping(gc,l,&wrapxc,&wrapyc,&wrapzc);

  const int nyc = (nelyc + 1);
  const int nzc = (nelzc + 1);

  (*gc).fixedDofs[l].n = 3 * nyc * nzc;
  (*gc).fixedDofs[l].idx = malloc(sizeof(uint_fast32_t) * (*gc).fixedDofs[l].n);
  int offset = 0;
  for (uint_fast32_t k = 1; k < (nzc + 1); k++)
    for (uint_fast32_t j = 1; j < (nyc + 1); j++) {
      (*gc).fixedDofs[l].idx[offset + 0] =
          3 * (wrapyc * wrapzc + wrapyc * k + j) + 0;
      (*gc).fixedDofs[l].idx[offset + 1] =
          3 * (wrapyc * wrapzc + wrapyc * k + j) + 1;
      (*gc).fixedDofs[l].idx[offset + 2] =
          3 * (wrapyc * wrapzc + wrapyc * k + j) + 2;
      offset += 3;
    }

  //const uint_fast32_t n = (*gc).fixedDofs[l].n;
  //const uint_fast32_t *pidx = (*gc).fixedDofs[l].idx;

  //#pragma omp target enter data map(to : pidx[:n])
}

void setupGC(struct gridContext *gc, const int nl,const int allocate) {
  compute_wrapping(gc,0,&(gc->wrapx),&(gc->wrapy),&(gc->wrapz));

  if (allocate){
    (*gc).precomputedKE = malloc(sizeof(MTYPE *) * nl);
    (*gc).fixedDofs = malloc(sizeof(struct FixedDofs) * nl);

    for (int l = 0; l < nl; l++) {
      const int ncell = pow(2, l);
      const int pKESize = 24 * 24 * ncell * ncell * ncell;
      (*gc).precomputedKE[l] = malloc(sizeof(MTYPE) * pKESize);
      getKEsubspace((*gc).precomputedKE[l], (*gc).nu, l);

      setFixedDof_halo(gc, l);

      //MTYPE *pKE = (*gc).precomputedKE[l];

      //#pragma omp target enter data map(to : pKE[:pKESize])
    }
  }
}

void freeGC(struct gridContext *gc, const int nl) {

  for (int l = 0; l < nl; l++) {
    free((*gc).precomputedKE[l]);
    free((*gc).fixedDofs[l].idx);
  }

  free((*gc).precomputedKE);
  free((*gc).fixedDofs);
}

void allocateZeroPaddedStateField(const struct gridContext gc, const int l,
                                  CTYPE **v) {
  const int ndof = compute_ndof(&gc,l);

  (*v) = malloc(sizeof(CTYPE) * ndof);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < ndof; i++)
    (*v)[i] = 0.0;
}

void allocateZeroPaddedStateField_MTYPE(const struct gridContext gc,
                                        const int l, MTYPE **v) {
  const int ndof = compute_ndof(&gc,l);

  (*v) = malloc(sizeof(MTYPE) * ndof);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < ndof; i++)
    (*v)[i] = 0.0;
}

void allocateZeroPaddedStateField_STYPE(const struct gridContext gc,
                                        const int l, STYPE **v) {
  const int ndof = compute_ndof(&gc,l);

  (*v) = malloc(sizeof(STYPE) * ndof);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < ndof; i++)
    (*v)[i] = 0.0;
}
