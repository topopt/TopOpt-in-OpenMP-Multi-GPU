#include "../include/eightColor_methods.h"

#include <cblas.h>

// Apply the global matrix vector product out = K * in
// temperature: very hot, called ~25 x (number of mg levels [1-5]) x
// (number of cg iterations [125-2500]) = [125-12500]  times pr design
// iteration
void applyStateOperatorSubspace_eightColor(const struct gridContext gc,
                                           const DTYPE *x, const int l,
                                           CTYPE *in, CTYPE *out) {

  const int ncell = pow(2, l);
  const int32_t nelxc = gc.nelx / ncell;
  const int32_t nelyc = gc.nely / ncell;
  const int32_t nelzc = gc.nelz / ncell;
  const int32_t nyc = nelyc + 1;
  const int32_t nzc = nelzc + 1;

  const uint32_t ndof = 3 * nyc * nzc * (nelxc + 1);
  for (uint32_t i = 0; i < ndof; i++)
    out[i] = 0.0;

// loop over elements, depends on the which level you are on. For the finest
// (level 0) nelx*nely*nelz = 100.000 or more, but for every level you go down
// the number of iterations reduce by a factor of 8. i.e. level 2 will only have
// ~1000. This specific loop accounts for ~90% runtime
#pragma omp parallel for collapse(3)
  for (int32_t i = 0; i < nelxc; i += 2)
    for (int32_t k = 0; k < nelzc; k += 2)
      for (int32_t j = 0; j < nelyc; j += 2) {

        uint_fast32_t edof[24];
        MTYPE u_local[24];
        MTYPE out_local[24];

        getEdof(edof, i, j, k, nyc, nzc);

        for (int ii = 0; ii < 24; ii++)
          u_local[ii] = (MTYPE)in[edof[ii]];

        // loop over interior subcells, depends on the level. total iterations
        // = (level+1)^3, i.e. only one iteration for the finest level 0, but
        // inreasing cubicly. Note that the total amount of inner iterations
        // nested by the inner and outer sets of loops is always constant  (
        // across all levels, that means that as the level number grows, the
        // parallelization available is shifted from the outer loops to the
        // inner loops.
        for (int ii = 0; ii < ncell; ii++)
          for (int kk = 0; kk < ncell; kk++)
            for (int jj = 0; jj < ncell; jj++) {
              const int ifine = i * ncell + ii;
              const int jfine = j * ncell + jj;
              const int kfine = k * ncell + kk;

              const int cellidx = ncell * ncell * ii + ncell * kk + jj;

              const uint_fast32_t elementIndex =
                  ifine * gc.nely * gc.nelz + kfine * gc.nely + jfine;
              const MTYPE elementScale =
                  gc.Emin + pow(x[elementIndex], gc.penal) * (gc.E0 - gc.Emin);

              cblas_dgemv(CblasRowMajor, CblasNoTrans, 24, 24, elementScale,
                          gc.precomputedKE[l] + 24 * 24 * cellidx, 24, u_local,
                          1, 0.0, out_local, 1);

              for (int iii = 0; iii < 24; iii++)
                out[edof[iii]] += (CTYPE)out_local[iii];
            }
      }

// same as above
#pragma omp parallel for collapse(3)
  for (int32_t i = 0; i < nelxc; i += 2)
    for (int32_t k = 0; k < nelzc; k += 2)
      for (int32_t j = 1; j < nelyc; j += 2) {

        uint_fast32_t edof[24];
        MTYPE u_local[24];
        MTYPE out_local[24];

        getEdof(edof, i, j, k, nyc, nzc);

        for (int ii = 0; ii < 24; ii++)
          u_local[ii] = in[edof[ii]];

        for (int ii = 0; ii < ncell; ii++)
          for (int kk = 0; kk < ncell; kk++)
            for (int jj = 0; jj < ncell; jj++) {
              const int ifine = i * ncell + ii;
              const int jfine = j * ncell + jj;
              const int kfine = k * ncell + kk;

              const int cellidx = ncell * ncell * ii + ncell * kk + jj;

              const uint_fast32_t elementIndex =
                  ifine * gc.nely * gc.nelz + kfine * gc.nely + jfine;
              const MTYPE elementScale =
                  gc.Emin + pow(x[elementIndex], gc.penal) * (gc.E0 - gc.Emin);

              cblas_dgemv(CblasRowMajor, CblasNoTrans, 24, 24, elementScale,
                          gc.precomputedKE[l] + 24 * 24 * cellidx, 24, u_local,
                          1, 0.0, out_local, 1);

              for (int iii = 0; iii < 24; iii++)
                out[edof[iii]] += out_local[iii];
            }
      }

// same as above
#pragma omp parallel for collapse(3)
  for (int32_t i = 0; i < nelxc; i += 2)
    for (int32_t k = 1; k < nelzc; k += 2)
      for (int32_t j = 0; j < nelyc; j += 2) {

        uint_fast32_t edof[24];
        MTYPE u_local[24];
        MTYPE out_local[24];

        getEdof(edof, i, j, k, nyc, nzc);

        for (int ii = 0; ii < 24; ii++)
          u_local[ii] = in[edof[ii]];

        for (int ii = 0; ii < ncell; ii++)
          for (int kk = 0; kk < ncell; kk++)
            for (int jj = 0; jj < ncell; jj++) {
              const int ifine = i * ncell + ii;
              const int jfine = j * ncell + jj;
              const int kfine = k * ncell + kk;

              const int cellidx = ncell * ncell * ii + ncell * kk + jj;

              const uint_fast32_t elementIndex =
                  ifine * gc.nely * gc.nelz + kfine * gc.nely + jfine;
              const MTYPE elementScale =
                  gc.Emin + pow(x[elementIndex], gc.penal) * (gc.E0 - gc.Emin);

              cblas_dgemv(CblasRowMajor, CblasNoTrans, 24, 24, elementScale,
                          gc.precomputedKE[l] + 24 * 24 * cellidx, 24, u_local,
                          1, 0.0, out_local, 1);

              for (int iii = 0; iii < 24; iii++)
                out[edof[iii]] += out_local[iii];
            }
      }

// same as above
#pragma omp parallel for collapse(3)
  for (int32_t i = 0; i < nelxc; i += 2)
    for (int32_t k = 1; k < nelzc; k += 2)
      for (int32_t j = 1; j < nelyc; j += 2) {

        uint_fast32_t edof[24];
        MTYPE u_local[24];
        MTYPE out_local[24];

        getEdof(edof, i, j, k, nyc, nzc);

        for (int ii = 0; ii < 24; ii++)
          u_local[ii] = in[edof[ii]];

        for (int ii = 0; ii < ncell; ii++)
          for (int kk = 0; kk < ncell; kk++)
            for (int jj = 0; jj < ncell; jj++) {
              const int ifine = i * ncell + ii;
              const int jfine = j * ncell + jj;
              const int kfine = k * ncell + kk;

              const int cellidx = ncell * ncell * ii + ncell * kk + jj;

              const uint_fast32_t elementIndex =
                  ifine * gc.nely * gc.nelz + kfine * gc.nely + jfine;
              const MTYPE elementScale =
                  gc.Emin + pow(x[elementIndex], gc.penal) * (gc.E0 - gc.Emin);

              cblas_dgemv(CblasRowMajor, CblasNoTrans, 24, 24, elementScale,
                          gc.precomputedKE[l] + 24 * 24 * cellidx, 24, u_local,
                          1, 0.0, out_local, 1);

              for (int iii = 0; iii < 24; iii++)
                out[edof[iii]] += out_local[iii];
            }
      }

// same as above
#pragma omp parallel for collapse(3)
  for (int32_t i = 1; i < nelxc; i += 2)
    for (int32_t k = 0; k < nelzc; k += 2)
      for (int32_t j = 0; j < nelyc; j += 2) {

        uint_fast32_t edof[24];
        MTYPE u_local[24];
        MTYPE out_local[24];

        getEdof(edof, i, j, k, nyc, nzc);

        for (int ii = 0; ii < 24; ii++)
          u_local[ii] = in[edof[ii]];

        for (int ii = 0; ii < ncell; ii++)
          for (int kk = 0; kk < ncell; kk++)
            for (int jj = 0; jj < ncell; jj++) {
              const int ifine = i * ncell + ii;
              const int jfine = j * ncell + jj;
              const int kfine = k * ncell + kk;

              const int cellidx = ncell * ncell * ii + ncell * kk + jj;

              const uint_fast32_t elementIndex =
                  ifine * gc.nely * gc.nelz + kfine * gc.nely + jfine;
              const MTYPE elementScale =
                  gc.Emin + pow(x[elementIndex], gc.penal) * (gc.E0 - gc.Emin);

              cblas_dgemv(CblasRowMajor, CblasNoTrans, 24, 24, elementScale,
                          gc.precomputedKE[l] + 24 * 24 * cellidx, 24, u_local,
                          1, 0.0, out_local, 1);

              for (int iii = 0; iii < 24; iii++)
                out[edof[iii]] += out_local[iii];
            }
      }

// same as above
#pragma omp parallel for collapse(3)
  for (int32_t i = 1; i < nelxc; i += 2)
    for (int32_t k = 0; k < nelzc; k += 2)
      for (int32_t j = 1; j < nelyc; j += 2) {

        uint_fast32_t edof[24];
        MTYPE u_local[24];
        MTYPE out_local[24];

        getEdof(edof, i, j, k, nyc, nzc);

        for (int ii = 0; ii < 24; ii++)
          u_local[ii] = in[edof[ii]];

        for (int ii = 0; ii < ncell; ii++)
          for (int kk = 0; kk < ncell; kk++)
            for (int jj = 0; jj < ncell; jj++) {
              const int ifine = i * ncell + ii;
              const int jfine = j * ncell + jj;
              const int kfine = k * ncell + kk;

              const int cellidx = ncell * ncell * ii + ncell * kk + jj;

              const uint_fast32_t elementIndex =
                  ifine * gc.nely * gc.nelz + kfine * gc.nely + jfine;
              const MTYPE elementScale =
                  gc.Emin + pow(x[elementIndex], gc.penal) * (gc.E0 - gc.Emin);

              cblas_dgemv(CblasRowMajor, CblasNoTrans, 24, 24, elementScale,
                          gc.precomputedKE[l] + 24 * 24 * cellidx, 24, u_local,
                          1, 0.0, out_local, 1);

              for (int iii = 0; iii < 24; iii++)
                out[edof[iii]] += out_local[iii];
            }
      }

// same as above
#pragma omp parallel for collapse(3)
  for (int32_t i = 1; i < nelxc; i += 2)
    for (int32_t k = 1; k < nelzc; k += 2)
      for (int32_t j = 0; j < nelyc; j += 2) {

        uint_fast32_t edof[24];
        MTYPE u_local[24];
        MTYPE out_local[24];

        getEdof(edof, i, j, k, nyc, nzc);

        for (int ii = 0; ii < 24; ii++)
          u_local[ii] = in[edof[ii]];

        for (int ii = 0; ii < ncell; ii++)
          for (int kk = 0; kk < ncell; kk++)
            for (int jj = 0; jj < ncell; jj++) {
              const int ifine = i * ncell + ii;
              const int jfine = j * ncell + jj;
              const int kfine = k * ncell + kk;

              const int cellidx = ncell * ncell * ii + ncell * kk + jj;

              const uint_fast32_t elementIndex =
                  ifine * gc.nely * gc.nelz + kfine * gc.nely + jfine;
              const MTYPE elementScale =
                  gc.Emin + pow(x[elementIndex], gc.penal) * (gc.E0 - gc.Emin);

              cblas_dgemv(CblasRowMajor, CblasNoTrans, 24, 24, elementScale,
                          gc.precomputedKE[l] + 24 * 24 * cellidx, 24, u_local,
                          1, 0.0, out_local, 1);

              for (int iii = 0; iii < 24; iii++)
                out[edof[iii]] += out_local[iii];
            }
      }

// same as above
#pragma omp parallel for collapse(3)
  for (int32_t i = 1; i < nelxc; i += 2)
    for (int32_t k = 1; k < nelzc; k += 2)
      for (int32_t j = 1; j < nelyc; j += 2) {

        uint_fast32_t edof[24];
        MTYPE u_local[24];
        MTYPE out_local[24];

        getEdof(edof, i, j, k, nyc, nzc);

        for (int ii = 0; ii < 24; ii++)
          u_local[ii] = in[edof[ii]];

        for (int ii = 0; ii < ncell; ii++)
          for (int kk = 0; kk < ncell; kk++)
            for (int jj = 0; jj < ncell; jj++) {
              const int ifine = i * ncell + ii;
              const int jfine = j * ncell + jj;
              const int kfine = k * ncell + kk;

              const int cellidx = ncell * ncell * ii + ncell * kk + jj;

              const uint_fast32_t elementIndex =
                  ifine * gc.nely * gc.nelz + kfine * gc.nely + jfine;
              const MTYPE elementScale =
                  gc.Emin + pow(x[elementIndex], gc.penal) * (gc.E0 - gc.Emin);

              cblas_dgemv(CblasRowMajor, CblasNoTrans, 24, 24, elementScale,
                          gc.precomputedKE[l] + 24 * 24 * cellidx, 24, u_local,
                          1, 0.0, out_local, 1);

              for (int iii = 0; iii < 24; iii++)
                out[edof[iii]] += out_local[iii];
            }
      }

  // apply boundaryConditions
  struct FixedDofs fd = getFixedDof(nelxc, nelyc, nelzc);
  for (int i = 0; i < fd.n; i++)
    out[fd.idx[i]] = in[fd.idx[i]];
  free(fd.idx);
  // free(KEarray);
}

struct FixedDofs getFixedDof(int nelx, int nely, int nelz) {
  struct FixedDofs fd;
  fd.n = 3 * (nely + 1) * (nelz + 1);
  fd.idx = malloc(sizeof(uint_fast32_t) * fd.n);
  for (uint_fast32_t i = 0; i < fd.n; i++)
    fd.idx[i] = i;
  return fd;
}

// projects a field to a finer grid ucoarse -> ufine
// temperature: medium, called (number of mg levels [1-5]) x (number of cg
// iterations [5-100]) = [5-500]  times pr design iteration
void projectToFinerGrid(const struct gridContext gc, /*in*/ const int l, /*in*/
                        const CTYPE *ucoarse,                            /*in*/
                        CTYPE *ufine /*out*/) {

  const int nelxf = gc.nelx / pow(2, l);
  const int nelyf = gc.nely / pow(2, l);
  const int nelzf = gc.nelz / pow(2, l);

  const int nelxc = gc.nelx / pow(2, l + 1);
  const int nelyc = gc.nely / pow(2, l + 1);
  const int nelzc = gc.nelz / pow(2, l + 1);

  const int nxc = nelxc + 1;
  const int nyc = nelyc + 1;
  const int nzc = nelzc + 1;

  const int nxf = nelxf + 1;
  const int nyf = nelyf + 1;
  const int nzf = nelzf + 1;

  MTYPE vals[4] = {1, 0.5, 0.25, 0.125};

  const uint32_t ndoff = 3 * nxf * nyf * nzf;

#pragma omp parallel for
  for (uint32_t i = 0; i < ndoff; i++)
    ufine[i] = 0.0;

  // loop over nodes, usually very large with nx*ny*nz = 100.000 or more
  for (int32_t nx = 0; nx < nxc; nx++)
    for (int32_t nz = 0; nz < nzc; nz++)
      for (int32_t ny = 0; ny < nyc; ny++) {

        const int coarseIndex = nx * nyc * nzc + nz * nyc + ny;

        // Node indices on fine grid
        const int nx1 = nx * 2;
        const int ny1 = ny * 2;
        const int nz1 = nz * 2;

        const int xmin = MAX(nx1 - 1, 0);
        const int ymin = MAX(ny1 - 1, 0);
        const int zmin = MAX(nz1 - 1, 0);

        const int xmax = MIN(nx1 + 2, nxf);
        const int ymax = MIN(ny1 + 2, nyf);
        const int zmax = MIN(nz1 + 2, nzf);

        // the three loops herein are over a constant neighbourhood. typically
        // 4x4x4 or something like that
        for (int32_t i = xmin; i < xmax; i++)
          for (int32_t k = zmin; k < zmax; k++)
            for (int32_t j = ymin; j < ymax; j++) {

              const uint32_t fineIndex = i * nyf * nzf + k * nyf + j;

              const int ind = (nx1 - i) * (nx1 - i) + (ny1 - j) * (ny1 - j) +
                              (nz1 - k) * (nz1 - k);

              ufine[3 * fineIndex + 0] +=
                  vals[ind] * ucoarse[3 * coarseIndex + 0];
              ufine[3 * fineIndex + 1] +=
                  vals[ind] * ucoarse[3 * coarseIndex + 1];
              ufine[3 * fineIndex + 2] +=
                  vals[ind] * ucoarse[3 * coarseIndex + 2];
            }
      }
}

// projects a field to a coarser grid ufine -> ucoarse
// temperature: medium, called (number of mg levels [1-5]) x (number of cg
// iterations [5-100]) = [5-500]  times pr design iteration
void projectToCoarserGrid(const struct gridContext gc,
                          /*in*/ const int l, /*in*/
                          const CTYPE *ufine, /*in*/
                          CTYPE *ucoarse /*out*/) {

  const int nelxf = gc.nelx / pow(2, l);
  const int nelyf = gc.nely / pow(2, l);
  const int nelzf = gc.nelz / pow(2, l);

  const int nelxc = gc.nelx / pow(2, l + 1);
  const int nelyc = gc.nely / pow(2, l + 1);
  const int nelzc = gc.nelz / pow(2, l + 1);

  const int nxc = nelxc + 1;
  const int nyc = nelyc + 1;
  const int nzc = nelzc + 1;

  const int nxf = nelxf + 1;
  const int nyf = nelyf + 1;
  const int nzf = nelzf + 1;

  MTYPE vals[4] = {1.0, 0.5, 0.25, 0.125};

  const uint32_t ndofc = 3 * nxc * nyc * nzc;

  // #pragma omp parallel for
  for (uint32_t i = 0; i < ndofc; i++)
    ucoarse[i] = 0.0;

  // loop over nodes, usually very large with nx*ny*nz = 100.000 or more
  // #pragma omp parallel for
  for (int32_t nx = 0; nx < nxc; nx++)
    for (int32_t nz = 0; nz < nzc; nz++)
      for (int32_t ny = 0; ny < nyc; ny++) {

        const int coarseIndex = nx * nyc * nzc + nz * nyc + ny;

        // Node indices on fine grid
        const int nx1 = nx * 2;
        const int ny1 = ny * 2;
        const int nz1 = nz * 2;

        const int xmin = MAX(nx1 - 1, 0);
        const int ymin = MAX(ny1 - 1, 0);
        const int zmin = MAX(nz1 - 1, 0);

        const int xmax = MIN(nx1 + 2, nxf);
        const int ymax = MIN(ny1 + 2, nyf);
        const int zmax = MIN(nz1 + 2, nzf);

        // the three loops herein are over a constant neighbourhood. typically
        // 4x4x4 or something like that
        for (int32_t i = xmin; i < xmax; i++)
          for (int32_t k = zmin; k < zmax; k++)
            for (int32_t j = ymin; j < ymax; j++) {

              const uint32_t fineIndex = i * nyf * nzf + k * nyf + j;

              const int ind = (nx1 - i) * (nx1 - i) + (ny1 - j) * (ny1 - j) +
                              (nz1 - k) * (nz1 - k);

              ucoarse[3 * coarseIndex + 0] +=
                  vals[ind] * ufine[3 * fineIndex + 0];
              ucoarse[3 * coarseIndex + 1] +=
                  vals[ind] * ufine[3 * fineIndex + 1];
              ucoarse[3 * coarseIndex + 2] +=
                  vals[ind] * ufine[3 * fineIndex + 2];
            }
      }
}

// generate the matrix diagonal for jacobi smoothing.
// temperature: low-medium, called number of levels for every design iteration.
void generateMatrixDiagonalSubspace(const struct gridContext gc, const DTYPE *x,
                                    const int l, MTYPE *diag) {

  const int ncell = pow(2, l);
  const int32_t nelxc = gc.nelx / ncell;
  const int32_t nelyc = gc.nely / ncell;
  const int32_t nelzc = gc.nelz / ncell;
  const int32_t nyc = nelyc + 1;
  const int32_t nzc = nelzc + 1;

  uint_fast32_t edof[24];

  const uint_fast32_t ndofc = 3 * nyc * nzc * (nelxc + 1);

#pragma omp parallel for
  for (unsigned int i = 0; i < ndofc; i++)
    diag[i] = 0.0;

  // see the comments in applyStateOperatorSubspace, as the three inner, and
  // three outer loops follow the exact same pattern.
  for (int32_t i = 0; i < nelxc; i++)
    for (int32_t k = 0; k < nelzc; k++)
      for (int32_t j = 0; j < nelyc; j++) {
        getEdof(edof, i, j, k, nyc, nzc);

        for (int ii = 0; ii < ncell; ii++)
          for (int kk = 0; kk < ncell; kk++)
            for (int jj = 0; jj < ncell; jj++) {
              const int ifine = i * ncell + ii;
              const int jfine = j * ncell + jj;
              const int kfine = k * ncell + kk;

              const int cellidx = ncell * ncell * ii + ncell * kk + jj;

              const uint_fast32_t elementIndex =
                  ifine * gc.nely * gc.nelz + kfine * gc.nely + jfine;
              const MTYPE elementScale =
                  gc.Emin + pow(x[elementIndex], gc.penal) * (gc.E0 - gc.Emin);

              for (int iii = 0; iii < 24; iii++)
                diag[edof[iii]] +=
                    elementScale *
                    gc.precomputedKE[l][24 * 24 * cellidx + iii * 24 + iii];
            }
      }

  // apply boundaryConditions
  struct FixedDofs fd = getFixedDof(nelxc, nelyc, nelzc);
  for (int i = 0; i < fd.n; i++)
    diag[fd.idx[i]] = 1.0;
  free(fd.idx);
  //  free(KEarray);
}

// jacobi smoothing/preconditioning
// temperature: hot, called 2x(number of levels)x(number of cg iterations) ~
// [20-1000] times every design iteration. Note that most compute time is spent
// in child function.
void smthdmpjacSubspace_eightColor(const struct gridContext gc, const DTYPE *x,
                                   const int l, const uint_fast32_t nswp,
                                   const CTYPE omega, const MTYPE *invD,
                                   CTYPE *u, const CTYPE *b, CTYPE *tmp) {

  const int ncell = pow(2, l);
  const uint_fast32_t nelxc = gc.nelx / ncell;
  const uint_fast32_t nelyc = gc.nely / ncell;
  const uint_fast32_t nelzc = gc.nelz / ncell;
  const uint_fast32_t ndof = 3 * (nelxc + 1) * (nelyc + 1) * (nelzc + 1);

  // usually nswp is between 1 and 5
  for (int s = 0; s < nswp; s++) {
    applyStateOperatorSubspace_eightColor(gc, x, l, u, tmp);

// long for loop, as ndof is typically 300.000 or more, but also trivially
// parallel.
#pragma omp parallel for
    for (int i = 0; i < ndof; i++)
      u[i] += omega * invD[i] * (b[i] - tmp[i]);
  }
}