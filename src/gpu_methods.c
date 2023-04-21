#include "../include/gpu_methods.h"
/**
 * Input: 
 *      node->solver.mg[l].z
 *      node->x
 * Output:
 *      node->solver.mg[l].d 
 **/

#include <cblas.h>

__force_inline inline void
loadStencilInput_gpu(const uint_fast32_t wrapy, const uint_fast32_t wrapz, const int i_center,
                 const int j_center, const int k_center,
                 const int nodeOffset[3], const CTYPE *in,
                 MTYPE buffer_x[STENCIL_SIZE_Y], MTYPE buffer_y[STENCIL_SIZE_Y],
                 MTYPE buffer_z[STENCIL_SIZE_Y]) {

  const int i_sender = i_center + nodeOffset[0];
  const int j_sender = j_center + nodeOffset[1];
  const int k_sender = k_center + nodeOffset[2];

#pragma omp simd safelen(STENCIL_SIZE_Y) simdlen(STENCIL_SIZE_Y)               \
    aligned(buffer_x, buffer_y, buffer_z                                       \
            : __alignBound)
  for (int jj = 0; jj < STENCIL_SIZE_Y; jj++) {

    const uint_fast32_t sendingNodeIndex =
        (i_sender)*wrapy * wrapz + (k_sender)*wrapy + (j_sender + jj);

    const int startSend = 3 * sendingNodeIndex;

    buffer_x[jj] = in[startSend + 0];
    buffer_y[jj] = in[startSend + 1];
    buffer_z[jj] = in[startSend + 2];
  }
}

__force_inline inline void applyStateStencilSpoke_finegrid_gpu(
    const uint_fast32_t nelx, const uint_fast32_t nely, const uint_fast32_t nelz,
    const uint_fast32_t wrapy,const uint_fast32_t wrapz, const double Emin, const double E0,
    const MTYPE *precomputedKE, const int i_center,
    const int j_center, const int k_center, const int nodeOffset[3],
    const int elementOffset[3], const uint32_t ny, const uint32_t nz, const DTYPE *x,
    MTYPE inBuffer_x[STENCIL_SIZE_Y], MTYPE inBuffer_y[STENCIL_SIZE_Y],
    MTYPE inBuffer_z[STENCIL_SIZE_Y], MTYPE outBuffer_x[STENCIL_SIZE_Y],
    MTYPE outBuffer_y[STENCIL_SIZE_Y], MTYPE outBuffer_z[STENCIL_SIZE_Y],
    const uint_fast32_t prev_is_null,const uint_fast32_t next_is_null) {

  // compute sending and recieving local node number, hopefully be evaluated at
  // compile-time

  const int recievingNodeOffset[3] = {-elementOffset[0], -elementOffset[1],
                                      -elementOffset[2]};
  const int nodeOffsetFromElement[3] = {nodeOffset[0] - elementOffset[0],
                                        nodeOffset[1] - elementOffset[1],
                                        nodeOffset[2] - elementOffset[2]};

  const int localRecievingNodeIdx = getLocalNodeIndex(recievingNodeOffset);
  const int localSendingNodeIdx = getLocalNodeIndex(nodeOffsetFromElement);

  // compute index for element
  const int i_element = i_center + elementOffset[0];
  const int j_element = j_center + elementOffset[1];
  const int k_element = k_center + elementOffset[2];

  MTYPE localBuf_x[STENCIL_SIZE_Y];
  MTYPE localBuf_y[STENCIL_SIZE_Y];
  MTYPE localBuf_z[STENCIL_SIZE_Y];

  const int startRecieve_local = 3 * localRecievingNodeIdx;
  const int startSend_local = 3 * localSendingNodeIdx;

  // loop over simd stencil size
// currently does not compile to simd instructions..
#pragma omp simd safelen(STENCIL_SIZE_Y) simdlen(STENCIL_SIZE_Y) aligned(      \
    inBuffer_x, inBuffer_y, inBuffer_z, outBuffer_x, outBuffer_y, outBuffer_z  \
    : __alignBound)
  for (int jj = 0; jj < STENCIL_SIZE_Y; jj++) {

    // local coordinates
    const uint_fast32_t elementIndex =
        (i_element) * (wrapy - 1) * (wrapz - 1) +
        (k_element) * (wrapy - 1) + (j_element + jj);
    MTYPE elementScale = Emin + x[elementIndex] * x[elementIndex] *
                                       x[elementIndex] * (E0 - Emin);

    // important, sets true zero to halo values. This is necessary for
    // correctness. Performance can be gained by removing the constant Emin, and
    // setting the minimum allowed density to a corresponding non-zero value.
    // But this is left for the future at the moment.
    if (((i_element == 0) && (prev_is_null == 1)) || ((i_element > nelx) && (next_is_null == 1)) || j_element + jj == 0 ||
        j_element + jj > nely || k_element == 0 || k_element > nelz)
      elementScale = 0.0;

    localBuf_x[jj] = 0.0;
    localBuf_y[jj] = 0.0;
    localBuf_z[jj] = 0.0;

    // add the spoke contribution
    localBuf_x[jj] +=
        precomputedKE[24 * (startRecieve_local + 0) + (startSend_local + 0)] *
        inBuffer_x[jj];
    localBuf_x[jj] +=
        precomputedKE[24 * (startRecieve_local + 0) + (startSend_local + 1)] *
        inBuffer_y[jj];
    localBuf_x[jj] +=
        precomputedKE[24 * (startRecieve_local + 0) + (startSend_local + 2)] *
        inBuffer_z[jj];

    localBuf_y[jj] +=
        precomputedKE[24 * (startRecieve_local + 1) + (startSend_local + 0)] *
        inBuffer_x[jj];
    localBuf_y[jj] +=
        precomputedKE[24 * (startRecieve_local + 1) + (startSend_local + 1)] *
        inBuffer_y[jj];
    localBuf_y[jj] +=
        precomputedKE[24 * (startRecieve_local + 1) + (startSend_local + 2)] *
        inBuffer_z[jj];

    localBuf_z[jj] +=
        precomputedKE[24 * (startRecieve_local + 2) + (startSend_local + 0)] *
        inBuffer_x[jj];
    localBuf_z[jj] +=
        precomputedKE[24 * (startRecieve_local + 2) + (startSend_local + 1)] *
        inBuffer_y[jj];
    localBuf_z[jj] +=
        precomputedKE[24 * (startRecieve_local + 2) + (startSend_local + 2)] *
        inBuffer_z[jj];

    outBuffer_x[jj] += elementScale * localBuf_x[jj];
    outBuffer_y[jj] += elementScale * localBuf_y[jj];
    outBuffer_z[jj] += elementScale * localBuf_z[jj];
  }
}

void applyStateOperator_stencil_kernel_1(gpuNode * node,const int l, const int xmin,const int xmax,const DTYPE *x,
                                const CTYPE *in, CTYPE *out) {

  const uint_fast32_t ny = node->dims.nely[l] + 1;
  const uint_fast32_t nz = node->dims.nelz[l] + 1;

  const uint_fast32_t nelx = node->dims.nelx[l];
  const uint_fast32_t nely = node->dims.nely[l];
  const uint_fast32_t nelz = node->dims.nelz[l];
  const uint_fast32_t wrapy = node->dims.wrapy[l];
  const uint_fast32_t wrapz = node->dims.wrapz[l];
  const double Emin = node->gc.Emin;
  const double E0 = node->gc.E0;

  const uint_fast32_t prev_is_null = (node->prev == NULL) ? 1 : 0;
  const uint_fast32_t next_is_null = (node->next == NULL) ? 1 : 0;

  // this is necessary for omp to recognize that gc.precomputedKE[0] is already
  // mapped
  const MTYPE *precomputedKE = node->gc.precomputedKE[l];

  // loop over elements, depends on the which level you are on. For the finest
  // (level 0) nelx*nely*nelz = 100.000 or more, but for every level you go down
  // the number of iterations reduce by a factor of 8. i.e. level 2 will only
  // have ~1000. This specific loop accounts for ~90% runtime
  //#pragma omp teams distribute parallel for collapse(3) schedule(static)

  //#pragma omp parallel for schedule(static) collapse(3) 
  #pragma omp target teams distribute parallel for schedule(static) collapse(3) device(node->id)
  for (int32_t i = xmin; i < xmax; i += STENCIL_SIZE_X) {
    for (int32_t k = 1; k < nz + 1; k += STENCIL_SIZE_Z) {
      for (int32_t j = 1; j < ny + 1; j += STENCIL_SIZE_Y) {

        alignas(__alignBound) MTYPE out_x[STENCIL_SIZE_Y];
        alignas(__alignBound) MTYPE out_y[STENCIL_SIZE_Y];
        alignas(__alignBound) MTYPE out_z[STENCIL_SIZE_Y];

        alignas(__alignBound) MTYPE in_x[STENCIL_SIZE_Y];
        alignas(__alignBound) MTYPE in_y[STENCIL_SIZE_Y];
        alignas(__alignBound) MTYPE in_z[STENCIL_SIZE_Y];

// zero the values about to be written in this
#pragma omp simd safelen(STENCIL_SIZE_Y) simdlen(STENCIL_SIZE_Y)               \
    aligned(out_x, out_y, out_z                                                \
            : __alignBound)
        for (int jj = 0; jj < STENCIL_SIZE_Y; jj++) {
          out_x[jj] = 0.0;
          out_y[jj] = 0.0;
          out_z[jj] = 0.0;
        }

        // center line, uses uses the same 15 doubles from in
        loadStencilInput_gpu(wrapy,wrapz, i, j, k, (const int[]){0, 0, 0}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 0},
                                        (const int[]){0, 0, 0}, ny, nz, x, in_x,
                                        in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 0},
                                        (const int[]){0, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 0},
                                        (const int[]){0, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 0},
                                        (const int[]){0, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 0},
                                        (const int[]){-1, 0, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 0},
                                        (const int[]){-1, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 0},
                                        (const int[]){-1, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 0},
                                        (const int[]){-1, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);

        loadStencilInput_gpu(wrapy,wrapz, i, j, k, (const int[]){0, 1, 0}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, 1, 0},
                                        (const int[]){0, 0, 0}, ny, nz, x, in_x,
                                        in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, 1, 0},
                                        (const int[]){0, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, 1, 0},
                                        (const int[]){-1, 0, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, 1, 0},
                                        (const int[]){-1, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);

        loadStencilInput_gpu(wrapy,wrapz, i, j, k, (const int[]){0, -1, 0}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, -1, 0},
                                        (const int[]){0, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, -1, 0},
                                        (const int[]){0, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, -1, 0},
                                        (const int[]){-1, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, -1, 0},
                                        (const int[]){-1, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);

        // side line, uses uses the same 15 doubles from in
        loadStencilInput_gpu(wrapy,wrapz, i, j, k, (const int[]){0, 0, 1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 1},
                                        (const int[]){0, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 1},
                                        (const int[]){0, 0, 0}, ny, nz, x, in_x,
                                        in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 1},
                                        (const int[]){-1, 0, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 1},
                                        (const int[]){-1, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);

        loadStencilInput_gpu(wrapy,wrapz, i, j, k, (const int[]){0, 1, 1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, 1, 1},
                                        (const int[]){-1, 0, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, 1, 1},
                                        (const int[]){0, 0, 0}, ny, nz, x, in_x,
                                        in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);

        loadStencilInput_gpu(wrapy,wrapz, i, j, k, (const int[]){0, -1, 1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, -1, 1},
                                        (const int[]){-1, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, -1, 1},
                                        (const int[]){0, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);

        // side line, uses uses the same 15 doubles from in
        loadStencilInput_gpu(wrapy,wrapz, i, j, k, (const int[]){0, 0, -1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, 0, -1},
                                        (const int[]){0, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, 0, -1},
                                        (const int[]){-1, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, 0, -1},
                                        (const int[]){-1, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, 0, -1},
                                        (const int[]){0, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);

        loadStencilInput_gpu(wrapy,wrapz, i, j, k, (const int[]){0, 1, -1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, 1, -1},
                                        (const int[]){0, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, 1, -1},
                                        (const int[]){-1, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);

        loadStencilInput_gpu(wrapy,wrapz, i, j, k, (const int[]){0, -1, -1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, -1, -1},
                                        (const int[]){0, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){0, -1, -1},
                                        (const int[]){-1, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);

        // side line, uses uses the same 15 doubles from in
        loadStencilInput_gpu(wrapy,wrapz, i, j, k, (const int[]){1, 0, 0}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){1, 0, 0},
                                        (const int[]){0, 0, 0}, ny, nz, x, in_x,
                                        in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){1, 0, 0},
                                        (const int[]){0, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){1, 0, 0},
                                        (const int[]){0, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){1, 0, 0},
                                        (const int[]){0, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);

        loadStencilInput_gpu(wrapy,wrapz, i, j, k, (const int[]){1, 1, 0}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){1, 1, 0},
                                        (const int[]){0, 0, 0}, ny, nz, x, in_x,
                                        in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){1, 1, 0},
                                        (const int[]){0, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);

        loadStencilInput_gpu(wrapy,wrapz, i, j, k, (const int[]){1, -1, 0}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){1, -1, 0},
                                        (const int[]){0, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){1, -1, 0},
                                        (const int[]){0, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);

        // side line, uses uses the same 15 doubles from in
        loadStencilInput_gpu(wrapy,wrapz, i, j, k, (const int[]){-1, 0, 0}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){-1, 0, 0},
                                        (const int[]){-1, 0, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){-1, 0, 0},
                                        (const int[]){-1, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){-1, 0, 0},
                                        (const int[]){-1, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){-1, 0, 0},
                                        (const int[]){-1, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);

        loadStencilInput_gpu(wrapy,wrapz, i, j, k, (const int[]){-1, -1, 0}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){-1, -1, 0},
                                        (const int[]){-1, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){-1, -1, 0},
                                        (const int[]){-1, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);

        loadStencilInput_gpu(wrapy,wrapz, i, j, k, (const int[]){-1, 1, 0}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){-1, 1, 0},
                                        (const int[]){-1, 0, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){-1, 1, 0},
                                        (const int[]){-1, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);

        // edge line, uses uses the same 15 doubles from in
        loadStencilInput_gpu(wrapy,wrapz, i, j, k, (const int[]){-1, 1, -1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){-1, 1, -1},
                                        (const int[]){-1, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);

        loadStencilInput_gpu(wrapy,wrapz, i, j, k, (const int[]){-1, 0, -1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){-1, 0, -1},
                                        (const int[]){-1, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){-1, 0, -1},
                                        (const int[]){-1, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);

        loadStencilInput_gpu(wrapy,wrapz, i, j, k, (const int[]){-1, -1, -1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){-1, -1, -1},
                                        (const int[]){-1, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);

        // edge line, uses uses the same 15 doubles from in
        loadStencilInput_gpu(wrapy,wrapz, i, j, k, (const int[]){1, 0, -1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){1, 0, -1},
                                        (const int[]){0, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){1, 0, -1},
                                        (const int[]){0, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);

        loadStencilInput_gpu(wrapy,wrapz, i, j, k, (const int[]){1, -1, -1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){1, -1, -1},
                                        (const int[]){0, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);

        loadStencilInput_gpu(wrapy,wrapz, i, j, k, (const int[]){1, 1, -1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){1, 1, -1},
                                        (const int[]){0, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);

        // edge line, uses uses the same 15 doubles from in
        loadStencilInput_gpu(wrapy,wrapz, i, j, k, (const int[]){1, 0, 1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){1, 0, 1},
                                        (const int[]){0, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){1, 0, 1},
                                        (const int[]){0, 0, 0}, ny, nz, x, in_x,
                                        in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);

        loadStencilInput_gpu(wrapy,wrapz, i, j, k, (const int[]){1, 1, 1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){1, 1, 1},
                                        (const int[]){0, 0, 0}, ny, nz, x, in_x,
                                        in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);

        loadStencilInput_gpu(wrapy,wrapz, i, j, k, (const int[]){1, -1, 1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){1, -1, 1},
                                        (const int[]){0, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);

        // edge line, uses uses the same 15 doubles from in
        loadStencilInput_gpu(wrapy,wrapz, i, j, k, (const int[]){-1, 0, 1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){-1, 0, 1},
                                        (const int[]){-1, 0, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){-1, 0, 1},
                                        (const int[]){-1, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);

        loadStencilInput_gpu(wrapy,wrapz, i, j, k, (const int[]){-1, -1, 1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){-1, -1, 1},
                                        (const int[]){-1, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);

        loadStencilInput_gpu(wrapy,wrapz, i, j, k, (const int[]){-1, 1, 1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid_gpu(nelx,nely,nelz,wrapy,wrapz,Emin,E0, precomputedKE, i, j, k,
                                        (const int[]){-1, 1, 1},
                                        (const int[]){-1, 0, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z,prev_is_null,next_is_null);

#pragma omp simd safelen(STENCIL_SIZE_Y) simdlen(STENCIL_SIZE_Y)
        for (int jj = 0; jj < STENCIL_SIZE_Y; jj++) {
          const uint_fast32_t offset =
              3 * (i * wrapy * wrapz + k * wrapy + j + jj);
              const
          // The mask makes sure that the padded part of the domain is set to zero
          CTYPE mask = (j+jj<ny + 1);
          out[offset + 0] = mask*out_x[jj];
          out[offset + 1] = mask*out_y[jj];
          out[offset + 2] = mask*out_z[jj];
        }
      }
    }
  }
}

// void zero_padding_kernel(gpuNode * node,const int l, CTYPE *out) {

//   const uint_fast32_t ny = node->dims.nely[l] + 1;

//   const uint_fast32_t wrapx = node->dims.wrapx[l];
//   const uint_fast32_t wrapy = node->dims.wrapy[l];
//   const uint_fast32_t wrapz = node->dims.wrapz[l];


// // zero out the extra padded nodes
//   #pragma omp target teams distribute parallel for collapse(3) device(node->id)
//   for (int32_t i = 0; i < wrapx; i++)
//     for (int32_t k = 0; k < wrapz; k++)
//       for (int32_t j = ny + 1; j < wrapy; j++) {

//         const uint_fast32_t offset =
//             3 * (i * wrapy * wrapz + k * wrapy + j);

//         out[offset + 0] = 0.0;
//         out[offset + 1] = 0.0;
//         out[offset + 2] = 0.0;
//       }
// }

void applyStateOperator_stencil_kernel_3(gpuNode * node,const int l,const DTYPE *x,
                                const CTYPE *in, CTYPE *out) {
  if (node->prev == NULL){
    const uint_fast32_t n = node->gc.fixedDofs[0].n;
    const uint_fast32_t *fidx = node->gc.fixedDofs[0].idx;

    // apply boundaryConditions
    #pragma omp target teams distribute parallel for device(node->id)
    for (int i = 0; i < n; i++) {
      out[fidx[i]] = in[fidx[i]];
    }
  }
}

void applyStateOperator_stencil_xz_d(gpuGrid * gpu_grid, const int l)
{
  #pragma omp parallel num_threads(gpu_grid->num_targets)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    int xmin = (node->prev == NULL) ? 1 : 2;
    int xmax = node->dims.nelx[l] + 2;
    if (node->prev != NULL){
      applyStateOperator_stencil_kernel_1(node,l,xmin,xmin+LENGTH_LEFT,node->x,node->solver.mg[l].z,node->solver.mg[l].d);
      xmin += LENGTH_LEFT;
    }
    if (node->next != NULL){
      applyStateOperator_stencil_kernel_1(node,l,xmax-LENGTH_RIGHT,xmax,node->x,node->solver.mg[l].z,node->solver.mg[l].d);
      xmax -= LENGTH_RIGHT;
    }

    #pragma omp barrier

    if (node->next != NULL){
      SEND_MG_NEXT(node,d,l)
    }
    if (node->prev != NULL){
      SEND_MG_PREV(node,d,l)
    }
    applyStateOperator_stencil_kernel_1(node,l,xmin,xmax,node->x,node->solver.mg[l].z,node->solver.mg[l].d);
//    zero_padding_kernel(node,l,node->solver.mg[l].d);
    if (node->prev == NULL){
      applyStateOperator_stencil_kernel_3(node,l,node->x,node->solver.mg[l].z,node->solver.mg[l].d);
    }
  }
}

void applyStateOperator_stencil_xz_r(gpuGrid * gpu_grid, const int l)
{
  #pragma omp parallel num_threads(gpu_grid->num_targets)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    int xmin = (node->prev == NULL) ? 1 : 2;
    int xmax = node->dims.nelx[l] + 2;
    if (node->prev != NULL){
      applyStateOperator_stencil_kernel_1(node,l,xmin,xmin+LENGTH_LEFT,node->x,node->solver.mg[l].z,node->solver.mg[l].r);
      xmin += LENGTH_LEFT;
    }
    if (node->next != NULL){
      applyStateOperator_stencil_kernel_1(node,l,xmax-LENGTH_RIGHT,xmax,node->x,node->solver.mg[l].z,node->solver.mg[l].r);
      xmax -= LENGTH_RIGHT;
    }

    #pragma omp barrier

    if (node->next != NULL){
      SEND_MG_NEXT(node,r,l)
    }
    if (node->prev != NULL){
      SEND_MG_PREV(node,r,l)
    }
    applyStateOperator_stencil_kernel_1(node,l,xmin,xmax,node->x,node->solver.mg[l].z,node->solver.mg[l].r);
//    zero_padding_kernel(node,l,node->solver.mg[l].r);
    if (node->prev == NULL){
      applyStateOperator_stencil_kernel_3(node,l,node->x,node->solver.mg[l].z,node->solver.mg[l].r);
    }
  }
}

void applyStateOperator_stencil_xp_q(gpuGrid * gpu_grid)
{
  #pragma omp parallel num_threads(gpu_grid->num_targets)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    int xmin = (node->prev == NULL) ? 1 : 2;
    int xmax = node->dims.nelx[0] + 2;
    if (node->prev != NULL){
      applyStateOperator_stencil_kernel_1(node,0,xmin,xmin+LENGTH_LEFT,node->x,node->solver.cg.p,node->solver.cg.q);
      xmin += LENGTH_LEFT;
    }
    if (node->next != NULL){
      applyStateOperator_stencil_kernel_1(node,0,xmax-LENGTH_RIGHT,xmax,node->x,node->solver.cg.p,node->solver.cg.q);
      xmax -= LENGTH_RIGHT;
    }

    #pragma omp barrier

    if (node->next != NULL){
      SEND_CG_NEXT(node,q)
    }
    if (node->prev != NULL){
      SEND_CG_PREV(node,q)
    }
    
    applyStateOperator_stencil_kernel_1(node,0,xmin,xmax,node->x,node->solver.cg.p,node->solver.cg.q);
//    zero_padding_kernel(node,0,node->solver.cg.q);
    if (node->prev == NULL){
      applyStateOperator_stencil_kernel_3(node,0,node->x,node->solver.cg.p,node->solver.cg.q);
    }
  }
}

void smoothDampedJacobi_halo_kernel(gpuNode * node, const int l, const CTYPE omega,const int xmin,const int xmax){
  const MTYPE *invD = node->solver.mg[l].invD;
  CTYPE *z = node->solver.mg[l].z;
  const CTYPE *r = node->solver.mg[l].r;
  CTYPE * d = node->solver.mg[l].d;
  const uint_fast32_t nely = node->dims.nely[l];
  const uint_fast32_t nelz = node->dims.nelz[l];
  const uint_fast32_t wrapy = node->dims.wrapy[l];
  const uint_fast32_t wrapz = node->dims.wrapz[l];

#ifdef SIMD
  // Looping over all non-halo lattices
  #pragma omp target teams distribute parallel for collapse(3) schedule(static) device(node->id)
    for (int i = xmin; i < xmax; i++)
      for (int k = 1; k < nelz + 2; k++)
        for (int j = 1; j < nely + 2; j += STENCIL_SIZE_Y) {
          const int nidx = i * wrapy * wrapz + wrapy * k + j;
          #pragma omp simd safelen(STENCIL_SIZE_Y) simdlen(STENCIL_SIZE_Y)
          for (int jj = 0; jj < STENCIL_SIZE_Y; jj++) {
            const uint32_t idx = 3 * (nidx + jj);
            CTYPE mask = (j+jj < nely + 2);
            z[idx] += mask*omega * invD[idx] * (r[idx] - d[idx]);
            z[idx+1] += mask*omega * invD[idx+1] * (r[idx+1] - d[idx+1]);
            z[idx+2] += mask*omega * invD[idx+2] * (r[idx+2] - d[idx+2]);
          }
        }
#else 
  // Looping over all non-halo lattices
  #pragma omp target teams distribute parallel for collapse(3) schedule(static) device(node->id)
    for (int i = xmin; i < xmax; i++)
      for (int k = 1; k < nelz + 2; k++)
        for (int j = 1; j < nely + 2; j++) {
          const int nidx = i * wrapy * wrapz + wrapy * k + j;
          for(int ii = 0;ii<3;ii++){
            const uint32_t idx = 3 * nidx + ii;
            z[idx] += omega * invD[idx] * (r[idx] - d[idx]);
          }
        }
#endif
}

void smoothDampedJacobi_halo_gpu(gpuGrid * gpu_grid, const uint_fast32_t l,const uint_fast32_t nswp, const CTYPE omega) {
  // usually nswp is between 1 and 5
  for (int s = 0; s < nswp; s++) {
    if (l>number_of_matrix_free_levels){
      applyStateOperatorSubspaceMatrix_gpu(gpu_grid,l);
    }
    else if (l>0){
      applyStateOperatorSubspace_halo_gpu(gpu_grid,l);
    }
    else {
      applyStateOperator_stencil_xz_d(gpu_grid,l);
    }

    #pragma omp parallel num_threads(gpu_grid->num_targets)
    {
      gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
      int xmin = (node->prev == NULL) ? 1 : 2;
      int xmax = node->dims.nelx[l] + 2;
      if (node->prev != NULL){
        smoothDampedJacobi_halo_kernel(node,l,omega,xmin,xmin+LENGTH_LEFT);
        xmin+=LENGTH_LEFT;
      }
      if (node->next != NULL){
        smoothDampedJacobi_halo_kernel(node,l,omega,xmax-LENGTH_RIGHT,xmax);
        xmax-=LENGTH_RIGHT;
      }

      #pragma omp barrier

        // The above kernels must finish before the data transfers of the boundaries may proceed.
        // If that was not the case, a data race could occur. Example:
        // Task A updates the east boundary on GPU 1 and sends the result to GPU 2.
        // Now task B updates the west boundary on GPU 2 hence it may access too new values.
      
      if (node->prev != NULL){
        SEND_MG_PREV(node,z,l)
      }
      if (node->next != NULL){
        SEND_MG_NEXT(node,z,l)
      }
      smoothDampedJacobi_halo_kernel(node,l,omega,xmin,xmax);
    }
  }
}

void set_z_to_zero_kernel(gpuNode * node,const uint_fast32_t l){
  CTYPE * z = node->solver.mg[l].z;
  const int ndof = node->dims.ndof[l];
  #pragma omp target teams distribute parallel for schedule(static) device(node->id)
  for(int i=0;i<ndof;i++){
    z[i] = 0.0;
  }
}

void set_z_to_zero(gpuGrid * gpu_grid, const uint_fast32_t l)
{
  #pragma omp parallel num_threads(gpu_grid->num_targets)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    set_z_to_zero_kernel(node,l);
  }
}

void add_d_to_z_kernel(gpuNode * node,const uint_fast32_t l, const int from,const int to){
  CTYPE * z = node->solver.mg[l].z;
  CTYPE * d = node->solver.mg[l].d;
#ifdef SIMD
  #pragma omp target teams distribute parallel for simd schedule(static) device(node->id)
#else
  #pragma omp target teams distribute parallel for schedule(static) device(node->id)
#endif
  for(int i=from;i<to;i++){
    z[i] += d[i];
  }
}

void add_d_to_z(gpuGrid * gpu_grid, const uint_fast32_t l)
{
  #pragma omp parallel num_threads(gpu_grid->num_targets)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    const int offset = node->dims.offset[l];
    const int length = node->dims.length[l];
    if (node->prev != NULL){
      add_d_to_z_kernel(node,l,offset,offset+NODE_SIZE(node,l)*LENGTH_LEFT);
    }
    if (node->next != NULL){
      add_d_to_z_kernel(node,l,offset+length-NODE_SIZE(node,l)*LENGTH_RIGHT,offset+length);
    }
    
    #pragma omp barrier

    int from = offset;
    int to = offset+length;
    if (node->prev != NULL){
      SEND_MG_PREV(node,z,l)
      from += LENGTH_LEFT*NODE_SIZE(node,l);
    }
    if (node->next != NULL){
      SEND_MG_NEXT(node,z,l)
      to -= LENGTH_RIGHT*NODE_SIZE(node,l);
    }
    add_d_to_z_kernel(node,l,from,to);
  }
}


void projectToFinerGrid_halo_gpu_kernel(gpuNode * node,const int l) {
  const CTYPE *z_coarse= node->solver.mg[l+1].z;
  CTYPE *d_fine = node->solver.mg[l].d;

  // Fine grid dimensions
  const uint_fast32_t nelxf = node->dims.nelx[l];
  const uint_fast32_t nelyf = node->dims.nely[l];
  const uint_fast32_t nelzf = node->dims.nelz[l];

  const uint_fast32_t wrapyf = node->dims.wrapy[l];
  const uint_fast32_t wrapzf = node->dims.wrapz[l];

  // Coarse grid dimensions
  const uint_fast32_t wrapyc = node->dims.wrapy[l+1];
  const uint_fast32_t wrapzc = node->dims.wrapz[l+1];

  const int nxf = nelxf + 1;
  const int nyf = nelyf + 1;
  const int nzf = nelzf + 1;

  // loop over nodes, usually very large with nx*ny*nz = 100.000 or more
  #pragma omp target teams distribute parallel for collapse(3) schedule(static) device(node->id)
  for (int32_t ifine = 1; ifine < nxf + 1; ifine++)
    for (int32_t kfine = 1; kfine < nzf + 1; kfine++)
      for (int32_t jfine = 1; jfine < nyf + 1; jfine++) {

        const uint32_t fineIndex =
            ifine * wrapyf * wrapzf + kfine * wrapyf + jfine;

        const uint32_t icoarse1 = (ifine - 1) / 2 + 1;
        const uint32_t icoarse2 = (ifine) / 2 + 1;
        const uint32_t jcoarse1 = (jfine - 1) / 2 + 1;
        const uint32_t jcoarse2 = (jfine) / 2 + 1;
        const uint32_t kcoarse1 = (kfine - 1) / 2 + 1;
        const uint32_t kcoarse2 = (kfine) / 2 + 1;

        // Node indices on coarse grid
        const uint_fast32_t coarseIndex1 =
            icoarse1 * wrapyc * wrapzc + kcoarse1 * wrapyc + jcoarse2;
        const uint_fast32_t coarseIndex2 =
            icoarse2 * wrapyc * wrapzc + kcoarse1 * wrapyc + jcoarse2;
        const uint_fast32_t coarseIndex3 =
            icoarse2 * wrapyc * wrapzc + kcoarse1 * wrapyc + jcoarse1;
        const uint_fast32_t coarseIndex4 =
            icoarse1 * wrapyc * wrapzc + kcoarse1 * wrapyc + jcoarse1;
        const uint_fast32_t coarseIndex5 =
            icoarse1 * wrapyc * wrapzc + kcoarse2 * wrapyc + jcoarse2;
        const uint_fast32_t coarseIndex6 =
            icoarse2 * wrapyc * wrapzc + kcoarse2 * wrapyc + jcoarse2;
        const uint_fast32_t coarseIndex7 =
            icoarse2 * wrapyc * wrapzc + kcoarse2 * wrapyc + jcoarse1;
        const uint_fast32_t coarseIndex8 =
            icoarse1 * wrapyc * wrapzc + kcoarse2 * wrapyc + jcoarse1;

        d_fine[3 * fineIndex + 0] = 0.125 * z_coarse[3 * coarseIndex1 + 0] +
                                   0.125 * z_coarse[3 * coarseIndex2 + 0] +
                                   0.125 * z_coarse[3 * coarseIndex3 + 0] +
                                   0.125 * z_coarse[3 * coarseIndex4 + 0] +
                                   0.125 * z_coarse[3 * coarseIndex5 + 0] +
                                   0.125 * z_coarse[3 * coarseIndex6 + 0] +
                                   0.125 * z_coarse[3 * coarseIndex7 + 0] +
                                   0.125 * z_coarse[3 * coarseIndex8 + 0];

        d_fine[3 * fineIndex + 1] = 0.125 * z_coarse[3 * coarseIndex1 + 1] +
                                   0.125 * z_coarse[3 * coarseIndex2 + 1] +
                                   0.125 * z_coarse[3 * coarseIndex3 + 1] +
                                   0.125 * z_coarse[3 * coarseIndex4 + 1] +
                                   0.125 * z_coarse[3 * coarseIndex5 + 1] +
                                   0.125 * z_coarse[3 * coarseIndex6 + 1] +
                                   0.125 * z_coarse[3 * coarseIndex7 + 1] +
                                   0.125 * z_coarse[3 * coarseIndex8 + 1];

        d_fine[3 * fineIndex + 2] = 0.125 * z_coarse[3 * coarseIndex1 + 2] +
                                   0.125 * z_coarse[3 * coarseIndex2 + 2] +
                                   0.125 * z_coarse[3 * coarseIndex3 + 2] +
                                   0.125 * z_coarse[3 * coarseIndex4 + 2] +
                                   0.125 * z_coarse[3 * coarseIndex5 + 2] +
                                   0.125 * z_coarse[3 * coarseIndex6 + 2] +
                                   0.125 * z_coarse[3 * coarseIndex7 + 2] +
                                   0.125 * z_coarse[3 * coarseIndex8 + 2];
      }
}

void projectToFinerGrid_halo_gpu(gpuGrid * gpu_grid, const uint_fast32_t l)
{
  #pragma omp parallel num_threads(gpu_grid->num_targets)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    projectToFinerGrid_halo_gpu_kernel(node,l);
    if (node->next != NULL){
      SEND_MG_NEXT(node,d,l)
    }
    if (node->prev != NULL){
      SEND_MG_PREV(node,d,l)
    }
  }
}

// projects a field to a coarser grid ufine -> r_coarse
// temperature: medium, called (number of mg levels [1-5]) x (number of cg
// iterations [5-100]) = [5-500]  times pr design iteration
void projectToCoarserGrid_halo_gpu_kernel(gpuNode * node,const int l) {
  CTYPE * d_fine = node->solver.mg[l].d;
  CTYPE * r_coarse = node->solver.mg[l+1].r;

  const uint_fast32_t wrapyf = node->dims.wrapy[l];
  const uint_fast32_t wrapzf = node->dims.wrapz[l];

  // Coarse grid dimensions
  const uint_fast32_t nelxc = node->dims.nelx[l+1];
  const uint_fast32_t nelyc = node->dims.nely[l+1];
  const uint_fast32_t nelzc = node->dims.nelz[l+1];

  const uint_fast32_t wrapyc = node->dims.wrapy[l+1];
  const uint_fast32_t wrapzc = node->dims.wrapz[l+1];

  const int nxc = nelxc + 1;
  const int nyc = nelyc + 1;
  const int nzc = nelzc + 1;

  const MTYPE vals[4] = {1.0, 0.5, 0.25, 0.125};

  // loop over nodes, usually very large with nx*ny*nz = 100.000 or more

  #pragma omp target teams distribute parallel for collapse(4) schedule(static) device(node->id) map(to:vals[:4])
  for (int32_t icoarse = 1; icoarse < nxc + 1; icoarse++)
    for (int32_t kcoarse = 1; kcoarse < nzc + 1; kcoarse++)
      for (int32_t jcoarse = 1; jcoarse < nyc + 1; jcoarse++) {
        for(int ii=0;ii<3;ii++){
          const int coarseIndex =
          3*(icoarse * wrapyc * wrapzc + kcoarse * wrapyc + jcoarse) + ii;

          // Node indices on fine grid
          const int nx1 = (icoarse - 1) * 2 + 1;
          const int ny1 = (jcoarse - 1) * 2 + 1;
          const int nz1 = (kcoarse - 1) * 2 + 1;

          CTYPE tmp = 0.0;
#ifdef SIMD
          #pragma omp simd collapse(3) reduction(+:tmp)
#endif
          for (int32_t ifine = nx1 - 1; ifine < nx1 + 2; ifine++)
            for (int32_t kfine = nz1 - 1; kfine < nz1 + 2; kfine++)
              for (int32_t jfine = ny1 - 1; jfine < ny1 + 2; jfine++) {
                const uint32_t fineIndex =
                  ifine * wrapyf * wrapzf + kfine * wrapyf + jfine;

                const int ind = (nx1 - ifine) * (nx1 - ifine) +
                              (ny1 - jfine) * (ny1 - jfine) +
                              (nz1 - kfine) * (nz1 - kfine);
                tmp +=
                  vals[ind] * d_fine[3 * fineIndex + ii];
              }
          r_coarse[coarseIndex] = tmp;
        }
      }
}

void projectToCoarserGrid_halo_gpu(gpuGrid * gpu_grid, const uint_fast32_t l)
{
  #pragma omp parallel num_threads(gpu_grid->num_targets)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    projectToCoarserGrid_halo_gpu_kernel(node,l);
    if (node->next != NULL){
      SEND_MG_NEXT(node,r,l+1)
    }
    if (node->prev != NULL){
      SEND_MG_PREV(node,r,l+1)
    }
  }
}

void to_coarse_gpu(gpuGrid * gpu_grid,const uint_fast32_t l)
{
  #pragma omp parallel num_threads(gpu_grid->num_targets)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    UPDATE_MG_TO(node,d,l)
  }
  d_to_d_minus_r(gpu_grid,l);

  // project residual down
  projectToCoarserGrid_halo_gpu(gpu_grid, l);

  #pragma omp parallel num_threads(gpu_grid->num_targets)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    UPDATE_MG_FROM(node,r,l+1)
    UPDATE_MG_FROM(node,d,l)
  }
}

void d_to_d_minus_r_kernel(gpuNode * node,const uint_fast32_t l, const int from,const int to){
  CTYPE * d = node->solver.mg[l].d;
  CTYPE * r = node->solver.mg[l].r;
  #pragma omp target teams distribute parallel for schedule(static) device(node->id)
  for(int i=from;i<to;i++){
    d[i] = r[i] - d[i];
  }
}

void d_to_d_minus_r(gpuGrid * gpu_grid, const uint_fast32_t l)
{
  #pragma omp parallel num_threads(gpu_grid->num_targets)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    const int offset = node->dims.offset[l];
    const int length = node->dims.length[l];
    if (node->prev != NULL){
      d_to_d_minus_r_kernel(node,l,offset,offset+NODE_SIZE(node,l)*LENGTH_LEFT);
    }
    if (node->next != NULL){
      d_to_d_minus_r_kernel(node,l,offset+length-NODE_SIZE(node,l)*LENGTH_RIGHT,offset+length);
    }
    #pragma omp barrier
    
    int from = offset;
    int to = offset+length;
    if (node->prev != NULL){
      SEND_MG_PREV(node,d,l)
      from += LENGTH_LEFT*NODE_SIZE(node,l);
    }
    if (node->next != NULL){
      SEND_MG_NEXT(node,d,l)
      to -= LENGTH_RIGHT*NODE_SIZE(node,l);
    }
    d_to_d_minus_r_kernel(node,l,from,to);
  }
}

void r_to_F_minus_r_kernel(gpuNode * node, const int from,const int to){
  CTYPE * F = node->F;
  CTYPE * r = node->solver.mg[0].r;
  #pragma omp target teams distribute parallel for schedule(static) device(node->id)
  for(int i=from;i<to;i++){
    r[i] = F[i] - r[i];
  }
}

void r_to_F_minus_r(gpuGrid * gpu_grid)
{
  #pragma omp parallel num_threads(gpu_grid->num_targets)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    const int offset = node->dims.offset[0];
    const int length = node->dims.length[0];
    if (node->prev != NULL){
      r_to_F_minus_r_kernel(node,offset,offset+NODE_SIZE(node,0)*LENGTH_LEFT);
    }
    if (node->next != NULL){
      r_to_F_minus_r_kernel(node,offset+length-NODE_SIZE(node,0)*LENGTH_RIGHT,offset+length);
    }

    #pragma omp barrier
    
    int from = offset;
    int to = offset+length;
    if (node->prev != NULL){
      SEND_MG_PREV(node,r,0)
      from += LENGTH_LEFT*NODE_SIZE(node,0);
    }
    if (node->next != NULL){
      SEND_MG_NEXT(node,r,0)
      to -= LENGTH_RIGHT*NODE_SIZE(node,0);
    }
    r_to_F_minus_r_kernel(node,from,to);
  }
}

void to_fine_gpu(gpuGrid * gpu_grid,const uint_fast32_t l)
{
  #pragma omp parallel num_threads(gpu_grid->num_targets)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    UPDATE_MG_TO(node,z,l+1)
  }

  // project residual up
  //projectToFinerGrid_halo(gc, l, mg[l+1].z, mg[l].d);
  projectToFinerGrid_halo_gpu(gpu_grid, l);

  #pragma omp parallel num_threads(gpu_grid->num_targets)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    UPDATE_MG_FROM(node,d,l)
  }
}

void applyStateOperatorSubspace_halo_gpu_kernel_1(const struct gpuNode * node, const int l) { 
  CTYPE * d = node->solver.mg[l].d;
  const uint_fast32_t ndofc = node->dims.ndof[l];
#ifdef SIMD
  #pragma omp target teams distribute parallel for simd schedule(static) device(node->id)
#else
  #pragma omp target teams distribute parallel for schedule(static) device(node->id)
#endif
  for (uint32_t i = 0; i < ndofc; i++)
    d[i] = 0.0;
}

void applyStateOperatorSubspace_halo_gpu_kernel_2(const struct gpuNode * node, const int l,const int32_t xmin,const int32_t xmax) { 
  const DTYPE *x = node->x;
  CTYPE * z = node->solver.mg[l].z;
  CTYPE * d = node->solver.mg[l].d;

  // On level 0 the dimensions are
  const uint_fast32_t wrapy = node->dims.wrapy[0];
  const uint_fast32_t wrapz = node->dims.wrapz[0];

  // Computing dimensions for coarse grid
  const uint_fast32_t wrapyc = node->dims.wrapy[l];
  const uint_fast32_t wrapzc = node->dims.wrapz[l];

  const uint_fast32_t nelyc = node->dims.nely[l];
  const uint_fast32_t nelzc = node->dims.nelz[l];

  const int ncell = pow(2, l);

  // Constants
  const double E0 = node->gc.E0;
  const double Emin = node->gc.Emin;

  const MTYPE* KE = node->gc.precomputedKE[l];

  //#pragma omp target teams device(node->id)
  for (int32_t bx = 0; bx < 2; bx++)
    for (int32_t bz = 0; bz < 2; bz++)
      for (int32_t by = 0; by < 2; by++)
        #pragma omp target teams distribute parallel for collapse(3) schedule(static) device(node->id)
        for (int32_t i = bx + xmin; i < xmax; i += 2)
          for (int32_t k = bz + 1; k < nelzc + 1; k += 2)
            for (int32_t j = by + 1; j < nelyc + 1; j += 2) {

              alignas(__alignBound) uint_fast32_t edof[24];
              alignas(__alignBound) MTYPE u_local[24];
              alignas(__alignBound) MTYPE out_local[24];

              getEdof_halo(edof, i, j, k, wrapyc, wrapzc);

              //#pragma omp simd safelen(24) aligned(out_local : __alignBound)
#ifdef SIMD
              #pragma omp simd simdlen(24) safelen(24) aligned(out_local:__alignBound)
#endif
              for (int ii = 0; ii < 24; ii++)
                out_local[ii] = 0.0;
#ifdef SIMD
              #pragma omp simd simdlen(24) safelen(24) aligned(edof,u_local:__alignBound)
#endif
              for (int ii = 0; ii < 24; ii++)
                u_local[ii] = (MTYPE)z[edof[ii]];

              // loop over interior subcells, depends on the level. total
              // iterations = (level+1)^3, i.e. only one iteration for the
              // finest level 0, but inreasing cubicly. Note that the total
              // amount of inner iterations nested by the inner and outer sets
              // of loops is always constant  ( across all levels, that means
              // that as the level number grows, the parallelization available
              // is shifted from the outer loops to the inner loops.
              for (int ii = 0; ii < ncell; ii++)
                for (int kk = 0; kk < ncell; kk++)
                  for (int jj = 0; jj < ncell; jj++) {
                    const int ifine = ((i - 1) * ncell) + ii + 1;
                    const int jfine = ((j - 1) * ncell) + jj + 1;
                    const int kfine = ((k - 1) * ncell) + kk + 1;

                    const int cellidx = ncell * ncell * ii + ncell * kk + jj;

                    const uint_fast32_t elementIndex =
                        ifine * (wrapy - 1) * (wrapz - 1) +
                        kfine * (wrapy - 1) + jfine;
                    const MTYPE elementScale =
                        Emin + x[elementIndex] * x[elementIndex] *
                                      x[elementIndex] * (E0 - Emin);

                    // cblas_dgemv(CblasRowMajor, CblasNoTrans, 24, 24,
                    //             elementScale,
                    //             &KE[24 * 24 * cellidx], 24,
                    //             u_local, 1, 1.0, out_local, 1);
#ifdef SIMD
              #pragma omp simd simdlen(24) safelen(24) aligned(u_local,out_local:__alignBound)
#endif
                    for (int i1=0;i1<24;i1++){
                      for (int i2=0;i2<24;i2++){
                        out_local[i1] += elementScale*KE[24 * 24 * cellidx+24*i1+i2]*u_local[i2];
                      }
                    }
                  }
#ifdef SIMD
              #pragma omp simd simdlen(24) safelen(24) aligned(out_local,edof:__alignBound)
#endif
              for (int iii = 0; iii < 24; iii++)
                d[edof[iii]] += (CTYPE)out_local[iii];
            }
}

void applyStateOperatorSubspace_halo_gpu_kernel_2_improved(const struct gpuNode * node, const int l,const int32_t xmin,const int32_t xmax) { 
  const DTYPE *x = node->x;
  CTYPE * z = node->solver.mg[l].z;
  CTYPE * d = node->solver.mg[l].d;

  // On level 0 the dimensions are
  const uint_fast32_t wrapy = node->dims.wrapy[0];
  const uint_fast32_t wrapz = node->dims.wrapz[0];

  // Computing dimensions for coarse grid
  const uint_fast32_t wrapyc = node->dims.wrapy[l];
  const uint_fast32_t wrapzc = node->dims.wrapz[l];

  const uint_fast32_t nelyc = node->dims.nely[l];
  const uint_fast32_t nelzc = node->dims.nelz[l];

  const int ncell = pow(2, l);

  // Constants
  const double E0 = node->gc.E0;
  const double Emin = node->gc.Emin;

  const MTYPE* KE = node->gc.precomputedKE[l];

  #pragma omp target teams device(node->id)
  for (int32_t bx = 0; bx < 2; bx++)
    for (int32_t bz = 0; bz < 2; bz++)
      for (int32_t by = 0; by < 2; by++)
        #pragma omp distribute parallel for collapse(3) schedule(static)
        for (int32_t i = bx + xmin; i < xmax; i += 2)
          for (int32_t k = bz + 1; k < nelzc + 1; k += 2)
            for (int32_t j = by + 1; j < nelyc + 1; j += 2)
#ifdef SIMD
              #pragma omp simd simdlen(24) safelen(24)
#endif
              for (int imv=0;imv<24;imv++){
                MTYPE out_local = 0.0;
                uint_fast32_t idx_update = 0;
                for (int kt=0;kt<2;kt++){
                  for (int jt=0;jt<2;jt++){
                    for (int it=0;it<2;it++){
                      int nx = i + it;
                      if (jt == 1){
                        nx = i + 1-it;
                      }
                      const int pos = 3*(kt*4+jt*2+it);
                      const int nz = k + kt;
                      const int ny = j + 1-jt;
                      const uint_fast32_t nIndex = nx * wrapyc * wrapzc + nz * wrapyc + ny;
                      uint_fast32_t edof = 3 * nIndex;
                      for (int off1 = 0;off1<3;off1++){
                        if (pos + off1 == imv){
                          idx_update = edof+off1;
                        }
                      }
                      for (int ii = 0; ii < ncell; ii++)
                        for (int kk = 0; kk < ncell; kk++)
                          for (int jj = 0; jj < ncell; jj++){
                            const int ifine = ((i - 1) * ncell) + ii + 1;
                            const int jfine = ((j - 1) * ncell) + jj + 1;
                            const int kfine = ((k - 1) * ncell) + kk + 1;

                            const int cellidx = 24 * 24 * (ncell * ncell * ii + ncell * kk + jj);

                            const uint_fast32_t elementIndex =
                                ifine * (wrapy - 1) * (wrapz - 1) +
                                kfine * (wrapy - 1) + jfine;
                            const MTYPE elementScale =
                                Emin + x[elementIndex] * x[elementIndex] *
                                x[elementIndex] * (E0 - Emin);
                            for (int off = 0;off<3;off++){
                              out_local += elementScale*KE[cellidx+imv*24+pos+off]*((MTYPE)z[edof+off]);
                            }
                          }
                    }
                  }
                }
                #pragma omp atomic update
                d[idx_update] += (CTYPE)out_local;
              }

}


void applyStateOperatorSubspace_halo_gpu_kernel_3(const struct gpuNode * node, const int l) { 
  CTYPE * z = node->solver.mg[l].z;
  CTYPE * d = node->solver.mg[l].d;

  uint_fast32_t * idx = node->gc.fixedDofs[l].idx;
  uint_fast32_t n = node->gc.fixedDofs[l].n;
                  
// apply boundaryConditions
#pragma omp target teams distribute parallel for schedule(static) device(node->id)
  for (int i = 0; i < n; i++) {
    d[idx[i]] = z[idx[i]];
  }
}

/**
 * This function is particularly tricky as it adds to the surrounding 24 lattices
 * for each lattice. Thus the contribution on the west boundary must be send to
 * the previos GPU before the previous GPU computes the values on the east boundary.
 * 
 */

void applyStateOperatorSubspace_halo_gpu(gpuGrid * gpu_grid, const int l)
{
  #pragma omp parallel num_threads(gpu_grid->num_targets)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    applyStateOperatorSubspace_halo_gpu_kernel_1(node,l);

    #pragma omp barrier

    if (node->prev != NULL){
      applyStateOperatorSubspace_halo_gpu_kernel_2(node,l,1,1+LENGTH_LEFT);
      send_buffer_to_prev_CTYPE(node,node->solver.mg[l].d,l);
    }
    if (node->next != NULL){
      int xmax = node->dims.nelx[l] + 1;
      applyStateOperatorSubspace_halo_gpu_kernel_2(node,l,xmax-LENGTH_RIGHT,xmax);
      send_buffer_to_next_CTYPE(node,node->solver.mg[l].d,l);
    }

    #pragma omp barrier

    if (node->prev != NULL){
      sum_west_halo_CTYPE(node,node->solver.mg[l].d,l);
    }
    if (node->next != NULL){
      sum_east_halo_CTYPE(node,node->solver.mg[l].d,l);
    }
    #pragma omp barrier
    
    int xmin = 1;
    int xmax = node->dims.nelx[l] + 1;
    if (node->prev != NULL){
      xmin += LENGTH_LEFT;
    }
    if (node->next != NULL){
      xmax -= LENGTH_RIGHT;
    }
    applyStateOperatorSubspace_halo_gpu_kernel_2(node,l,xmin,xmax);
    if (node->prev == NULL){
      applyStateOperatorSubspace_halo_gpu_kernel_3(node,l);
    }
  }
}

void assemble_invD_kernel_1(gpuNode * node,const int l) {
  MTYPE *invD = node->solver.mg[l].invD;
  const uint_fast32_t ndofc = node->dims.ndof[l];

  #pragma omp target teams distribute parallel for schedule(static) device(node->id)
  for (unsigned int i = 0; i < ndofc; i++)
    invD[i] = 0.0;
}

void assemble_invD_kernel_2(gpuNode * node,const int l,const int xmin,const int xmax) {
  const DTYPE *x = node->x;
  MTYPE *invD = node->solver.mg[l].invD;
  // Coarse grid dimensions
  const int ncell = pow(2, l);
  const uint_fast32_t wrapyc = node->dims.wrapy[l];
  const uint_fast32_t wrapzc = node->dims.wrapz[l];

  const uint_fast32_t nelyc = node->dims.nely[l];
  const uint_fast32_t nelzc = node->dims.nelz[l];

  const uint_fast32_t wrapy = node->dims.wrapy[0];
  const uint_fast32_t wrapz = node->dims.wrapz[0];

  const double Emin = node->gc.Emin;
  const double E0 = node->gc.E0;
  const MTYPE * KE = node->gc.precomputedKE[l];

  #pragma omp target teams device(node->id)
  for (int32_t bx = 0; bx < 2; bx++)
    for (int32_t bz = 0; bz < 2; bz++)
      for (int32_t by = 0; by < 2; by++)

        #pragma omp distribute parallel for collapse(3) schedule(static)
        for (int32_t i = bx + xmin; i < xmax; i += 2)
          for (int32_t k = bz + 1; k < nelzc + 1; k += 2)
            for (int32_t j = by + 1; j < nelyc + 1; j += 2) {

              alignas(__alignBound) uint_fast32_t edof[24];
              getEdof_halo(edof, i, j, k, wrapyc, wrapzc);

              for (int ii = 0; ii < ncell; ii++)
                for (int kk = 0; kk < ncell; kk++)
                  for (int jj = 0; jj < ncell; jj++) {
                    const int ifine = ((i - 1) * ncell) + ii + 1;
                    const int jfine = ((j - 1) * ncell) + jj + 1;
                    const int kfine = ((k - 1) * ncell) + kk + 1;

                    const int cellidx = ncell * ncell * ii + ncell * kk + jj;

                    const uint_fast32_t elementIndex =
                        ifine * (wrapy - 1) * (wrapz - 1) +
                        kfine * (wrapy - 1) + jfine;
                    const MTYPE elementScale =
                        Emin + x[elementIndex] * x[elementIndex] *
                                      x[elementIndex] * (E0 - Emin);
#ifdef SIMD
                    #pragma omp simd safelen(24) aligned(edof:__alignBound)
#endif
                    for (int iii = 0; iii < 24; iii++)
                      invD[edof[iii]] +=
                          elementScale * KE[24 * 24 * cellidx +
                                                             iii * 24 + iii];
                  }
            }
}

void assemble_invD_kernel_2_improved(gpuNode * node,const int l,const int xmin,const int xmax) {
  const DTYPE *x = node->x;
  MTYPE *invD = node->solver.mg[l].invD;
  // Coarse grid dimensions
  const int ncell = pow(2, l);
  const uint_fast32_t wrapyc = node->dims.wrapy[l];
  const uint_fast32_t wrapzc = node->dims.wrapz[l];

  const uint_fast32_t nelyc = node->dims.nely[l];
  const uint_fast32_t nelzc = node->dims.nelz[l];

  const uint_fast32_t wrapy = node->dims.wrapy[0];
  const uint_fast32_t wrapz = node->dims.wrapz[0];

  const double Emin = node->gc.Emin;
  const double E0 = node->gc.E0;
  const MTYPE * KE = node->gc.precomputedKE[l];

  for (int32_t bx = 0; bx < 2; bx++)
    for (int32_t bz = 0; bz < 2; bz++)
      for (int32_t by = 0; by < 2; by++)

        #pragma omp target teams distribute parallel for collapse(6) schedule(static) device(node->id)
        for (int32_t i = bx + xmin; i < xmax; i += 2)
          for (int32_t k = bz + 1; k < nelzc + 1; k += 2)
            for (int32_t j = by + 1; j < nelyc + 1; j += 2) {
              for (int32_t kt = 0; kt<2; kt++) {
                for (int32_t jt = 1; jt>=0; jt--) {
                  for (int32_t itemp = 0; itemp<2; itemp++) {
                    int32_t it = (jt == 1) ? itemp : 1-itemp;
                    int32_t idx_edof = 3*(kt*4+(1-jt)*2+itemp);
                    int32_t idx_global = 3*((i+it) * wrapyc * wrapzc + (k+kt) * wrapyc + (j+jt));
                    MTYPE tmp[3] = {0.0,0.0,0.0};
                    for (int ii = 0; ii < ncell; ii++)
                      for (int kk = 0; kk < ncell; kk++)
                        for (int jj = 0; jj < ncell; jj++) {
                          const int ifine = ((i - 1) * ncell) + ii + 1;
                          const int jfine = ((j - 1) * ncell) + jj + 1;
                          const int kfine = ((k - 1) * ncell) + kk + 1;

                          const int cellidx = ncell * ncell * ii + ncell * kk + jj;

                          const uint_fast32_t elementIndex =
                              ifine * (wrapy - 1) * (wrapz - 1) +
                              kfine * (wrapy - 1) + jfine;
                          const MTYPE elementScale =
                              Emin + x[elementIndex] * x[elementIndex] *
                                            x[elementIndex] * (E0 - Emin);
                          tmp[0] += elementScale * KE[24 * 24 * cellidx + idx_edof * 24 + idx_edof];
                          tmp[1] += elementScale * KE[24 * 24 * cellidx + (idx_edof+1) * 24 + (idx_edof+1)];
                          tmp[2] += elementScale * KE[24 * 24 * cellidx + (idx_edof+2) * 24 + (idx_edof+2)];
                        }
                    invD[idx_global] += tmp[0];
                    invD[idx_global+1] += tmp[1];
                    invD[idx_global+2] += tmp[2];
                  }
                }
              }
            }
}

void assemble_invD_kernel_3(gpuNode * node,const int l) {
  MTYPE *invD = node->solver.mg[l].invD;

  // apply boundaryConditions
  //if (node->prev == NULL){
    const uint_fast32_t * idx = node->gc.fixedDofs[l].idx;
    const uint_fast32_t n = node->gc.fixedDofs[l].n;
    #pragma omp target teams distribute parallel for schedule(static) device(node->id)
      for (int i = 0; i < n; i++)
        invD[idx[i]] = 1.0;
  //}
}

void assemble_invD_kernel_4(gpuNode * node,const int l) {
  MTYPE *invD = node->solver.mg[l].invD;
  // Coarse grid dimensions
  const uint_fast32_t wrapyc = node->dims.wrapy[l];
  const uint_fast32_t wrapzc = node->dims.wrapz[l];

  const uint_fast32_t nelxc = node->dims.nelx[l];
  const uint_fast32_t nelyc = node->dims.nely[l];
  const uint_fast32_t nelzc = node->dims.nelz[l];

  #pragma omp target teams distribute parallel for collapse(3) schedule(static) device(node->id)
  for (int i = 1; i < nelxc +2; i++)
    for (int k = 1; k < nelzc + 2; k++)
      for (int j = 1; j < nelyc + 2; j++) {
        const int nidx = (i * wrapyc * wrapzc + wrapyc * k + j);

        invD[3 * nidx + 0] = 1.0 / invD[3 * nidx + 0];
        invD[3 * nidx + 1] = 1.0 / invD[3 * nidx + 1];
        invD[3 * nidx + 2] = 1.0 / invD[3 * nidx + 2];
      }
}

void assembleInvertedMatrixDiagonalSubspace_halo_gpu(gpuGrid * gpu_grid,const int nl)
{
  #pragma omp parallel num_threads(gpu_grid->num_targets)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    for (int l=0;l<nl;l++){
      assemble_invD_kernel_1(node,l);
    }
    #pragma omp barrier

    for (int l=0;l<nl;l++){
      if (node->prev != NULL){
        assemble_invD_kernel_2_improved(node,l,1,1+LENGTH_LEFT);
        send_buffer_to_prev_MTYPE(node,node->solver.mg[l].invD,l);
      }
      if (node->next != NULL){
        int xmax = node->dims.nelx[l] + 1;
        assemble_invD_kernel_2_improved(node,l,xmax-LENGTH_RIGHT,xmax);
        send_buffer_to_next_MTYPE(node,node->solver.mg[l].invD,l);
      }

      #pragma omp barrier

      if (node->prev != NULL){
        sum_west_halo_MTYPE(node,node->solver.mg[l].invD,l);
      }
      if (node->next != NULL){
        sum_east_halo_MTYPE(node,node->solver.mg[l].invD,l);
      }

      #pragma omp barrier
      
      int xmin = 1;
      int xmax = node->dims.nelx[l] + 1;
      if (node->prev != NULL){
        xmin += LENGTH_LEFT;
      }
      if (node->next != NULL){
        xmax -= LENGTH_RIGHT;
      }
      assemble_invD_kernel_2_improved(node,l,xmin,xmax);
      if (node->prev == NULL){
        assemble_invD_kernel_3(node,l);
      }
      assemble_invD_kernel_4(node,l);

      #pragma omp barrier

      if (node->next != NULL){
        SEND_MG_NEXT(node,invD,l)
      }
      if (node->prev != NULL){
        SEND_MG_PREV(node,invD,l)
      }
    }
  }
}

CTYPE sum_d_gpus(gpuGrid * gpu_grid,const int l){
  CTYPE sum = 0;
  for(int it= 0;it<gpu_grid->num_targets;it++){
    gpuNode * node = &(gpu_grid->targets[it]);
    CTYPE sum_tmp = 0;
    CTYPE * d = node->solver.mg[l].d;
    const int offset = node->dims.offset[l];
    const int length = node->dims.length[l];
    #pragma omp target teams distribute parallel for schedule(static) device(node->id) reduction(+:sum_tmp)
    for(int i = offset;i<offset+length;i++){
      sum_tmp+=d[i];
    }
    sum += sum_tmp;
  }
  return sum;
}

CTYPE sum_F_gpus(gpuGrid * gpu_grid){
  CTYPE sum = 0;
  for(int it= 0;it<gpu_grid->num_targets;it++){
    gpuNode * node = &(gpu_grid->targets[it]);
    CTYPE sum_tmp = 0;
    CTYPE * F = node->F;
    const int offset = node->dims.offset[0];
    const int length = node->dims.length[0];
    #pragma omp target teams distribute parallel for schedule(static) device(node->id) reduction(+:sum_tmp)
    for(int i = offset;i<offset+length;i++){
      sum_tmp+=F[i];
    }
    sum += sum_tmp;
  }
  return sum;
}

CTYPE reduce_inner_product(CTYPE *x, CTYPE *y, gpuNode * node, const int l) {
  CTYPE val = 0.0;
  const int start = node->dims.offset[l];
  const int end = start+node->dims.length[l];
  #pragma omp target teams distribute parallel for reduction(+ : val) map(always,tofrom: val) device(node->id)
  for (uint_fast32_t i = start; i < end; i++)
    val += x[i] * y[i];
  return val;
}

void F_norm(gpuGrid * gpu_grid,CTYPE * val) {
  CTYPE tmp[gpu_grid->num_targets];
  #pragma omp parallel num_threads(gpu_grid->num_targets) shared(tmp)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    tmp[omp_get_thread_num()] = reduce_inner_product(node->F,node->F,node,0);
  }
  *val = 0.0;
  for (int it=0;it < gpu_grid->num_targets;it++){
    *val += tmp[it];
  }
  *val = sqrt(*val);
}

void r_norm(gpuGrid * gpu_grid,CTYPE * val) {
  CTYPE tmp[gpu_grid->num_targets];
  #pragma omp parallel num_threads(gpu_grid->num_targets) shared(tmp)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    tmp[omp_get_thread_num()] = reduce_inner_product(node->solver.mg[0].r,node->solver.mg[0].r,node,0);
  }
  *val = 0.0;
  for (int it=0;it < gpu_grid->num_targets;it++){
    *val += tmp[it];
  }
  *val = sqrt(*val);
}

void get_rho(gpuGrid * gpu_grid,CTYPE * val) {
  CTYPE tmp[gpu_grid->num_targets];
  #pragma omp parallel num_threads(gpu_grid->num_targets) shared(tmp)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    tmp[omp_get_thread_num()] = reduce_inner_product(node->solver.mg[0].r,node->solver.mg[0].z,node,0);
  }
  *val = 0.0;
  for (int it=0;it < gpu_grid->num_targets;it++){
    *val += tmp[it];
  }
}

void p_q_inner_product(gpuGrid * gpu_grid,CTYPE * val) {
  CTYPE tmp[gpu_grid->num_targets];
  #pragma omp parallel num_threads(gpu_grid->num_targets) shared(tmp)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    tmp[omp_get_thread_num()] = reduce_inner_product(node->solver.cg.p,node->solver.cg.q,node,0);
  }
  *val = 0.0;
  for (int it=0;it < gpu_grid->num_targets;it++){
    *val += tmp[it];
  }
}

void copy_u_kernel(gpuNode * node){
  CTYPE * z = node->solver.mg[0].z;
  STYPE * U = node->U;
  const int ndof = node->dims.ndof[0];
  #pragma omp target teams distribute parallel for schedule(static) device(node->id)
  for (uint_fast32_t i = 0; i < ndof; i++)
      z[i] = (CTYPE)U[i];
}

void copy_U_to_z(gpuGrid * gpu_grid) {
  #pragma omp parallel num_threads(gpu_grid->num_targets)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    copy_u_kernel(node);
  }
}

void beta_p_plus_z_kernel(gpuNode * node,const CTYPE beta, const int from,const int to){
  CTYPE * p = node->solver.cg.p;
  CTYPE * z = node->solver.mg[0].z;
  #pragma omp target teams distribute parallel for schedule(static) device(node->id) firstprivate(beta)
  for(int i=from;i<to;i++){
    p[i] = beta * p[i] + z[i];
  }
}

void beta_p_plus_z(gpuGrid * gpu_grid, const CTYPE beta)
{
  #pragma omp parallel num_threads(gpu_grid->num_targets)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    const int offset = node->dims.offset[0];
    const int length = node->dims.length[0];
    if (node->prev != NULL){
      beta_p_plus_z_kernel(node,beta,offset,offset+NODE_SIZE(node,0)*LENGTH_LEFT);
    }
    if (node->next != NULL){
      beta_p_plus_z_kernel(node,beta,offset+length-NODE_SIZE(node,0)*LENGTH_RIGHT,offset+length);
    }
    #pragma omp barrier

    int from = offset;
    int to = offset+length;
    if (node->prev != NULL){
      SEND_CG_PREV(node,p)
      from += LENGTH_LEFT*NODE_SIZE(node,0);
    }
    if (node->next != NULL){
      SEND_CG_NEXT(node,p)
      to -= LENGTH_RIGHT*NODE_SIZE(node,0);
    }
    beta_p_plus_z_kernel(node,beta,from,to);
  }
}

void add_p_alpha_to_U_kernel(gpuNode * node,const CTYPE alpha, const int from,const int to){
  CTYPE * p = node->solver.cg.p;
  STYPE * U = node->U;
  #pragma omp target teams distribute parallel for schedule(static) device(node->id) firstprivate(alpha)
  for(int i=from;i<to;i++){
    U[i] += (STYPE)(alpha * p[i]);
  }
}

void add_q_alpha_to_r(gpuNode * node,const CTYPE alpha, const int from,const int to){
  CTYPE * q = node->solver.cg.q;
  CTYPE * r = node->solver.mg[0].r;
  #pragma omp target teams distribute parallel for schedule(static) device(node->id) firstprivate(alpha)
  for(int i=from;i<to;i++){
    r[i] -= alpha * q[i];
  }
}

void cg_step_u_and_r(gpuGrid * gpu_grid, const CTYPE alpha)
{
  #pragma omp parallel num_threads(gpu_grid->num_targets)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    const int offset = node->dims.offset[0];
    const int length = node->dims.length[0];
    if (node->prev != NULL){
      add_p_alpha_to_U_kernel(node,alpha,offset,offset+NODE_SIZE(node,0)*LENGTH_LEFT);
      add_q_alpha_to_r(node,alpha,offset,offset+NODE_SIZE(node,0)*LENGTH_LEFT);
    }
    if (node->next != NULL){
      add_p_alpha_to_U_kernel(node,alpha,offset+length-NODE_SIZE(node,0)*LENGTH_RIGHT,offset+length);
       add_q_alpha_to_r(node,alpha,offset+length-NODE_SIZE(node,0)*LENGTH_RIGHT,offset+length);
    }

    #pragma omp barrier
    
    int from = offset;
    int to = offset+length;
    if (node->prev != NULL){
      SEND_MG_PREV(node,r,0)
      SEND_UF_PREV(node,U)
      from += LENGTH_LEFT*NODE_SIZE(node,0);
    }
    if (node->next != NULL){
      SEND_MG_NEXT(node,r,0)
      SEND_UF_NEXT(node,U)
      to -= LENGTH_RIGHT*NODE_SIZE(node,0);
    }
    add_p_alpha_to_U_kernel(node,alpha,from,to);
    add_q_alpha_to_r(node,alpha,from,to);
  }
}

DTYPE get_volume(gpuNode * node){
  DTYPE vol = 0.0;
  DTYPE *x = node->x;
  const uint_fast32_t nelx = node->dims.nelx[0];
  const uint_fast32_t nely = node->dims.nely[0];
  const uint_fast32_t nelz = node->dims.nelz[0];
  const uint_fast32_t wrapy = node->dims.wrapy[0];
  const uint_fast32_t wrapz = node->dims.wrapz[0];
  #pragma omp target teams distribute parallel for collapse(3) schedule(static)\
  reduction(+ : vol) map(always,tofrom: vol) device(node->id)
  for (int i = 1; i < nelx + 1; i++)
    for (int k = 1; k < nelz + 1; k++)
      for (int j = 1; j < nely + 1; j++) {
        const int idx = i * (wrapy - 1) * (wrapz - 1) + k * (wrapy - 1) + j;
        vol += x[idx];
      }
  return vol;
}

DTYPE get_xnew(gpuNode * node,const DTYPE move,const DTYPE lmid,const DTYPE volfrac){
  DTYPE gt = 0.0;
  DTYPE * xnew = node->xnew;
  DTYPE * xtmp = node->xtmp;
  DTYPE * dc = node->dc;
  DTYPE * dv = node->dv;
  const int from = (node->prev != NULL)*X_SIZE(node);
  const int to = node->design_size-(node->next != NULL)*X_SIZE(node);
#ifdef SIMD
  #pragma omp target teams distribute parallel for simd schedule(static)\
  reduction(+ : gt) map(always,tofrom: gt) device(node->id)
#else
  #pragma omp target teams distribute parallel for schedule(static)\
  reduction(+ : gt) map(always,tofrom: gt) device(node->id)
#endif
  for (int i = from; i < to; i++) {
    xnew[i] =
        MAX(0.0, MAX(xtmp[i] - move,
                     MIN(1.0, MIN(xtmp[i] + move,
                                  xtmp[i] * sqrt(-dc[i] / (dv[i] * lmid))))));
    gt += dv[i] * (xnew[i] - xtmp[i]);
  }
  return gt;
}

DTYPE get_xtmp(gpuNode * node)
{
  DTYPE change = 0.0;
  DTYPE * xnew = node->xnew;
  DTYPE * xtmp = node->xtmp;
  const int from = (node->prev != NULL)*X_SIZE(node);
  const int to = node->design_size-(node->next != NULL)*X_SIZE(node);
  #pragma omp target teams distribute parallel for schedule(static)\
  reduction(max : change) map(always,tofrom: change) device(node->id)
  for (int i = from; i < to; i++) {
    change = MAX(change, fabs(xtmp[i] - xnew[i]));
    xtmp[i] = xnew[i];
  }
  return change;
}

void conjugate_gradient_step(gpuGrid * gpu_grid,DTYPE * vol,float * change,const DTYPE volfrac)
{
  *vol = 0.0;
  #pragma omp parallel num_threads(gpu_grid->num_targets) shared(vol)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    DTYPE vol_tmp = get_volume(node);
    #pragma omp atomic update
    (*vol) += vol_tmp;
  }
  (*vol) /= (DTYPE)(gpu_grid->gc->nelx * gpu_grid->gc->nely * gpu_grid->gc->nelz);

  // Avoiding catastrophic cancellation when (*vol)=0.11999999999999999999 and volfrac=0.12
  DTYPE g = MAX(MIN((*vol) - volfrac,1e-4),1e-4);

  DTYPE l1 = 0.0, l2 = 1e9, move = 0.2;
  while ((l2 - l1) / (l1 + l2) > 1e-6) {
    DTYPE lmid = 0.5 * (l2 + l1);
    DTYPE gt = 0.0;


    #pragma omp parallel num_threads(gpu_grid->num_targets) shared(gt)
    {
      gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
      DTYPE gt_temp = get_xnew(node,move,lmid,volfrac);
      #pragma omp atomic update
      gt += gt_temp;
    }
    //printf("g is %lf,gt is %lf\n",g,gt);
    gt += g;
    if (gt > 0)
      l1 = lmid;
    else
      l2 = lmid;
  }

  float change_tmp = 0.0;
  #pragma omp parallel num_threads(gpu_grid->num_targets) reduction(max:change_tmp)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    CTYPE tmp = get_xtmp(node);
    change_tmp =  MAX(change_tmp,tmp);
  }
  *change = change_tmp;
}

// generate elementwise gradients from displacement.
// temperature: cold, called once for every design iteration.
DTYPE getComplianceAndSensetivity_halo_gpu_kernel(gpuNode * node) {
  const struct gridContext gc = node->gc;
  const DTYPE *x = node->x;
  STYPE *u = node->U;
  DTYPE *dcdx = node->dc;
  DTYPE c = 0.0;

  const uint_fast32_t nelx = node->dims.nelx[0];
  const uint_fast32_t nely = node->dims.nely[0];
  const uint_fast32_t nelz = node->dims.nelz[0];

  const uint_fast32_t wrapy = node->dims.wrapy[0];
  const uint_fast32_t wrapz = node->dims.wrapz[0];

  const MTYPE* KE = node->gc.precomputedKE[0];

  const double E0 = gc.E0;
  const double Emin = gc.Emin;

// loops over all elements, typically 100.000 or more. Note that there are no
// write dependencies, other than the reduction.
#pragma omp target teams distribute parallel for collapse(3) reduction(+ : c) map(always,tofrom:c) device(node->id)
  for (int32_t i = 1; i < nelx + 1; i++)
    for (int32_t k = 1; k < nelz + 1; k++)
      for (int32_t j = 1; j < nely + 1; j++) {

        alignas(__alignBound) uint_fast32_t edof[24];
        alignas(__alignBound) MTYPE u_local[24];
        //MTYPE tmp[24];

        getEdof_halo(edof, i, j, k, wrapy, wrapz);
        const uint_fast32_t elementIndex =
            i * (wrapy - 1) * (wrapz - 1) + k * (wrapy - 1) + j;

        // copy to local buffer for blas use
#ifdef SIMD
        #pragma omp simd simdlen(24) safelen(24) aligned(edof,u_local:__alignBound)
#endif
        for (int ii = 0; ii < 24; ii++)
          u_local[ii] = u[edof[ii]];

        // clocal = ulocal^T * ke * ulocal
        //cblas_dsymv(CblasRowMajor, CblasUpper, 24, 1.0, KE, 24,
        //            u_local, 1, 0.0, tmp, 1);
        // I am not taking advantage of the fact that KE is symmetric
        STYPE clocal = 0.0;
        for(int ii=0;ii<24;ii++){
          STYPE tmp = 0.0;
#ifdef SIMD
          #pragma omp simd safelen(24) reduction(+:tmp) aligned(u_local:__alignBound)
#endif
          for (int jj=0;jj<24;jj++){
            tmp += KE[ii*24+jj]*u_local[jj];
          }
          clocal += u_local[ii]*tmp;
        }

        //STYPE clocal = 0.0;
        //for (int ii = 0; ii < 24; ii++)
        //  clocal += u_local[ii] * tmp[ii];

        // apply contribution to c and dcdx
        c += clocal * (Emin + x[elementIndex] * x[elementIndex] *
                                      x[elementIndex] * (E0 - Emin));
        dcdx[elementIndex] = clocal * (-3.0 * (E0 - Emin) *
                                       x[elementIndex] * x[elementIndex]);
      }
  return c;
}

void getComplianceAndSensetivity_halo_gpu(gpuGrid * gpu_grid, DTYPE * c)
{
  (*c) = 0.0;
  #pragma omp parallel num_threads(gpu_grid->num_targets)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    DTYPE ctmp = getComplianceAndSensetivity_halo_gpu_kernel(node);
    #pragma omp atomic update
    (*c) += ctmp;
  }
}

void applyStateOperatorSubspaceMatrix_kernel_1(gpuNode * node,const int l) {
  const CTYPE * in = node->solver.mg[l].z;
  CTYPE * out = node->solver.mg[l].d;

  const struct CSRMatrix  * M = &(node->solver.coarseMatrices[l]);

  const uint_fast32_t nelxc = node->dims.nelx[l];
  const uint_fast32_t nelyc = node->dims.nely[l];
  const uint_fast32_t nelzc = node->dims.nelz[l];

  const uint_fast32_t wrapyc = node->dims.wrapy[l];
  const uint_fast32_t wrapzc = node->dims.wrapz[l];

  const int32_t nxc = nelxc + 1;
  const int32_t nyc = nelyc + 1;
  const int32_t nzc = nelzc + 1;

  const int ndofc = node->dims.ndof[l];

  int * rowOffsets = M->rowOffsets;
  MTYPE * vals = M->vals;

  int gpu_offset = 0;
  gpuNode * prev = node->prev;
  while(prev != NULL){
    gpu_offset += (prev->dims.nelx[l])*(prev->dims.nely[l]+1)*(prev->dims.nelz[l]+1);
    prev = prev->prev;
  }
  
  const int prev_is_null = (node->prev) == NULL;
  const int next_is_null = (node->next) == NULL;

#pragma omp target teams distribute parallel for schedule(static) device(node->id)
for (int i = 0; i < ndofc; i++)
  out[i] = 0.0;

#pragma omp target teams distribute parallel for collapse(3) schedule(static) device(node->id)
  for (int32_t i = 1; i < nxc + 1; i++)
    for (int32_t k = 1; k < nzc + 1; k++)
      for (int32_t j = 1; j < nyc + 1; j++) {

        const int haloNodeIndex = i * wrapyc * wrapzc + wrapyc * k + j;
        const uint32_t rowHaloIndex1 = 3 * haloNodeIndex + 0;
        const uint32_t rowHaloIndex2 = 3 * haloNodeIndex + 1;
        const uint32_t rowHaloIndex3 = 3 * haloNodeIndex + 2;

        const uint32_t i_no_halo = i - 1;
        const uint32_t j_no_halo = j - 1;
        const uint32_t k_no_halo = k - 1;

        const int rowNodeIndex = gpu_offset +
            i_no_halo * nyc * nzc + nyc * k_no_halo + j_no_halo;
        const uint32_t rowIndex1 = 3 * rowNodeIndex + 0;
        const uint32_t rowIndex2 = 3 * rowNodeIndex + 1;
        const uint32_t rowIndex3 = 3 * rowNodeIndex + 2;

        double outBufferRow1 = 0.0;
        double outBufferRow2 = 0.0;
        double outBufferRow3 = 0.0;

        const int32_t xmin = (prev_is_null) ? MAX(i - 1, 1) : i-1;
        const int32_t ymin = MAX(j - 1, 1);
        const int32_t zmin = MAX(k - 1, 1);

        const int32_t xmax = (next_is_null) ? MIN(i + 2, nxc + 1) : i + 2;
        const int32_t ymax = MIN(j + 2, nyc + 1);
        const int32_t zmax = MIN(k + 2, nzc + 1);

        // recalculate indices instead of recreating i,j,k without halo
        int offsetRow1 = rowOffsets[rowIndex1];
        int offsetRow2 = rowOffsets[rowIndex2];
        int offsetRow3 = rowOffsets[rowIndex3];

        for (int ii = xmin; ii < xmax; ii++)
          for (int kk = zmin; kk < zmax; kk++)
            for (int jj = ymin; jj < ymax; jj++) {

              const int haloColNodeIndex =
                  ii * wrapyc * wrapzc + wrapyc * kk + jj;

              const uint32_t colHaloIndex1 = 3 * haloColNodeIndex + 0;
              const uint32_t colHaloIndex2 = 3 * haloColNodeIndex + 1;
              const uint32_t colHaloIndex3 = 3 * haloColNodeIndex + 2;

              outBufferRow1 +=
                  (double)in[colHaloIndex1] * vals[offsetRow1 + 0] +
                  (double)in[colHaloIndex2] * vals[offsetRow1 + 1] +
                  (double)in[colHaloIndex3] * vals[offsetRow1 + 2];

              outBufferRow2 +=
                  (double)in[colHaloIndex1] * vals[offsetRow2 + 0] +
                  (double)in[colHaloIndex2] * vals[offsetRow2 + 1] +
                  (double)in[colHaloIndex3] * vals[offsetRow2 + 2];

              outBufferRow3 +=
                  (double)in[colHaloIndex1] * vals[offsetRow3 + 0] +
                  (double)in[colHaloIndex2] * vals[offsetRow3 + 1] +
                  (double)in[colHaloIndex3] * vals[offsetRow3 + 2];

              offsetRow1 += 3;
              offsetRow2 += 3;
              offsetRow3 += 3;
            }

        out[rowHaloIndex1] = outBufferRow1;
        out[rowHaloIndex2] = outBufferRow2;
        out[rowHaloIndex3] = outBufferRow3;
      }
}

void applyStateOperatorSubspaceMatrix_gpu(gpuGrid * gpu_grid,const int l)
{
  #pragma omp parallel num_threads(gpu_grid->num_targets)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    applyStateOperatorSubspaceMatrix_kernel_1(node,l);
    #pragma omp barrier
    if (node->next != NULL){
      SEND_MG_NEXT(node,d,l)
    }
    if (node->prev != NULL){
      SEND_MG_PREV(node,d,l)
    }
  }
}

#ifdef USE_CHOLMOD
void solveSubspaceMatrix_gpu_kernel_1(gpuNode * node,const int nl)
{
  const CTYPE *in = node->solver.mg[nl-1].r;

  const int32_t nelxc = node->dims.nelx[nl-1];
  const int32_t nelyc = node->dims.nely[nl-1];
  const int32_t nelzc = node->dims.nelz[nl-1];

  const int32_t nxc = nelxc + 1;
  const int32_t nyc = nelyc + 1;
  const int32_t nzc = nelzc + 1;

  const int wrapyc = node->dims.wrapy[nl-1];
  const int wrapzc = node->dims.wrapz[nl-1];

  double * rhs = node->rhs;

  int gpu_offset = 0;
  gpuNode * prev = node->prev;
  while(prev != NULL){
    gpu_offset += (prev->dims.nelx[nl-1])*(prev->dims.nely[nl-1]+1)*(prev->dims.nelz[nl-1]+1);
    prev = prev->prev;
  }

  // copy grid data to vector
  #pragma omp target teams distribute parallel for collapse(3) schedule(static) device(node->id)
  for (int i = 1; i < nxc + 1; i++)
    for (int k = 1; k < nzc + 1; k++)
      for (int j = 1; j < nyc + 1; j++) {
        const int nidx = gpu_offset + ((i - 1) * nyc * nzc + (k - 1) * nyc + (j - 1));
        const int nidx_s = (i * wrapyc * wrapzc + wrapyc * k + j);

        rhs[3 * nidx + 0] = in[3 * nidx_s + 0];
        rhs[3 * nidx + 1] = in[3 * nidx_s + 1];
        rhs[3 * nidx + 2] = in[3 * nidx_s + 2];
      }
  // The first gpu owns one more slice than the others in the 
  // non-padded domain used for cholmod
  if (node->prev == NULL){
    #pragma omp target update from(rhs[:3*nxc*nyc*nzc]) device(node->id)
  }
  else {
    #pragma omp target update from(rhs[3*(nxc*nyc*nzc+(node->id-1)*nelxc*nyc*nzc):3*nelxc*nyc*nzc]) device(node->id)
  }
}

void solveSubspaceMatrix_gpu_kernel_2(gpuNode * node,const int nl)
{
  CTYPE *out = node->solver.mg[nl-1].z;

  const int32_t nelxc = node->dims.nelx[nl-1];
  const int32_t nelyc = node->dims.nely[nl-1];
  const int32_t nelzc = node->dims.nelz[nl-1];

  const int32_t nxc = nelxc + 1;
  const int32_t nyc = nelyc + 1;
  const int32_t nzc = nelzc + 1;

  const int wrapyc = node->dims.wrapy[nl-1];
  const int wrapzc = node->dims.wrapz[nl-1];

  double * sol = node->sol;

  int gpu_offset = 0;
  gpuNode * prev = node->prev;
  while(prev != NULL){
    gpu_offset += (prev->dims.nelx[nl-1])*(prev->dims.nely[nl-1]+1)*(prev->dims.nelz[nl-1]+1);
    prev = prev->prev;
  }
  int nrows = node->solver.coarseMatrices[nl-1].nrows;
  #pragma omp target update to(sol[:nrows]) device(node->id)

  // copy grid data to vector
  #pragma omp target teams distribute parallel for collapse(3) schedule(static) device(node->id)
  for (int i = 1; i < nxc + 1; i++)
    for (int k = 1; k < nzc + 1; k++)
      for (int j = 1; j < nyc + 1; j++) {
        const int nidx = gpu_offset +  ((i - 1) * nyc * nzc + (k - 1) * nyc + (j - 1));
        const int nidx_s = (i * wrapyc * wrapzc + wrapyc * k + j);

        out[3 * nidx_s + 0] = sol[3 * nidx + 0];
        out[3 * nidx_s + 1] = sol[3 * nidx + 1];
        out[3 * nidx_s + 2] = sol[3 * nidx + 2];
      }

}

void solveSubspaceMatrix_gpu(gpuGrid * gpu_grid, const int nl,
                         struct CoarseSolverData solverData) {

  #pragma omp parallel num_threads(gpu_grid->num_targets)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    solveSubspaceMatrix_gpu_kernel_1(node,nl);
  }

  // cholmod_print_dense(solverData.rhs, "rhs", &solverData.cholmodCommon);

  // call cholmod solve

  cholmod_solve2(CHOLMOD_A,                 // system to solve 
                 solverData.factoredMatrix, // factorization to use 
                 solverData.rhs,            // right-hand-side 
                 NULL,                      // handle 
                 &solverData.solution,      // solution, allocated if need be 
                 NULL,                      // handle
                 &solverData.Y_workspace,   // workspace, or NULL 
                 &solverData.E_workspace,   // workspace, or NULL 
                 solverData.cholmodCommon);
  
  CTYPE *out = gpu_grid->targets[0].solver.mg[nl-1].z;
  const struct gridContext gc = *(gpu_grid->gc);
  const int ncell = pow(2, nl-1);
  const int32_t nelxc = gc.nelx / ncell;
  const int32_t nelyc = gc.nely / ncell;
  const int32_t nelzc = gc.nelz / ncell;

  const int32_t nxc = nelxc + 1;
  const int32_t nyc = nelyc + 1;
  const int32_t nzc = nelzc + 1;

  const int paddingyc =
      (STENCIL_SIZE_Y - ((nelyc + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
  const int paddingzc =
      (STENCIL_SIZE_Z - ((nelzc + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;

  const int wrapyc = nelyc + paddingyc + 3;
  const int wrapzc = nelzc + paddingzc + 3;

// copy data back to grid format
#pragma omp for collapse(3)
  for (int i = 1; i < nxc + 1; i++)
    for (int k = 1; k < nzc + 1; k++)
      for (int j = 1; j < nyc + 1; j++) {
        const int nidx = ((i - 1) * nyc * nzc + (k - 1) * nyc + (j - 1));
        const int nidx_s = (i * wrapyc * wrapzc + wrapyc * k + j);

        out[3 * nidx_s + 0] = ((double *)solverData.solution->x)[3 * nidx + 0];
        out[3 * nidx_s + 1] = ((double *)solverData.solution->x)[3 * nidx + 1];
        out[3 * nidx_s + 2] = ((double *)solverData.solution->x)[3 * nidx + 2];
      }
}

#endif
