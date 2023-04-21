#pragma once

#include "definitions.h"

// compute indices of displacement for a given element number
// temperature: very very hot, called as part of the hot kernels in the
// program, should be inlined always.
#pragma omp declare target
__force_inline inline void getEdof_halo(uint_fast32_t edof[24], const int i,
                                        const int j, const int k,
                                        const int wrapy, const int wrapz) {

  const int nx_1 = i;
  const int nx_2 = i + 1;
  const int nz_1 = k;
  const int nz_2 = k + 1;
  const int ny_1 = j;
  const int ny_2 = j + 1;

  const uint_fast32_t nIndex1 = nx_1 * wrapy * wrapz + nz_1 * wrapy + ny_2;
  const uint_fast32_t nIndex2 = nx_2 * wrapy * wrapz + nz_1 * wrapy + ny_2;
  const uint_fast32_t nIndex3 = nx_2 * wrapy * wrapz + nz_1 * wrapy + ny_1;
  const uint_fast32_t nIndex4 = nx_1 * wrapy * wrapz + nz_1 * wrapy + ny_1;
  const uint_fast32_t nIndex5 = nx_1 * wrapy * wrapz + nz_2 * wrapy + ny_2;
  const uint_fast32_t nIndex6 = nx_2 * wrapy * wrapz + nz_2 * wrapy + ny_2;
  const uint_fast32_t nIndex7 = nx_2 * wrapy * wrapz + nz_2 * wrapy + ny_1;
  const uint_fast32_t nIndex8 = nx_1 * wrapy * wrapz + nz_2 * wrapy + ny_1;

  edof[0] = 3 * nIndex1 + 0;
  edof[1] = 3 * nIndex1 + 1;
  edof[2] = 3 * nIndex1 + 2;
  edof[3] = 3 * nIndex2 + 0;
  edof[4] = 3 * nIndex2 + 1;
  edof[5] = 3 * nIndex2 + 2;
  edof[6] = 3 * nIndex3 + 0;
  edof[7] = 3 * nIndex3 + 1;
  edof[8] = 3 * nIndex3 + 2;
  edof[9] = 3 * nIndex4 + 0;
  edof[10] = 3 * nIndex4 + 1;
  edof[11] = 3 * nIndex4 + 2;

  edof[12] = 3 * nIndex5 + 0;
  edof[13] = 3 * nIndex5 + 1;
  edof[14] = 3 * nIndex5 + 2;
  edof[15] = 3 * nIndex6 + 0;
  edof[16] = 3 * nIndex6 + 1;
  edof[17] = 3 * nIndex6 + 2;
  edof[18] = 3 * nIndex7 + 0;
  edof[19] = 3 * nIndex7 + 1;
  edof[20] = 3 * nIndex7 + 2;
  edof[21] = 3 * nIndex8 + 0;
  edof[22] = 3 * nIndex8 + 1;
  edof[23] = 3 * nIndex8 + 2;
}

// convert the node index from coordinates with halo padding to a grid without.
// requires the wrapping parameters from the grid with halo and size of the grid
// without
__force_inline inline int haloToTrue(const int index, const int wrapy,
                                     const int wrapz, const int ny,
                                     const int nz) {

  const int nodeNumber = index / 3;
  const int dofOffset = index % 3;

  const int i_halo = nodeNumber / (wrapy * wrapz);
  const int j_halo = nodeNumber % wrapy;
  const int k_halo = (nodeNumber % (wrapy * wrapz)) / wrapy;

  const int i = i_halo - 1;
  const int j = j_halo - 1;
  const int k = k_halo - 1;

  const int newNodeNumber = i * ny * nz + k * ny + j;

  return 3 * newNodeNumber + dofOffset;
}
#pragma omp end declare target
