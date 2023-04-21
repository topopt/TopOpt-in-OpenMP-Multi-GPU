#pragma once

#include "definitions.h"

// main function
void top3dmgcg(const uint_fast32_t nelx, const uint_fast32_t nely,
               const uint_fast32_t nelz, const DTYPE volfrac, const DTYPE rmin,
               const uint_fast32_t nl, const float cgtol,
               const uint_fast32_t cgmax,const uint_fast32_t verbose,
               const uint_fast32_t write_result,const uint_fast32_t max_iterations);


void applyDensityFilter(const struct gridContext gc, const DTYPE rmin,
                        const DTYPE *rho, DTYPE *out);

void applyDensityFilterGradient(const struct gridContext gc, const DTYPE rmin,
                                DTYPE *v);

void writeDensityVtkFile(const int nelx, const int nely, const int nelz,
                         const DTYPE *densityArray, const char *filename);

void writeDensityVtkFileWithHalo(const int nelx, const int nely, const int nelz,
                                 const DTYPE *densityArray,
                                 const char *filename);
