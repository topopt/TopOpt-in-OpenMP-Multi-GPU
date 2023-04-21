#pragma once

#include "definitions.h"

void compute_padding(const struct gridContext *gc,const unsigned int l, uint_fast32_t * paddingx,uint_fast32_t * paddingy, uint_fast32_t * paddingz);

void compute_num_cells(const struct gridContext *gc,const unsigned int l,uint_fast32_t * nelx,uint_fast32_t * nely,uint_fast32_t * nelz);

void compute_wrapping(const struct gridContext *gc,const unsigned int l,uint_fast32_t * wrapx,uint_fast32_t * wrapy,uint_fast32_t * wrapz);

uint_fast32_t compute_ndof(const struct gridContext *gc,const unsigned int l);

void setFixedDof_halo(struct gridContext *gc, const int l);

void setupGC(struct gridContext *gc, const int nl,const int allocate);

void freeGC(struct gridContext *gc, const int nl);

void allocateZeroPaddedStateField(const struct gridContext gc, const int l,
                                  CTYPE **v);
void allocateZeroPaddedStateField_MTYPE(const struct gridContext gc,
                                        const int l, MTYPE **v);

void allocateZeroPaddedStateField_STYPE(const struct gridContext gc,
                                        const int l, STYPE **v);
