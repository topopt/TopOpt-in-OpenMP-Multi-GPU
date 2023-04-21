#ifndef GPU_METHODS
#define GPU_METHODS

#include "definitions.h"
#include "gpu_grid.h"
#include "stencil_methods.h"
#include "stencil_utility.h"
#include "cholmod.h"
#include <omp.h>

/**
 * Input: 
 *      node->solver.mg[l].z
 *      node->x
 * Output:
 *      node->solver.mg[l].d 
 **/

void applyStateOperator_stencil_xz_d(gpuGrid * gpu_grid, const int l);

void applyStateOperator_stencil_xz_r(gpuGrid * gpu_grid, const int l);

void applyStateOperator_stencil_xp_q(gpuGrid * gpu_grid);

void smoothDampedJacobi_halo_gpu(gpuGrid * gpu_grid, const uint_fast32_t l,const uint_fast32_t nswp, const CTYPE omega);

void set_z_to_zero(gpuGrid * gpu_grid, const uint_fast32_t l);

void add_d_to_z(gpuGrid * gpu_grid, const uint_fast32_t l);

void projectToFinerGrid_halo_gpu(gpuGrid * gpu_grid, const uint_fast32_t l);

void projectToCoarserGrid_halo_gpu(gpuGrid * gpu_grid, const uint_fast32_t l);

void to_coarse_gpu(gpuGrid * gpu_grid,const uint_fast32_t l);

void d_to_d_minus_r(gpuGrid * gpu_grid, const uint_fast32_t l);

void r_to_F_minus_r(gpuGrid * gpu_grid);

void to_fine_gpu(gpuGrid * gpu_grid,const uint_fast32_t l);

void applyStateOperatorSubspace_halo_gpu(gpuGrid * gpu_grid, const int l);

void assembleInvertedMatrixDiagonalSubspace_halo_gpu(gpuGrid * gpu_grid,const int nl);

CTYPE sum_d_gpus(gpuGrid * gpu_grid,const int l);

CTYPE sum_F_gpus(gpuGrid * gpu_grid);

CTYPE reduce_inner_product(CTYPE *x, CTYPE *y, gpuNode * node,const int l);

void F_norm(gpuGrid * gpu_grid,CTYPE * val);

void r_norm(gpuGrid * gpu_grid,CTYPE * val);

void get_rho(gpuGrid * gpu_grid,CTYPE * val);

void p_q_inner_product(gpuGrid * gpu_grid,CTYPE * val);

void copy_U_to_z(gpuGrid * gpu_grid);

void beta_p_plus_z(gpuGrid * gpu_grid, const CTYPE beta);

void cg_step_u_and_r(gpuGrid * gpu_grid, const CTYPE alpha);

void conjugate_gradient_step(gpuGrid * gpu_grid,DTYPE * vol,float * change,const DTYPE volfrac);

void getComplianceAndSensetivity_halo_gpu(gpuGrid * gpu_grid, DTYPE * c);

void applyStateOperatorSubspaceMatrix_gpu(gpuGrid * gpu_grid,const int l);

void solveSubspaceMatrix_gpu(gpuGrid * gpu_grid, const int nl,
                         struct CoarseSolverData solverData);

#endif
