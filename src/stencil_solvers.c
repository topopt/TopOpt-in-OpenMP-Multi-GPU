#include "../include/stencil_solvers.h"

#include "../include/stencil_grid_utility.h"
#include "../include/stencil_methods.h"
#include "../include/gpu_methods.h"

// jacobi smoothing/preconditioning
// temperature: hot, called 2x(number of levels)x(number of cg iterations) ~
// [20-1000] times every design iteration. Note that most compute time is spent
// in child function.

/**
 * This function updates:
 * mg[l].z
 * mg[l].d
 */

void smoothDampedJacobiSubspace_halo(const struct gridContext gc,
                                     const DTYPE *x, const int l,
                                     const uint_fast32_t nswp,
                                     const CTYPE omega,mg_ptrs * mg) {
  const MTYPE *invD = mg[l].invD;
  CTYPE *z = mg[l].z;
  const CTYPE *r = mg[l].r;
  CTYPE *d = mg[l].d;

  uint_fast32_t nelxc,nelyc,nelzc;
  compute_num_cells(&gc,l,&nelxc,&nelyc,&nelzc);

  uint_fast32_t wrapxc,wrapyc,wrapzc;
  compute_wrapping(&gc,l,&wrapxc,&wrapyc,&wrapzc);

  // usually nswp is between 1 and 5
  for (int s = 0; s < nswp; s++) {
    applyStateOperatorSubspace_halo(gc, l, x, z, d);

// long for loop, as ndof is typically 300.000 or more, but also trivially
// parallel.
#pragma omp parallel for collapse(3) schedule(static)
    for (int i = 1; i < nelxc + 2; i++)
      for (int k = 1; k < nelzc + 2; k++)
        for (int j = 1; j < nelyc + 2; j++) {
          const int nidx = (i * wrapyc * wrapzc + wrapyc * k + j);

          const uint32_t idx1 = 3 * nidx + 0;
          const uint32_t idx2 = 3 * nidx + 1;
          const uint32_t idx3 = 3 * nidx + 2;

          z[idx1] += omega * invD[idx1] * (r[idx1] - d[idx1]);
          z[idx2] += omega * invD[idx2] * (r[idx2] - d[idx2]);
          z[idx3] += omega * invD[idx3] * (r[idx3] - d[idx3]);
        }
  }
}

// jacobi smoothing/preconditioning
// temperature: hot, called 2x(number of levels)x(number of cg iterations) ~
// [20-1000] times every design iteration. Note that most compute time is spent
// in child function.

/**
 * This function updates:
 * mg[l].z
 * mg[l].d
 */

void smoothDampedJacobiSubspaceMatrix_halo(
    const struct gridContext gc, const struct CSRMatrix M, const int l,
    const uint_fast32_t nswp, const CTYPE omega,mg_ptrs * mg) {

    const MTYPE *invD = mg[l].invD;
    CTYPE *z = mg[l].z;
    const CTYPE *r = mg[l].r;
    CTYPE *d = mg[l].d;

  uint_fast32_t nelxc,nelyc,nelzc;
  compute_num_cells(&gc,l,&nelxc,&nelyc,&nelzc);

  uint_fast32_t wrapxc,wrapyc,wrapzc;
  compute_wrapping(&gc,l,&wrapxc,&wrapyc,&wrapzc);

  // usually nswp is between 1 and 5
  for (int s = 0; s < nswp; s++) {
    applyStateOperatorSubspaceMatrix(gc, l, M, z, d);

// long for loop, as ndof is typically 300.000 or more, but also trivially
// parallel.
#pragma omp parallel for collapse(3) schedule(static)
    for (int i = 1; i < nelxc + 2; i++)
      for (int k = 1; k < nelzc + 2; k++)
        for (int j = 1; j < nelyc + 2; j++) {
          const int nidx = (i * wrapyc * wrapzc + wrapyc * k + j);

          const uint32_t idx1 = 3 * nidx + 0;
          const uint32_t idx2 = 3 * nidx + 1;
          const uint32_t idx3 = 3 * nidx + 2;

          z[idx1] += omega * invD[idx1] * (r[idx1] - d[idx1]);
          z[idx2] += omega * invD[idx2] * (r[idx2] - d[idx2]);
          z[idx3] += omega * invD[idx3] * (r[idx3] - d[idx3]);
        }
  }
}

// jacobi smoothing/preconditioning
// temperature: hot, called 2x(number of levels)x(number of cg iterations) ~
// [20-1000] times every design iteration. Note that most compute time is spent
// in child function.

/**
 * This function updates:
 * mg.z
 * mg.d
 */

void smoothDampedJacobi_halo(gpuGrid * gpu_grid,const struct gridContext gc, const DTYPE *x,
                             const uint_fast32_t nswp, const CTYPE omega,
                             mg_ptrs mg) {
  const MTYPE *invD = mg.invD;
  CTYPE *z = mg.z;
  const CTYPE *r = mg.r;
  CTYPE * d = mg.d;

  // usually nswp is between 1 and 5
  for (int s = 0; s < nswp; s++) {
    applyStateOperator_stencil(gc, x, z, d);

//#pragma omp target teams distribute parallel for collapse(3) schedule(static)
    #pragma omp parallel for collapse(3) schedule(static)
    for (int i = 1; i < gc.nelx + 2; i++)
      for (int k = 1; k < gc.nelz + 2; k++)
        for (int j = 1; j < gc.nely + 2; j++) {
          const int nidx = i * gc.wrapy * gc.wrapz + gc.wrapy * k + j;

          const uint32_t idx1 = 3 * nidx + 0;
          const uint32_t idx2 = 3 * nidx + 1;
          const uint32_t idx3 = 3 * nidx + 2;

          z[idx1] += omega * invD[idx1] * (r[idx1] - d[idx1]);
          z[idx2] += omega * invD[idx2] * (r[idx2] - d[idx2]);
          z[idx3] += omega * invD[idx3] * (r[idx3] - d[idx3]);
        }
  }
}

// Vcycle preconditioner. recursive function.
// temperature: medium, called (number of levels)x(number of cg iterations ~
// 5 - 100) every design iteration. Much of the compute time is spent in
// this function, although in children functions.
void VcyclePreconditioner(gpuGrid * gpu_grid,const struct gridContext gc,solver_ptrs *data, const DTYPE *x,
                          const int nl, CTYPE omega,
                          const int nswp,double * cpu_time) {
  int last_layer_on_cpu = 0;
  #ifdef USE_CHOLMOD
    last_layer_on_cpu  = 1;
  #endif

  for (int l = 0;l<nl-last_layer_on_cpu;l++){
    set_z_to_zero(gpu_grid,l);

    if (l < nl-1){
      smoothDampedJacobi_halo_gpu(gpu_grid,l,nswp,omega);
    }
    else {
      smoothDampedJacobi_halo_gpu(gpu_grid,l,50,omega);
    }

    if (l>number_of_matrix_free_levels){
      applyStateOperatorSubspaceMatrix_gpu(gpu_grid,l);
    }
    else if (l>0){
      applyStateOperatorSubspace_halo_gpu(gpu_grid,l);
    }
    else {
      applyStateOperator_stencil_xz_d(gpu_grid,l);
    }

    d_to_d_minus_r(gpu_grid,l);

    // project residual down
    if (l < nl-1){
      projectToCoarserGrid_halo_gpu(gpu_grid, l);
    }
  }

  #ifdef USE_CHOLMOD
    #pragma omp parallel num_threads(gpu_grid->num_targets)
    {
      gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
      UPDATE_MG_FROM(node,r,nl-1)
    }

    // Coarsest level
    set_z_to_zero(gpu_grid,nl-1);

    //solveSubspaceMatrix(gc, nl-1, data->bottomSolver, mg[nl-1].r, mg[nl-1].z);
    double tmp_time = omp_get_wtime();
    solveSubspaceMatrix_gpu(gpu_grid,nl,data->bottomSolver);
    *cpu_time += omp_get_wtime()-tmp_time;

    #pragma omp parallel num_threads(gpu_grid->num_targets)
    {
      gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
      UPDATE_MG_TO(node,z,nl-1)
    }
  #endif

  // Level number_of_matrix_free_levels - 1 to 0
  for (int l = nl-last_layer_on_cpu-1;l>=0;l--){
    if (l < nl-1){
      projectToFinerGrid_halo_gpu(gpu_grid, l);
    }

    add_d_to_z(gpu_grid,l);

    smoothDampedJacobi_halo_gpu(gpu_grid,l,nswp,omega);

  }
}

// solves the linear system of Ku = b.
// temperature: medium, accounts for 95% or more of runtime, but this time is
// spent in children functions. The iter loop of this funciton is a good
// candidate for GPU parallel region scope, as it is only performed once every
// design iteration (and thus only 100 times during a program)
void solveStateMG_halo(gpuGrid * gpu_grid,const struct gridContext gc, DTYPE *x, const int nswp,
                       const int nl, const CTYPE tol,
                       solver_ptrs *data, int *finalIter,
                       float *finalRes, CTYPE *b, STYPE *u, const int verbose,double * cpu_time) {

  // Multigrid variables
  mg_ptrs * mg = data->mg;

  // Assembling the inverse matrices needed on the CPU while transferring data to the CPU
  for (int l = number_of_matrix_free_levels; l < nl; l++) {
    // printf("assemble mat l:%i\n", l);
    double tmp_time = omp_get_wtime();
    assembleSubspaceMatrix(gc, l, x, data->coarseMatrices[l], mg[l].invD);
    *cpu_time += omp_get_wtime()-tmp_time;
    #pragma omp parallel num_threads(gpu_grid->num_targets)
    {
      gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
      //UPDATE_CRSMATRIX_TO(node,l)
      MTYPE * vals = node->solver.coarseMatrices[l].vals;
      const int nnz = node->solver.coarseMatrices[l].nnz;
      #pragma omp target update to(vals[:nnz]) device(node->id)
    }
  }

  //Assemling the inverted matrix diagonal for all levels at once
  //Updating data needed in VcyclePreconditioner
  #pragma omp parallel num_threads(gpu_grid->num_targets)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    UPDATE_X_TO(node,x)
    UPDATE_U_TO(node)
  }

  #ifdef USE_CHOLMOD
  double tmp_time = omp_get_wtime();
  factorizeSubspaceMatrix(gc, nl - 1, data->bottomSolver,
                          data->coarseMatrices[nl - 1]);
  *cpu_time += omp_get_wtime()-tmp_time;
  #endif
  
  assembleInvertedMatrixDiagonalSubspace_halo_gpu(gpu_grid,nl);

  #pragma omp parallel num_threads(gpu_grid->num_targets)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    for (int l = 0; l < nl; l++) {
      UPDATE_MG_FROM(node,invD,l)
    }
  }

  CTYPE rhoold = 0.0;
  CTYPE dpr;
  CTYPE alpha;
  CTYPE rho;


    // setup residual vector
    copy_U_to_z(gpu_grid);

    //applyStateOperator_stencil(gc, x, z, r);
    applyStateOperator_stencil_xz_r(gpu_grid,0);

    r_to_F_minus_r(gpu_grid);

    // setup scalars
    const MTYPE omega = 0.6;
    CTYPE bnorm = 0.0;
    F_norm(gpu_grid,&bnorm);

    const int maxIter = 200;
  
    // begin cg loop - usually spans 5 - 300 iterations will be reduced to 5 -
    // 20 iterations once direct solver is included for coarse subproblem.
    for (int iter = 0; iter < maxIter; iter++) {

      // get preconditioned vector
      VcyclePreconditioner(gpu_grid,gc, data, x, nl, omega, nswp,cpu_time);

      rho = 0.0;
      get_rho(gpu_grid,&rho);

      CTYPE beta = (iter == 0) ? 0.0 : rho / rhoold;

      beta_p_plus_z(gpu_grid,beta);

      applyStateOperator_stencil_xp_q(gpu_grid);

      dpr = 0.0;
      p_q_inner_product(gpu_grid,&dpr);
      alpha = rho / dpr;
      rhoold = rho;

      cg_step_u_and_r(gpu_grid,alpha);

      CTYPE rnorm = 0.0;
      r_norm(gpu_grid,&rnorm);
      const CTYPE relres = rnorm / bnorm;

      (*finalIter) = iter;
      (*finalRes) = relres;
      if (verbose == 2){
        printf("it: %i, res=%e\n", iter, relres);
      }

      if (relres < tol)
        break;
    }

  #pragma omp parallel num_threads(gpu_grid->num_targets)
  {
    gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
    UPDATE_U_FROM(node)
  }
}

void allocateSolverData(const struct gridContext gc, const int nl,
                        solver_ptrs *data) {

  // Allocating conjugate gradient buffers
  allocateZeroPaddedStateField(gc, 0, &(*data).cg.p);
  allocateZeroPaddedStateField(gc, 0, &(*data).cg.q);

  // Allocating multigrid buffers
  data->mg = (mg_ptrs *) malloc(sizeof(mg_ptrs)*nl);
  for (int l = 0; l < nl; l++) {
    allocateZeroPaddedStateField(gc, l, &((*data).mg[l].d));
    allocateZeroPaddedStateField(gc, l, &((*data).mg[l].r));
    allocateZeroPaddedStateField(gc, l, &((*data).mg[l].z));
    allocateZeroPaddedStateField_MTYPE(gc, l, &((*data).mg[l].invD));
  }

  // allocate for all levels for easy indces
  (*data).coarseMatrices = malloc(sizeof(struct CSRMatrix) * nl);
  for (int l = number_of_matrix_free_levels; l < nl; l++) {
    allocateSubspaceMatrix(gc, l, &((*data).coarseMatrices[l]));
  }
}

void freeSolverData(solver_ptrs *data, const int nl) {

  // Deallocating conjugate gradient buffers
  free((*data).cg.p);
  free((*data).cg.q);

  // Deallocating multigrid buffers
  for (int l = 0; l < nl; l++) {
    free((*data).mg[l].invD);
    free((*data).mg[l].d);
    free((*data).mg[l].r);
    free((*data).mg[l].z);
  }
  free(data->mg);

  for (int l = number_of_matrix_free_levels; l < nl; l++) {
    freeSubspaceMatrix(&((*data).coarseMatrices[l]));
  }
  free((*data).coarseMatrices);
}
