#include "../include/stencil_optimization.h"

#include "../include/stencil_grid_utility.h"
#include "../include/stencil_methods.h"
#include "../include/stencil_solvers.h"
#include "../include/gpu_grid.h"
#include "../include/gpu_methods.h"

#include <omp.h>

// main function
void top3dmgcg(const uint_fast32_t nelx, const uint_fast32_t nely,
               const uint_fast32_t nelz, const DTYPE volfrac, const DTYPE rmin,
               const uint_fast32_t nl, const float cgtol,
               const uint_fast32_t cgmax,const uint_fast32_t verbose,
               const uint_fast32_t write_result,const uint_fast32_t max_iterations) {

  struct gridContext gridContext;
  gridContext.E0 = 1;
  gridContext.Emin = 1e-6;
  gridContext.nu = 0.3;
  gridContext.nelx = nelx;
  gridContext.nely = nely;
  gridContext.nelz = nelz;
  gridContext.penal = 3; // dummy variable, does nothing

  gridContext.elementSizeX = 0.5;
  gridContext.elementSizeY = 0.5;
  gridContext.elementSizeZ = 0.5;

  const int allocate= 1;
  setupGC(&gridContext, nl,allocate);

  const uint_fast64_t nelem = (gridContext.wrapx - 1) *
                              (gridContext.wrapy - 1) * (gridContext.wrapz - 1);

  CTYPE *F;
  STYPE *U;
  allocateZeroPaddedStateField(gridContext, 0, &F);
  allocateZeroPaddedStateField_STYPE(gridContext, 0, &U);

// for (int i = 1; i < gridContext.nelx + 2; i++)
// for (int k = 1; k < gridContext.nelz + 2; k++)
#pragma omp parallel for
  for (int j = 1; j < gridContext.nely + 2; j++) {
    const int i = gridContext.nelx;
    const int k = 1;

    const uint_fast32_t nidx =
        i * gridContext.wrapy * gridContext.wrapz + gridContext.wrapy * k + j;
    F[3 * nidx + 2] = -1.0;
  }

  F[3 * (gridContext.nelx * gridContext.wrapy * gridContext.wrapz +
         gridContext.wrapy + 1) +
    2] = -0.5;
  F[3 * (gridContext.nelx * gridContext.wrapy * gridContext.wrapz +
         gridContext.wrapy + (gridContext.nely + 1)) +
    2] = -0.5;

  DTYPE *dc = malloc(sizeof(DTYPE) * nelem);
  DTYPE *dv = malloc(sizeof(DTYPE) * nelem);
  DTYPE *xtmp = malloc(sizeof(DTYPE) * nelem);
  DTYPE *x = malloc(sizeof(DTYPE) * nelem);
  DTYPE *xnew = malloc(sizeof(DTYPE) * nelem);
  DTYPE c = 0.0;

#pragma omp parallel for
  for (uint_fast64_t i = 0; i < nelem; i++) {
    xtmp[i] = 0.0;
    x[i] = 0.0;
    dv[i] = 1.0;
  }

#pragma omp parallel for collapse(3)
  for (int i = 1; i < gridContext.nelx + 1; i++)
    for (int k = 1; k < gridContext.nelz + 1; k++)
      for (int j = 1; j < gridContext.nely + 1; j++) {
        const int idx = i * (gridContext.wrapy - 1) * (gridContext.wrapz - 1) +
                        k * (gridContext.wrapy - 1) + j;

        xtmp[idx] = volfrac;
        x[idx] = volfrac;
      }

  applyDensityFilterGradient(gridContext, rmin, dv);

  // allocate needed memory for solver
  solver_ptrs solverData;
  allocateSolverData(gridContext, nl, &solverData);
  #ifdef USE_CHOLMOD
  initializeCholmod(gridContext, nl - 1, &solverData.bottomSolver,
                    solverData.coarseMatrices[nl - 1]);
  #endif


  unsigned int loop = 0;
  float change = 1.0;

  if (verbose > 0){
    printf("OpenMP enabled with %d threads and %d devices.\n",
    omp_get_max_threads(),omp_get_num_devices());
  }

  const int num_devices = omp_get_num_devices();
  gpuGrid gpu_grid = init_gpu_grid(&gridContext,nl,num_devices,x,xtmp,xnew,dv,dc,F,U,&solverData);
  target_enter_data(&gpu_grid,nl);

  const double start_wtime = omp_get_wtime();
  double cpu_time = 0.0;

  DTYPE vol = 0.0;
  /* %% START ITERATION */
  while ((change > 1e-2) && (loop < max_iterations)) {

    loop++;

    int cgiter;
    float cgres;
    const int nswp = 5;
    double time = omp_get_wtime();
    solveStateMG_halo(&gpu_grid,gridContext, x, nswp, nl, cgtol, &solverData, &cgiter,
                      &cgres, F, U,verbose,&cpu_time);
    if (verbose == 2){
      printf("\tsolveStateMG_halo took %4.2lf seconds\n",omp_get_wtime()-time);
    }

    time = omp_get_wtime();
    //getComplianceAndSensetivity_halo(gridContext, x, U, &c, dc);
    #pragma omp parallel num_threads(gpu_grid.num_targets)
    {
      gpuNode * node = &(gpu_grid.targets[omp_get_thread_num()]);
      //UPDATE_X_TO(node,x)
      //UPDATE_U_TO(node)
    }
    getComplianceAndSensetivity_halo_gpu(&gpu_grid,&c);

    #pragma omp parallel num_threads(gpu_grid.num_targets)
    {
      gpuNode * node = &(gpu_grid.targets[omp_get_thread_num()]);
      UPDATE_X_FROM(node,dc)
    }

    if (verbose == 2){
      printf("\tgetComplianceAndSensetivity_halo took %4.2lf seconds\n",omp_get_wtime()-time);
    }

    time = omp_get_wtime();
    applyDensityFilterGradient(gridContext, rmin, dc);
    if (verbose == 2){
      printf("\tapplyDensityFilterGradient took %4.2lf seconds\n",omp_get_wtime()-time);
    }
    cpu_time += omp_get_wtime()-time;

    time = omp_get_wtime();
    vol = 0.0;
    change = 0.0;

    #pragma omp parallel num_threads(gpu_grid.num_targets)
    {
      gpuNode * node = &(gpu_grid.targets[omp_get_thread_num()]);
      //UPDATE_X_TO(node,x)
      //UPDATE_X_TO(node,xtmp)
      //UPDATE_X_TO(node,xnew)
      //UPDATE_X_TO(node,dv)
      UPDATE_X_TO(node,dc)
    }

    conjugate_gradient_step(&gpu_grid,&vol,&change,volfrac);

    #pragma omp parallel num_threads(gpu_grid.num_targets)
    {
      gpuNode * node = &(gpu_grid.targets[omp_get_thread_num()]);
      //UPDATE_X_FROM(node,xnew)
      UPDATE_X_FROM(node,xtmp)
    }
    if (verbose == 2){
      printf("\tconjugate_gradient_step took %4.2lf seconds\n",omp_get_wtime()-time);
    }

    time = omp_get_wtime();
    applyDensityFilter(gridContext, rmin, xtmp, x);
    if (verbose == 2){
      printf("\tapplyDensityFilter took %4.2lf seconds\n",omp_get_wtime()-time);
    }
    cpu_time += omp_get_wtime()-time;

    if (verbose > 0){
      printf("It.:%4i Obj.:%6.3e Vol.:%6.3f ch.:%4.2f relres: %4.2e iters: %4i ",
           loop, c, vol, change, cgres, cgiter);
      printf("time: %6.3f \n", omp_get_wtime() - start_wtime);
    }

  }

  printf("# Iterations | Domain size | Objective value | Volume | Wall time | Threads | Devices | MF levels | Total levels | CPU time\n");
  printf("%12d   %11lu   %15.3e   %6.3f   %9.3f   %7d   %7d   %9d   %12u   %8.3f\n",
          loop,nelx*nely*nelz,c, vol, omp_get_wtime() - start_wtime,omp_get_max_threads(),omp_get_num_devices(),number_of_matrix_free_levels,(unsigned int) nl,cpu_time);
  target_exit_data(&gpu_grid,nl);
  free_gpu_grid(&gpu_grid);


  if (write_result){
    char name1[60];
    snprintf(name1, 60, "out_%d_%d_%d.vtu",(int)nelx, (int)nely, (int)nelz);
    char name2[60];
    snprintf(name2, 60, "out_%d_%d_%d_halo.vtu",(int)nelx, (int)nely, (int)nelz);
    writeDensityVtkFile(nelx, nely, nelz, x,name1);
    printf("Saved result to file %s.\n",name1);
    writeDensityVtkFileWithHalo(nelx, nely, nelz, x,name2);
    printf("Saved result to file %s.\n",name2);
  }

  freeSolverData(&solverData, nl);
  freeGC(&gridContext, nl);
}

// this function acts as a matrix-free replacement for out = (H*rho(:))./Hs
// note that rho and out cannot be the same pointer!
// temperature: cold, called once pr design iteration
void applyDensityFilter(const struct gridContext gc, const DTYPE rmin,
                        const DTYPE *rho, DTYPE *out) {

  const uint32_t nelx = gc.nelx;
  const uint32_t nely = gc.nely;
  const uint32_t nelz = gc.nelz;

  const uint32_t elWrapy = gc.wrapy - 1;
  const uint32_t elWrapz = gc.wrapz - 1;

// loop over elements, usually very large with nelx*nely*nelz = 100.000 or
// more
#pragma omp parallel for collapse(3)
  for (unsigned int i1 = 1; i1 < nelx + 1; i1++)
    for (unsigned int k1 = 1; k1 < nelz + 1; k1++)
      for (unsigned int j1 = 1; j1 < nely + 1; j1++) {

        const uint64_t e1 = i1 * elWrapy * elWrapz + k1 * elWrapy + j1;

        out[e1] = 0.0;
        DTYPE unityScale = 0.0;

        // loop over neighbourhood
        const uint32_t i2max = MIN(i1 + (ceil(rmin) + 1), nelx + 1);
        const uint32_t i2min = MAX(i1 - (ceil(rmin) - 1), 1);

        // the three loops herein are over a constant neighbourhood. typically
        // 4x4x4 or something like that
        for (uint32_t i2 = i2min; i2 < i2max; i2++) {

          const uint32_t k2max = MIN(k1 + (ceil(rmin) + 1), nelz + 1);
          const uint32_t k2min = MAX(k1 - (ceil(rmin) - 1), 1);

          for (uint32_t k2 = k2min; k2 < k2max; k2++) {

            const uint32_t j2max = MIN(j1 + (ceil(rmin) + 1), nely + 1);
            const uint32_t j2min = MAX(j1 - (ceil(rmin) - 1), 1);

            for (uint32_t j2 = j2min; j2 < j2max; j2++) {

              const uint64_t e2 = i2 * elWrapy * elWrapz + k2 * elWrapy + j2;

              const DTYPE filterWeight =
                  MAX(0.0, rmin - sqrt((i1 - i2) * (i1 - i2) +
                                       (j1 - j2) * (j1 - j2) +
                                       (k1 - k2) * (k1 - k2)));

              out[e1] += filterWeight * rho[e2];
              unityScale += filterWeight;
            }
          }
        }

        out[e1] /= unityScale;
      }
}

// this function acts as a matrix-free replacement for v = H* (v(:)./Hs)
// note that rho and out cannot be the same pointer!
// temperature: cold, called twice pr design iteration
void applyDensityFilterGradient(const struct gridContext gc, const DTYPE rmin,
                                DTYPE *v) {
  const uint32_t nelx = gc.nelx;
  const uint32_t nely = gc.nely;
  const uint32_t nelz = gc.nelz;
  const uint32_t elWrapy = gc.wrapy - 1;
  const uint32_t elWrapz = gc.wrapz - 1;
  DTYPE *tmp = malloc(sizeof(DTYPE) * (gc.wrapx - 1) * elWrapy * elWrapz);

// loop over elements, usually very large with nelx*nely*nelz = 100.000 or
// more
#pragma omp parallel for collapse(3)
  for (unsigned int i1 = 1; i1 < nelx + 1; i1++)
    for (unsigned int k1 = 1; k1 < nelz + 1; k1++)
      for (unsigned int j1 = 1; j1 < nely + 1; j1++) {

        const uint64_t e1 = i1 * elWrapy * elWrapz + k1 * elWrapy + j1;

        DTYPE unityScale = 0.0;

        // loop over neighbourhood
        const uint32_t i2max = MIN(i1 + (ceil(rmin) + 1), nelx + 1);
        const uint32_t i2min = MAX(i1 - (ceil(rmin) - 1), 1);

        // the three loops herein are over a constant neighbourhood. typically
        // 4x4x4 or something like that
        for (uint32_t i2 = i2min; i2 < i2max; i2++) {

          const uint32_t k2max = MIN(k1 + (ceil(rmin) + 1), nelz + 1);
          const uint32_t k2min = MAX(k1 - (ceil(rmin) - 1), 1);

          for (uint32_t k2 = k2min; k2 < k2max; k2++) {

            const uint32_t j2max = MIN(j1 + (ceil(rmin) + 1), nely + 1);
            const uint32_t j2min = MAX(j1 - (ceil(rmin) - 1), 1);

            for (uint32_t j2 = j2min; j2 < j2max; j2++) {

              const DTYPE filterWeight =
                  MAX(0.0, rmin - sqrt((i1 - i2) * (i1 - i2) +
                                       (j1 - j2) * (j1 - j2) +
                                       (k1 - k2) * (k1 - k2)));

              unityScale += filterWeight;
            }
          }
        }

        tmp[e1] = v[e1] / unityScale;
      }

// loop over elements, usually very large with nelx*nely*nelz = 100.000 or
// more
#pragma omp parallel for collapse(3)
  for (unsigned int i1 = 1; i1 < nelx + 1; i1++)
    for (unsigned int k1 = 1; k1 < nelz + 1; k1++)
      for (unsigned int j1 = 1; j1 < nely + 1; j1++) {

        const uint64_t e1 = i1 * elWrapy * elWrapz + k1 * elWrapy + j1;

        v[e1] = 0.0;

        // loop over neighbourhood
        const uint32_t i2max = MIN(i1 + (ceil(rmin) + 1), nelx + 1);
        const uint32_t i2min = MAX(i1 - (ceil(rmin) - 1), 1);

        // the three loops herein are over a constant neighbourhood. typically
        // 4x4x4 or something like that
        for (uint32_t i2 = i2min; i2 < i2max; i2++) {

          const uint32_t k2max = MIN(k1 + (ceil(rmin) + 1), nelz + 1);
          const uint32_t k2min = MAX(k1 - (ceil(rmin) - 1), 1);

          for (uint32_t k2 = k2min; k2 < k2max; k2++) {

            const uint32_t j2max = MIN(j1 + (ceil(rmin) + 1), nely + 1);
            const uint32_t j2min = MAX(j1 - (ceil(rmin) - 1), 1);

            for (uint32_t j2 = j2min; j2 < j2max; j2++) {

              const uint64_t e2 = i2 * elWrapy * elWrapz + k2 * elWrapy + j2;

              const DTYPE filterWeight =
                  MAX(0.0, rmin - sqrt((i1 - i2) * (i1 - i2) +
                                       (j1 - j2) * (j1 - j2) +
                                       (k1 - k2) * (k1 - k2)));

              v[e1] += filterWeight * tmp[e2];
            }
          }
        }
      }

  free(tmp);
}

// writes a file with a snapshot of the density field (x,xPhys), can be opened
// with paraview temperature: very cold, usually called once only
void writeDensityVtkFile(const int nelx, const int nely, const int nelz,
                         const DTYPE *densityArray, const char *filename) {
  int nx = nelx + 1;
  int ny = nely + 1;
  int nz = nelz + 1;

  const int paddingy =
      (STENCIL_SIZE_Y - ((nely + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
  const int paddingz =
      (STENCIL_SIZE_Z - ((nelz + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;

  const int elWrapy = nely + paddingy + 3 - 1;
  const int elWrapz = nelz + paddingz + 3 - 1;

  int numberOfNodes = nx * ny * nz;
  int numberOfElements = nelx * nely * nelz;

  FILE *fid = fopen(filename, "w");

  // write header
  fprintf(fid, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" "
               "byte_order=\"LittleEndian\">\n");
  fprintf(fid, "<UnstructuredGrid>\n");
  fprintf(fid, "<Piece NumberOfPoints=\"%i\" NumberOfCells=\"%i\">\n",
          numberOfNodes, numberOfElements);

  // points
  fprintf(fid, "<Points>\n");
  fprintf(fid,
          "<DataArray type=\"Float32\" NumberOfComponents=\"%i\" "
          "format=\"ascii\">\n",
          3);
  for (int i = 0; i < nx; i++)
    for (int k = 0; k < nz; k++)
      for (int j = 0; j < ny; j++)
        fprintf(fid, "%e %e %e\n", (float)i, (float)j, (float)k);
  fprintf(fid, "</DataArray>\n");
  fprintf(fid, "</Points>\n");

  fprintf(fid, "<Cells>\n");

  fprintf(
      fid,
      "<DataArray type=\"Int32\" Name=\"connectivity\" format= \"ascii\">\n");
  for (int i = 0; i < nelx; i++)
    for (int k = 0; k < nelz; k++)
      for (int j = 0; j < nely; j++) {
        const int nx_1 = i;
        const int nx_2 = i + 1;
        const int nz_1 = k;
        const int nz_2 = k + 1;
        const int ny_1 = j;
        const int ny_2 = j + 1;
        fprintf(fid, "%d %d %d %d %d %d %d %d\n",
                nx_1 * ny * nz + nz_1 * ny + ny_2,
                nx_2 * ny * nz + nz_1 * ny + ny_2,
                nx_2 * ny * nz + nz_1 * ny + ny_1,
                nx_1 * ny * nz + nz_1 * ny + ny_1,
                nx_1 * ny * nz + nz_2 * ny + ny_2,
                nx_2 * ny * nz + nz_2 * ny + ny_2,
                nx_2 * ny * nz + nz_2 * ny + ny_1,
                nx_1 * ny * nz + nz_2 * ny + ny_1);
      }

  fprintf(fid, "</DataArray>\n");

  fprintf(fid,
          "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");
  for (int i = 1; i < numberOfElements + 1; i++)
    fprintf(fid, "%d\n", i * 8);
  fprintf(fid, "</DataArray>\n");

  fprintf(fid, "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
  for (int i = 0; i < numberOfElements; i++)
    fprintf(fid, "%d\n", 12);
  fprintf(fid, "</DataArray>\n");
  fprintf(fid, "</Cells>\n");

  fprintf(fid, "<CellData>\n");
  fprintf(fid, "<DataArray type=\"Float32\" NumberOfComponents=\"1\" "
               "Name=\"density\" format=\"ascii\">\n");
  for (unsigned int i1 = 1; i1 < nelx + 1; i1++)
    for (unsigned int k1 = 1; k1 < nelz + 1; k1++)
      for (unsigned int j1 = 1; j1 < nely + 1; j1++) {
        const uint64_t idx = i1 * elWrapy * elWrapz + k1 * elWrapy + j1;
        fprintf(fid, "%e\n", densityArray[idx]);
      }
  fprintf(fid, "</DataArray>\n");
  fprintf(fid, "</CellData>\n");

  fprintf(fid, "</Piece>\n");
  fprintf(fid, "</UnstructuredGrid>\n");
  fprintf(fid, "</VTKFile>\n");

  fclose(fid);
}

// writes a file with a snapshot of the density field (x,xPhys), can be opened
// with paraview temperature: very cold, usually called once only
void writeDensityVtkFileWithHalo(const int nelx, const int nely, const int nelz,
                                 const DTYPE *densityArray,
                                 const char *filename) {

  const int paddingx =
      (STENCIL_SIZE_X - ((nelx + 1) % STENCIL_SIZE_X)) % STENCIL_SIZE_X;
  const int paddingy =
      (STENCIL_SIZE_Y - ((nely + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
  const int paddingz =
      (STENCIL_SIZE_Z - ((nelz + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;

  const int wrapx = nelx + paddingx + 3;
  const int wrapy = nely + paddingy + 3;
  const int wrapz = nelz + paddingz + 3;

  const int elWrapx = wrapx - 1;
  const int elWrapy = wrapy - 1;
  const int elWrapz = wrapz - 1;

  int numberOfNodes = wrapx * wrapy * wrapz;
  int numberOfElements = elWrapx * elWrapy * elWrapz;

  FILE *fid = fopen(filename, "w");

  // write header
  fprintf(fid, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" "
               "byte_order=\"LittleEndian\">\n");
  fprintf(fid, "<UnstructuredGrid>\n");
  fprintf(fid, "<Piece NumberOfPoints=\"%i\" NumberOfCells=\"%i\">\n",
          numberOfNodes, numberOfElements);

  // points
  fprintf(fid, "<Points>\n");
  fprintf(fid,
          "<DataArray type=\"Float32\" NumberOfComponents=\"%i\" "
          "format=\"ascii\">\n",
          3);
  for (int i = 0; i < wrapx; i++)
    for (int k = 0; k < wrapz; k++)
      for (int j = 0; j < wrapy; j++)
        fprintf(fid, "%e %e %e\n", (float)i, (float)j, (float)k);
  fprintf(fid, "</DataArray>\n");
  fprintf(fid, "</Points>\n");

  fprintf(fid, "<Cells>\n");

  fprintf(
      fid,
      "<DataArray type=\"Int32\" Name=\"connectivity\" format= \"ascii\">\n");
  for (int i = 0; i < elWrapx; i++)
    for (int k = 0; k < elWrapz; k++)
      for (int j = 0; j < elWrapy; j++) {
        const int nx_1 = i;
        const int nx_2 = i + 1;
        const int nz_1 = k;
        const int nz_2 = k + 1;
        const int ny_1 = j;
        const int ny_2 = j + 1;
        fprintf(fid, "%d %d %d %d %d %d %d %d\n",
                nx_1 * wrapy * wrapz + nz_1 * wrapy + ny_2,
                nx_2 * wrapy * wrapz + nz_1 * wrapy + ny_2,
                nx_2 * wrapy * wrapz + nz_1 * wrapy + ny_1,
                nx_1 * wrapy * wrapz + nz_1 * wrapy + ny_1,
                nx_1 * wrapy * wrapz + nz_2 * wrapy + ny_2,
                nx_2 * wrapy * wrapz + nz_2 * wrapy + ny_2,
                nx_2 * wrapy * wrapz + nz_2 * wrapy + ny_1,
                nx_1 * wrapy * wrapz + nz_2 * wrapy + ny_1);
      }

  fprintf(fid, "</DataArray>\n");

  fprintf(fid,
          "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");
  for (int i = 1; i < numberOfElements + 1; i++)
    fprintf(fid, "%d\n", i * 8);
  fprintf(fid, "</DataArray>\n");

  fprintf(fid, "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
  for (int i = 0; i < numberOfElements; i++)
    fprintf(fid, "%d\n", 12);
  fprintf(fid, "</DataArray>\n");
  fprintf(fid, "</Cells>\n");

  fprintf(fid, "<CellData>\n");
  fprintf(fid, "<DataArray type=\"Float32\" NumberOfComponents=\"1\" "
               "Name=\"density\" format=\"ascii\">\n");
  for (unsigned int i1 = 0; i1 < elWrapx; i1++)
    for (unsigned int k1 = 0; k1 < elWrapz; k1++)
      for (unsigned int j1 = 0; j1 < elWrapy; j1++) {
        const uint64_t idx = i1 * elWrapy * elWrapz + k1 * elWrapy + j1;
        fprintf(fid, "%e\n", densityArray[idx]);
      }
  fprintf(fid, "</DataArray>\n");
  fprintf(fid, "</CellData>\n");

  fprintf(fid, "</Piece>\n");
  fprintf(fid, "</UnstructuredGrid>\n");
  fprintf(fid, "</VTKFile>\n");

  fclose(fid);
}
