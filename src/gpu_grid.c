#include "../include/gpu_grid.h"
#include <omp.h>
#include <math.h>

void init_mg_dims(mg_dims * dims,const int nl){
    dims->nelx = (uint_fast32_t *) malloc(nl*sizeof(uint_fast32_t));
    dims->nely = (uint_fast32_t *) malloc(nl*sizeof(uint_fast32_t));
    dims->nelz = (uint_fast32_t *) malloc(nl*sizeof(uint_fast32_t));
    dims->wrapx = (uint_fast32_t *) malloc(nl*sizeof(uint_fast32_t));
    dims->wrapy = (uint_fast32_t *) malloc(nl*sizeof(uint_fast32_t));
    dims->wrapz = (uint_fast32_t *) malloc(nl*sizeof(uint_fast32_t));
    dims->offset = (uint_fast32_t *) malloc(nl*sizeof(uint_fast32_t));
    dims->length = (uint_fast32_t *) malloc(nl*sizeof(uint_fast32_t));
    dims->ndof = (uint_fast32_t *) malloc(nl*sizeof(uint_fast32_t));
    dims->pKEsize = (uint_fast32_t *) malloc(nl*sizeof(uint_fast32_t));
}

void free_mg_dims(mg_dims * dims){
    free(dims->nelx);
    dims->nelx = NULL;
    free(dims->nely);
    dims->nely = NULL;
    free(dims->nelz);
    dims->nelz = NULL;
    free(dims->wrapx);
    dims->wrapx = NULL;
    free(dims->wrapy);
    dims->wrapy = NULL;
    free(dims->wrapz);
    dims->wrapz = NULL;
    free(dims->offset);
    dims->offset = NULL;
    free(dims->length);
    dims->length = NULL;
    free(dims->ndof);
    dims->ndof = NULL;
    free(dims->pKEsize);
    dims->pKEsize = NULL;
}

void compute_overlap(gpuNode * node,const int l){
    node->dims.offset[l] = (node->prev != NULL) ? 2*NODE_SIZE(node,l) : 0;
    node->dims.ndof[l] = 3*node->dims.wrapx[l]*node->dims.wrapy[l]*node->dims.wrapz[l];
    const int offset_right = (node->next != NULL) ? NODE_SIZE(node,l) : 0;
    node->dims.length[l] = node->dims.ndof[l] - (node->dims.offset[l] + offset_right);
    const int ncell = pow(2, l);
    node->dims.pKEsize[l] = 24 * 24 * ncell * ncell * ncell;
}

void compute_dims(struct gridContext * gc,gpuNode * node,const int nl){
    for (int l=0;l<nl;l++){
        compute_num_cells(gc,l,&(node->dims.nelx[l]),&(node->dims.nely[l]),&(node->dims.nelz[l]));
        compute_wrapping(gc,l,&(node->dims.wrapx[l]),&(node->dims.wrapy[l]),&(node->dims.wrapz[l]));
        compute_overlap(node,l);
        if (DEBUGGING_MODE) {
            printf("Grid %d has dimensions:\n\tElements: (%4lu,%4lu,%4lu)\n\tWrapping: (%4lu,%4lu,%4lu)\n\tOffset: %lu\n\tLength: %lu \n\tNdof %lu\n",
                l,node->dims.nelx[l],node->dims.nely[l],node->dims.nelz[l],
                node->dims.wrapx[l],node->dims.wrapy[l],node->dims.wrapz[l],
                node->dims.offset[l],node->dims.length[l],node->dims.ndof[l]);
        }
    }
}

void partition_multigrid(gpuNode * node,const int nl, solver_ptrs * solver){
    for (int l=0;l<nl;l++){
        if (node->prev == NULL){
            node->solver.mg[l].invD = solver->mg[l].invD;
            node->solver.mg[l].d = solver->mg[l].d;
            node->solver.mg[l].r = solver->mg[l].r;
            node->solver.mg[l].z = solver->mg[l].z;
        }
        else {
            gpuNode * prev = node->prev;
            const int offset = prev->dims.ndof[l]-3*3*prev->dims.wrapy[l]*prev->dims.wrapz[l];
            node->solver.mg[l].invD = &(prev->solver.mg[l].invD[offset]);
            node->solver.mg[l].d = &(prev->solver.mg[l].d[offset]);
            node->solver.mg[l].r = &(prev->solver.mg[l].r[offset]);
            node->solver.mg[l].z = &(prev->solver.mg[l].z[offset]);
        }
    }
}

void partition_coarseMatrices(gpuNode * node, struct CSRMatrix * M)
{
    node->solver.coarseMatrices = M;
}

void partition_conjugate_gradient(gpuNode * node, solver_ptrs * solver,CTYPE *F,STYPE *U){
    if (node->prev == NULL){
        node->solver.cg.p = solver->cg.p;
        node->solver.cg.q = solver->cg.q;
        node->U = U;
        node->F = F;
    }
    else {
        gpuNode * prev = node->prev;
        const int offset = prev->dims.ndof[0]-3*3*prev->dims.wrapy[0]*prev->dims.wrapz[0];
        node->solver.cg.p = &(prev->solver.cg.p[offset]);
        node->solver.cg.q = &(prev->solver.cg.q[offset]);
        node->U = &(prev->U[offset]);
        node->F = &(prev->F[offset]);
    }
}

gpuGrid init_gpu_grid(struct gridContext * gc,const int nl, const int num_devices,
                    DTYPE *x,DTYPE *xtmp,DTYPE *xnew,DTYPE *dv,DTYPE *dc,
                    CTYPE *F,STYPE *U,solver_ptrs * solver){
    gpuGrid gpu_grid;
    gpu_grid.targets = malloc(num_devices*sizeof(gpuNode));
    gpu_grid.num_targets = num_devices;
    gpu_grid.gc = gc;
    //int offset_uf = 0;
    int offset_x = 0;
    const int nelx = gc->nelx;
    const int nelx_block = (int) nelx/num_devices;
    for (int i=0;i<num_devices;i++){
        gpuNode * node = &(gpu_grid.targets[i]);
        node->id = i;

        // Is the gpu the first, last or a middle GPU in the grid?
        node->prev = (i > 0) ? &(gpu_grid.targets[i-1]) : NULL;
        node->next = (i < num_devices-1) ? &(gpu_grid.targets[i+1]) : NULL;

        // Copying constants in the grid context
        node->gc = *gc;

        // Blocking over number of elements in the x dimension
        int nelx_local = (i == num_devices-1) ? nelx-i*nelx_block : nelx_block;
        node->gc.nelx = nelx_local;
        setupGC(&(node->gc),nl,0);

        // Initializing multi grid data structure
        init_mg_dims(&(node->dims),nl);
        compute_dims(&(node->gc),node,nl);
        
        node->solver.mg = (mg_ptrs *) malloc(sizeof(mg_ptrs)*nl);
        partition_multigrid(node,nl,solver);

        // Conjugate gradient data structure
        partition_conjugate_gradient(node,solver,F,U);

        partition_coarseMatrices(node,solver->coarseMatrices);

        // Prtitioning of global arrays
        const int designSize = (node->dims.wrapx[0] - 1) * (node->dims.wrapy[0] - 1) * (node->dims.wrapz[0] - 1);
        node->design_size = designSize;

        if (DEBUGGING_MODE){
            printf("node->gc: nel: (%4lu,%4lu,%4lu) wrap: (%4lu,%4lu,%4lu)\n",node->gc.nelx,node->gc.nely,node->gc.nelz,node->gc.wrapx,node->gc.wrapy,node->gc.wrapz);
        }

        node->x = &x[offset_x];
        node->xtmp = &xtmp[offset_x];
        node->xnew = &xnew[offset_x];
        node->dv = &dv[offset_x];
        node->dc = &dc[offset_x];

        // Allocating buffer for exchanging boundaries in mg
        node->CTYPE_buffer = (CTYPE * ) malloc(2*(LENGTH_LEFT+LENGTH_RIGHT)*NODE_SIZE(node,0)*sizeof(CTYPE));
        node->MTYPE_buffer = (MTYPE * ) malloc(2*(LENGTH_LEFT+LENGTH_RIGHT)*NODE_SIZE(node,0)*sizeof(MTYPE));

        // Incrementing position in conjugate gradient grid / multi grid level 0
        //offset_uf += node->dims.ndof[0]-3*3*node->dims.wrapy[0]*node->dims.wrapz[0];
        offset_x += node->design_size - 2*(node->dims.wrapy[0] - 1) * (node->dims.wrapz[0] - 1);

        // The offset in the cholmod layer is different
        #ifdef USE_CHOLMOD
        // int rhs_offset = 0;
        // gpuNode * prev = node->prev;
        // while(prev != NULL){
        //     rhs_offset += 3*(prev->dims.nelx[nl-1])*(prev->dims.nely[nl-1]+1)*(prev->dims.nelz[nl-1]+1);
        //     prev = prev->prev;
        // }
        node->rhs = (double *) (solver->bottomSolver.rhs->x);
        node->sol = NULL;
        #endif
    }
    return gpu_grid;
}

void free_gpu_grid(gpuGrid * gpu_grid) {
    for (int i=0;i<gpu_grid->num_targets;i++){
        gpuNode * node = &(gpu_grid->targets[i]);
        free_mg_dims(&(node->dims));
        free(node->solver.mg);
        free(node->CTYPE_buffer);
        free(node->MTYPE_buffer);
    }
}

void target_enter_data(gpuGrid * gpu_grid,const int nl){
    #pragma omp parallel num_threads(gpu_grid->num_targets)
    {
        gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
        int ndof = node->dims.ndof[0];
        #pragma omp target enter data map(to:\
            node->solver.cg.p[:ndof],\
            node->solver.cg.q[:ndof]\
            ) device(node->id) 
        if (DEBUGGING_MODE) printf("Mapped cg to gpu %d\n",node->id);

        #pragma omp target enter data map(to:\
            node->U[:ndof],\
            node->F[:ndof],\
            node->x[:node->design_size],\
            node->xtmp[:node->design_size],\
            node->xnew[:node->design_size],\
            node->dv[:node->design_size],\
            node->dc[:node->design_size]\
            ) device(node->id) 
        if (DEBUGGING_MODE) printf("Mapped U,F, and x to gpu %d\n",node->id);

        // Mapping buffers
        const int buffer_size = 6*NODE_SIZE(node,0);
        #pragma omp target enter data map(to:\
            node->CTYPE_buffer[:buffer_size],\
            node->MTYPE_buffer[:buffer_size]) device(node->id) 
        if (node->prev != NULL){
            #pragma omp target enter data map(to:\
                node->CTYPE_buffer[:buffer_size],\
                node->MTYPE_buffer[:buffer_size]) device(node->prev->id) 
        }
        if (node->next != NULL){
            #pragma omp target enter data map(to:\
                node->CTYPE_buffer[:buffer_size],\
                node->MTYPE_buffer[:buffer_size]) device(node->next->id) 
        }

        for (int l=0;l<nl;l++){
            const int ndofmg = node->dims.ndof[l];
            #pragma omp target enter data map(to:\
                node->solver.mg[l].invD[:ndofmg],\
                node->solver.mg[l].d[:ndofmg],\
                node->solver.mg[l].r[:ndofmg],\
                node->solver.mg[l].z[:ndofmg]\
                ) device(node->id)
            if (DEBUGGING_MODE) printf("Mapped mg level %d to gpu %d\n",l,node->id);
            
            const uint_fast32_t n = node->gc.fixedDofs[l].n;
            const uint_fast32_t *pidx = node->gc.fixedDofs[l].idx;
            MTYPE *pKE = node->gc.precomputedKE[l];
            const int nnz = node->solver.coarseMatrices[l].nnz;
            const int nrows = node->solver.coarseMatrices[l].nrows;
            int * rowOffsets = node->solver.coarseMatrices[l].rowOffsets;
            MTYPE * vals = node->solver.coarseMatrices[l].vals;
            #pragma omp target enter data map(to :\
                pidx[:n],\
                pKE[:node->dims.pKEsize[l]]\
                ) device(node->id) 
            if (l >= number_of_matrix_free_levels){
                #pragma omp target enter data map(to :\
                    rowOffsets[:nrows+1],\
                    vals[:nnz]\
                    ) device(node->id) 
            }
            if (DEBUGGING_MODE) printf("Mapped pKE level %d to gpu %d\n",l,node->id);
            #ifdef USE_CHOLMOD
            if (l == nl-1){
                #pragma omp target enter data map(to :node->rhs[:nrows]) device(node->id) 
            }
            #endif
        }
    }
}

void target_exit_data(gpuGrid * gpu_grid,const int nl){
    #pragma omp parallel num_threads(gpu_grid->num_targets)
    {
        gpuNode * node = &(gpu_grid->targets[omp_get_thread_num()]);
        int ndof = node->dims.ndof[0];
        #pragma omp target exit data map(delete:\
            node->solver.cg.p[:ndof],\
            node->solver.cg.q[:ndof]\
            ) device(node->id) 
        if (DEBUGGING_MODE) printf("Unmapped cg from gpu %d\n",node->id);

        #pragma omp target exit data map(delete:\
            node->U[:ndof],\
            node->F[:ndof],\
            node->x[:node->design_size],\
            node->xtmp[:node->design_size],\
            node->xnew[:node->design_size],\
            node->dv[:node->design_size],\
            node->dc[:node->design_size]\
            ) device(node->id) 
        if (DEBUGGING_MODE) printf("Unmapped U,F, and x from gpu %d\n",node->id);

        // Mapping buffers
        const int buffer_size = 6*NODE_SIZE(node,0);
        #pragma omp target exit data map(delete:\
            node->CTYPE_buffer[:buffer_size],\
            node->MTYPE_buffer[:buffer_size]) device(node->id) 
        if (node->prev != NULL){
            #pragma omp target exit data map(delete:\
                node->CTYPE_buffer[:buffer_size],\
                node->MTYPE_buffer[:buffer_size]) device(node->prev->id) 
        }
        if (node->next != NULL){
            #pragma omp target exit data map(delete:\
                node->CTYPE_buffer[:buffer_size],\
                node->MTYPE_buffer[:buffer_size]) device(node->next->id) 
        }

        for (int l=0;l<nl;l++){
            const int ndofmg = node->dims.ndof[l];
            #pragma omp target exit data map(delete:\
                node->solver.mg[l].invD[:ndofmg],\
                node->solver.mg[l].d[:ndofmg],\
                node->solver.mg[l].r[:ndofmg],\
                node->solver.mg[l].z[:ndofmg]\
                ) device(node->id)
            if (DEBUGGING_MODE) printf("Unmapped mg level %d from gpu %d\n",l,node->id);
            
            const uint_fast32_t n = node->gc.fixedDofs[l].n;
            const uint_fast32_t *pidx = node->gc.fixedDofs[l].idx;
            MTYPE *pKE = node->gc.precomputedKE[l];
            const int nnz = node->solver.coarseMatrices[l].nnz;
            const int nrows = node->solver.coarseMatrices[l].nrows;
            int * rowOffsets = node->solver.coarseMatrices[l].rowOffsets;
            MTYPE * vals = node->solver.coarseMatrices[l].vals;
            #pragma omp target exit data map(delete :\
                pidx[:n],\
                pKE[:node->dims.pKEsize[l]]\
                ) device(node->id)
            if (l >= number_of_matrix_free_levels){
                #pragma omp target exit data map(delete :\
                    rowOffsets[:nrows+1],\
                    vals[:nnz]\
                    ) device(node->id)
            }
            if (DEBUGGING_MODE) printf("Unmapped pKE level %d from gpu %d\n",l,node->id);
            #ifdef USE_CHOLMOD
            if (l == nl-1){
                #pragma omp target exit data map(delete :node->rhs[:nrows]) device(node->id)
            }
            #endif
        }
    }
}

void init_z_to_zero(gpuNode * node,const int l){
    CTYPE *z = node->solver.mg[l].z;
    const uint_fast32_t start = node->dims.offset[l];
    const uint_fast32_t end = start + node->dims.length[l];
    #pragma omp target teams distribute parallel for schedule(static) device(node->id)
    for(int i = start;i<end;i++){
        z[i] = 0.0;
    }
}

void r_minus_d(gpuNode * node,const int l){
    CTYPE *d = node->solver.mg[l].d;
    CTYPE *r = node->solver.mg[l].r;
    const uint_fast32_t start = node->dims.offset[l];
    const uint_fast32_t end = start + node->dims.length[l];
    #pragma omp target teams distribute parallel for schedule(static) device(node->id)
    for(int i = start;i<end;i++){
        d[i] = r[i]-d[i];
    }
}

void send_buffer_to_prev_CTYPE(gpuNode * node,CTYPE * d, const int l)
{
  const uint_fast32_t size = NODE_SIZE(node,l);
  CTYPE * prev_buffer = node->prev->CTYPE_buffer;
  #pragma omp target teams distribute parallel for schedule(static) device(node->id)
  for (int i = 0;i<3*size;i++){
    prev_buffer[3*size+i] = d[i];
  }
  #pragma omp target update from(prev_buffer[3*size:3*size]) device(node->id)
  #pragma omp target update to(prev_buffer[3*size:3*size]) device(node->prev->id)
}

void send_buffer_to_next_CTYPE(gpuNode * node,CTYPE * d, const int l)
{
  const uint_fast32_t size = NODE_SIZE(node,l);
  const uint_fast32_t ndof = node->dims.ndof[l];
  CTYPE * next_buffer = node->next->CTYPE_buffer;
  #pragma omp target teams distribute parallel for schedule(static) device(node->id)
  for (int i = 0;i<3*size;i++){
    next_buffer[i] = d[ndof-3*size+i];
  }
  #pragma omp target update from(next_buffer[:3*size]) device(node->id)
  #pragma omp target update to(next_buffer[:3*size]) device(node->next->id)
}

void sum_west_halo_CTYPE(gpuNode * node,CTYPE * d, const int l)
{
  const uint_fast32_t size = NODE_SIZE(node,l);
  CTYPE * buffer = node->CTYPE_buffer;
  #pragma omp target teams distribute parallel for schedule(static) device(node->id)
  for (int i = 0;i<3*size;i++){
    d[i] += buffer[i];
  }
}

void sum_east_halo_CTYPE(gpuNode * node,CTYPE * d, const int l)
{
  const uint_fast32_t size = NODE_SIZE(node,l);
  const uint_fast32_t ndof = node->dims.ndof[l];
  CTYPE * buffer = node->CTYPE_buffer;
  #pragma omp target teams distribute parallel for schedule(static) device(node->id)
  for (int i = 0;i<3*size;i++){
    d[ndof-3*size+i] += buffer[3*size+i];
  }
}

void send_buffer_to_prev_MTYPE(gpuNode * node,MTYPE * d, const int l)
{
  const uint_fast32_t size = NODE_SIZE(node,l);
  MTYPE * prev_buffer = node->prev->MTYPE_buffer;
  #pragma omp target teams distribute parallel for schedule(static) device(node->id)
  for (int i = 0;i<3*size;i++){
    prev_buffer[3*size+i] = d[i];
  }
  #pragma omp target update from(prev_buffer[3*size:3*size]) device(node->id)
  #pragma omp target update to(prev_buffer[3*size:3*size]) device(node->prev->id)
}

void send_buffer_to_next_MTYPE(gpuNode * node,MTYPE * d, const int l)
{
  const uint_fast32_t size = NODE_SIZE(node,l);
  const uint_fast32_t ndof = node->dims.ndof[l];
  MTYPE * next_buffer = node->next->MTYPE_buffer;
  #pragma omp target teams distribute parallel for schedule(static) device(node->id)
  for (int i = 0;i<3*size;i++){
    next_buffer[i] = d[ndof-3*size+i];
  }
  #pragma omp target update from(next_buffer[:3*size]) device(node->id)
  #pragma omp target update to(next_buffer[:3*size]) device(node->next->id)
}

void sum_west_halo_MTYPE(gpuNode * node,MTYPE * d, const int l)
{
  const uint_fast32_t size = NODE_SIZE(node,l);
  MTYPE * buffer = node->MTYPE_buffer;
  #pragma omp target teams distribute parallel for schedule(static) device(node->id)
  for (int i = 0;i<3*size;i++){
    d[i] += buffer[i];
  }
}

void sum_east_halo_MTYPE(gpuNode * node,MTYPE * d, const int l)
{
  const uint_fast32_t size = NODE_SIZE(node,l);
  const uint_fast32_t ndof = node->dims.ndof[l];
  MTYPE * buffer = node->MTYPE_buffer;
  #pragma omp target teams distribute parallel for schedule(static) device(node->id)
  for (int i = 0;i<3*size;i++){
    d[ndof-3*size+i] += buffer[3*size+i];
  }
}
