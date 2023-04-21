#ifndef GPU_GRID_LIB
#define GPU_GRID_LIB

#include <stdio.h>
#include <stdlib.h>
#include "definitions.h"
#include "stencil_grid_utility.h"
#include "stencil_solvers.h"

// Defining what to send from each GPU to its neighbors
#define OFFSET_RIGHT 3
#define LENGTH_RIGHT 2
#define LENGTH_LEFT 1
#define OFFSET_LEFT 2

// Defining macros generally needed to declare pragmas dynamically
#define STRINGIFY(a) #a
#define NODE_SIZE(node,l) (3*(*(node->dims.wrapy+l))*(*(node->dims.wrapz+l)))

// Defining macros for updating boundary values to the next GPU
#define MG_NEXT_DEV_2_HOST(node,size,U,L,...) _Pragma(STRINGIFY(omp target update from(node->solver.mg[L].U[*(node->dims.ndof+L)-OFFSET_RIGHT*size:LENGTH_RIGHT*size]) device(node->id) __VA_ARGS__))
#define MG_NEXT_HOST_2_DEV(node,size,U,L,...) _Pragma(STRINGIFY(omp target update to(node->solver.mg[L].U[*(node->dims.ndof+L)-OFFSET_RIGHT*size:LENGTH_RIGHT*size]) device(node->next->id) __VA_ARGS__))
#define SEND_MG_NEXT(node,u,l,...) \
    MG_NEXT_DEV_2_HOST(node,NODE_SIZE(node,l),u,l) \
    MG_NEXT_HOST_2_DEV(node,NODE_SIZE(node,l),u,l,__VA_ARGS__)

// Defining macros for updating boundary values to the previous GPU
#define MG_PREV_DEV_2_HOST(node,size,U,L,...) _Pragma(STRINGIFY(omp target update from(node->solver.mg[L].U[OFFSET_LEFT*size:LENGTH_LEFT*size]) device(node->id) __VA_ARGS__))
#define MG_PREV_HOST_2_DEV(node,size,U,L,...) _Pragma(STRINGIFY(omp target update to(node->solver.mg[L].U[OFFSET_LEFT*size:LENGTH_LEFT*size]) device(node->prev->id) __VA_ARGS__))
#define SEND_MG_PREV(node,u,l,...) \
    MG_PREV_DEV_2_HOST(node,NODE_SIZE(node,l),u,l) \
    MG_PREV_HOST_2_DEV(node,NODE_SIZE(node,l),u,l,__VA_ARGS__)

#define EXCHANGE_MG(node,U,L,...) \
    if(node->next != NULL){ \
        SEND_MG_NEXT(node,U,L,__VA_ARGS__) \
    }\
    if(node->prev != NULL){ \
        SEND_MG_PREV(node,U,L,__VA_ARGS__) \
    }

// Defining macros for updating boundary values to the next GPU
#define CG_NEXT_DEV_2_HOST(node,size,U,...) _Pragma(STRINGIFY(omp target update from(node->solver.cg.U[*(node->dims.ndof)-OFFSET_RIGHT*size:LENGTH_RIGHT*size]) device(node->id) __VA_ARGS__))
#define CG_NEXT_HOST_2_DEV(node,size,U,...) _Pragma(STRINGIFY(omp target update to(node->solver.cg.U[*(node->dims.ndof)-OFFSET_RIGHT*size:LENGTH_RIGHT*size]) device(node->next->id) __VA_ARGS__))
#define SEND_CG_NEXT(node,u,...) \
    CG_NEXT_DEV_2_HOST(node,NODE_SIZE(node,0),u) \
    CG_NEXT_HOST_2_DEV(node,NODE_SIZE(node,0),u,__VA_ARGS__)

// Defining macros for updating boundary values to the previous GPU
#define CG_PREV_DEV_2_HOST(node,size,U,...) _Pragma(STRINGIFY(omp target update from(node->solver.cg.U[OFFSET_LEFT*size:LENGTH_LEFT*size]) device(node->id) __VA_ARGS__))
#define CG_PREV_HOST_2_DEV(node,size,U,...) _Pragma(STRINGIFY(omp target update to(node->solver.cg.U[OFFSET_LEFT*size:LENGTH_LEFT*size]) device(node->prev->id) __VA_ARGS__))
#define SEND_CG_PREV(node,u,...) \
    CG_PREV_DEV_2_HOST(node,NODE_SIZE(node,0),u) \
    CG_PREV_HOST_2_DEV(node,NODE_SIZE(node,0),u,__VA_ARGS__)

// Defining macros for updating boundary values to the next GPU
#define UF_NEXT_DEV_2_HOST(node,size,U,...) _Pragma(STRINGIFY(omp target update from(node->U[*(node->dims.ndof)-OFFSET_RIGHT*size:LENGTH_RIGHT*size]) device(node->id) __VA_ARGS__))
#define UF_NEXT_HOST_2_DEV(node,size,U,...) _Pragma(STRINGIFY(omp target update to(node->U[*(node->dims.ndof)-OFFSET_RIGHT*size:LENGTH_RIGHT*size]) device(node->next->id) __VA_ARGS__))
#define SEND_UF_NEXT(node,u,...) \
    UF_NEXT_DEV_2_HOST(node,NODE_SIZE(node,0),u) \
    UF_NEXT_HOST_2_DEV(node,NODE_SIZE(node,0),u,__VA_ARGS__)

// Defining macros for updating boundary values to the previous GPU
#define UF_PREV_DEV_2_HOST(node,size,U,...) _Pragma(STRINGIFY(omp target update from(node->U[OFFSET_LEFT*size:LENGTH_LEFT*size]) device(node->id) __VA_ARGS__))
#define UF_PREV_HOST_2_DEV(node,size,U,...) _Pragma(STRINGIFY(omp target update to(node->U[OFFSET_LEFT*size:LENGTH_LEFT*size]) device(node->prev->id) __VA_ARGS__))
#define SEND_UF_PREV(node,u,...) \
    UF_PREV_DEV_2_HOST(node,NODE_SIZE(node,0),u) \
    UF_PREV_HOST_2_DEV(node,NODE_SIZE(node,0),u,__VA_ARGS__)

// Defining macro to update from device to host
#define UPDATE_MG_FROM(node,U,L) _Pragma(STRINGIFY(omp target update from(node->solver.mg[L].U[*(node->dims.offset+L):*(node->dims.length+L)]) device(node->id)));

// Defining macro to update host to device
#define UPDATE_MG_TO(node,U,L) _Pragma(STRINGIFY(omp target update to(node->solver.mg[L].U[:*(node->dims.ndof+L)]) device(node->id)))

// Defining macro to update from device to host
#define UPDATE_CG_FROM(node,U) _Pragma(STRINGIFY(omp target update from(node->solver.cg.U[*(node->dims.offset):*(node->dims.length)]) device(node->id)));

// Defining macro to update host to device
#define UPDATE_CG_TO(node,U) _Pragma(STRINGIFY(omp target update to(node->solver.cg.U[:*(node->dims.ndof)]) device(node->id)))

// Since x is smaller than all other arrays it needs a separate function
#define UPDATE_X_TO(node,X) _Pragma(STRINGIFY(omp target update to(node->X[:node->design_size]) device(node->id)))

#define X_SIZE(node) (*(node->dims.wrapy)-1)*(*(node->dims.wrapz)-1)

#define UPDATE_X_FROM(node,X) _Pragma(STRINGIFY(omp target update from(node->X[(node->prev!=NULL)*X_SIZE(node):node->design_size-(node->prev!=NULL)*X_SIZE(node)-(node->next!=NULL)*X_SIZE(node)]) device(node->id)))

#define UPDATE_U_TO(node) _Pragma(STRINGIFY(omp target update to(node->U[:*(node->dims.ndof)]) device(node->id)))

#define UPDATE_F_TO(node) _Pragma(STRINGIFY(omp target update to(node->F[:*(node->dims.ndof)]) device(node->id)))

#define UPDATE_U_FROM(node) _Pragma(STRINGIFY(omp target update from(node->U[*(node->dims.offset):*(node->dims.length)]) device(node->id)));

#define UPDATE_F_FROM(node) _Pragma(STRINGIFY(omp target update from(node->F[*(node->dims.offset):*(node->dims.length)]) device(node->id)));

#define UPDATE_CRSMATRIX_TO(node,L) _Pragma(STRINGIFY(omp target update to(node->solver.coarseMatrices[L].vals[:node->solver.coarseMatrices[L].nnz]) device(node->id)))

// Defining a taskgroup loop
#define TASK_FIRSTPRIVATE(...) _Pragma(STRINGIFY(omp task firstprivate(__VA_ARGS__)))
#define TASKGROUP_LOOP(todo) _Pragma(STRINGIFY(omp taskgroup)) \
    for (int it=0;it<gpu_grid->num_targets;it++){ \
    gpuNode * node = &(gpu_grid->targets[it]); \
    todo \
    }

#define TASKGROUP_LOOP_TASK(todo)\
    TASKGROUP_LOOP(\
        TASK_FIRSTPRIVATE(node)\
        {\
            todo\
        }\
    )

// Defining a single region in a parallel region to spawn tasks
#define PARALLEL_SINGLE(todo) _Pragma(STRINGIFY(omp parallel num_threads(TASKING_THREADS))) \
    _Pragma(STRINGIFY(omp single nowait)) \
    {\
    todo\
    }

gpuGrid init_gpu_grid(struct gridContext * gc,const int nl, const int num_devices,
                    DTYPE *x,DTYPE *xtmp,DTYPE *xnew,DTYPE *dv,DTYPE *dc,
                    CTYPE *F,STYPE *U,solver_ptrs * solver);

void free_gpu_grid(gpuGrid * gpu_grid);

void target_enter_data(gpuGrid * gpu_grid,const int nl);

void target_exit_data(gpuGrid * gpu_grid,const int nl);

void init_z_to_zero(gpuNode * node,const int l);

void r_minus_d(gpuNode * node,const int l);

void send_buffer_to_prev_CTYPE(gpuNode * node,CTYPE * d, const int l);

void send_buffer_to_next_CTYPE(gpuNode * node,CTYPE * d, const int l);

void sum_west_halo_CTYPE(gpuNode * node,CTYPE * d, const int l);

void sum_east_halo_CTYPE(gpuNode * node,CTYPE * d, const int l);

void send_buffer_to_prev_MTYPE(gpuNode * node,MTYPE * d, const int l);

void send_buffer_to_next_MTYPE(gpuNode * node,MTYPE * d, const int l);

void sum_west_halo_MTYPE(gpuNode * node,MTYPE * d, const int l);

void sum_east_halo_MTYPE(gpuNode * node,MTYPE * d, const int l);

#endif
