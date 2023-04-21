#include "include/definitions.h"

#include "include/eightColor_methods.h"
#include "include/local_matrix.h"
#include "include/stencil_grid_utility.h"
#include "include/stencil_methods.h"

#include "include/stencil_assembly.h"
#include "include/stencil_optimization.h"
#include "include/stencil_solvers.h"

int main(int argc, char *argv[]) {
  if (argc < 4){
    printf("Please provide arguments \n\t%s nelx nely nelz <verbose:0|1> <write_file:0|1> <max_iterations> <num_layers>\n",argv[0]);
    return 0; 
  }
  int verbose = 0;
  int write_result = 0;
  int max_iterations = 25;
  const int nelx = atoi(argv[1]);
  const int nely = atoi(argv[2]);
  const int nelz = atoi(argv[3]);
  if (argc >= 5){
    verbose = atoi(argv[4]);
  }
  if (argc >= 6){
    write_result = atoi(argv[5]);
  }
  if (argc >= 7){
    max_iterations = atoi(argv[6]);
  }
  int nl = number_of_matrix_free_levels+2;
  if (argc >= 8){
    nl = atoi(argv[7]);
  }
  const float volfrac = 0.12;
  const float rmin = 1.5;
  float cgtol = 1e-4;
  if (argc >= 9){
    cgtol = atof(argv[8]);
  }
  const int cgmax = 200;

  top3dmgcg(nelx, nely, nelz, volfrac, rmin, nl, cgtol, cgmax,verbose,write_result,max_iterations);
  return EXIT_SUCCESS;
}
