#include <iostream>
#include <cmath>
#include <chrono>
#include "utils.h"
using namespace std;
// define the finest mesh level for readbility
const int FINE_MESH = 0;

/******************************************************************************/
/* Set the variable parameters for the simulation here. Note that             */
/* the number of nodes and levels of multigrid can also be set                */
/* at the command line.                                                       */
/******************************************************************************/
// number of nodes in each direction, we choose to have odd nodes
// so that after divide into 4 blocks using MPI the coarse operation remains the same.
// definition of multigrid cycles,for best coarsening, num of nodes n should be in the form 2^k*(5-1)+1
#define NUM_NODES 17
#define MG_CYCLES 1
// number of sweeps in jacobi smoother
#define NUM_SWEEP 3

#define TOLERANCE 10
//! Dynamically allocates arrays needed for the life of the solver
int allocate_arrays(double**** phi, double**** f, double**** aux, int n_nodes);
//free memory
void deallocate_arrays(double*** phi, double*** f, double*** aux, int n_nodes, int n_levels);
//! Initializes the values for phi and the f term on the fine mesh
void intitialize_solution(double** phi, double** f, int n_nodes);

//! Recursive function for completing a multigrid V-cycle
void multigrid_cycle(double*** phi, double*** f, double*** aux, int n_nodes, int n_sweeps, int n_levels, int level);
//! Smooth the linear system using the Jacobi method
void smooth_jacobi(double** phi, double** f, double** aux, int n_nodes, int n_sweeps);
// full weight restrict
void restrict_weighted(double*** phi, double*** f, double*** aux, int n_nodes, int level);
// prolongation
void prolongate_weighted(double*** phi, double*** aux, int n_nodes, int level);
// compute the residual
double compute_residual(double** phi, double** f, double** residual, int n_nodes);

int allocate_arrays(double**** phi, double**** f,
    double**** aux, int n_nodes)
{
    // phi is the value we want to solve for at each level, f is the value right side
    // x,y are just coordinates at each level, aux is the auxilarry variable, n_ nodes is
    // number of nodes at finest level, level 0.
    int i_max, j_max;
    double*** phi_temp, *** aux_temp, *** f_temp;
    bool coarsen = true;
    int n_levels = 1;
    int nodes = n_nodes;
    // compute the number of levels of coarsen we want to perform
    while (coarsen)
    { // make sure each level has odd number of nodes
        if ((nodes - 1) % 2 == 0 && (nodes - 1) / 2 + 1 >= 5)
        {
            nodes = (nodes - 1) / 2 + 1;
            n_levels++;
        }
        else
        {
            coarsen = false;
        }
    }
    // allocate for different levels first
    phi_temp = new double** [n_levels];
    aux_temp = new double** [n_levels];
    f_temp = new double** [n_levels];
    // allocate for arrays in x,y direction.
    nodes = n_nodes;
    for (int i_level = 0; i_level < n_levels; i_level++)
    {

        //! Allocate space for the i dimension
        phi_temp[i_level] = new double* [nodes];
        aux_temp[i_level] = new double* [nodes];
        f_temp[i_level] = new double* [nodes];

        //! Allocate space for the j dimension

        for (int i = 0; i < nodes; i++)
        {
            phi_temp[i_level][i] = new double[nodes];
            aux_temp[i_level][i] = new double[nodes];
            f_temp[i_level][i] = new double[nodes];
        }

        //! Compute number of nodes on the next coarse level

        nodes = (nodes - 1) / 2 + 1;
    }
    //! Set the pointers to the correct memory for use outside the function
    *phi = phi_temp;
    *aux = aux_temp;
    *f = f_temp;
    return n_levels;
}
void deallocate_arrays(double*** phi, double*** f, double*** aux, int n_nodes, int n_levels)
{
    int nodes = n_nodes;
    for (int i_level = 0;i_level < n_levels; i_level++)
    {
        //delete j dimension
        for (int i = 0;i < nodes;i++)
        {
            delete[] phi[i_level][i];
            delete[] aux[i_level][i];
            delete[] f[i_level][i];
        }
        //delete i dimension
        delete[] phi[i_level];
        delete[] aux[i_level];
        delete[] f[i_level];
        nodes = (nodes - 1) / 2 + 1;

    }
    //delete the levels dimension
    delete[] phi;
    delete[] aux;
    delete[] f;
}
void initialize_solution(double** phi, double** f, int n_nodes)
{
    // set boundary condition and initial guess for phi and f
    for (int i = 0; i < n_nodes; i++)
    {
        for (int j = 0; j < n_nodes; j++)
        {
            if (i == 0 || j == 0 || i == n_nodes - 1 || j == n_nodes - 1)
            {
                // boundarycondition goes here
                phi[i][j] = 0;
            }
            else
            {
                // initial guess and value of f goes here
                phi[i][j] = 0;
                f[i][j] = 1;
            }
        }
    }
    printf("Solution initialized.\n");
}
void smooth_jacobi(double** phi, double** f, double** aux, int n_nodes, int n_sweeps)
{
    // we are solving in (0,1) square, calculate h^2 here
    double h2 = pow(1.0 / ((double)n_nodes - 1.0), 2.0);
    // perform jacobi iteration here, we store new value of each iteration in aux
    for (int iter = 0; iter < n_sweeps; iter++)
    {
        for (int i = 1; i < n_nodes - 1; i++)
        {
            for (int j = 1; j < n_nodes - 1; j++)
            {
                aux[i][j] = (phi[i][j - 1] + phi[i - 1][j] + phi[i + 1][j] + phi[i][j + 1] + h2 * f[i][j]) / 4.0;
            }
        }
        // store the new value from aux into phi
        for (int i = 1; i < n_nodes - 1; i++)
        {
            for (int j = 1; j < n_nodes - 1; j++)
            {
                phi[i][j] = aux[i][j];
            }
        }
    }
}
double compute_residual(double** phi, double** f, double** residual, int n_nodes)
{
    // compute the residual which is f-A*phi,so we can use it as the f in next level
    // and return the 2 norm
    double norm;
    double h2 = pow(1.0 / ((double)n_nodes - 1.0), 2.0);
    for (int i = 1; i < n_nodes - 1; i++)
    {
        for (int j = 1; j < n_nodes - 1; j++)
        {
            residual[i][j] = f[i][j] + (phi[i][j - 1] + phi[i - 1][j] + phi[i + 1][j] + phi[i][j + 1]
                - 4.0 * phi[i][j]) / h2;

            norm += residual[i][j] * residual[i][j];
        }
    }
    // NOTE: when using, the residual is always stored in the aux variable.
    norm = sqrt(norm);
    return norm;
}

void restrict_weighted(double*** phi, double*** f, double*** aux, int n_nodes, int level)
{
    // restriction, which means we transfer from a fine grid to a coarser grid
    // we restrict the residual, so compute residual first
    compute_residual(phi[level], f[level], aux[level], n_nodes);
    // compute information of coarse level, i.e. the next level
    int n_coarse = (n_nodes - 1) / (2) + 1;
    int level_coarse = level + 1;
    int i_fine, j_fine;
    // initialize phi and f on next level
    for (int i = 0; i < n_coarse; i++)
    {
        for (int j = 0; j < n_coarse; j++)
        {
            phi[level_coarse][i][j] = 0.0;
            f[level_coarse][i][j] = 0.0;
        }
    }
    // perform restrict in full weight for all interior grids
    for (int i = 1; i < n_coarse - 1; i++)
    {
        for (int j = 1; j < n_coarse - 1; j++)
        {

            //! Calculate the indices on the fine mesh for clarity

            i_fine = (i * 2);
            j_fine = (j * 2);

            //! Perform the restriction operation for this node by injection
            f[level_coarse][i][j] = aux[level][i_fine - 1][j_fine + 1] * (1.0 / 16.0)
                + aux[level][i_fine][j_fine + 1] * (1.0 / 8.0)
                + aux[level][i_fine + 1][j_fine + 1] * (1.0 / 16.0)
                + aux[level][i_fine - 1][j_fine] * (1.0 / 8.0)
                + aux[level][i_fine][j_fine] * (1.0 / 4.0)
                + aux[level][i_fine + 1][j_fine] * (1.0 / 8.0)
                + aux[level][i_fine - 1][j_fine - 1] * (1.0 / 16.0)
                + aux[level][i_fine][j_fine - 1] * (1.0 / 8.0)
                + aux[level][i_fine + 1][j_fine - 1] * (1.0 / 16.0);
        }
    }
}

void prolongate_weighted(double*** phi, double*** aux, int n_nodes, int level)
{
    //prolongate from coarse to fine grid
    int n_coarse = (n_nodes - 1) / (2) + 1;
    int level_coarse = level + 1;
    int i_fine, j_fine;
    //initialize correction to zero
    for (int i = 0; i < n_nodes; i++) {
        for (int j = 0; j < n_nodes; j++) {
            aux[level][i][j] = 0.0;
        }
    }
    //perform full weight prolongation for interior point
    for (int i = 1; i < n_coarse - 1; i++) {
        for (int j = 1; j < n_coarse - 1; j++) {

            //! Calculate the indices on the fine mesh for clarity

            i_fine = i * 2; j_fine = j * 2;

            //! Perform the prolongation operation by copying the value for
            //! a coincident node on the fine mesh and also incrementing the
            //! values for the neighbors.

            aux[level][i_fine - 1][j_fine + 1] += phi[level_coarse][i][j] * (1.0 / 4.0);
            aux[level][i_fine][j_fine + 1] += phi[level_coarse][i][j] * (1.0 / 2.0);
            aux[level][i_fine + 1][j_fine + 1] += phi[level_coarse][i][j] * (1.0 / 4.0);
            aux[level][i_fine - 1][j_fine] += phi[level_coarse][i][j] * (1.0 / 2.0);
            aux[level][i_fine][j_fine] = phi[level_coarse][i][j];
            aux[level][i_fine + 1][j_fine] += phi[level_coarse][i][j] * (1.0 / 2.0);
            aux[level][i_fine - 1][j_fine - 1] += phi[level_coarse][i][j] * (1.0 / 4.0);
            aux[level][i_fine][j_fine - 1] += phi[level_coarse][i][j] * (1.0 / 2.0);
            aux[level][i_fine + 1][j_fine - 1] += phi[level_coarse][i][j] * (1.0 / 4.0);

        }
    }
    //add the correction to fine grid
    for (int i = 0; i < n_nodes; i++) {
        for (int j = 0; j < n_nodes; j++) {
            phi[level][i][j] += aux[level][i][j];
        }
    }
}

void multigrid_cycle(double*** phi, double*** f, double*** aux, int n_nodes, int n_sweeps, int n_levels, int level)
{
    //do presmooth first
    smooth_jacobi(phi[level], f[level], aux[level], n_nodes, n_sweeps);
    //do multigrid steps until we have been at the coarsest level
    if (level < n_levels - 1) {
        //restrict first
        restrict_weighted(phi, f, aux, n_nodes, level);
        //get information at coarse level
        int n_coarse = (n_nodes - 1) / 2 + 1;
        int level_coarse = level + 1;
        //recursively call multigrid
        multigrid_cycle(phi, f, aux, n_coarse, n_sweeps, n_levels, level_coarse);
        //prolongation then
        prolongate_weighted(phi, aux, n_nodes, level);
    }
    //post smooth
    smooth_jacobi(phi[level], f[level], aux[level], n_nodes, n_sweeps);
}

int main(int argc, char* argv[])
{
    int n_mgcycles = MG_CYCLES, n_levels, n_sweeps = NUM_SWEEP;
    int i_mgcycles = 0, n_nodes = NUM_NODES;
    double  residual_0, residual;

    //check if we have custome defined number of nodes
    if (argc == 2) {
        n_nodes = atoi(argv[1]);
    }
    //check if we have custome defind numebr of mg_cycles
    if (argc == 3) {
        n_nodes = atoi(argv[1]);
        n_mgcycles = atoi(argv[2]);
    }
    //allocate arrays we need for the solution
    double*** phi, *** f, *** aux;
    n_levels = allocate_arrays(&phi, &f, &aux, n_nodes);
    //initialize the problem at the arrays
    initialize_solution(phi[FINE_MESH], f[FINE_MESH], n_nodes);
    //compute the initial value of the residual before any smoothing
    residual_0 = compute_residual(phi[FINE_MESH], f[FINE_MESH], aux[FINE_MESH], n_nodes);
    //begin count time
    Timer t;
    t.tic();
    for (i_mgcycles = 1;i_mgcycles <= n_mgcycles;i_mgcycles++)
    {
        //call the recursive multigrid
        multigrid_cycle(phi, f, aux, n_nodes, n_sweeps, n_levels, FINE_MESH);

    }
    //timer ends here
    double timespent = t.toc();
    //compute residual to report
    residual = compute_residual(phi[FINE_MESH], f[FINE_MESH], aux[FINE_MESH], n_nodes);
    //free all memory
    deallocate_arrays(phi, f, aux, n_nodes, n_levels);
    //print information
    printf("with %d nodes,original error is %f,after %d multicycles with %d sweeps, the error is %f\n", n_nodes
        , residual_0, n_mgcycles, n_sweeps, residual);
    printf("time spent: %f\n", timespent);
}