#include "utility.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <stdio.h>
using namespace std;

const int FINE_MESH = 0;
// definition of multigrid cycles,for best coarsening, num of nodes n should be in the form 2^k*(5-1)+1
#define NUM_NODES 17
#define MG_CYCLES 1
// number of sweeps in jacobi smoother
#define NUM_SWEEP 10

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm comm = MPI_COMM_WORLD;
    int p; // number of nodes/processes we have.
    MPI_Comm_size(comm, &p);
    if (p != 4)
    {
        printf("number of processes must be 4");
        abort();
    }
    int n_mgcycles = MG_CYCLES, n_levels, n_sweeps = NUM_SWEEP;
    int i_mgcycles = 0, n_nodes = NUM_NODES;
    double residual_0, residual;
    // check if we have custome defined number of nodes
    if (argc == 2)
    {
        n_nodes = atoi(argv[1]);
    }
    // check if we have custome defind numebr of mg_cycles
    if (argc == 3)
    {
        n_nodes = atoi(argv[1]);
        n_mgcycles = atoi(argv[2]);
    }
    int n = (n_nodes - 1) / 2 + 1;
    // allocate arrays we need for the solution
    double ***phi, ***f, ***aux;
    n_levels = allocate_arrays(&phi, &f, &aux, n);
    int jr, jc;
    if (rank == 0)
    {
        jr = 0;
        jc = 0;
    }
    else if (rank == 1)
    {
        jr = 0;
        jc = n - 1;
    }
    else if (rank == 2)
    {
        jr = n - 1;
        jc = n - 1;
    }
    else
    {
        jr = n - 1;
        jc = 0;
    }
    // initialize the solution according to rank
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            phi[0][i][j] = 0.;
            if (i != jr && j != jc)
            {
                f[0][i][j] = 1;
            }
        }
    }
    // compute initial value of the residual before any smoothing
    residual_0 = compute_residual(phi[0], aux[0], f[0], n, comm);
    MPI_Barrier(comm);
    for (i_mgcycles = 1; i_mgcycles <= n_mgcycles; i_mgcycles++)
    {
        multigrid_cycle(phi, f, aux, n, n_sweeps, n_levels, 0, comm);
        MPI_Barrier(comm);
    }
    // compute residual to report
    residual = compute_residual(phi[0], aux[0], f[0], n, comm);
    deallocate_arrays(phi, f, aux, n, n_levels);
    // print
    if (rank == 0)
    {
        printf("with %d nodes,original error is %f,after %d multicycles with %d sweeps, the error is %f\n", n_nodes,
               residual_0, n_mgcycles, n_sweeps, residual);
    }
    MPI_Finalize();
}