#ifndef UTILITY_H
#define UTILITY_H

#include "jacobi.h"
#include "prolongate.h"
#include "restrict.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <stdio.h>
using namespace std;
int allocate_arrays(double ****phi, double ****f, double ****aux, int n_nodes)
{
    int i_max, j_max;
    double ***phi_temp, ***aux_temp, ***f_temp;
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
    phi_temp = new double **[n_levels];
    aux_temp = new double **[n_levels];
    f_temp = new double **[n_levels];
    // allocate for arrays in x,y direction.
    nodes = n_nodes;
    for (int i_level = 0; i_level < n_levels; i_level++)
    {

        //! Allocate space for the i dimension
        phi_temp[i_level] = new double *[nodes];
        aux_temp[i_level] = new double *[nodes];
        f_temp[i_level] = new double *[nodes];

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
void deallocate_arrays(double ***phi, double ***f, double ***aux, int n_nodes, int n_levels)
{
    int nodes = n_nodes;
    for (int i_level = 0; i_level < n_levels; i_level++)
    {
        // delete j dimension
        for (int i = 0; i < nodes; i++)
        {
            delete[] phi[i_level][i];
            delete[] aux[i_level][i];
            delete[] f[i_level][i];
        }
        // delete i dimension
        delete[] phi[i_level];
        delete[] aux[i_level];
        delete[] f[i_level];
        nodes = (nodes - 1) / 2 + 1;
    }
    // delete the levels dimension
    delete[] phi;
    delete[] aux;
    delete[] f;
}
void multigrid_cycle(double ***phi, double ***f, double ***aux, int n_nodes, int n_sweeps, int n_levels, int level,
                     MPI_Comm comm)
{
    jacobi(phi[level], aux[level], f[level], n_nodes, n_sweeps, comm);
    MPI_Barrier(comm);
    // do multigrid steps until we have been at the coarsest level
    if (level < n_levels - 1)
    {
        restrict(phi, f, aux, n_nodes, level, comm);
        MPI_Barrier(comm);
        int n_coarse = (n_nodes - 1) / 2 + 1;
        int level_coarse = level + 1;
        multigrid_cycle(phi, f, aux, n_coarse, n_sweeps, n_levels, level_coarse, comm);
        MPI_Barrier(comm);
        prolongate(phi, aux, n_nodes, level, comm);
        MPI_Barrier(comm);
    }
    jacobi(phi[level], aux[level], f[level], n_nodes, n_sweeps, comm);
    MPI_Barrier(comm);
}
#endif