#ifndef PROBLONGATE_H
#define PROLONGATE_H
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <stdio.h>

using namespace std;

void prolongate(double ***phi, double ***aux, int n, int level, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    // prolongate from coarse to fine grid
    int n_coarse = (n - 1) / 2 + 1;
    int level_coarse = level + 1;
    int i_fine, j_fine;
    // initialize correction to zero, in case
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            aux[level][i][j] = 0.0;
        }
    }
    int jr, jc;
    if (rank == 0)
    {
        jr = 0;
        jc = 0;
    }
    else if (rank == 1)
    {
        jr = 0;
        jc = n_coarse - 1;
    }
    else if (rank == 2)
    {
        jr = n_coarse - 1;
        jc = n_coarse - 1;
    }
    else
    {
        jr = n_coarse - 1;
        jc = 0;
    }
    // perform full weight prolongation for interior point
    for (int i = 0; i < n_coarse; i++)
    {
        for (int j = 0; j < n_coarse; j++)
        {
            if (i == jr || j == jc)
            {
                continue;
            }

            //! Calculate the indices on the fine mesh for clarity

            i_fine = i * 2;
            j_fine = j * 2;

            //! Perform the prolongation operation by copying the value for
            //! a coincident node on the fine mesh and also incrementing the
            //! values for the neighbors.

            if (i_fine - 1 >= 0 && j_fine + 1 < n)
            {
                aux[level][i_fine - 1][j_fine + 1] += phi[level_coarse][i][j] * (1.0 / 4.0);
            }
            if (j_fine + 1 < n)
            {
                aux[level][i_fine][j_fine + 1] += phi[level_coarse][i][j] * (1.0 / 2.0);
            }
            if (i_fine + 1 < n && j_fine + 1 < n)
            {
                aux[level][i_fine + 1][j_fine + 1] += phi[level_coarse][i][j] * (1.0 / 4.0);
            }
            if (i_fine - 1 >= 0)
            {
                aux[level][i_fine - 1][j_fine] += phi[level_coarse][i][j] * (1.0 / 2.0);
            }

            aux[level][i_fine][j_fine] = phi[level_coarse][i][j];
            if (i_fine + 1 < n)
            {
                aux[level][i_fine + 1][j_fine] += phi[level_coarse][i][j] * (1.0 / 2.0);
            }
            if (i_fine - 1 >= 0 && j_fine - 1 >= 0)
            {
                aux[level][i_fine - 1][j_fine - 1] += phi[level_coarse][i][j] * (1.0 / 4.0);
            }
            if (j_fine - 1 >= 0)
            {
                aux[level][i_fine][j_fine - 1] += phi[level_coarse][i][j] * (1.0 / 2.0);
            }
            if (i_fine + 1 < n && j_fine - 1 >= 0)
            {
                aux[level][i_fine + 1][j_fine - 1] += phi[level_coarse][i][j] * (1.0 / 4.0);
            }
        }
    }
    // add the correction to fine grid
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            phi[level][i][j] += aux[level][i][j];
        }
    }
}

#endif